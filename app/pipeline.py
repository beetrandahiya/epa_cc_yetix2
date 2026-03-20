from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from app.anomaly import detect_anomalies
from app.config import load_settings
from app.enterprise_processing import normalize_nursing_notes, process_pdf_inbox
from app.ingestion import (
    attach_case_fk,
    build_case_table,
    drop_missing_mandatory,
    load_device_1hz,
    load_device_motion,
    load_epa_data_1,
    load_epa_data_2,
    load_epa_data_3,
    load_icd_ops,
    load_labs_data,
    load_medication,
    load_nursing,
    merge_epa_sources,
    standardize_target_columns,
)
from app.mappings import item_name_to_iid_lookup, load_iid_sid_map, sid_to_iid_lookup
from app.quality import compute_completeness, detect_data_quality_issues
from app.source_discovery import discover_data_files
from app.utils import read_csv_flexible


def _file_signature(paths: list[str]) -> str:
    payload = []
    for p in paths:
        path = Path(p)
        if path.exists():
            stat = path.stat()
            payload.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
        else:
            payload.append(f"{path}:missing")
    joined = "|".join(payload)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _pdf_folder_signature(pdf_dir: str) -> str:
    folder = Path(pdf_dir)
    if not folder.exists():
        return "pdf_dir_missing"
    payload = []
    for file in sorted(folder.glob("*.pdf")):
        stat = file.stat()
        payload.append(f"{file.name}:{stat.st_size}:{stat.st_mtime_ns}")
    raw = "|".join(payload) if payload else "pdf_dir_empty"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_or_none(loader, *args):
    try:
        return loader(*args)
    except FileNotFoundError:
        return pd.DataFrame()


def _ensure_processed_dir(processed_root: str) -> Path:
    out = Path(processed_root)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_parquet_tables(processed_root: Path, tables: dict[str, pd.DataFrame]) -> None:
    for name, table in tables.items():
        table.to_parquet(processed_root / f"{name}.parquet", index=False)


def _persist_to_duckdb(duckdb_file: str, tables: dict[str, pd.DataFrame]) -> None:
    conn = duckdb.connect(duckdb_file)
    try:
        for name, df in tables.items():
            conn.register("tmp_df", df)
            conn.execute(f"create or replace table {name} as select * from tmp_df")
            conn.unregister("tmp_df")
    finally:
        conn.close()


def _duckdb_tables_exist(duckdb_file: str, table_names: set[str]) -> bool:
    path = Path(duckdb_file)
    if not path.exists():
        return False
    try:
        conn = duckdb.connect(duckdb_file, read_only=True)
    except Exception:
        return False
    try:
        rows = conn.execute(
            """
            select table_name
            from information_schema.tables
            where table_name in ({})
            """.format(
                ",".join(["?"] * len(table_names))
            ),
            list(table_names),
        ).fetchdf()
        existing = set(rows["table_name"].tolist()) if not rows.empty else set()
        return table_names.issubset(existing)
    except Exception:
        return False
    finally:
        conn.close()


def _postprocess_import_table(df: pd.DataFrame, case_table: pd.DataFrame) -> pd.DataFrame:
    df = standardize_target_columns(df)
    df = attach_case_fk(df, case_table)
    if "coId" not in df.columns:
        df = df.reset_index(drop=True)
        df.insert(0, "coId", df.index + 1)
    return df


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty and len(df.columns) == 0:
        return pd.DataFrame(columns=columns)
    return df


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        frame = frame.loc[:, ~frame.columns.duplicated()]
        valid.append(frame)
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True, sort=False)


def _build_auto_lineage(source_table: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["source_table", "source_field", "target_field", "rule", "source_file"])
    src_file_col = "source_file" if "source_file" in df.columns else None
    rows = []
    excluded = {"coid", "cocaseid", "case_id", "patient_id", "source_file"}
    target_cols = [c for c in df.columns if c.lower() not in excluded]
    target_cols = [
        c
        for c in target_cols
        if not re.match(r"^(col\d+|unnamed_\d+|extra_\d+)$", str(c).lower())
    ]
    files = df[src_file_col].dropna().astype(str).unique().tolist() if src_file_col else ["unknown"]
    for sf in files:
        for col in target_cols:
            rows.append(
                {
                    "source_table": source_table,
                    "source_field": col,
                    "target_field": col,
                    "rule": "standardized multi-source mapping",
                    "source_file": sf,
                }
            )
    return pd.DataFrame(rows)


def _drop_placeholder_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keep_always = {"coId", "coCaseId", "case_id", "patient_id", "source_file", "coSource_file"}
    drop_cols = []
    for col in df.columns:
        col_l = str(col).lower()
        if col in keep_always:
            continue
        if re.match(r"^(col\d+|unnamed_\d+|extra_\d+)$", col_l):
            drop_cols.append(col)
    if drop_cols:
        return df.drop(columns=drop_cols, errors="ignore")
    return df


def _enforce_machine_readable_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    always_keep = {
        "coId",
        "coCaseId",
        "case_id",
        "patient_id",
        "source_file",
        "coSource_file",
    }
    allowed_lowercase = {
        "prescriber_role",
        "order_status",
        "administration_datetime",
        "administered_dose",
        "administered_unit",
        "administration_status",
        "note",
    }

    keep_cols = []
    for col in df.columns:
        col_str = str(col)
        if col_str in always_keep:
            keep_cols.append(col)
            continue
        if col_str.lower() in {c.lower() for c in allowed_lowercase}:
            keep_cols.append(col)
            continue
        if col_str.startswith("co"):
            keep_cols.append(col)

    keep_cols = [c for c in keep_cols if c in df.columns]
    if not keep_cols:
        return df
    return df[keep_cols].copy()


def _sha256_for_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    stat = path.stat()
    payload = f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_bronze_raw_blob(discovered: dict[str, list[str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain, files in discovered.items():
        for file in files:
            path = Path(file)
            sample_json = "[]"
            row_count = 0
            error_message = None
            try:
                if path.suffix.lower() == ".csv":
                    raw = read_csv_flexible(str(path))
                    row_count = int(len(raw))
                    sample = raw.head(50)
                    sample_json = json.dumps(sample.to_dict(orient="records"), ensure_ascii=False, default=str)
                else:
                    sample_json = json.dumps([{"file": path.name, "note": "non-csv raw blob"}], ensure_ascii=False)
            except Exception as ex:
                error_message = str(ex)

            rows.append(
                {
                    "layer": "bronze",
                    "source_domain": domain,
                    "source_file": path.name,
                    "source_path": str(path),
                    "lineage_hash": _sha256_for_file(path),
                    "raw_row_count": row_count,
                    "raw_blob_sample_json": sample_json,
                    "error_message": error_message,
                }
            )

    return pd.DataFrame(rows)


def _build_gold_case_analytics(case_table: pd.DataFrame, import_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if case_table.empty:
        return pd.DataFrame(columns=["coCaseId", "case_id", "patient_id"])

    out = case_table[["coId", "case_id", "patient_id"]].copy()
    out = out.rename(columns={"coId": "coCaseId"})

    def add_count(df: pd.DataFrame, out_col: str) -> None:
        if df.empty or "coCaseId" not in df.columns:
            out[out_col] = 0
            return
        counts = (
            pd.to_numeric(df["coCaseId"], errors="coerce")
            .dropna()
            .astype(int)
            .value_counts()
            .rename_axis("coCaseId")
            .reset_index(name=out_col)
        )
        merged = out.merge(counts, on="coCaseId", how="left")
        out[out_col] = merged[out_col]
        out[out_col] = out[out_col].fillna(0).astype(int)

    add_count(import_tables.get("tbImportAcData", pd.DataFrame()), "assessment_count")
    add_count(import_tables.get("tbImportLabsData", pd.DataFrame()), "lab_count")
    add_count(import_tables.get("tbImportNursingDailyReportsData", pd.DataFrame()), "nursing_count")
    add_count(import_tables.get("tbImportMedicationInpatientData", pd.DataFrame()), "medication_count")
    add_count(import_tables.get("tbImportDeviceMotionData", pd.DataFrame()), "device_motion_count")
    add_count(import_tables.get("tbImportDevice1HzMotionData", pd.DataFrame()), "device_1hz_count")
    add_count(import_tables.get("tbImportIcd10Data", pd.DataFrame()), "icd_count")

    count_cols = [c for c in out.columns if c.endswith("_count")]
    out["total_events"] = out[count_cols].sum(axis=1)
    out["is_sparse_case"] = out["total_events"] < 3
    return out


def _build_data_contract_results(
    bronze_raw_blob: pd.DataFrame,
    tb_case_data: pd.DataFrame,
    import_tables: dict[str, pd.DataFrame],
    gold_case_analytics: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    def push(layer: str, contract_name: str, target: str, passed: bool, details: str, metric: float | None = None) -> None:
        checks.append(
            {
                "layer": layer,
                "contract_name": contract_name,
                "target_object": target,
                "status": "pass" if passed else "fail",
                "details": details,
                "metric_value": metric,
                "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    bronze_required = {"source_domain", "source_file", "source_path", "lineage_hash", "raw_blob_sample_json"}
    push(
        "bronze",
        "required_columns",
        "bronze_raw_blob",
        bronze_required.issubset(set(bronze_raw_blob.columns)),
        f"required={sorted(bronze_required)}",
    )
    push("bronze", "non_empty", "bronze_raw_blob", len(bronze_raw_blob) > 0, f"rows={len(bronze_raw_blob)}", float(len(bronze_raw_blob)))

    silver_tables = {"tbCaseData": tb_case_data, **import_tables}
    for name, table in silver_tables.items():
        if name == "tbCaseData":
            required = {"coId", "coE2I222", "coPatientId"}
        else:
            required = {"coId", "coCaseId"}
        passed = required.issubset(set(table.columns))
        push("silver", "required_columns", name, passed, f"required={sorted(required)}")
        push("silver", "non_empty", name, len(table) > 0, f"rows={len(table)}", float(len(table)))

    gold_required = {"coCaseId", "case_id", "patient_id", "total_events"}
    push(
        "gold",
        "required_columns",
        "gold_case_analytics",
        gold_required.issubset(set(gold_case_analytics.columns)),
        f"required={sorted(gold_required)}",
    )
    push(
        "gold",
        "non_empty",
        "gold_case_analytics",
        len(gold_case_analytics) > 0,
        f"rows={len(gold_case_analytics)}",
        float(len(gold_case_analytics)),
    )

    return pd.DataFrame(checks)


def _build_benchmark_metrics(
    mapping_df: pd.DataFrame,
    lineage: pd.DataFrame,
    dq_completeness: pd.DataFrame,
    bronze_raw_blob: pd.DataFrame,
    pre_validation_tables: dict[str, pd.DataFrame],
    import_tables: dict[str, pd.DataFrame],
    required_fields: list[str],
    duckdb_file: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_targets = {f"CO{str(i).upper()}" for i in mapping_df["iid_clean"].dropna().astype(str).tolist()}

    lineage_eval = lineage.copy() if not lineage.empty else pd.DataFrame(columns=["source_table", "target_field"])
    if "target_field" not in lineage_eval.columns:
        lineage_eval["target_field"] = pd.Series(dtype=str)
    if "source_table" not in lineage_eval.columns:
        lineage_eval["source_table"] = pd.Series(dtype=str)

    lineage_eval["target_field"] = lineage_eval["target_field"].astype(str).str.upper()
    lineage_eval = lineage_eval[lineage_eval["source_table"].astype(str).str.contains("epa_data_", case=False, na=False)].copy()
    lineage_eval["is_mapped_to_dictionary"] = lineage_eval["target_field"].isin(valid_targets)

    mapping_total = int(len(lineage_eval))
    mapping_matched = int(lineage_eval["is_mapped_to_dictionary"].sum()) if mapping_total else 0
    mapping_accuracy = (mapping_matched / mapping_total) if mapping_total else 0.0

    mapping_detail = lineage_eval.groupby("source_table", as_index=False).agg(
        mapped_total=("is_mapped_to_dictionary", "count"),
        mapped_matched=("is_mapped_to_dictionary", "sum"),
    )
    if not mapping_detail.empty:
        mapping_detail["mapping_accuracy"] = (mapping_detail["mapped_matched"] / mapping_detail["mapped_total"]).round(4)

    before_non_null = 0.0
    if not bronze_raw_blob.empty and "raw_blob_sample_json" in bronze_raw_blob.columns:
        ratios = []
        for blob in bronze_raw_blob["raw_blob_sample_json"].dropna().astype(str).head(200):
            try:
                rows = json.loads(blob)
                if isinstance(rows, list) and rows:
                    sample_df = pd.DataFrame(rows)
                    if not sample_df.empty:
                        ratios.append(float(sample_df.notna().mean().mean()))
            except Exception:
                continue
        before_non_null = float(sum(ratios) / len(ratios)) if ratios else 0.0

    silver_tables = set(import_tables.keys()).union({"tbCaseData"})
    after_df = dq_completeness[dq_completeness["table_name"].isin(silver_tables)] if (not dq_completeness.empty and "table_name" in dq_completeness.columns) else pd.DataFrame()
    after_non_null = float(pd.to_numeric(after_df["non_null_rate"], errors="coerce").dropna().mean()) if not after_df.empty else 0.0
    field_density_delta = after_non_null - before_non_null

    required_fields_norm = [str(f).strip() for f in required_fields if str(f).strip()]

    def _avg_required_ratio(tables: dict[str, pd.DataFrame], fields: list[str]) -> tuple[float, int]:
        vals: list[float] = []
        for _, df in tables.items():
            if df is None or df.empty:
                continue
            for fld in fields:
                if fld in df.columns:
                    vals.append(float(df[fld].notna().mean()))
                else:
                    vals.append(0.0)
        return (float(sum(vals) / len(vals)) if vals else 0.0, len(vals))

    before_required, before_required_samples = _avg_required_ratio(pre_validation_tables, required_fields_norm)
    after_required, after_required_samples = _avg_required_ratio(import_tables, required_fields_norm)

    fk_vals: list[float] = []
    for _, df in import_tables.items():
        if df is None or df.empty:
            continue
        if "coCaseId" in df.columns:
            fk_vals.append(float(df["coCaseId"].notna().mean()))
    after_fk = float(sum(fk_vals) / len(fk_vals)) if fk_vals else 0.0

    key_quality_before = before_required
    key_quality_after = (after_required + after_fk) / 2.0 if fk_vals else after_required
    missingness_improvement = key_quality_after - key_quality_before
    key_quality_before_samples = int(before_required_samples)
    key_quality_after_samples = int(after_required_samples + (len(fk_vals) if fk_vals else 0))

    extraction_total = 0
    extraction_success = 0
    try:
        conn = duckdb.connect(duckdb_file)
        try:
            exists = conn.execute("select count(*) from information_schema.tables where table_name='ai_extraction_log'").fetchone()[0] > 0
            if exists:
                stats = conn.execute(
                    """
                    select count(*) as total,
                           sum(case when lower(status)='success' then 1 else 0 end) as success
                    from ai_extraction_log
                    """
                ).fetchone()
                extraction_total = int(stats[0] or 0)
                extraction_success = int(stats[1] or 0)
        finally:
            conn.close()
    except Exception:
        extraction_total = 0
        extraction_success = 0

    extraction_accuracy = (extraction_success / extraction_total) if extraction_total else 0.0

    metrics = pd.DataFrame(
        [
            {
                "metric_name": "mapping_accuracy_epa",
                "metric_value": mapping_accuracy,
                "numerator": mapping_matched,
                "denominator": mapping_total,
                "details": "EPA lineage targets aligned to IID-SID-ITEM dictionary",
            },
            {
                "metric_name": "extraction_accuracy_ai",
                "metric_value": extraction_accuracy,
                "numerator": extraction_success,
                "denominator": extraction_total,
                "details": "Success ratio from ai_extraction_log",
            },
            {
                "metric_name": "non_null_before_bronze_sample",
                "metric_value": before_non_null,
                "numerator": None,
                "denominator": None,
                "details": "Average non-null ratio from Bronze raw blob samples",
            },
            {
                "metric_name": "non_null_after_silver",
                "metric_value": after_non_null,
                "numerator": None,
                "denominator": None,
                "details": "Average non-null ratio from Silver dq_completeness",
            },
            {
                "metric_name": "field_density_delta_all_fields",
                "metric_value": field_density_delta,
                "numerator": None,
                "denominator": None,
                "details": "All-field density delta (Silver average non-null - Bronze sampled non-null)",
            },
            {
                "metric_name": "key_quality_before",
                "metric_value": key_quality_before,
                "numerator": float(key_quality_before * key_quality_before_samples),
                "denominator": key_quality_before_samples,
                "details": "Average completeness of required key fields before harmonization",
            },
            {
                "metric_name": "key_quality_after",
                "metric_value": key_quality_after,
                "numerator": float(key_quality_after * key_quality_after_samples),
                "denominator": key_quality_after_samples,
                "details": "Average completeness of required keys and coCaseId after harmonization",
            },
            {
                "metric_name": "missingness_improvement_after_harmonization",
                "metric_value": missingness_improvement,
                "numerator": None,
                "denominator": key_quality_after_samples,
                "details": "Key-field quality uplift after harmonization (required fields + coCaseId linkage)",
            },
        ]
    )

    mapping_detail = mapping_detail if not mapping_detail.empty else pd.DataFrame(
        columns=["source_table", "mapped_total", "mapped_matched", "mapping_accuracy"]
    )
    return metrics, mapping_detail


def run_pipeline(force: bool = False) -> dict[str, object]:
    pipeline_t0 = time.perf_counter()
    step_metrics: list[dict[str, object]] = []

    def _record_step(step_name: str, started_at: float, row_count: int | None = None) -> None:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        step_metrics.append(
            {
                "step_name": step_name,
                "duration_ms": elapsed_ms,
                "row_count": row_count,
                "run_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    t0 = time.perf_counter()
    cfg = load_settings()
    processed_root = _ensure_processed_dir(cfg["paths"]["processed_root"])
    duckdb_file = cfg["paths"]["duckdb_file"]
    metadata_path = processed_root / "pipeline_metadata.json"
    _record_step("load_config", t0)

    t0 = time.perf_counter()
    discovered = discover_data_files(cfg.get("input_roots", []))
    source_paths = [p for files in discovered.values() for p in files] + [cfg["paths"]["iid_sid_map_file"]]
    if not source_paths:
        source_paths = list(cfg["input_files"].values()) + [cfg["paths"]["iid_sid_map_file"]]
    signature = _file_signature(source_paths)
    pdf_signature = _pdf_folder_signature(cfg["paths"].get("pdf_inbox_dir", ""))
    signature = hashlib.sha256(f"{signature}|{pdf_signature}".encode("utf-8")).hexdigest()
    _record_step("build_source_signature", t0, row_count=len(source_paths))

    if metadata_path.exists() and not force:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("source_signature") == signature:
            required_tables = {
                "benchmark_metrics",
                "benchmark_mapping_detail",
                "data_contract_results",
                "bronze_raw_blob",
                "gold_case_analytics",
            }
            if _duckdb_tables_exist(duckdb_file, required_tables):
                print("Pipeline skipped: source files unchanged. Using cached processed tables.")
                return {"status": "skipped", "reason": "source_unchanged_and_required_tables_present"}
            print("Source unchanged, but required benchmark/contract tables missing. Running backfill build.")

    null_like_values = cfg["rules"]["null_like_values"]
    required_fields = cfg["rules"]["required_fields"]

    t0 = time.perf_counter()
    mapping_df = load_iid_sid_map(cfg["paths"]["iid_sid_map_file"])
    sid_to_iid = sid_to_iid_lookup(mapping_df)
    item_name_lookup = item_name_to_iid_lookup(mapping_df)
    _record_step("load_iid_sid_mapping", t0, row_count=len(mapping_df))

    t0 = time.perf_counter()
    epa1_frames, epa2_frames, epa3_frames = [], [], []
    lineage_frames = []

    epa1_files = discovered.get("epa1", []) or [cfg["input_files"]["epa_data_1"]]
    epa2_files = discovered.get("epa2", []) or [cfg["input_files"]["epa_data_2"]]
    epa3_files = discovered.get("epa3", []) or [cfg["input_files"]["epa_data_3"]]

    for path in epa1_files:
        df, lin = load_epa_data_1(path, sid_to_iid, null_like_values)
        epa1_frames.append(df)
        lineage_frames.append(lin)

    for path in epa2_files:
        df, lin = load_epa_data_2(path, sid_to_iid, null_like_values)
        epa2_frames.append(df)
        lineage_frames.append(lin)

    for path in epa3_files:
        df, lin = load_epa_data_3(path, null_like_values, item_name_to_iid=item_name_lookup)
        epa3_frames.append(df)
        lineage_frames.append(lin)

    epa1 = _concat_frames(epa1_frames)
    epa2 = _concat_frames(epa2_frames)
    epa3 = _concat_frames(epa3_frames)
    _record_step("load_epa_sources", t0, row_count=len(epa1) + len(epa2) + len(epa3))

    t0 = time.perf_counter()
    epa = merge_epa_sources(epa1, epa2, epa3)
    epa = standardize_target_columns(epa)
    _record_step("merge_epa_sources", t0, row_count=len(epa))

    if "coE2I222" in epa.columns and "case_id" not in epa.columns:
        epa["case_id"] = pd.to_numeric(epa["coE2I222"], errors="coerce").astype("Int64").astype(str)
        epa.loc[epa["case_id"] == "<NA>", "case_id"] = pd.NA

    if "coPatientId" in epa.columns and "patient_id" not in epa.columns:
        epa["patient_id"] = pd.to_numeric(epa["coPatientId"], errors="coerce").astype("Int64").astype(str)
        epa.loc[epa["patient_id"] == "<NA>", "patient_id"] = pd.NA

    t0 = time.perf_counter()
    labs_files = discovered.get("labs", []) or [cfg["input_files"]["labs"]]
    device_motion_files = discovered.get("device_motion", []) or [cfg["input_files"]["device_motion"]]
    device_1hz_files = discovered.get("device_1hz", []) or [cfg["input_files"]["device_1hz"]]
    medication_files = discovered.get("medication", []) or [cfg["input_files"]["medication"]]
    nursing_files = discovered.get("nursing", []) or [cfg["input_files"]["nursing"]]
    icd_files = discovered.get("icd", []) or [cfg["input_files"]["icd_ops"]]

    labs = _concat_frames([load_labs_data(p, null_like_values) for p in labs_files])
    device_motion = _concat_frames([load_device_motion(p, null_like_values) for p in device_motion_files])
    device_1hz = _concat_frames([load_device_1hz(p, null_like_values) for p in device_1hz_files])
    medication = _concat_frames([load_medication(p, null_like_values) for p in medication_files])
    nursing = _concat_frames([load_nursing(p, null_like_values) for p in nursing_files])
    icd_ops = _concat_frames([load_icd_ops(p, null_like_values) for p in icd_files])
    _record_step(
        "load_non_epa_sources",
        t0,
        row_count=len(labs) + len(device_motion) + len(device_1hz) + len(medication) + len(nursing) + len(icd_ops),
    )

    t0 = time.perf_counter()
    nursing_nlp = normalize_nursing_notes(
        nursing_df=nursing,
        model=cfg["ai"]["anthropic_model"],
        enable_ai_enrichment=cfg["ai"].get("enable_ai_enrichment", True),
        max_ai_rows_per_run=cfg["ai"].get("max_ai_rows_per_run", 20),
        strict_validation=cfg["ai"].get("strict_validation", False),
        strict_max_retries=cfg["ai"].get("strict_max_retries", 2),
    )
    _record_step("normalize_nursing_notes", t0, row_count=len(nursing_nlp))

    t0 = time.perf_counter()
    pdf_clinical = process_pdf_inbox(
        pdf_dir=cfg["paths"].get("pdf_inbox_dir", ""),
        processed_root=cfg["paths"]["processed_root"],
        model=cfg["ai"]["anthropic_model"],
        enable_ai_enrichment=cfg["ai"].get("enable_ai_enrichment", True),
        max_ai_rows_per_run=cfg["ai"].get("max_ai_rows_per_run", 20),
        strict_validation=cfg["ai"].get("strict_validation", False),
        strict_max_retries=cfg["ai"].get("strict_max_retries", 2),
        additional_pdf_roots=cfg.get("input_roots", []),
    )
    _record_step("process_pdf_inbox", t0, row_count=len(pdf_clinical))

    mandatory_issues = []
    table_candidates = {
        "tbImportAcData": epa,
        "tbImportLabsData": labs,
        "tbImportDeviceMotionData": device_motion,
        "tbImportDevice1HzMotionData": device_1hz,
        "tbImportMedicationInpatientData": medication,
        "tbImportNursingDailyReportsData": nursing,
        "tbImportIcd10Data": icd_ops,
    }

    t0 = time.perf_counter()
    validated_tables = {}
    for name, table in table_candidates.items():
        filtered, issue = drop_missing_mandatory(table, required_fields, name)
        validated_tables[name] = filtered
        mandatory_issues.append(issue)
    _record_step("validate_required_fields", t0, row_count=sum(len(df) for df in validated_tables.values()))

    t0 = time.perf_counter()
    case_table = build_case_table(validated_tables)
    _record_step("build_case_table", t0, row_count=len(case_table))

    t0 = time.perf_counter()
    import_tables = {name: _postprocess_import_table(table, case_table) for name, table in validated_tables.items()}
    import_tables["tbImportNursingNlpData"] = _postprocess_import_table(nursing_nlp, case_table)
    import_tables["tbImportPdfClinicalData"] = _postprocess_import_table(pdf_clinical, case_table)

    epa5_files = discovered.get("epa5", [])
    epa5_frames = []
    for path in epa5_files:
        raw = read_csv_flexible(path)
        raw["source_file"] = Path(path).name
        epa5_frames.append(raw)
    import_tables["tbImportAcDataEncryptedRaw"] = _postprocess_import_table(_concat_frames(epa5_frames), case_table)

    for table_name in list(import_tables.keys()):
        if table_name == "tbImportAcDataEncryptedRaw":
            continue
        cleaned = _drop_placeholder_columns(import_tables[table_name])
        cleaned = _enforce_machine_readable_columns(cleaned)
        import_tables[table_name] = cleaned
    _record_step("attach_case_foreign_keys", t0, row_count=sum(len(df) for df in import_tables.values()))

    tb_case_data = case_table[["coId", "coE2I222", "coPatientId"]].copy()

    t0 = time.perf_counter()
    lineage = _concat_frames(lineage_frames)
    lineage = pd.concat(
        [
            lineage,
            _build_auto_lineage("tbImportLabsData", import_tables.get("tbImportLabsData", pd.DataFrame())),
            _build_auto_lineage("tbImportDeviceMotionData", import_tables.get("tbImportDeviceMotionData", pd.DataFrame())),
            _build_auto_lineage("tbImportDevice1HzMotionData", import_tables.get("tbImportDevice1HzMotionData", pd.DataFrame())),
            _build_auto_lineage("tbImportMedicationInpatientData", import_tables.get("tbImportMedicationInpatientData", pd.DataFrame())),
            _build_auto_lineage("tbImportNursingDailyReportsData", import_tables.get("tbImportNursingDailyReportsData", pd.DataFrame())),
            _build_auto_lineage("tbImportIcd10Data", import_tables.get("tbImportIcd10Data", pd.DataFrame())),
            _build_auto_lineage("tbImportNursingNlpData", import_tables.get("tbImportNursingNlpData", pd.DataFrame())),
            _build_auto_lineage("tbImportPdfClinicalData", import_tables.get("tbImportPdfClinicalData", pd.DataFrame())),
        ],
        ignore_index=True,
    ).drop_duplicates()
    _record_step("build_mapping_lineage", t0, row_count=len(lineage))

    quality_frames = []
    anomaly_frames = []
    issue_frames = mandatory_issues.copy()

    t0 = time.perf_counter()
    for table_name, table in {"tbCaseData": tb_case_data, **import_tables}.items():
        quality_frames.append(compute_completeness(table_name, table))
        quality_issue_df = detect_data_quality_issues(table_name, table)
        if not quality_issue_df.empty:
            issue_frames.append(quality_issue_df)
        anomaly_frames.append(detect_anomalies(table_name, table))
    _record_step("compute_quality_and_anomalies", t0)

    quality_frames = [df for df in quality_frames if not df.empty]
    issue_frames = [df for df in issue_frames if not df.empty]
    anomaly_frames = [df for df in anomaly_frames if not df.empty]

    dq_completeness = pd.concat(quality_frames, ignore_index=True) if quality_frames else pd.DataFrame()
    dq_issues = pd.concat(issue_frames, ignore_index=True) if issue_frames else pd.DataFrame()
    dq_anomalies = pd.concat(anomaly_frames, ignore_index=True) if anomaly_frames else pd.DataFrame()

    dq_completeness = _ensure_columns(
        dq_completeness,
        ["table_name", "field", "non_null_rate", "null_count", "row_count"],
    )
    dq_issues = _ensure_columns(dq_issues, ["source_table", "issue_type", "field", "count", "severity"])
    dq_anomalies = _ensure_columns(dq_anomalies, ["source_table", "anomaly_type", "field", "count"])

    run_metadata = pd.DataFrame(
        [
            {
                "run_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_signature": signature,
                "pipeline_duration_ms": None,
                "records_case": len(tb_case_data),
                "records_ac": len(import_tables["tbImportAcData"]),
                "records_labs": len(import_tables["tbImportLabsData"]),
                "records_nursing": len(import_tables["tbImportNursingDailyReportsData"]),
                "records_medication": len(import_tables["tbImportMedicationInpatientData"]),
                "records_device_hourly": len(import_tables["tbImportDeviceMotionData"]),
                "records_device_1hz": len(import_tables["tbImportDevice1HzMotionData"]),
                "records_icd": len(import_tables["tbImportIcd10Data"]),
            }
        ]
    )

    pipeline_step_metrics = pd.DataFrame(step_metrics)

    source_catalog_rows = []
    for domain, files in discovered.items():
        for file in files:
            source_catalog_rows.append({"source_domain": domain, "source_file": Path(file).name, "source_path": file})
    source_file_catalog = pd.DataFrame(source_catalog_rows)

    t0 = time.perf_counter()
    bronze_raw_blob = _build_bronze_raw_blob(discovered)
    gold_case_analytics = _build_gold_case_analytics(case_table, import_tables)
    benchmark_metrics, benchmark_mapping_detail = _build_benchmark_metrics(
        mapping_df=mapping_df,
        lineage=lineage,
        dq_completeness=dq_completeness,
        bronze_raw_blob=bronze_raw_blob,
        pre_validation_tables=table_candidates,
        import_tables=import_tables,
        required_fields=required_fields,
        duckdb_file=duckdb_file,
    )
    data_contract_results = _build_data_contract_results(
        bronze_raw_blob=bronze_raw_blob,
        tb_case_data=tb_case_data,
        import_tables=import_tables,
        gold_case_analytics=gold_case_analytics,
    )
    _record_step(
        "build_benchmark_and_contracts",
        t0,
        row_count=len(benchmark_metrics) + len(data_contract_results),
    )

    tables_to_persist = {
        "bronze_raw_blob": bronze_raw_blob,
        "tbCaseData": tb_case_data,
        **import_tables,
        "gold_case_analytics": gold_case_analytics,
        "mapping_lineage": lineage,
        "dq_completeness": dq_completeness,
        "dq_issues": dq_issues,
        "dq_anomalies": dq_anomalies,
        "benchmark_metrics": benchmark_metrics,
        "benchmark_mapping_detail": benchmark_mapping_detail,
        "data_contract_results": data_contract_results,
        "pipeline_step_metrics": pipeline_step_metrics,
        "pipeline_run_metadata": run_metadata,
        "source_file_catalog": source_file_catalog,
    }

    _save_parquet_tables(processed_root, tables_to_persist)
    _persist_to_duckdb(duckdb_file, tables_to_persist)

    run_metadata.loc[0, "pipeline_duration_ms"] = round((time.perf_counter() - pipeline_t0) * 1000, 2)
    _persist_to_duckdb(duckdb_file, {"pipeline_run_metadata": run_metadata})

    metadata_path.write_text(
        json.dumps(
            {
                "source_signature": signature,
                "last_run_utc": run_metadata.iloc[0]["run_at_utc"],
                "duckdb_file": duckdb_file,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Pipeline completed. Processed data written to: {processed_root}")
    return {
        "status": "completed",
        "reason": "processed",
        "records_case": len(tb_case_data),
        "duration_ms": float(run_metadata.loc[0, "pipeline_duration_ms"]),
    }
