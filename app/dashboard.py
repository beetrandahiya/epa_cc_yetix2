from __future__ import annotations

import io
import json
import os
import random
import re
import tempfile
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from app.ai_extraction import anthropic_extract_structured, extract_from_pdf_with_ai
from app.anomaly import detect_anomalies
from app.config import load_settings
from app.ingestion import (
    load_device_1hz,
    load_device_motion,
    load_epa_data_1,
    load_epa_data_2,
    load_epa_data_3,
    load_icd_ops,
    load_labs_data,
    load_medication,
    load_nursing,
)
from app.mappings import item_name_to_iid_lookup, load_iid_sid_map, sid_to_iid_lookup
from app.pipeline import run_pipeline
from app.quality import compute_completeness, detect_data_quality_issues
from app.source_discovery import classify_file
from app.utils import read_csv_flexible

st.set_page_config(page_title="Smart Health Data Mapping", layout="wide")


@st.cache_resource
def get_conn(db_path: str):
    return duckdb.connect(db_path, read_only=False)


def table_exists(conn: duckdb.DuckDBPyConnection, name: str) -> bool:
    return conn.execute(
        "select count(*) from information_schema.tables where table_name = ?", [name]
    ).fetchone()[0] > 0


def list_import_tables(db_path: str) -> list[str]:
    conn = get_conn(db_path)
    rows = conn.execute(
        """
        select table_name
        from information_schema.tables
        where table_name like 'tbImport%'
        order by table_name
        """
    ).fetchdf()
    return rows["table_name"].tolist() if not rows.empty else []


@st.cache_data(ttl=120)
def get_sid_lookup(iid_sid_map_file: str) -> dict[str, str]:
    mapping = load_iid_sid_map(iid_sid_map_file)
    return sid_to_iid_lookup(mapping)


@st.cache_data(ttl=120)
def get_item_name_lookup(iid_sid_map_file: str) -> dict[str, str]:
    mapping = load_iid_sid_map(iid_sid_map_file)
    return item_name_to_iid_lookup(mapping)


@st.cache_data(ttl=120)
def get_coe_label_lookup(iid_sid_map_file: str, language: str = "de") -> dict[str, str]:
    mapping = load_iid_sid_map(iid_sid_map_file)
    label_col = "itmname255_de" if language.lower() == "de" else "itmname255_en"
    out: dict[str, str] = {}
    for _, row in mapping.dropna(subset=["iid_clean"]).iterrows():
        iid = str(row["iid_clean"]).upper()
        coe = f"CO{iid}"
        label = row.get(label_col) or row.get("itmname255_de") or row.get("itmname255_en") or ""
        label = str(label).strip()
        if label:
            out[coe] = label
    return out


def to_display_name(name: str, display_mode: str, coe_labels: dict[str, str]) -> str:
    if display_mode == "machine":
        return str(name)

    text = str(name)
    match = re.match(r"^(COE\d+I\d+)(.*)$", text, flags=re.IGNORECASE)
    if not match:
        return text

    base = match.group(1).upper()
    suffix = match.group(2) or ""
    label = coe_labels.get(base)
    if not label:
        return text
    return f"{base} · {label}{suffix}"


def apply_display_to_columns(df: pd.DataFrame, display_mode: str, coe_labels: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return df
    renamed = {col: to_display_name(str(col), display_mode, coe_labels) for col in df.columns}
    return df.rename(columns=renamed)


def apply_display_to_field_values(df: pd.DataFrame, field_col: str, display_mode: str, coe_labels: dict[str, str]) -> pd.DataFrame:
    if df.empty or field_col not in df.columns:
        return df
    out = df.copy()
    out[field_col] = out[field_col].astype(str).map(lambda x: to_display_name(x, display_mode, coe_labels))
    return out


def parse_probe_fields(raw_text: str) -> list[str]:
    if not raw_text:
        return []
    tokens = re.split(r"[\n,;]+", raw_text)
    cleaned = []
    seen = set()
    for tok in tokens:
        val = str(tok).strip()
        if not val:
            continue
        key = val.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(val)
    return cleaned


def _generate_sensor_profiles(device_count: int, seed: int = 42) -> list[dict[str, object]]:
    rng = random.Random(seed)
    profiles: list[dict[str, object]] = []
    for idx in range(device_count):
        ward = f"ward_{(idx % 4) + 1}"
        profiles.append(
            {
                "device_id": f"dev_{idx + 1:02d}",
                "ward": ward,
                "heart_rate": 72.0 + rng.uniform(-8.0, 8.0),
                "spo2": 97.0 + rng.uniform(-1.5, 1.0),
                "resp_rate": 16.0 + rng.uniform(-2.0, 2.0),
                "skin_temp_c": 36.6 + rng.uniform(-0.4, 0.4),
                "accel_g": 1.0 + rng.uniform(-0.05, 0.05),
            }
        )
    return profiles


def _simulate_sensor_stream(
    profiles: list[dict[str, object]],
    start_ts: datetime,
    seconds: int,
    seed: int = 7,
) -> tuple[pd.DataFrame, datetime]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    current_ts = start_ts

    for _ in range(max(0, int(seconds))):
        current_ts = current_ts + timedelta(seconds=1)
        for profile in profiles:
            hr = float(profile["heart_rate"]) + rng.uniform(-1.8, 1.8)
            spo2 = float(profile["spo2"]) + rng.uniform(-0.3, 0.2)
            resp = float(profile["resp_rate"]) + rng.uniform(-0.8, 0.8)
            temp = float(profile["skin_temp_c"]) + rng.uniform(-0.04, 0.04)
            accel = max(0.0, float(profile["accel_g"]) + rng.uniform(-0.08, 0.08))

            if rng.random() < 0.01:
                hr += rng.uniform(30, 55)
            if rng.random() < 0.008:
                spo2 -= rng.uniform(5, 10)
            if rng.random() < 0.006:
                accel += rng.uniform(1.5, 3.2)

            hr = max(35.0, min(165.0, hr))
            spo2 = max(80.0, min(100.0, spo2))
            resp = max(6.0, min(35.0, resp))
            temp = max(35.0, min(39.5, temp))

            profile["heart_rate"] = hr
            profile["spo2"] = spo2
            profile["resp_rate"] = resp
            profile["skin_temp_c"] = temp
            profile["accel_g"] = accel

            anomaly_type = ""
            if accel >= 2.5:
                anomaly_type = "fall_suspected"
            elif spo2 <= 90.0:
                anomaly_type = "hypoxia_risk"
            elif hr >= 125.0:
                anomaly_type = "tachycardia"
            elif hr <= 45.0:
                anomaly_type = "bradycardia"

            rows.append(
                {
                    "timestamp": current_ts,
                    "device_id": str(profile["device_id"]),
                    "ward": str(profile["ward"]),
                    "heart_rate": round(hr, 1),
                    "spo2": round(spo2, 1),
                    "resp_rate": round(resp, 1),
                    "skin_temp_c": round(temp, 2),
                    "accel_g": round(accel, 3),
                    "anomaly_type": anomaly_type,
                    "is_alert": bool(anomaly_type),
                }
            )

    return pd.DataFrame(rows), current_ts


def _normalize_probe_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


CANONICAL_ALIAS_MAP: dict[str, set[str]] = {
    "patient_id": {"patientid", "userid", "userid", "pid", "patid", "idpat", "copatientid", "pat"},
    "case_id": {"caseid", "fallnr", "fall", "encounterid", "patfal", "cocaseid"},
    "ward": {"ward", "station", "unit"},
    "specimen_datetime": {"specimendatetime", "labdatetime", "samplingtime"},
    "medication_name": {"medicationname", "drugname", "arznei", "medname"},
}


def _canonical_from_alias(probe_field: str) -> str | None:
    token = _normalize_probe_token(probe_field)
    for canonical, aliases in CANONICAL_ALIAS_MAP.items():
        if token in aliases:
            return canonical
    return None


def calibrate_mapping_assessments(assessments: list[dict[str, object]]) -> list[dict[str, object]]:
    calibrated: list[dict[str, object]] = []
    for item in assessments:
        rec = dict(item)
        probe = str(rec.get("probe_field", ""))
        inferred = _canonical_from_alias(probe)
        canonical = str(rec.get("canonical_name") or "").strip()
        confidence = str(rec.get("confidence") or "").lower().strip() or "medium"

        if inferred and (not canonical or canonical == "None"):
            rec["canonical_name"] = inferred
            confidence = "high"
            rec["rationale"] = f"Alias-based calibration inferred {inferred} from source field '{probe}'."
        elif inferred and canonical == inferred and confidence == "low":
            confidence = "high"
            existing = str(rec.get("rationale") or "")
            rec["rationale"] = (existing + " " if existing else "") + "Confidence upgraded by alias calibration."

        rec["confidence"] = confidence
        calibrated.append(rec)
    return calibrated


def ensure_accepted_mappings_table(db_path: str) -> None:
    conn = get_conn(db_path)
    conn.execute(
        """
        create table if not exists accepted_mappings (
            source_field varchar,
            canonical_name varchar,
            source_scope varchar,
            decision varchar,
            confidence varchar,
            rationale varchar,
            decided_at_utc varchar
        )
        """
    )


def load_accepted_mappings(db_path: str, source_scope: str | None = None) -> pd.DataFrame:
    ensure_accepted_mappings_table(db_path)
    conn = get_conn(db_path)
    if source_scope:
        return conn.execute(
            """
            select *
            from accepted_mappings
            where source_scope = ? or source_scope = '*'
            """,
            [source_scope],
        ).df()
    return conn.execute("select * from accepted_mappings").df()


def save_accepted_mapping(
    db_path: str,
    source_field: str,
    canonical_name: str,
    source_scope: str,
    confidence: str,
    rationale: str,
) -> None:
    ensure_accepted_mappings_table(db_path)
    conn = get_conn(db_path)
    conn.execute(
        "delete from accepted_mappings where lower(source_field) = lower(?) and source_scope = ?",
        [source_field, source_scope],
    )
    row = pd.DataFrame(
        [
            {
                "source_field": source_field,
                "canonical_name": canonical_name,
                "source_scope": source_scope,
                "decision": "accepted",
                "confidence": confidence,
                "rationale": rationale,
                "decided_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )
    conn.register("tmp_map_accept", row)
    conn.execute("insert into accepted_mappings select * from tmp_map_accept")
    conn.unregister("tmp_map_accept")


def ensure_governance_tables(db_path: str) -> None:
    conn = get_conn(db_path)
    conn.execute(
        """
        create table if not exists audit_log (
            audit_id bigint,
            event_time_utc varchar,
            user_id varchar,
            user_role varchar,
            event_type varchar,
            entity_type varchar,
            entity_key varchar,
            before_value varchar,
            after_value varchar,
            note varchar,
            status varchar
        )
        """
    )
    conn.execute(
        """
        create table if not exists ai_review_queue (
            review_id bigint,
            created_at_utc varchar,
            source_type varchar,
            source_scope varchar,
            probe_field varchar,
            proposed_value varchar,
            confidence varchar,
            rationale varchar,
            status varchar,
            reviewed_by varchar,
            reviewed_at_utc varchar,
            decision_note varchar
        )
        """
    )


def has_permission(role: str, action: str) -> bool:
    perm_map = {
        "admin": {"mapping_accept", "review_approve", "correction_edit", "view_audit"},
        "data_steward": {"mapping_accept", "review_approve", "correction_edit", "view_audit"},
        "analyst": {"view_audit"},
        "viewer": set(),
    }
    return action in perm_map.get(str(role).lower(), set())


def log_audit_event(
    db_path: str,
    user_id: str,
    user_role: str,
    event_type: str,
    entity_type: str,
    entity_key: str,
    before_value: str | None,
    after_value: str | None,
    note: str,
    status: str = "success",
) -> None:
    ensure_governance_tables(db_path)
    conn = get_conn(db_path)
    last_id = conn.execute("select coalesce(max(audit_id), 0) from audit_log").fetchone()[0]
    row = pd.DataFrame(
        [
            {
                "audit_id": int(last_id) + 1,
                "event_time_utc": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "user_role": user_role,
                "event_type": event_type,
                "entity_type": entity_type,
                "entity_key": entity_key,
                "before_value": (before_value or "")[:12000],
                "after_value": (after_value or "")[:12000],
                "note": note[:1000],
                "status": status,
            }
        ]
    )
    conn.register("tmp_audit_log", row)
    conn.execute("insert into audit_log select * from tmp_audit_log")
    conn.unregister("tmp_audit_log")


def load_audit_log(db_path: str) -> pd.DataFrame:
    ensure_governance_tables(db_path)
    return load_table(db_path, "audit_log")


def enqueue_mapping_review(
    db_path: str,
    source_scope: str,
    probe_field: str,
    proposed_value: str,
    confidence: str,
    rationale: str,
) -> None:
    ensure_governance_tables(db_path)
    conn = get_conn(db_path)
    exists = conn.execute(
        """
        select count(*)
        from ai_review_queue
        where lower(probe_field) = lower(?)
          and source_scope = ?
          and lower(status) = 'pending'
        """,
        [probe_field, source_scope],
    ).fetchone()[0]
    if int(exists) > 0:
        return

    last_id = conn.execute("select coalesce(max(review_id), 0) from ai_review_queue").fetchone()[0]
    row = pd.DataFrame(
        [
            {
                "review_id": int(last_id) + 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_type": "mapping_studio",
                "source_scope": source_scope,
                "probe_field": probe_field,
                "proposed_value": proposed_value,
                "confidence": confidence,
                "rationale": rationale,
                "status": "pending",
                "reviewed_by": None,
                "reviewed_at_utc": None,
                "decision_note": None,
            }
        ]
    )
    conn.register("tmp_review_q", row)
    conn.execute("insert into ai_review_queue select * from tmp_review_q")
    conn.unregister("tmp_review_q")


def load_review_queue(db_path: str, source_scope: str | None = None) -> pd.DataFrame:
    ensure_governance_tables(db_path)
    conn = get_conn(db_path)
    if source_scope:
        return conn.execute(
            """
            select *
            from ai_review_queue
            where source_scope = ? or source_scope = '*'
            order by review_id desc
            """,
            [source_scope],
        ).df()
    return conn.execute("select * from ai_review_queue order by review_id desc").df()


def decide_review_item(db_path: str, review_id: int, decision: str, reviewer: str, note: str) -> None:
    ensure_governance_tables(db_path)
    conn = get_conn(db_path)
    conn.execute(
        """
        update ai_review_queue
        set status = ?,
            reviewed_by = ?,
            reviewed_at_utc = ?,
            decision_note = ?
        where review_id = ?
        """,
        [decision, reviewer, datetime.now(timezone.utc).isoformat(), note, int(review_id)],
    )


def build_change_audit_payload(before_df: pd.DataFrame, after_df: pd.DataFrame, key_col: str = "coId") -> tuple[str, str, int]:
    if key_col not in before_df.columns or key_col not in after_df.columns:
        return "[]", "[]", 0

    before_idx = before_df.set_index(key_col, drop=False)
    after_idx = after_df.set_index(key_col, drop=False)
    common = before_idx.index.intersection(after_idx.index)
    if len(common) == 0:
        return "[]", "[]", 0

    compare_cols = [c for c in after_idx.columns if c in before_idx.columns and c != key_col]
    before_c = before_idx.loc[common, compare_cols].fillna("<NA>").astype(str)
    after_c = after_idx.loc[common, compare_cols].fillna("<NA>").astype(str)
    changed_mask = before_c.ne(after_c)
    changed_ids = changed_mask.any(axis=1)
    changed_index = changed_ids[changed_ids].index.tolist()

    if not changed_index:
        return "[]", "[]", 0

    sample_ids = changed_index[:20]
    before_rows = before_idx.loc[sample_ids].reset_index(drop=True).to_dict(orient="records")
    after_rows = after_idx.loc[sample_ids].reset_index(drop=True).to_dict(orient="records")
    return (
        json.dumps(before_rows, ensure_ascii=False, default=str),
        json.dumps(after_rows, ensure_ascii=False, default=str),
        len(changed_index),
    )


def compute_business_kpis(db_path: str, benchmark_metrics: pd.DataFrame) -> dict[str, float]:
    accepted = load_accepted_mappings(db_path)
    ai_logs = load_table(db_path, "ai_extraction_log")
    audit = load_audit_log(db_path)
    queue = load_review_queue(db_path)

    accepted_count = float(len(accepted)) if not accepted.empty else 0.0
    ai_success = float((ai_logs["status"].astype(str).str.lower() == "success").sum()) if not ai_logs.empty and "status" in ai_logs.columns else 0.0
    ai_total = float(len(ai_logs)) if not ai_logs.empty else 0.0
    missingness_improvement = 0.0
    if not benchmark_metrics.empty and "metric_name" in benchmark_metrics.columns:
        hit = benchmark_metrics[benchmark_metrics["metric_name"] == "missingness_improvement_after_harmonization"]
        if not hit.empty:
            missingness_improvement = float(pd.to_numeric(hit.iloc[0].get("metric_value"), errors="coerce") or 0.0)

    review_approved = float((audit["event_type"].astype(str) == "mapping_review_approved").sum()) if not audit.empty and "event_type" in audit.columns else 0.0
    review_rejected = float((audit["event_type"].astype(str) == "mapping_review_rejected").sum()) if not audit.empty and "event_type" in audit.columns else 0.0
    corrections_applied = float((audit["event_type"].astype(str) == "correction_applied").sum()) if not audit.empty and "event_type" in audit.columns else 0.0
    pending_reviews = float((queue["status"].astype(str).str.lower() == "pending").sum()) if not queue.empty and "status" in queue.columns else 0.0

    auto_decisions = accepted_count + max(0.0, ai_success - (review_approved + review_rejected))
    total_decisions = auto_decisions + review_approved + review_rejected + pending_reviews
    automation_rate = (auto_decisions / total_decisions) if total_decisions else 0.0
    review_approval_rate = (review_approved / (review_approved + review_rejected)) if (review_approved + review_rejected) else 0.0
    extraction_success_rate = (ai_success / ai_total) if ai_total else 0.0

    effort_saved_hours = (accepted_count * 2.0 + ai_success * 4.0) / 60.0
    prevented_error_proxy = review_approved + corrections_applied
    monthly_saved_hours = effort_saved_hours * 4.0
    annualized_savings_eur = monthly_saved_hours * 12.0 * 45.0
    return {
        "estimated_effort_saved_hours": effort_saved_hours,
        "estimated_effort_saved_hours_monthly": monthly_saved_hours,
        "annualized_efficiency_savings_eur": annualized_savings_eur,
        "data_quality_uplift": missingness_improvement,
        "prevented_error_proxy": prevented_error_proxy,
        "automation_rate_proxy": automation_rate,
        "review_approval_rate": review_approval_rate,
        "extraction_success_rate": extraction_success_rate,
        "pending_reviews": pending_reviews,
    }


def build_clinic_executive_report_markdown(
    clinic_id: str,
    clinic_catalog: pd.DataFrame,
    coverage_df: pd.DataFrame,
    clinic_cases: pd.DataFrame,
    clinic_ai_out: dict | None,
    kpis: dict[str, float] | None = None,
) -> str:
    domains = clinic_catalog["source_domain"].dropna().astype(str).unique().tolist() if "source_domain" in clinic_catalog.columns else []
    files = clinic_catalog["source_file"].dropna().astype(str).unique().tolist() if "source_file" in clinic_catalog.columns else []
    top_tables = coverage_df.sort_values("rows", ascending=False).head(5) if not coverage_df.empty else pd.DataFrame()
    case_count = int(len(clinic_cases)) if not clinic_cases.empty else 0
    patient_count = int(clinic_cases["coPatientId"].dropna().nunique()) if (not clinic_cases.empty and "coPatientId" in clinic_cases.columns) else 0

    lines = [
        f"# Executive Data Report — {clinic_id}",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## 1) Executive Summary",
        f"- This clinic currently contributes **{len(files)} source files** across **{len(domains)} domains**.",
        f"- The harmonized dataset links **{case_count} cases** and **{patient_count} patients** into the unified model.",
        "- Data quality and mapping controls are active, including anomaly checks and manual correction workflows.",
        "",
        "## 2) Source Coverage",
        f"- Source domains: {', '.join(domains) if domains else 'No domains detected'}",
        f"- Source files tracked: {len(files)}",
        "",
        "### Top Tables by Volume",
    ]

    if top_tables.empty:
        lines.append("- No processed table coverage available yet.")
    else:
        for _, row in top_tables.iterrows():
            lines.append(f"- {row['table']}: {int(row['rows'])} rows from {int(row['files'])} files")

    lines += ["", "## 3) AI-Derived Insights", ""]
    if isinstance(clinic_ai_out, dict) and clinic_ai_out:
        summary = str(clinic_ai_out.get("summary", "")).strip()
        if summary:
            lines.append(textwrap.fill(summary, width=100))
            lines.append("")

        insights = clinic_ai_out.get("insights", []) if isinstance(clinic_ai_out.get("insights", []), list) else []
        if insights:
            for item in insights[:8]:
                title = str(item.get("title", "Insight"))
                why = str(item.get("why_it_matters", ""))
                evidence = str(item.get("evidence", ""))
                conf = str(item.get("confidence", ""))
                lines.append(f"- **{title}** ({conf})")
                if why:
                    lines.append(f"  - Why it matters: {why}")
                if evidence:
                    lines.append(f"  - Evidence: {evidence}")
        else:
            lines.append("- AI insights not generated yet for this clinic.")

        actions = clinic_ai_out.get("actions", []) if isinstance(clinic_ai_out.get("actions", []), list) else []
        lines += ["", "## 4) Recommended Actions", ""]
        if actions:
            for action in actions[:8]:
                lines.append(
                    f"- **{action.get('action', 'Action')}** | Priority: {action.get('priority', 'n/a')} | Owner: {action.get('owner', 'n/a')}"
                )
                if action.get("expected_impact"):
                    lines.append(f"  - Expected impact: {action.get('expected_impact')}")
        else:
            lines.append("- No AI actions available yet.")
    else:
        lines.append("- AI insights have not been run yet for this clinic.")
        lines += ["", "## 4) Recommended Actions", "", "- Run Clinic 360 AI insights (Analyze/Ideate/Both) to populate strategic recommendations."]

    lines += [
        "",
        "## 5) Business Impact Snapshot",
    ]

    if kpis:
        lines += [
            f"- Estimated annualized efficiency potential: **€{kpis.get('annualized_efficiency_savings_eur', 0.0):,.0f}**",
            f"- Automation rate (proxy): **{kpis.get('automation_rate_proxy', 0.0):.1%}**",
            f"- Data quality uplift (non-null): **{kpis.get('data_quality_uplift', 0.0):+.2%}**",
            f"- Prevented-error proxy events: **{int(kpis.get('prevented_error_proxy', 0.0))}**",
        ]
    else:
        lines.append("- KPI snapshot unavailable for this export session.")

    lines += [
        "",
        "## 6) Governance Notes",
        "- This report is generated from on-prem harmonized data and dashboard-derived governance controls.",
        "- Review low-confidence AI mappings in the review queue before promoting to accepted mappings.",
    ]

    return "\n".join(lines).strip() + "\n"


def markdown_to_pdf_bytes(markdown_text: str) -> bytes | None:
    try:
        import importlib

        pagesizes = importlib.import_module("reportlab.lib.pagesizes")
        canvas_mod = importlib.import_module("reportlab.pdfgen.canvas")
        A4 = pagesizes.A4
        Canvas = canvas_mod.Canvas
    except Exception:
        return None

    buffer = io.BytesIO()
    pdf = Canvas(buffer, pagesize=A4)
    width, height = A4
    margin_x = 40
    y = height - 40
    line_height = 14

    for raw_line in markdown_text.splitlines():
        line = raw_line.replace("**", "").replace("#", "").strip()
        wrapped = textwrap.wrap(line, width=110) if line else [""]
        for part in wrapped:
            if y < 40:
                pdf.showPage()
                y = height - 40
            pdf.drawString(margin_x, y, part)
            y -= line_height

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def _build_lineage_context(lineage: pd.DataFrame, probe_fields: list[str]) -> list[dict[str, object]]:
    if lineage.empty or "source_field" not in lineage.columns or "target_field" not in lineage.columns:
        return []

    out: list[dict[str, object]] = []
    src = lineage.copy()
    src["source_field_norm"] = src["source_field"].astype(str).map(_normalize_probe_token)
    for probe in probe_fields:
        p_norm = _normalize_probe_token(probe)
        exact = src[src["source_field_norm"] == p_norm]
        hints = []
        if not exact.empty:
            grp = exact.groupby("target_field", as_index=False).size().rename(columns={"size": "count"})
            grp = grp.sort_values("count", ascending=False).head(8)
            hints = grp.to_dict(orient="records")
        out.append({"probe_field": probe, "lineage_hints": hints})
    return out


def ai_map_fields(
    probe_fields: list[str],
    candidate_targets: list[str],
    lineage_context: list[dict[str, object]],
    model: str,
    strict_validation: bool = False,
    max_retries: int = 0,
) -> dict:
    if not probe_fields:
        return {"assessments": []}

    schema = json.dumps(
        {
            "assessments": [
                {
                    "probe_field": "string",
                    "canonical_name": "string|null",
                    "confidence": "low|medium|high",
                    "rationale": "string",
                    "alternatives": ["string"],
                }
            ]
        },
        ensure_ascii=False,
    )

    prompt = (
        "You are an AI data mapper for healthcare data. "
        "CONTEXT: The data is German. 'Fall' means 'Case/Encounter', NOT a physical floor-fall. "
        "TASK: For each source column, map to one canonical target from the allowed list. "
        "Use lineage hints if available, otherwise infer from semantic meaning. "
        "If unclear, set canonical_name to null and confidence low. "
        "RETURN strict JSON only.\n\n"
        f"Allowed canonical targets: {json.dumps(candidate_targets, ensure_ascii=False)}\n"
        f"Source columns to map: {json.dumps(probe_fields, ensure_ascii=False)}\n"
        f"Lineage context: {json.dumps(lineage_context, ensure_ascii=False)}"
    )
    return anthropic_extract_structured(
        prompt,
        schema_hint=schema,
        model=model,
        strict_validation=strict_validation,
        max_retries=max_retries,
        required_keys=["assessments"],
    )


@st.cache_data(ttl=120)
def read_raw_preview(file_path: str) -> pd.DataFrame:
    p = Path(file_path)
    if p.suffix.lower() == ".pdf":
        return pd.DataFrame({"pdf_file": [p.name], "note": ["PDF detected; use AI tab for full extraction"]})
    return read_csv_flexible(file_path).head(200)


def normalize_single_file(
    file_path: str,
    cfg: dict,
    sid_lookup: dict[str, str],
    item_name_lookup: dict[str, str],
) -> pd.DataFrame:
    domain = classify_file(Path(file_path))
    nulls = cfg["rules"]["null_like_values"]

    if domain == "epa1":
        return load_epa_data_1(file_path, sid_lookup, nulls)[0]
    if domain == "epa2":
        return load_epa_data_2(file_path, sid_lookup, nulls)[0]
    if domain == "epa3":
        return load_epa_data_3(file_path, nulls, item_name_to_iid=item_name_lookup)[0]
    if domain == "labs":
        return load_labs_data(file_path, nulls)
    if domain == "device_motion":
        return load_device_motion(file_path, nulls)
    if domain == "device_1hz":
        return load_device_1hz(file_path, nulls)
    if domain == "medication":
        return load_medication(file_path, nulls)
    if domain == "nursing":
        return load_nursing(file_path, nulls)
    if domain == "icd":
        return load_icd_ops(file_path, nulls)

    if Path(file_path).suffix.lower() == ".pdf":
        return pd.DataFrame({"pdf_file": [Path(file_path).name], "normalized_status": ["handled by PDF pipeline"]})

    return read_csv_flexible(file_path)


def find_processed_rows_for_file(db_path: str, source_file: str) -> dict[str, pd.DataFrame]:
    conn = get_conn(db_path)
    outputs: dict[str, pd.DataFrame] = {}
    for table in list_import_tables(db_path):
        cols = conn.execute(f"describe {table}").fetchdf()["column_name"].tolist()
        if "source_file" in cols:
            df = conn.execute(f"select * from {table} where source_file = ? limit 200", [source_file]).df()
            if not df.empty:
                outputs[table] = df
        elif "coSource_file" in cols:
            df = conn.execute(f"select * from {table} where coSource_file = ? limit 200", [source_file]).df()
            if not df.empty:
                outputs[table] = df
    return outputs


@st.cache_data(ttl=120)
def load_table(db_path: str, table_name: str) -> pd.DataFrame:
    conn = get_conn(db_path)
    if not table_exists(conn, table_name):
        return pd.DataFrame()
    return conn.execute(f"select * from {table_name}").df()


def extract_clinic_id(text: str) -> str:
    value = str(text or "")
    match = re.search(r"clinic[_\- ]?(\d+)", value, flags=re.IGNORECASE)
    if match:
        return f"clinic_{match.group(1)}"
    return "unassigned"


def annotate_catalog_with_clinic(source_catalog: pd.DataFrame) -> pd.DataFrame:
    if source_catalog.empty:
        return source_catalog
    out = source_catalog.copy()
    out["clinic_id"] = out.apply(
        lambda r: extract_clinic_id(f"{r.get('source_file', '')} {r.get('source_path', '')}"),
        axis=1,
    )
    return out


def build_clinic_unified_data(
    db_path: str,
    clinic_source_files: list[str],
    max_rows_per_table: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = ["tbCaseData"] + list_import_tables(db_path)
    file_set = set(str(x) for x in clinic_source_files)

    coverage_rows: list[dict[str, object]] = []
    unified_rows: list[pd.DataFrame] = []
    linked_case_ids: set[int] = set()

    for table_name in tables:
        df = load_table(db_path, table_name)
        if df.empty:
            coverage_rows.append({"table": table_name, "rows": 0, "files": 0})
            continue

        source_col = "source_file" if "source_file" in df.columns else ("coSource_file" if "coSource_file" in df.columns else None)

        if table_name == "tbCaseData":
            clinic_df = df.copy()
            if linked_case_ids and "coId" in clinic_df.columns:
                clinic_df = clinic_df[clinic_df["coId"].isin(linked_case_ids)].copy()
            else:
                clinic_df = clinic_df.iloc[0:0].copy()
            files_count = 0
        elif source_col is not None:
            clinic_df = df[df[source_col].astype(str).isin(file_set)].copy()
            if "coCaseId" in clinic_df.columns:
                linked_case_ids.update(
                    pd.to_numeric(clinic_df["coCaseId"], errors="coerce").dropna().astype(int).unique().tolist()
                )
            files_count = clinic_df[source_col].dropna().astype(str).nunique() if not clinic_df.empty else 0
        else:
            clinic_df = df.iloc[0:0].copy()
            files_count = 0

        coverage_rows.append({"table": table_name, "rows": int(len(clinic_df)), "files": int(files_count)})

        if clinic_df.empty:
            continue

        subset = clinic_df.head(max_rows_per_table).copy()
        if source_col is None:
            subset["source_file"] = pd.NA
        elif source_col != "source_file":
            subset = subset.rename(columns={source_col: "source_file"})

        payload = subset.drop(columns=[c for c in ["source_file"] if c in subset.columns], errors="ignore")
        unified = pd.DataFrame(
            {
                "source_table": table_name,
                "source_file": subset["source_file"] if "source_file" in subset.columns else pd.NA,
                "coId": subset["coId"] if "coId" in subset.columns else pd.NA,
                "coCaseId": subset["coCaseId"] if "coCaseId" in subset.columns else pd.NA,
                "case_id": subset["case_id"] if "case_id" in subset.columns else pd.NA,
                "patient_id": subset["patient_id"] if "patient_id" in subset.columns else pd.NA,
                "row_payload_json": payload.apply(lambda r: json.dumps(r.to_dict(), ensure_ascii=False, default=str), axis=1),
            }
        )
        unified_rows.append(unified)

    coverage_df = pd.DataFrame(coverage_rows).sort_values(["rows", "table"], ascending=[False, True])
    unified_df = pd.concat(unified_rows, ignore_index=True) if unified_rows else pd.DataFrame(
        columns=["source_table", "source_file", "coId", "coCaseId", "case_id", "patient_id", "row_payload_json"]
    )

    if linked_case_ids:
        case_df = load_table(db_path, "tbCaseData")
        if not case_df.empty and "coId" in case_df.columns:
            clinic_cases = case_df[case_df["coId"].isin(linked_case_ids)].copy()
        else:
            clinic_cases = pd.DataFrame()
    else:
        clinic_cases = pd.DataFrame()

    return coverage_df, unified_df, clinic_cases


def save_uploaded_files(
    uploaded_files: list,
    target_dir: str,
    subfolder: str = "",
    overwrite: bool = False,
) -> pd.DataFrame:
    base = Path(target_dir)
    if subfolder.strip():
        safe_sub = subfolder.strip().replace("\\", "/").strip("/")
        base = base / safe_sub
    base.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for up in uploaded_files:
        file_name = Path(str(up.name)).name
        destination = base / file_name

        if destination.exists() and not overwrite:
            rows.append(
                {
                    "file_name": file_name,
                    "status": "skipped_exists",
                    "saved_path": str(destination),
                    "bytes": int(getattr(up, "size", 0) or 0),
                    "detected_domain": classify_file(destination) if destination.suffix.lower() == ".csv" else "pdf",
                }
            )
            continue

        with destination.open("wb") as f:
            f.write(up.getbuffer())

        domain = classify_file(destination) if destination.suffix.lower() == ".csv" else ("pdf" if destination.suffix.lower() == ".pdf" else "unknown")
        rows.append(
            {
                "file_name": file_name,
                "status": "saved",
                "saved_path": str(destination),
                "bytes": int(getattr(up, "size", 0) or 0),
                "detected_domain": domain,
            }
        )

    return pd.DataFrame(rows)


def build_clinic_ai_context(
    selected_clinic: str,
    clinic_catalog: pd.DataFrame,
    coverage_df: pd.DataFrame,
    clinic_cases: pd.DataFrame,
    unified_df: pd.DataFrame,
    max_samples: int = 40,
) -> dict[str, object]:
    sample_rows = unified_df.head(max_samples).copy() if not unified_df.empty else pd.DataFrame()
    sample_payloads = sample_rows[[c for c in ["source_table", "source_file", "row_payload_json"] if c in sample_rows.columns]]
    return {
        "clinic_id": selected_clinic,
        "source_files": clinic_catalog["source_file"].dropna().astype(str).unique().tolist() if "source_file" in clinic_catalog.columns else [],
        "source_domains": clinic_catalog["source_domain"].dropna().astype(str).unique().tolist() if "source_domain" in clinic_catalog.columns else [],
        "coverage": coverage_df.to_dict(orient="records") if not coverage_df.empty else [],
        "case_count": int(len(clinic_cases)),
        "patient_count": int(clinic_cases["coPatientId"].dropna().nunique()) if (not clinic_cases.empty and "coPatientId" in clinic_cases.columns) else 0,
        "sample_rows": sample_payloads.to_dict(orient="records") if not sample_payloads.empty else [],
    }


def ai_clinic_insights(
    mode: str,
    context_payload: dict[str, object],
    user_focus: str,
    model: str,
    strict_validation: bool = False,
    max_retries: int = 0,
) -> dict:
    schema = json.dumps(
        {
            "mode": "analyze|ideate|both",
            "summary": "string",
            "insights": [
                {
                    "title": "string",
                    "why_it_matters": "string",
                    "evidence": "string",
                    "confidence": "low|medium|high",
                }
            ],
            "actions": [
                {
                    "action": "string",
                    "priority": "low|medium|high",
                    "owner": "clinical|operations|data",
                    "expected_impact": "string",
                }
            ],
        },
        ensure_ascii=False,
    )

    objective = {
        "analyze": "Find concrete patterns, quality risks, and operational signals grounded in the clinic data.",
        "ideate": "Propose practical, high-value hypotheses, care/process experiments, and data products this clinic could test.",
        "both": "First analyze current signals, then propose actionable ideas based on those signals.",
    }.get(mode, "Analyze and ideate from clinic data.")

    prompt = (
        "You are a healthcare analytics copilot for clinic-level data review. "
        "Use only provided context; do not fabricate metrics. "
        "CONTEXT: In German data, 'Fall' means case/encounter, not a physical floor-fall.\n\n"
        f"Objective: {objective}\n"
        f"Optional user focus: {user_focus or 'none'}\n\n"
        f"Clinic context JSON:\n{json.dumps(context_payload, ensure_ascii=False)}"
    )
    return anthropic_extract_structured(
        prompt,
        schema_hint=schema,
        model=model,
        strict_validation=strict_validation,
        max_retries=max_retries,
        required_keys=["mode", "summary", "insights", "actions"],
    )


def show_kpi_row(metadata: pd.DataFrame) -> None:
    if metadata.empty:
        st.warning("No pipeline metadata found. Run preprocessing first.")
        return

    row = metadata.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cases", int(row.get("records_case", 0)))
    c2.metric("Assessments", int(row.get("records_ac", 0)))
    c3.metric("Lab Records", int(row.get("records_labs", 0)))
    c4.metric("Sensor Records (1Hz)", int(row.get("records_device_1hz", 0)))


def refresh_monitoring_tables(db_path: str) -> None:
    conn = get_conn(db_path)
    table_names = conn.execute(
        """
        select table_name
        from information_schema.tables
        where table_name = 'tbCaseData' or table_name like 'tbImport%'
        order by table_name
        """
    ).fetchdf()["table_name"].tolist()

    quality_frames = []
    issue_frames = []
    anomaly_frames = []

    for name in table_names:
        df = conn.execute(f"select * from {name}").df()
        quality_frames.append(compute_completeness(name, df))
        issues = detect_data_quality_issues(name, df)
        anomalies = detect_anomalies(name, df)
        if not issues.empty:
            issue_frames.append(issues)
        if not anomalies.empty:
            anomaly_frames.append(anomalies)

    dq_completeness = pd.concat([df for df in quality_frames if not df.empty], ignore_index=True) if quality_frames else pd.DataFrame()
    dq_issues = pd.concat([df for df in issue_frames if not df.empty], ignore_index=True) if issue_frames else pd.DataFrame()
    dq_anomalies = pd.concat([df for df in anomaly_frames if not df.empty], ignore_index=True) if anomaly_frames else pd.DataFrame()

    conn.register("tmp_dq", dq_completeness)
    conn.execute("create or replace table dq_completeness as select * from tmp_dq")
    conn.unregister("tmp_dq")

    conn.register("tmp_issue", dq_issues)
    conn.execute("create or replace table dq_issues as select * from tmp_issue")
    conn.unregister("tmp_issue")

    conn.register("tmp_anom", dq_anomalies)
    conn.execute("create or replace table dq_anomalies as select * from tmp_anom")
    conn.unregister("tmp_anom")


def persist_table_edits(db_path: str, processed_root: str, table_name: str, edited_df: pd.DataFrame) -> tuple[bool, str]:
    conn = get_conn(db_path)
    current = conn.execute(f"select * from {table_name}").df()
    if current.empty:
        return False, "Target table is empty."

    if "coId" not in current.columns or "coId" not in edited_df.columns:
        return False, "Cannot persist edits: coId key is required."

    current_idx = current.set_index("coId", drop=False)
    edited_idx = edited_df.set_index("coId", drop=False)

    common_ids = current_idx.index.intersection(edited_idx.index)
    if len(common_ids) == 0:
        return False, "No matching coId rows to update."

    update_cols = [c for c in edited_idx.columns if c in current_idx.columns]
    current_idx.loc[common_ids, update_cols] = edited_idx.loc[common_ids, update_cols]

    updated = current_idx.reset_index(drop=True)
    conn.register("tmp_update", updated)
    conn.execute(f"create or replace table {table_name} as select * from tmp_update")
    conn.unregister("tmp_update")

    parquet_path = Path(processed_root) / f"{table_name}.parquet"
    updated.to_parquet(parquet_path, index=False)

    refresh_monitoring_tables(db_path)

    return True, f"Updated {len(common_ids)} rows in {table_name}."


def log_ai_extraction(
    db_path: str,
    source_type: str,
    input_excerpt: str,
    schema_hint: str,
    result_json: str | None,
    status: str,
    error_message: str | None,
) -> None:
    conn = get_conn(db_path)
    conn.execute(
        """
        create table if not exists ai_extraction_log (
            log_id bigint,
            run_at_utc varchar,
            source_type varchar,
            input_excerpt varchar,
            schema_hint varchar,
            result_json varchar,
            status varchar,
            error_message varchar
        )
        """
    )

    last_id = conn.execute("select coalesce(max(log_id), 0) from ai_extraction_log").fetchone()[0]
    row = pd.DataFrame(
        [
            {
                "log_id": int(last_id) + 1,
                "run_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_type": source_type,
                "input_excerpt": input_excerpt[:500],
                "schema_hint": schema_hint[:2000],
                "result_json": (result_json or "")[:12000],
                "status": status,
                "error_message": (error_message or "")[:1000],
            }
        ]
    )

    conn.register("tmp_ai_log", row)
    conn.execute("insert into ai_extraction_log select * from tmp_ai_log")
    conn.unregister("tmp_ai_log")


def persist_api_key_to_env(api_key: str) -> None:
    from app.config import BASE_DIR

    env_path = BASE_DIR / ".env"
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    updated = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("ANTHROPIC_API_KEY="):
            new_lines.append(f"ANTHROPIC_API_KEY={api_key}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"ANTHROPIC_API_KEY={api_key}")

    env_path.write_text("\n".join(new_lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    cfg = load_settings()
    db_path = cfg["paths"]["duckdb_file"]
    model = cfg["ai"]["anthropic_model"]
    strict_ai_validation = bool(cfg.get("ai", {}).get("strict_validation", False))
    strict_ai_max_retries = int(cfg.get("ai", {}).get("strict_max_retries", 0) or 0)

    st.title("Smart Health Data Mapping")
    st.caption("On-prem, case-centric harmonization and quality monitoring dashboard")

    with st.sidebar:
        st.subheader("Pipeline")
        st.write("Preprocessing runs once and is cached until source files change.")
        force = st.checkbox("Force rebuild", value=False)
        if st.button("Run preprocessing", type="primary"):
            result = run_pipeline(force=force)
            st.cache_data.clear()
            if isinstance(result, dict) and result.get("status") == "skipped":
                st.info("Preprocessing skipped: sources unchanged and artifacts already present.")
            else:
                st.success("Preprocessing finished.")

        st.divider()
        st.subheader("Access")
        active_user = st.text_input("User ID", value=st.session_state.get("active_user", "analyst1"))
        role_options = ["admin", "data_steward", "analyst", "viewer"]
        current_role = st.session_state.get("active_role", "analyst")
        active_role = st.selectbox(
            "Role",
            options=role_options,
            index=role_options.index(current_role) if current_role in role_options else 2,
        )
        st.session_state["active_user"] = active_user.strip() or "anonymous"
        st.session_state["active_role"] = active_role
        st.caption(f"Active identity: {st.session_state['active_user']} ({active_role})")

        st.divider()
        st.subheader("AI Settings")
        model = st.text_input("Anthropic model", value=model)
        api_key_override = st.text_input("Anthropic API key", value="", type="password")
        if api_key_override.strip():
            os.environ["ANTHROPIC_API_KEY"] = api_key_override.strip()
        if st.button("Save API key locally"):
            if api_key_override.strip():
                persist_api_key_to_env(api_key_override.strip())
                st.success("API key saved to local .env for future sessions.")
            else:
                st.warning("Enter an API key first, then click save.")
        st.write("Use environment variable ANTHROPIC_API_KEY or paste a key above for this session.")

        st.divider()
        st.subheader("Display Mode")
        display_mode_label = st.radio(
            "Field naming",
            options=["Machine-readable (COE)", "Human-readable (IID-SID-ITEM)"],
            index=0,
        )
        label_language = st.radio("Label language", options=["de", "en"], index=0, horizontal=True)

    display_mode = "human" if "Human-readable" in display_mode_label else "machine"
    coe_labels = get_coe_label_lookup(cfg["paths"]["iid_sid_map_file"], language=label_language)

    if not Path(db_path).exists():
        st.info("Processed database not found. Click 'Run preprocessing' in the sidebar.")
        return

    ensure_governance_tables(db_path)
    active_user = str(st.session_state.get("active_user", "anonymous"))
    active_role = str(st.session_state.get("active_role", "viewer"))

    metadata = load_table(db_path, "pipeline_run_metadata")
    show_kpi_row(metadata)

    if not metadata.empty and "pipeline_duration_ms" in metadata.columns:
        st.caption(f"Last full pipeline duration: {metadata.iloc[-1]['pipeline_duration_ms']} ms")

    tab_overview, tab_benchmark, tab_quality, tab_anomaly, tab_sensor_live, tab_lineage, tab_file_inspector, tab_dataset_inspector, tab_clinic_360, tab_data_upload, tab_mapping_studio, tab_governance, tab_corrections, tab_ai = st.tabs(
        [
            "Overview",
            "Benchmark",
            "Quality & Completeness",
            "Anomaly Detection",
            "Sensor Live Demo",
            "Data Origin & Mapping",
            "File Inspector",
            "Dataset Inspector",
            "Clinic 360",
            "Data Upload",
            "Mapping Studio",
            "Governance",
            "Alerts & Corrections",
            "PDF / Text AI Extraction",
        ]
    )

    with tab_overview:
        case_df = load_table(db_path, "tbCaseData")
        labs_df = load_table(db_path, "tbImportLabsData")
        nursing_df = load_table(db_path, "tbImportNursingDailyReportsData")
        device_df = load_table(db_path, "tbImportDeviceMotionData")
        nlp_df = load_table(db_path, "tbImportNursingNlpData")
        pdf_df = load_table(db_path, "tbImportPdfClinicalData")

        left, right = st.columns(2)

        with left:
            st.subheader("Source Record Volume")
            volumes = pd.DataFrame(
                {
                    "table": ["Case", "Labs", "Nursing", "Device Hourly", "Nursing NLP", "PDF Clinical"],
                    "rows": [len(case_df), len(labs_df), len(nursing_df), len(device_df), len(nlp_df), len(pdf_df)],
                }
            )
            fig = px.bar(volumes, x="table", y="rows", color="table")
            st.plotly_chart(fig, width="stretch")

        with right:
            st.subheader("Case Timeline Coverage")
            if "coSpecimen_datetime" in labs_df.columns:
                timeline = labs_df.copy()
                timeline["dt"] = pd.to_datetime(timeline["coSpecimen_datetime"], errors="coerce")
                timeline = timeline.dropna(subset=["dt"])
                by_day = timeline.groupby(timeline["dt"].dt.date).size().reset_index(name="count")
                if not by_day.empty:
                    fig = px.line(by_day, x="dt", y="count", markers=True)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No valid lab timestamps available.")
            else:
                st.info("Lab timestamp column unavailable.")

        step_metrics = load_table(db_path, "pipeline_step_metrics")
        st.subheader("Pipeline Runtime by Step")
        if not step_metrics.empty:
            fig = px.bar(step_metrics, x="step_name", y="duration_ms", color="step_name", title="Step Duration (ms)")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No step metrics available yet. Run preprocessing once.")

    with tab_benchmark:
        st.subheader("Benchmark")
        st.write("Track mapping accuracy, AI extraction accuracy, and missingness improvement before/after harmonization.")

        metrics = load_table(db_path, "benchmark_metrics")
        mapping_detail = load_table(db_path, "benchmark_mapping_detail")
        contracts = load_table(db_path, "data_contract_results")
        bronze = load_table(db_path, "bronze_raw_blob")
        gold = load_table(db_path, "gold_case_analytics")

        if metrics.empty:
            st.info("Benchmark metrics are not available yet. Run preprocessing to generate them.")
        else:
            metric_lookup = {r["metric_name"]: float(r["metric_value"]) for _, r in metrics.iterrows() if pd.notna(r.get("metric_value"))}
            metrics_idx = metrics.set_index("metric_name") if "metric_name" in metrics.columns else pd.DataFrame()

            def _metric_denominator(metric_name: str) -> int:
                if isinstance(metrics_idx, pd.DataFrame) and (not metrics_idx.empty) and metric_name in metrics_idx.index:
                    raw = pd.to_numeric(metrics_idx.loc[metric_name].get("denominator"), errors="coerce")
                    return int(raw) if pd.notna(raw) else 0
                return 0

            baseline_quality = metric_lookup.get("key_quality_before", 0.0)
            baseline_n = _metric_denominator("key_quality_before")
            post_quality = metric_lookup.get("key_quality_after", 0.0)
            post_n = _metric_denominator("key_quality_after")
            uplift_quality = metric_lookup.get("missingness_improvement_after_harmonization", 0.0)
            uplift_n = _metric_denominator("missingness_improvement_after_harmonization")

            extraction_accuracy = metric_lookup.get("extraction_accuracy_ai", 0.0)
            extraction_n = _metric_denominator("extraction_accuracy_ai")
            extraction_label = f"{extraction_accuracy:.2%}" if extraction_n > 0 else "insufficient sample"

            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline Quality (Key Fields)", f"{baseline_quality:.2%}", delta=f"n={baseline_n}")
            c2.metric("Post-Harmonization Quality", f"{post_quality:.2%}", delta=f"n={post_n}")
            c3.metric("Quality Uplift (pp)", f"{uplift_quality * 100:+.2f} pp", delta=f"n={uplift_n}")

            c4, c5 = st.columns(2)
            c4.metric("Mapping Accuracy (EPA)", f"{metric_lookup.get('mapping_accuracy_epa', 0.0):.2%}")
            c5.metric("Extraction Accuracy (AI)", extraction_label, delta=(f"n={extraction_n}" if extraction_n > 0 else "n=0"))

            st.caption(
                f"Supporting density delta (all fields): {metric_lookup.get('field_density_delta_all_fields', 0.0):+.2%}. "
                "Headline uplift is based on required keys and case-link completeness."
            )

            left, right = st.columns(2)
            with left:
                if not mapping_detail.empty:
                    st.plotly_chart(
                        px.bar(mapping_detail, x="source_table", y="mapping_accuracy", title="Mapping Accuracy by EPA Source"),
                        width="stretch",
                    )
                else:
                    st.info("No per-source mapping detail available.")
            with right:
                st.markdown("**Metric Details**")
                st.dataframe(metrics, width="stretch", height=260)

        st.markdown("**Data Contracts (Bronze / Silver / Gold)**")
        if contracts.empty:
            st.info("No contract validation output available yet.")
        else:
            pass_rate = float((contracts["status"].astype(str).str.lower() == "pass").mean()) if len(contracts) else 0.0
            st.caption(f"Contract pass rate: {pass_rate:.2%}")
            st.dataframe(contracts.sort_values(["layer", "target_object", "contract_name"]), width="stretch", height=300)

        l1, l2 = st.columns(2)
        with l1:
            st.markdown("**Bronze Raw Blob Preview**")
            if bronze.empty:
                st.info("Bronze layer not available yet.")
            else:
                st.dataframe(bronze[[c for c in ["source_domain", "source_file", "raw_row_count", "lineage_hash", "error_message"] if c in bronze.columns]], width="stretch", height=220)
        with l2:
            st.markdown("**Gold Analytics Preview**")
            if gold.empty:
                st.info("Gold analytics view not available yet.")
            else:
                st.dataframe(gold.head(200), width="stretch", height=220)

        st.markdown("**Business KPI Panel**")
        kpis = compute_business_kpis(db_path, metrics)
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Annualized Efficiency (€)", f"€{kpis.get('annualized_efficiency_savings_eur', 0.0):,.0f}")
        b2.metric("Automation Rate (proxy)", f"{kpis.get('automation_rate_proxy', 0.0):.1%}")
        b3.metric("Data Quality Uplift", f"{kpis.get('data_quality_uplift', 0.0):+.2%}")
        b4.metric("Prevented-Error Proxy", f"{int(kpis.get('prevented_error_proxy', 0.0))}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Monthly Hours Saved (est.)", f"{kpis.get('estimated_effort_saved_hours_monthly', 0.0):.1f}")
        extraction_denominator = 0
        if not metrics.empty and "metric_name" in metrics.columns:
            hit = metrics[metrics["metric_name"] == "extraction_accuracy_ai"]
            if not hit.empty:
                den_raw = pd.to_numeric(hit.iloc[0].get("denominator"), errors="coerce")
                extraction_denominator = int(den_raw) if pd.notna(den_raw) else 0
        c2.metric(
            "AI Extraction Success",
            f"{kpis.get('extraction_success_rate', 0.0):.1%}" if extraction_denominator > 0 else "insufficient sample",
            delta=(f"n={extraction_denominator}" if extraction_denominator > 0 else "n=0"),
        )
        c3.metric("Pending Manual Reviews", f"{int(kpis.get('pending_reviews', 0.0))}")

        with st.expander("KPI assumptions", expanded=False):
            st.write(
                "These are operational proxies, not accounting-certified financials: "
                "accepted mappings and successful AI extractions are translated into saved analyst minutes; "
                "annualized € uses a configurable default labor rate of 45 €/hour; prevented-error proxy combines approved review interventions and applied corrections."
            )

    with tab_quality:
        st.subheader("Completeness Metrics")
        completeness = load_table(db_path, "dq_completeness")
        issues = load_table(db_path, "dq_issues")

        if not completeness.empty:
            completeness_plot = completeness.copy()
            completeness_plot["non_null_rate"] = pd.to_numeric(completeness_plot["non_null_rate"], errors="coerce").fillna(0.0)
            completeness_plot["null_rate"] = 1.0 - completeness_plot["non_null_rate"]

            show_technical = st.checkbox("Include technical/placeholder fields (col1, unnamed_*, ids)", value=False)
            if not show_technical:
                field_series = completeness_plot["field"].astype(str).str.lower()
                technical_mask = (
                    field_series.str.match(r"^col\d+$")
                    | field_series.str.match(r"^unnamed_\d+$")
                    | field_series.isin(["coid", "cocaseid", "source_file", "cosource_file"])
                )
                completeness_plot = completeness_plot.loc[~technical_mask].copy()

            completeness_plot = apply_display_to_field_values(completeness_plot, "field", display_mode, coe_labels)

            top_incomplete = completeness_plot.sort_values(["null_rate", "null_count"], ascending=[False, False]).head(30)
            top_incomplete["field_label"] = top_incomplete["table_name"].astype(str) + " :: " + top_incomplete["field"].astype(str)
            fig = px.bar(
                top_incomplete,
                x="null_rate",
                y="field_label",
                color="table_name",
                orientation="h",
                title="Highest Missingness Fields",
            )
            fig.update_layout(barmode="group")
            fig.update_xaxes(range=[0, 1])
            fig.update_xaxes(tickformat=".0%", title="Missing Rate")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(apply_display_to_field_values(completeness, "field", display_mode, coe_labels), width="stretch", height=320)
        else:
            st.info("No completeness data available.")

        st.subheader("Quality Issues")
        if not issues.empty:
            st.dataframe(apply_display_to_field_values(issues, "field", display_mode, coe_labels), width="stretch", height=260)
        else:
            st.success("No quality issues detected.")

    with tab_anomaly:
        st.subheader("Detected Anomalies")
        anomalies = load_table(db_path, "dq_anomalies")
        if anomalies.empty:
            st.success("No anomalies detected by current rules.")
        else:
            anomalies_view = apply_display_to_field_values(anomalies, "field", display_mode, coe_labels)
            fig = px.bar(anomalies_view, x="field", y="count", color="source_table", barmode="group")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(anomalies_view, width="stretch")

    with tab_sensor_live:
        st.subheader("Sensor Live Demo")
        st.write("Simulate realistic multi-device patient monitoring streams with real-time style anomaly alerts.")

        if "sensor_profiles" not in st.session_state:
            st.session_state["sensor_profiles"] = _generate_sensor_profiles(device_count=4, seed=42)
        if "sensor_last_ts" not in st.session_state:
            st.session_state["sensor_last_ts"] = datetime.now(timezone.utc) - timedelta(minutes=5)
        if "sensor_stream_df" not in st.session_state:
            base_df, last_ts = _simulate_sensor_stream(
                profiles=st.session_state["sensor_profiles"],
                start_ts=st.session_state["sensor_last_ts"],
                seconds=300,
                seed=17,
            )
            st.session_state["sensor_stream_df"] = base_df
            st.session_state["sensor_last_ts"] = last_ts

        current_profiles = st.session_state.get("sensor_profiles", [])
        current_df = st.session_state.get("sensor_stream_df", pd.DataFrame())

        c1, c2, c3, c4 = st.columns(4)
        device_count = c1.slider("Devices", min_value=1, max_value=12, value=max(1, len(current_profiles)), step=1)
        if c2.button("Initialize stream", type="primary"):
            st.session_state["sensor_profiles"] = _generate_sensor_profiles(device_count=device_count, seed=42)
            start_ts = datetime.now(timezone.utc) - timedelta(minutes=5)
            reset_df, last_ts = _simulate_sensor_stream(
                profiles=st.session_state["sensor_profiles"],
                start_ts=start_ts,
                seconds=300,
                seed=17,
            )
            st.session_state["sensor_stream_df"] = reset_df
            st.session_state["sensor_last_ts"] = last_ts
            current_df = reset_df
            current_profiles = st.session_state["sensor_profiles"]

        if c3.button("Advance +30s"):
            add_df, last_ts = _simulate_sensor_stream(
                profiles=st.session_state["sensor_profiles"],
                start_ts=st.session_state["sensor_last_ts"],
                seconds=30,
                seed=random.randint(1, 1_000_000),
            )
            st.session_state["sensor_stream_df"] = pd.concat([st.session_state["sensor_stream_df"], add_df], ignore_index=True)
            st.session_state["sensor_last_ts"] = last_ts
            current_df = st.session_state["sensor_stream_df"]

        if c4.button("Advance +5m"):
            add_df, last_ts = _simulate_sensor_stream(
                profiles=st.session_state["sensor_profiles"],
                start_ts=st.session_state["sensor_last_ts"],
                seconds=300,
                seed=random.randint(1, 1_000_000),
            )
            st.session_state["sensor_stream_df"] = pd.concat([st.session_state["sensor_stream_df"], add_df], ignore_index=True)
            st.session_state["sensor_last_ts"] = last_ts
            current_df = st.session_state["sensor_stream_df"]

        if current_df.empty:
            st.info("No stream data yet. Click 'Initialize stream'.")
        else:
            current_df = current_df.copy()
            current_df["timestamp"] = pd.to_datetime(current_df["timestamp"], errors="coerce")
            current_df = current_df.dropna(subset=["timestamp"])

            window_min = st.slider("Display window (minutes)", min_value=1, max_value=30, value=10, step=1)
            max_ts = current_df["timestamp"].max()
            view_df = current_df[current_df["timestamp"] >= (max_ts - pd.Timedelta(minutes=window_min))].copy()
            alerts_df = view_df[view_df["is_alert"] == True].copy()

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Active devices", f"{len(current_profiles)}")
            k2.metric("Data points (window)", f"{len(view_df):,}")
            k3.metric("Alerts (window)", f"{len(alerts_df):,}")
            alert_rate = (len(alerts_df) / len(view_df)) if len(view_df) else 0.0
            k4.metric("Alert rate", f"{alert_rate:.2%}")

            lcol, rcol = st.columns(2)
            with lcol:
                hr_fig = px.line(
                    view_df,
                    x="timestamp",
                    y="heart_rate",
                    color="device_id",
                    title="Heart Rate Stream",
                )
                st.plotly_chart(hr_fig, width="stretch")
            with rcol:
                spo2_fig = px.line(
                    view_df,
                    x="timestamp",
                    y="spo2",
                    color="device_id",
                    title="SpO₂ Stream",
                )
                st.plotly_chart(spo2_fig, width="stretch")

            accel_fig = px.scatter(
                view_df,
                x="timestamp",
                y="accel_g",
                color="anomaly_type",
                symbol="device_id",
                title="Motion Signal & Fall Indicators",
            )
            st.plotly_chart(accel_fig, width="stretch")

            st.markdown("**Device-Specific Alert View**")
            if alerts_df.empty:
                st.success("No active alerts in selected window.")
            else:
                per_device = (
                    alerts_df.groupby(["device_id", "ward", "anomaly_type"], as_index=False)
                    .size()
                    .rename(columns={"size": "alert_count"})
                    .sort_values(["alert_count", "device_id"], ascending=[False, True])
                )
                st.dataframe(per_device, width="stretch", height=220)
                st.dataframe(
                    alerts_df[["timestamp", "device_id", "ward", "heart_rate", "spo2", "resp_rate", "accel_g", "anomaly_type"]]
                    .sort_values("timestamp", ascending=False)
                    .head(50),
                    width="stretch",
                    height=240,
                )

            st.download_button(
                "Download simulated sensor stream (CSV)",
                data=current_df.to_csv(index=False).encode("utf-8"),
                file_name="sensor_live_demo_stream.csv",
                mime="text/csv",
            )

    with tab_lineage:
        st.subheader("Field Origin and Mapping")
        lineage = load_table(db_path, "mapping_lineage")
        source_catalog = load_table(db_path, "source_file_catalog")
        if lineage.empty:
            st.info("No mapping lineage found.")
        else:
            left, right = st.columns(2)
            with left:
                src_counts = lineage.groupby("source_table", as_index=False).size().rename(columns={"size": "count"})
                st.plotly_chart(px.pie(src_counts, names="source_table", values="count", title="Mapping by Source"), width="stretch")
            with right:
                meaningful_lineage = lineage.copy()
                tf = meaningful_lineage["target_field"].astype(str).str.lower()
                tech_mask = tf.str.match(r"^(col\d+|unnamed_\d+|extra_\d+)$")
                meaningful_lineage = meaningful_lineage.loc[~tech_mask]
                meaningful_lineage = apply_display_to_field_values(meaningful_lineage, "target_field", display_mode, coe_labels)

                top_targets = meaningful_lineage["target_field"].value_counts().head(20).reset_index()
                top_targets.columns = ["target_field", "count"]
                st.plotly_chart(
                    px.bar(top_targets, x="count", y="target_field", orientation="h", title="Most Mapped Target Fields"),
                    width="stretch",
                )

            st.dataframe(apply_display_to_field_values(lineage, "target_field", display_mode, coe_labels), width="stretch", height=420)
            st.caption("Lineage tracks mapping from raw source fields to target unified columns.")

        st.subheader("Source File Catalog")
        if source_catalog.empty:
            st.info("No source file catalog available.")
        else:
            domain_counts = source_catalog.groupby("source_domain", as_index=False).size().rename(columns={"size": "count"})
            st.plotly_chart(px.bar(domain_counts, x="source_domain", y="count", title="Discovered Files by Domain"), width="stretch")
            st.dataframe(source_catalog, width="stretch", height=260)

    with tab_file_inspector:
        st.subheader("Raw → Normalized → Processed File Inspector")
        source_catalog = load_table(db_path, "source_file_catalog")
        if source_catalog.empty:
            st.info("No discovered source files available. Run preprocessing first.")
        else:
            choices = source_catalog.sort_values(["source_domain", "source_file"])
            label_map = {
                idx: f"[{row['source_domain']}] {row['source_file']}"
                for idx, row in choices.iterrows()
            }
            selected_idx = st.selectbox("Select source file", options=list(label_map.keys()), format_func=lambda x: label_map[x])
            selected = choices.loc[selected_idx]
            source_path = selected["source_path"]
            source_file = selected["source_file"]
            source_domain = selected["source_domain"]

            st.caption(f"Domain: {source_domain} | Path: {source_path}")

            sid_lookup = get_sid_lookup(cfg["paths"]["iid_sid_map_file"])
            item_name_lookup = get_item_name_lookup(cfg["paths"]["iid_sid_map_file"])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Raw preview**")
                raw_df = read_raw_preview(source_path)
                st.write(f"Rows previewed: {len(raw_df)}")
                st.dataframe(apply_display_to_columns(raw_df, display_mode, coe_labels), width="stretch", height=280)

            with col2:
                st.markdown("**Normalized preview**")
                try:
                    norm_df = normalize_single_file(source_path, cfg, sid_lookup, item_name_lookup).head(200)
                    st.write(f"Rows previewed: {len(norm_df)}")
                    st.dataframe(apply_display_to_columns(norm_df, display_mode, coe_labels), width="stretch", height=280)
                except Exception as ex:
                    st.error(f"Normalization failed: {ex}")

            with col3:
                st.markdown("**Processed preview**")
                processed = find_processed_rows_for_file(db_path, source_file)
                if not processed:
                    st.info("No processed rows found for this source file.")
                else:
                    table_pick = st.selectbox("Processed table", options=list(processed.keys()), key=f"proc_{source_file}")
                    st.write(f"Rows previewed: {len(processed[table_pick])}")
                    st.dataframe(apply_display_to_columns(processed[table_pick], display_mode, coe_labels), width="stretch", height=280)

    with tab_dataset_inspector:
        st.subheader("Dataset Health & Origin Inspector")
        dataset_choices = ["tbCaseData"] + list_import_tables(db_path)
        dataset = st.selectbox("Select dataset", dataset_choices)
        df = load_table(db_path, dataset)

        if df.empty:
            st.warning("Selected dataset is empty.")
        else:
            total_rows = len(df)
            total_cols = len(df.columns)
            avg_non_null = float((1.0 - (df.isna().sum().sum() / (total_rows * max(total_cols, 1)))))
            distinct_cases = int(df["case_id"].nunique()) if "case_id" in df.columns else 0
            distinct_patients = int(df["patient_id"].nunique()) if "patient_id" in df.columns else 0

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Rows", total_rows)
            c2.metric("Columns", total_cols)
            c3.metric("Distinct Cases", distinct_cases)
            c4.metric("Distinct Patients", distinct_patients)
            c5.metric("Avg Non-Null", f"{avg_non_null:.2%}")

            st.markdown("**Schema**")
            schema_df = pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]})
            schema_df = apply_display_to_field_values(schema_df, "column", display_mode, coe_labels)
            st.dataframe(schema_df, width="stretch", height=220)

            left, right = st.columns(2)
            with left:
                st.markdown("**Origins**")
                src_col = "source_file" if "source_file" in df.columns else ("coSource_file" if "coSource_file" in df.columns else None)
                if src_col:
                    src = df[src_col].fillna("(missing)").value_counts().head(20).reset_index()
                    src.columns = ["source_file", "rows"]
                    st.plotly_chart(px.bar(src, x="source_file", y="rows", title="Top Source Files"), width="stretch")
                else:
                    st.info("No source file column on this dataset.")

            with right:
                st.markdown("**Health Distribution**")
                null_by_col = (df.isna().mean().sort_values(ascending=False).head(20) * 100).reset_index()
                null_by_col.columns = ["field", "null_pct"]
                null_by_col = apply_display_to_field_values(null_by_col, "field", display_mode, coe_labels)
                st.plotly_chart(px.bar(null_by_col, x="field", y="null_pct", title="Top Null % Fields"), width="stretch")

            st.markdown("**Quality & Anomaly Signals**")
            issues = load_table(db_path, "dq_issues")
            anomalies = load_table(db_path, "dq_anomalies")
            completeness = load_table(db_path, "dq_completeness")

            issue_view = issues[issues["source_table"] == dataset] if (not issues.empty and "source_table" in issues.columns) else pd.DataFrame()
            anomaly_view = anomalies[anomalies["source_table"] == dataset] if (not anomalies.empty and "source_table" in anomalies.columns) else pd.DataFrame()
            completeness_view = completeness[completeness["table_name"] == dataset] if (not completeness.empty and "table_name" in completeness.columns) else pd.DataFrame()
            completeness_view = apply_display_to_field_values(completeness_view, "field", display_mode, coe_labels)
            issue_view = apply_display_to_field_values(issue_view, "field", display_mode, coe_labels)
            anomaly_view = apply_display_to_field_values(anomaly_view, "field", display_mode, coe_labels)

            qa1, qa2, qa3 = st.columns(3)
            with qa1:
                st.markdown("**Completeness**")
                st.dataframe(completeness_view, width="stretch", height=220)
            with qa2:
                st.markdown("**Issues**")
                st.dataframe(issue_view, width="stretch", height=220)
            with qa3:
                st.markdown("**Anomalies**")
                st.dataframe(anomaly_view, width="stretch", height=220)

    with tab_clinic_360:
        st.subheader("Clinic 360")
        st.write("View all discovered source files and consolidated records for a selected clinic.")

        source_catalog = annotate_catalog_with_clinic(load_table(db_path, "source_file_catalog"))
        if source_catalog.empty:
            st.info("No source catalog available. Run preprocessing first.")
        else:
            clinics = sorted(source_catalog["clinic_id"].dropna().astype(str).unique().tolist())
            selected_clinic = st.selectbox("Select clinic", clinics)

            clinic_catalog = source_catalog[source_catalog["clinic_id"] == selected_clinic].copy()
            clinic_files = clinic_catalog["source_file"].dropna().astype(str).unique().tolist()

            c1, c2, c3 = st.columns(3)
            c1.metric("Source Files", int(len(clinic_files)))
            c2.metric("Domains", int(clinic_catalog["source_domain"].nunique() if "source_domain" in clinic_catalog.columns else 0))
            c3.metric("Catalog Rows", int(len(clinic_catalog)))

            st.markdown("**Source Files for Clinic**")
            st.dataframe(clinic_catalog.sort_values(["source_domain", "source_file"]), width="stretch", height=220)

            coverage_df, unified_df, clinic_cases = build_clinic_unified_data(
                db_path=db_path,
                clinic_source_files=clinic_files,
            )

            left, right = st.columns(2)
            with left:
                st.markdown("**Per-Table Coverage**")
                st.dataframe(coverage_df, width="stretch", height=260)
            with right:
                nonzero = coverage_df[coverage_df["rows"] > 0].copy()
                if nonzero.empty:
                    st.info("No processed table rows linked to this clinic yet.")
                else:
                    st.plotly_chart(px.bar(nonzero, x="table", y="rows", title="Rows by Table"), width="stretch")

            st.markdown("**Linked Case Master Data (tbCaseData)**")
            if clinic_cases.empty:
                st.info("No linked case rows found for this clinic.")
            else:
                clinic_cases_view = apply_display_to_columns(clinic_cases, display_mode, coe_labels)
                st.dataframe(clinic_cases_view, width="stretch", height=220)

            st.markdown("**Unified Clinic Records (All Tables)**")
            if unified_df.empty:
                st.info("No unified rows found for this clinic.")
            else:
                unified_view = apply_display_to_columns(unified_df, display_mode, coe_labels)
                st.dataframe(unified_view, width="stretch", height=340)
                st.download_button(
                    "Download unified clinic table (CSV)",
                    data=unified_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected_clinic}_unified_records.csv",
                    mime="text/csv",
                )

            st.markdown("**AI Clinic Insights**")
            insight_mode_label = st.radio(
                "Insight mode",
                options=["Analyze", "Ideate", "Both"],
                horizontal=True,
                key=f"clinic_ai_mode_{selected_clinic}",
            )
            insight_mode = insight_mode_label.lower()
            focus_prompt = st.text_area(
                "Optional focus for AI",
                value="",
                placeholder="e.g., focus on data quality risks, medication safety, and workflow bottlenecks",
                key=f"clinic_ai_focus_{selected_clinic}",
                height=80,
            )

            if st.button("Run AI Clinic Insights", key=f"clinic_ai_run_{selected_clinic}"):
                try:
                    context_payload = build_clinic_ai_context(
                        selected_clinic=selected_clinic,
                        clinic_catalog=clinic_catalog,
                        coverage_df=coverage_df,
                        clinic_cases=clinic_cases,
                        unified_df=unified_df,
                    )
                    insights = ai_clinic_insights(
                        mode=insight_mode,
                        context_payload=context_payload,
                        user_focus=focus_prompt,
                        model=model,
                        strict_validation=strict_ai_validation,
                        max_retries=strict_ai_max_retries,
                    )
                    st.session_state[f"clinic_ai_out_{selected_clinic}"] = insights
                    log_ai_extraction(
                        db_path=db_path,
                        source_type="clinic_insights",
                        input_excerpt=f"clinic={selected_clinic}; mode={insight_mode}; focus={focus_prompt}",
                        schema_hint='{"mode":null,"summary":null,"insights":[],"actions":[]}',
                        result_json=json.dumps(insights, ensure_ascii=False),
                        status="success",
                        error_message=None,
                    )
                except Exception as e:
                    log_ai_extraction(
                        db_path=db_path,
                        source_type="clinic_insights",
                        input_excerpt=f"clinic={selected_clinic}; mode={insight_mode}; focus={focus_prompt}",
                        schema_hint='{"mode":null,"summary":null,"insights":[],"actions":[]}',
                        result_json=None,
                        status="error",
                        error_message=str(e),
                    )
                    st.error(str(e))

            clinic_ai_out = st.session_state.get(f"clinic_ai_out_{selected_clinic}")
            if clinic_ai_out:
                if isinstance(clinic_ai_out, dict) and clinic_ai_out.get("summary"):
                    st.info(str(clinic_ai_out.get("summary")))

                insights_rows = clinic_ai_out.get("insights", []) if isinstance(clinic_ai_out, dict) else []
                if insights_rows:
                    st.markdown("**Insights**")
                    st.dataframe(pd.DataFrame(insights_rows), width="stretch", height=220)

                actions_rows = clinic_ai_out.get("actions", []) if isinstance(clinic_ai_out, dict) else []
                if actions_rows:
                    st.markdown("**Suggested Actions**")
                    st.dataframe(pd.DataFrame(actions_rows), width="stretch", height=220)

                st.json(clinic_ai_out)

            st.markdown("**Executive Report Export**")
            report_ai_mode = st.selectbox(
                "Report AI mode",
                options=["Use current insights", "Regenerate (Both) for report"],
                key=f"report_ai_mode_{selected_clinic}",
            )
            clinic_ai_for_report = clinic_ai_out if isinstance(clinic_ai_out, dict) else None

            if report_ai_mode == "Regenerate (Both) for report" and st.button(
                "Generate AI Analysis for Report", key=f"report_ai_generate_{selected_clinic}"
            ):
                try:
                    context_payload = build_clinic_ai_context(
                        selected_clinic=selected_clinic,
                        clinic_catalog=clinic_catalog,
                        coverage_df=coverage_df,
                        clinic_cases=clinic_cases,
                        unified_df=unified_df,
                    )
                    clinic_ai_for_report = ai_clinic_insights(
                        mode="both",
                        context_payload=context_payload,
                        user_focus="Executive report generation with concrete, high-impact recommendations.",
                        model=model,
                        strict_validation=strict_ai_validation,
                        max_retries=strict_ai_max_retries,
                    )
                    st.session_state[f"clinic_ai_out_{selected_clinic}"] = clinic_ai_for_report
                    log_ai_extraction(
                        db_path=db_path,
                        source_type="clinic_report_ai",
                        input_excerpt=f"clinic={selected_clinic}; mode=both; report_generation=true",
                        schema_hint='{"mode":null,"summary":null,"insights":[],"actions":[]}',
                        result_json=json.dumps(clinic_ai_for_report, ensure_ascii=False),
                        status="success",
                        error_message=None,
                    )
                    st.success("AI analysis refreshed for report export.")
                except Exception as e:
                    st.error(str(e))

            benchmark_metrics = load_table(db_path, "benchmark_metrics")
            kpis_for_report = compute_business_kpis(db_path, benchmark_metrics)
            report_md = build_clinic_executive_report_markdown(
                clinic_id=selected_clinic,
                clinic_catalog=clinic_catalog,
                coverage_df=coverage_df,
                clinic_cases=clinic_cases,
                clinic_ai_out=clinic_ai_for_report if isinstance(clinic_ai_for_report, dict) else None,
                kpis=kpis_for_report,
            )
            st.download_button(
                "Download Executive Report (Markdown)",
                data=report_md.encode("utf-8"),
                file_name=f"{selected_clinic}_executive_report.md",
                mime="text/markdown",
            )

            pdf_bytes = markdown_to_pdf_bytes(report_md)
            if pdf_bytes:
                st.download_button(
                    "Download Executive Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"{selected_clinic}_executive_report.pdf",
                    mime="application/pdf",
                )
            else:
                st.caption("PDF export requires reportlab in the environment. Markdown export is fully available.")

    with tab_data_upload:
        st.subheader("Upload New Data")
        st.write("Upload new CSV/PDF files into ingestion folders and optionally trigger preprocessing.")

        input_roots = cfg.get("input_roots", [])
        target_options = list(input_roots)
        pdf_inbox = cfg.get("paths", {}).get("pdf_inbox_dir")
        if pdf_inbox and pdf_inbox not in target_options:
            target_options.append(pdf_inbox)

        if not target_options:
            st.warning("No upload target folder configured.")
        else:
            target_dir = st.selectbox("Target folder", options=target_options, format_func=lambda p: f"{Path(p).name} ({p})")
            subfolder = st.text_input("Optional subfolder", value="")
            overwrite = st.checkbox("Overwrite if file exists", value=False)
            run_after = st.checkbox("Run preprocessing after upload", value=True)

            uploads = st.file_uploader(
                "Select files",
                type=["csv", "pdf", "txt", "json"],
                accept_multiple_files=True,
            )

            if st.button("Save uploaded files", type="primary"):
                if not uploads:
                    st.warning("Select at least one file first.")
                else:
                    result_df = save_uploaded_files(
                        uploaded_files=uploads,
                        target_dir=target_dir,
                        subfolder=subfolder,
                        overwrite=overwrite,
                    )
                    st.session_state["upload_results"] = result_df
                    saved_count = int((result_df["status"] == "saved").sum()) if not result_df.empty else 0
                    st.success(f"Upload complete. Saved {saved_count} file(s).")

                    if run_after and saved_count > 0:
                        run_pipeline(force=False)
                        st.cache_data.clear()
                        st.success("Preprocessing completed with new files.")

            upload_results = st.session_state.get("upload_results", pd.DataFrame())
            if not upload_results.empty:
                st.markdown("**Upload Results**")
                st.dataframe(upload_results, width="stretch", height=240)

    with tab_mapping_studio:
        st.subheader("Mapping Studio")
        st.write("AI-first column harmonization with human approval. Accepted mappings are persisted and reused.")
        st.caption("Context rule enforced in AI prompt: German 'Fall' means case/encounter, not a physical floor-fall.")

        lineage = load_table(db_path, "mapping_lineage")
        if lineage.empty:
            st.info("No mapping lineage available. Run preprocessing first.")
        else:
            all_sources = sorted(lineage["source_table"].dropna().astype(str).unique().tolist()) if "source_table" in lineage.columns else []
            source_filter = st.multiselect("Limit to source tables", options=all_sources, default=all_sources)

            probe_raw = st.text_area(
                "Fields to evaluate (one per line, or comma-separated)",
                value="case_id\npatient_id\nward\nhemoglobin\ndiagnosis",
                height=140,
            )

            filtered_lineage = lineage.copy()
            if source_filter and "source_table" in filtered_lineage.columns:
                filtered_lineage = filtered_lineage[filtered_lineage["source_table"].astype(str).isin(source_filter)]

            source_scope = ",".join(source_filter) if source_filter else "*"
            accepted = load_accepted_mappings(db_path, source_scope=source_scope)

            st.markdown("**Accepted mappings (reused before AI)**")
            if accepted.empty:
                st.caption("No accepted mappings stored for this source scope yet.")
            else:
                st.dataframe(accepted.sort_values("decided_at_utc", ascending=False), width="stretch", height=200)

            probes = parse_probe_fields(probe_raw)
            accepted_lookup = {
                _normalize_probe_token(r["source_field"]): r["canonical_name"]
                for _, r in accepted.iterrows()
                if str(r.get("decision", "")).lower() == "accepted"
            }
            pre_mapped = []
            unresolved = []
            for p in probes:
                mapped = accepted_lookup.get(_normalize_probe_token(p))
                if mapped:
                    pre_mapped.append(
                        {
                            "probe_field": p,
                            "canonical_name": mapped,
                            "confidence": "accepted",
                            "rationale": "Loaded from accepted_mappings.",
                            "alternatives": [],
                        }
                    )
                else:
                    unresolved.append(p)

            canonical_targets = [
                "patient_id",
                "case_id",
                "iid",
                "sex",
                "age_years",
                "ward",
                "admission_date",
                "discharge_date",
                "specimen_datetime",
                "medication_name",
                "medication_code_atc",
                "dose",
                "dose_unit",
                "frequency",
                "timestamp",
                "lab_name",
                "lab_value",
            ]

            if st.button("Run AI Mapping", type="primary"):
                try:
                    ai_result = {"assessments": []}
                    if unresolved:
                        lineage_context = _build_lineage_context(filtered_lineage, unresolved)
                        ai_result = ai_map_fields(
                            probe_fields=unresolved,
                            candidate_targets=canonical_targets,
                            lineage_context=lineage_context,
                            model=model,
                            strict_validation=strict_ai_validation,
                            max_retries=strict_ai_max_retries,
                        )

                    merged = list(pre_mapped)
                    if isinstance(ai_result, dict):
                        merged.extend(ai_result.get("assessments", []))

                    merged = calibrate_mapping_assessments(merged)

                    for rec in merged:
                        conf = str(rec.get("confidence", "")).lower()
                        if conf == "low":
                            enqueue_mapping_review(
                                db_path=db_path,
                                source_scope=source_scope,
                                probe_field=str(rec.get("probe_field", "")),
                                proposed_value=str(rec.get("canonical_name", "")),
                                confidence=conf,
                                rationale=str(rec.get("rationale", "Low-confidence mapping requires review")),
                            )

                    final = {"assessments": merged}
                    st.session_state["mapping_studio_ai"] = final

                    log_ai_extraction(
                        db_path=db_path,
                        source_type="mapping_studio",
                        input_excerpt="; ".join(probes),
                        schema_hint='{"assessments":[{"probe_field":null,"canonical_name":null,"confidence":null,"rationale":null,"alternatives":[]}]}',
                        result_json=json.dumps(final, ensure_ascii=False),
                        status="success",
                        error_message=None,
                    )
                except Exception as e:
                    log_ai_extraction(
                        db_path=db_path,
                        source_type="mapping_studio",
                        input_excerpt="; ".join(probes),
                        schema_hint='{"assessments":[]}',
                        result_json=None,
                        status="error",
                        error_message=str(e),
                    )
                    st.error(str(e))

            ai_out = st.session_state.get("mapping_studio_ai")
            if ai_out:
                assess = ai_out.get("assessments", []) if isinstance(ai_out, dict) else []
                if assess:
                    ai_df = pd.DataFrame(assess)
                    ai_df = apply_display_to_field_values(ai_df, "canonical_name", display_mode, coe_labels)
                    st.markdown("**AI mapping output**")
                    st.dataframe(ai_df, width="stretch", height=280)

                    st.markdown("**Accept mapping**")
                    pick_probe = st.selectbox("Source field", options=ai_df["probe_field"].astype(str).tolist(), key="mapping_accept_probe")
                    recommended = ai_df.loc[ai_df["probe_field"] == pick_probe, "canonical_name"].astype(str).head(1)
                    default_target = recommended.iloc[0] if not recommended.empty and recommended.iloc[0] != "None" else canonical_targets[0]
                    target_pick = st.selectbox(
                        "Canonical target",
                        options=canonical_targets,
                        index=canonical_targets.index(default_target) if default_target in canonical_targets else 0,
                        key="mapping_accept_target",
                    )
                    rationale_pick = ai_df.loc[ai_df["probe_field"] == pick_probe, "rationale"].astype(str).head(1)
                    confidence_pick = ai_df.loc[ai_df["probe_field"] == pick_probe, "confidence"].astype(str).head(1)
                    selected_confidence = confidence_pick.iloc[0].lower() if not confidence_pick.empty else "medium"

                    if selected_confidence == "low":
                        st.warning("Low-confidence mapping: mandatory manual approval required via review queue.")

                    if st.button("Accept selected mapping"):
                        if not has_permission(active_role, "mapping_accept"):
                            st.error("Your role is not allowed to accept mappings.")
                        elif selected_confidence == "low":
                            enqueue_mapping_review(
                                db_path=db_path,
                                source_scope=source_scope,
                                probe_field=pick_probe,
                                proposed_value=target_pick,
                                confidence=selected_confidence,
                                rationale=rationale_pick.iloc[0] if not rationale_pick.empty else "Low-confidence mapping requires manual approval",
                            )
                            log_audit_event(
                                db_path=db_path,
                                user_id=active_user,
                                user_role=active_role,
                                event_type="mapping_queued_for_review",
                                entity_type="mapping",
                                entity_key=f"{source_scope}:{pick_probe}",
                                before_value=None,
                                after_value=json.dumps({"canonical_name": target_pick, "confidence": selected_confidence}, ensure_ascii=False),
                                note="Low-confidence mapping sent to review queue",
                                status="pending",
                            )
                            st.warning("Queued for manual approval.")
                        else:
                            save_accepted_mapping(
                                db_path=db_path,
                                source_field=pick_probe,
                                canonical_name=target_pick,
                                source_scope=source_scope,
                                confidence=selected_confidence,
                                rationale=rationale_pick.iloc[0] if not rationale_pick.empty else "Accepted in Mapping Studio",
                            )
                            log_audit_event(
                                db_path=db_path,
                                user_id=active_user,
                                user_role=active_role,
                                event_type="mapping_accepted",
                                entity_type="mapping",
                                entity_key=f"{source_scope}:{pick_probe}",
                                before_value=None,
                                after_value=json.dumps({"canonical_name": target_pick, "confidence": selected_confidence}, ensure_ascii=False),
                                note="Mapping accepted",
                                status="success",
                            )
                            st.success(f"Accepted mapping saved: {pick_probe} -> {target_pick}")
                            st.cache_data.clear()

                st.json(ai_out)

            st.markdown("**Confidence-Gated Review Queue**")
            queue_df = load_review_queue(db_path, source_scope=source_scope)
            if queue_df.empty:
                st.caption("No review items currently in queue.")
            else:
                st.dataframe(queue_df, width="stretch", height=220)
                pending = queue_df[queue_df["status"].astype(str).str.lower() == "pending"]
                if not pending.empty:
                    review_id = st.selectbox("Pending review item", options=pending["review_id"].astype(int).tolist())
                    chosen = pending[pending["review_id"] == review_id].iloc[0]
                    st.write(
                        f"Field: {chosen['probe_field']} | Proposed: {chosen['proposed_value']} | Confidence: {chosen['confidence']}"
                    )
                    review_note = st.text_input("Review note", value="", key=f"review_note_{review_id}")
                    c1, c2 = st.columns(2)
                    if c1.button("Approve", key=f"review_approve_{review_id}"):
                        if not has_permission(active_role, "review_approve"):
                            st.error("Your role is not allowed to approve queued mappings.")
                        else:
                            save_accepted_mapping(
                                db_path=db_path,
                                source_field=str(chosen["probe_field"]),
                                canonical_name=str(chosen["proposed_value"]),
                                source_scope=str(chosen["source_scope"]),
                                confidence=str(chosen["confidence"]),
                                rationale=str(chosen.get("rationale", "Approved from review queue")),
                            )
                            decide_review_item(db_path, int(review_id), "approved", active_user, review_note)
                            log_audit_event(
                                db_path=db_path,
                                user_id=active_user,
                                user_role=active_role,
                                event_type="mapping_review_approved",
                                entity_type="mapping_review",
                                entity_key=str(review_id),
                                before_value=None,
                                after_value=json.dumps({"probe_field": chosen["probe_field"], "canonical_name": chosen["proposed_value"]}, ensure_ascii=False),
                                note=review_note or "Approved from queue",
                                status="success",
                            )
                            st.success("Review approved and mapping promoted to accepted mappings.")
                            st.cache_data.clear()
                    if c2.button("Reject", key=f"review_reject_{review_id}"):
                        if not has_permission(active_role, "review_approve"):
                            st.error("Your role is not allowed to reject queued mappings.")
                        else:
                            decide_review_item(db_path, int(review_id), "rejected", active_user, review_note)
                            log_audit_event(
                                db_path=db_path,
                                user_id=active_user,
                                user_role=active_role,
                                event_type="mapping_review_rejected",
                                entity_type="mapping_review",
                                entity_key=str(review_id),
                                before_value=None,
                                after_value=None,
                                note=review_note or "Rejected from queue",
                                status="success",
                            )
                            st.info("Review item rejected.")
                            st.cache_data.clear()

    with tab_governance:
        st.subheader("Governance & Audit")
        st.write("Track access role, mapping/correction actions, and confidence-gated review decisions.")

        st.markdown("**Current Access Context**")
        st.write(f"User: {active_user} | Role: {active_role}")
        role_matrix = pd.DataFrame(
            [
                {"role": "admin", "mapping_accept": True, "review_approve": True, "correction_edit": True, "view_audit": True},
                {"role": "data_steward", "mapping_accept": True, "review_approve": True, "correction_edit": True, "view_audit": True},
                {"role": "analyst", "mapping_accept": False, "review_approve": False, "correction_edit": False, "view_audit": True},
                {"role": "viewer", "mapping_accept": False, "review_approve": False, "correction_edit": False, "view_audit": False},
            ]
        )
        st.dataframe(role_matrix, width="stretch", height=170)

        if not has_permission(active_role, "view_audit"):
            st.warning("Your role cannot view governance logs.")
        else:
            audit_df = load_audit_log(db_path)
            queue_df = load_review_queue(db_path)

            st.markdown("**Audit Trail**")
            if audit_df.empty:
                st.info("No audit events logged yet.")
            else:
                event_options = sorted(audit_df["event_type"].dropna().astype(str).unique().tolist())
                event_filter = st.multiselect("Filter events", options=event_options, default=event_options)
                view = audit_df[audit_df["event_type"].astype(str).isin(event_filter)] if event_filter else audit_df
                st.dataframe(view.sort_values("audit_id", ascending=False), width="stretch", height=260)

            st.markdown("**Confidence Review Queue**")
            if queue_df.empty:
                st.info("No queue items available.")
            else:
                st.dataframe(queue_df, width="stretch", height=220)

    with tab_corrections:
        st.subheader("Alerts and Manual Corrections")
        issues = load_table(db_path, "dq_issues")
        st.write("Use this panel to review flagged records and create a correction patch file.")
        can_edit_corrections = has_permission(active_role, "correction_edit")
        if not can_edit_corrections:
            st.warning("Your current role is read-only for database corrections.")

        if issues.empty:
            st.success("No alerts available.")
        else:
            st.dataframe(apply_display_to_field_values(issues, "field", display_mode, coe_labels), width="stretch")

        table_choice = st.selectbox("Choose target table", list_import_tables(db_path))
        target_df = load_table(db_path, table_choice)
        st.write(f"Rows: {len(target_df)}")

        if not target_df.empty:
            preview = target_df.head(200).copy()
            edited = st.data_editor(preview, width="stretch", num_rows="fixed")

            if st.button("Apply edits to database", type="primary", disabled=not can_edit_corrections):
                ok, message = persist_table_edits(
                    db_path=db_path,
                    processed_root=cfg["paths"]["processed_root"],
                    table_name=table_choice,
                    edited_df=edited,
                )
                st.cache_data.clear()
                if ok:
                    before_payload, after_payload, changed_rows = build_change_audit_payload(preview, edited)
                    log_audit_event(
                        db_path=db_path,
                        user_id=active_user,
                        user_role=active_role,
                        event_type="correction_applied",
                        entity_type="table",
                        entity_key=table_choice,
                        before_value=before_payload,
                        after_value=after_payload,
                        note=f"Manual correction applied on {table_choice}; changed_rows={changed_rows}",
                        status="success",
                    )
                    st.success(message)
                else:
                    st.error(message)

            patch_csv = edited.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download correction patch (CSV)",
                data=patch_csv,
                file_name=f"{table_choice}_corrections.csv",
                mime="text/csv",
            )

    with tab_ai:
        st.subheader("AI Extraction for PDF and Free Text")
        schema_hint = st.text_area(
            "Target schema hint (JSON template or field list)",
            value='{"case_id": null, "patient_id": null, "ward": null, "findings": [], "actions": []}',
            height=120,
        )

        txt = st.text_area("Clinical free text", height=180)
        if st.button("Extract from text with Anthropic"):
            try:
                result = anthropic_extract_structured(
                    txt,
                    schema_hint=schema_hint,
                    model=model,
                    strict_validation=strict_ai_validation,
                    max_retries=strict_ai_max_retries,
                )
                st.json(result)
                log_ai_extraction(
                    db_path=db_path,
                    source_type="text",
                    input_excerpt=txt,
                    schema_hint=schema_hint,
                    result_json=json.dumps(result, ensure_ascii=False),
                    status="success",
                    error_message=None,
                )
            except Exception as e:
                log_ai_extraction(
                    db_path=db_path,
                    source_type="text",
                    input_excerpt=txt,
                    schema_hint=schema_hint,
                    result_json=None,
                    status="error",
                    error_message=str(e),
                )
                st.error(str(e))

        pdf_file = st.file_uploader("Upload PDF document", type=["pdf"])
        if st.button("Extract from PDF with Anthropic"):
            if pdf_file is None:
                st.warning("Please upload a PDF file first.")
            else:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(pdf_file.getvalue())
                        tmp_path = tmp.name
                    result = extract_from_pdf_with_ai(
                        tmp_path,
                        schema_hint=schema_hint,
                        model=model,
                        strict_validation=strict_ai_validation,
                        max_retries=strict_ai_max_retries,
                    )
                    st.json(result)
                    log_ai_extraction(
                        db_path=db_path,
                        source_type="pdf",
                        input_excerpt=pdf_file.name,
                        schema_hint=schema_hint,
                        result_json=json.dumps(result, ensure_ascii=False),
                        status="success",
                        error_message=None,
                    )
                except Exception as e:
                    log_ai_extraction(
                        db_path=db_path,
                        source_type="pdf",
                        input_excerpt=pdf_file.name if pdf_file else "",
                        schema_hint=schema_hint,
                        result_json=None,
                        status="error",
                        error_message=str(e),
                    )
                    st.error(str(e))

        st.subheader("AI Extraction History")
        ai_logs = load_table(db_path, "ai_extraction_log")
        if ai_logs.empty:
            st.info("No AI extraction calls logged yet.")
        else:
            st.dataframe(ai_logs.sort_values("log_id", ascending=False), width="stretch", height=240)

        st.caption(
            "This deployment is designed for on-prem execution. No cloud storage is required; data stays local unless you route API calls externally."
        )


if __name__ == "__main__":
    main()
