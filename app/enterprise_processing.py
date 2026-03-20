from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd

from app.ai_extraction import anthropic_extract_structured, extract_pdf_text
from app.utils import normalize_case_id


def _file_sig(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_case_patient_ward(text: str) -> tuple[str | None, str | None, str | None]:
    case_match = re.search(r"(?:case[_\s-]?id|fallnr|fall|encounter[_\s-]?id)\s*[:=]?\s*([A-Za-z0-9\-]+)", text, flags=re.IGNORECASE)
    patient_match = re.search(r"(?:patient[_\s-]?id)\s*[:=]?\s*([A-Za-z0-9\-]+)", text, flags=re.IGNORECASE)
    ward_match = re.search(r"(?:ward|station)\s*[:=]?\s*([A-Za-z0-9\-_/ ]{2,})", text, flags=re.IGNORECASE)

    case_id = normalize_case_id(case_match.group(1)) if case_match else None
    patient_id = patient_match.group(1).strip() if patient_match else None
    ward = ward_match.group(1).strip() if ward_match else None
    return case_id, patient_id, ward


def _load_pdf_manifest(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_pdf_manifest(manifest_path: Path, manifest: dict[str, str]) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def process_pdf_inbox(
    pdf_dir: str,
    processed_root: str,
    model: str,
    enable_ai_enrichment: bool,
    max_ai_rows_per_run: int,
    strict_validation: bool = False,
    strict_max_retries: int = 2,
    additional_pdf_roots: list[str] | None = None,
) -> pd.DataFrame:
    pdf_path = Path(pdf_dir)
    pdf_path.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(processed_root) / "pdf_manifest.json"

    known = _load_pdf_manifest(manifest_path)
    rows: list[dict[str, object]] = []

    files = list(sorted(pdf_path.glob("*.pdf")))
    for root in additional_pdf_roots or []:
        root_path = Path(root)
        if root_path.exists():
            files.extend(sorted(root_path.rglob("*.pdf")))

    dedup = {}
    for f in files:
        dedup[str(f.resolve())] = f
    files = list(dedup.values())

    ai_budget = max(0, int(max_ai_rows_per_run))

    for idx, f in enumerate(files, start=1):
        sig = _file_sig(f)
        file_key = str(f.resolve())
        if known.get(file_key) == sig:
            continue

        raw_text = ""
        case_id = None
        patient_id = None
        ward = None
        ai_json = None
        ai_status = "not_requested"
        error = None

        try:
            raw_text = extract_pdf_text(str(f))
            case_id, patient_id, ward = _extract_case_patient_ward(raw_text)

            if enable_ai_enrichment and ai_budget > 0 and os.getenv("ANTHROPIC_API_KEY"):
                schema_hint = (
                    '{"case_id": null, "patient_id": null, "ward": null, "document_type": null, '
                    '"key_findings": [], "procedures": [], "medications": []}'
                )
                ai_payload = anthropic_extract_structured(
                    raw_text,
                    schema_hint=schema_hint,
                    model=model,
                    strict_validation=strict_validation,
                    max_retries=strict_max_retries,
                    required_keys=["case_id", "patient_id", "ward"],
                )
                ai_json = json.dumps(ai_payload, ensure_ascii=False)
                case_id = case_id or normalize_case_id(ai_payload.get("case_id"))
                patient_id = patient_id or (str(ai_payload.get("patient_id")).strip() if ai_payload.get("patient_id") else None)
                ward = ward or ai_payload.get("ward")
                ai_status = "success"
                ai_budget -= 1
            elif enable_ai_enrichment and not os.getenv("ANTHROPIC_API_KEY"):
                ai_status = "skipped_no_api_key"

            known[file_key] = sig
        except Exception as ex:
            ai_status = "error"
            error = str(ex)

        rows.append(
            {
                "coId": idx,
                "case_id": case_id,
                "patient_id": patient_id,
                "coWard": ward,
                "coSource_file": f.name,
                "coDocument_text": raw_text[:12000],
                "coStructured_json": ai_json,
                "coAi_status": ai_status,
                "coError": error,
            }
        )

    _save_pdf_manifest(manifest_path, known)
    if not rows:
        return pd.DataFrame(
            columns=[
                "coId",
                "case_id",
                "patient_id",
                "coWard",
                "coSource_file",
                "coDocument_text",
                "coStructured_json",
                "coAi_status",
                "coError",
            ]
        )
    return pd.DataFrame(rows)


def normalize_nursing_notes(
    nursing_df: pd.DataFrame,
    model: str,
    enable_ai_enrichment: bool,
    max_ai_rows_per_run: int,
    strict_validation: bool = False,
    strict_max_retries: int = 2,
) -> pd.DataFrame:
    if nursing_df.empty or "coNursing_note_free_text" not in nursing_df.columns:
        return pd.DataFrame(
            columns=[
                "case_id",
                "patient_id",
                "coNlpPainFlag",
                "coNlpFeverFlag",
                "coNlpMobilityIssueFlag",
                "coNlpOrientationIssueFlag",
                "coNlpWoundFlag",
                "coNlpMedicationActionFlag",
                "coNlpMonitoringActionFlag",
                "coNlpImprovementFlag",
                "coNlpSummary",
                "coNlpAiStatus",
            ]
        )

    rules = {
        "coNlpPainFlag": [r"\bpain\b", r"\bschmerz"],
        "coNlpFeverFlag": [r"\bfever\b", r"\bfieber"],
        "coNlpMobilityIssueFlag": [r"mobility", r"immobil", r"transfer", r"geh"],
        "coNlpOrientationIssueFlag": [r"orientation", r"disorient", r"delir", r"verwirr"],
        "coNlpWoundFlag": [r"wound", r"ulcer", r"dekub"],
        "coNlpMedicationActionFlag": [r"medic", r"drug", r"dose", r"verord", r"gabe"],
        "coNlpMonitoringActionFlag": [r"monitor", r"vitals", r"beobacht", r"kontroll"],
        "coNlpImprovementFlag": [r"improv", r"stabil", r"better", r"gebessert"],
    }

    out = nursing_df[[c for c in ["case_id", "patient_id", "coNursing_note_free_text"] if c in nursing_df.columns]].copy()
    txt = out["coNursing_note_free_text"].fillna("").astype(str)

    for target, patterns in rules.items():
        regex = "|".join(patterns)
        out[target] = txt.str.contains(regex, case=False, regex=True).astype("Int64")

    out["coNlpSummary"] = txt.str.slice(0, 240)
    out["coNlpAiStatus"] = "not_requested"

    ai_budget = max(0, int(max_ai_rows_per_run))
    if enable_ai_enrichment and ai_budget > 0 and os.getenv("ANTHROPIC_API_KEY"):
        schema_hint = '{"clinical_summary": null}'
        ai_indices = out.index[:ai_budget]
        for i in ai_indices:
            try:
                result = anthropic_extract_structured(
                    txt.iloc[i],
                    schema_hint=schema_hint,
                    model=model,
                    strict_validation=strict_validation,
                    max_retries=strict_max_retries,
                    required_keys=["clinical_summary"],
                )
                summary = result.get("clinical_summary") if isinstance(result, dict) else None
                if summary:
                    out.at[i, "coNlpSummary"] = str(summary)[:240]
                out.at[i, "coNlpAiStatus"] = "success"
            except Exception:
                out.at[i, "coNlpAiStatus"] = "error"
    elif enable_ai_enrichment and not os.getenv("ANTHROPIC_API_KEY"):
        out["coNlpAiStatus"] = "skipped_no_api_key"

    return out.drop(columns=["coNursing_note_free_text"], errors="ignore")
