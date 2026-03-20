from __future__ import annotations

import pandas as pd


def compute_completeness(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["table_name", "field", "non_null_rate", "null_count", "row_count"])

    row_count = len(df)
    rows = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        non_null_rate = float((row_count - null_count) / row_count) if row_count > 0 else 0.0
        rows.append(
            {
                "table_name": table_name,
                "field": col,
                "non_null_rate": non_null_rate,
                "null_count": null_count,
                "row_count": row_count,
            }
        )
    return pd.DataFrame(rows)


def detect_data_quality_issues(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    issues = []
    if df.empty:
        return pd.DataFrame(columns=["source_table", "issue_type", "field", "count", "severity"])

    if "case_id" in df.columns:
        bad_case = df["case_id"].isna().sum()
        if bad_case:
            issues.append(
                {
                    "source_table": table_name,
                    "issue_type": "missing_case_id",
                    "field": "case_id",
                    "count": int(bad_case),
                    "severity": "high",
                }
            )

    if "patient_id" in df.columns:
        bad_patient = df["patient_id"].isna().sum()
        if bad_patient:
            issues.append(
                {
                    "source_table": table_name,
                    "issue_type": "missing_patient_id",
                    "field": "patient_id",
                    "count": int(bad_patient),
                    "severity": "high",
                }
            )

    for col in [c for c in df.columns if c.lower().endswith("_datetime") or c.lower().endswith("_date")]:
        parsed = pd.to_datetime(df[col], errors="coerce")
        invalid = parsed.isna().sum() - df[col].isna().sum()
        if invalid > 0:
            issues.append(
                {
                    "source_table": table_name,
                    "issue_type": "invalid_datetime",
                    "field": col,
                    "count": int(invalid),
                    "severity": "medium",
                }
            )

    return pd.DataFrame(issues)
