from __future__ import annotations

import pandas as pd


NUMERIC_CANDIDATES = [
    "coMovement_index_0_100",
    "coMicro_movements_count",
    "coImpact_magnitude_g",
    "coPost_fall_immobility_minutes",
    "coMovement_score_0_100",
    "coAccel_magnitude_g",
    "coSodium_mmol_L",
    "coPotassium_mmol_L",
    "coCreatinine_mg_dL",
    "coGlucose_mg_dL",
    "coCrp_mg_L",
    "coLactate_mmol_L",
]


def _iqr_outlier_mask(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.dropna().empty:
        return pd.Series(False, index=series.index)
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=series.index)
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (x < lower) | (x > upper)


def detect_anomalies(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["source_table", "anomaly_type", "field", "count"])

    rows = []
    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            mask = _iqr_outlier_mask(df[col])
            cnt = int(mask.sum())
            if cnt > 0:
                rows.append({"source_table": table_name, "anomaly_type": "outlier_iqr", "field": col, "count": cnt})

    if {"coFall_event_0_1", "coImpact_magnitude_g"}.issubset(set(df.columns)):
        fall_flag = pd.to_numeric(df["coFall_event_0_1"], errors="coerce").fillna(0)
        impact = pd.to_numeric(df["coImpact_magnitude_g"], errors="coerce")
        inconsistent = ((fall_flag == 1) & (impact.isna() | (impact <= 0))).sum()
        if inconsistent:
            rows.append(
                {
                    "source_table": table_name,
                    "anomaly_type": "event_inconsistency",
                    "field": "coFall_event_0_1/coImpact_magnitude_g",
                    "count": int(inconsistent),
                }
            )

    return pd.DataFrame(rows)
