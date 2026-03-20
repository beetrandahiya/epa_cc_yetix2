from __future__ import annotations

import re

import pandas as pd

from app.utils import normalize_col_name, normalize_columns, read_csv_flexible


def load_iid_sid_map(iid_sid_file: str) -> pd.DataFrame:
    mapping = read_csv_flexible(iid_sid_file)
    mapping = normalize_columns(mapping)
    required = ["itmiid", "itmsid"]
    for col in required:
        if col not in mapping.columns:
            raise ValueError(f"Missing required mapping column: {col}")
    mapping = mapping[["itmiid", "itmsid", "itmname255_de", "itmname255_en"]].copy()
    mapping["iid_clean"] = mapping["itmiid"].astype(str).str.replace("_", "", regex=False).str.upper()
    mapping["sid_clean"] = mapping["itmsid"].astype(str).str.upper()
    mapping = mapping.drop_duplicates(subset=["iid_clean", "sid_clean"])
    return mapping


def sid_to_iid_lookup(mapping_df: pd.DataFrame) -> dict[str, str]:
    return {
        row["sid_clean"]: row["iid_clean"]
        for _, row in mapping_df.dropna(subset=["sid_clean", "iid_clean"]).iterrows()
    }


def item_name_to_iid_lookup(mapping_df: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for _, row in mapping_df.dropna(subset=["iid_clean"]).iterrows():
        iid = str(row["iid_clean"])
        for name_col in ["itmname255_de", "itmname255_en"]:
            raw_name = row.get(name_col)
            if pd.notna(raw_name):
                normalized = normalize_col_name(str(raw_name))
                if normalized:
                    lookup[normalized] = iid
    return lookup


def ensure_co_prefix_iid(iid: str) -> str:
    iid_clean = iid.replace("_", "").upper()
    if iid_clean.startswith("CO"):
        return iid_clean
    return f"CO{iid_clean}"


def looks_like_sid_column(col_name: str) -> bool:
    return bool(re.match(r"^\d{2}(?:_\d{2}){1,2}$", col_name.upper()))
