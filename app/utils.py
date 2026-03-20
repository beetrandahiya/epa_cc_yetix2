from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


def detect_csv_delimiter(file_path: str | Path) -> str:
    with Path(file_path).open("r", encoding="utf-8", errors="ignore") as f:
        sample = "".join([f.readline() for _ in range(10)])
    candidates = {";": sample.count(";"), ",": sample.count(","), "\t": sample.count("\t")}
    return max(candidates, key=candidates.get)


def read_csv_flexible(file_path: str | Path, header: int | None = 0) -> pd.DataFrame:
    delimiter = detect_csv_delimiter(file_path)
    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(
                file_path,
                sep=delimiter,
                dtype=str,
                encoding=encoding,
                keep_default_na=False,
                header=header,
                on_bad_lines="skip",
                engine="python",
            )
        except UnicodeDecodeError:
            continue
    return pd.read_csv(
        file_path,
        sep=delimiter,
        dtype=str,
        encoding="latin1",
        keep_default_na=False,
        header=header,
        on_bad_lines="skip",
        engine="python",
    )


def normalize_col_name(col: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(col)).strip("_").lower()
    return cleaned


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: normalize_col_name(col) for col in df.columns}
    return df.rename(columns=renamed)


def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: dict[str, int] = {}
    unique_cols: list[str] = []
    for col in [str(c) for c in df.columns]:
        if col not in seen:
            seen[col] = 0
            unique_cols.append(col)
        else:
            seen[col] += 1
            unique_cols.append(f"{col}_{seen[col]}")
    if unique_cols != [str(c) for c in df.columns]:
        df = df.copy()
        df.columns = unique_cols
    return df


def normalize_missing_values(df: pd.DataFrame, null_like_values: Iterable[str]) -> pd.DataFrame:
    null_set = {str(v).strip().lower() for v in null_like_values}

    def clean_value(x: object) -> object:
        if pd.isna(x):
            return pd.NA
        sx = str(x).strip()
        if sx.lower() in null_set:
            return pd.NA
        return sx

    return df.map(clean_value)


def normalize_case_id(raw_case_id: object) -> str | None:
    if raw_case_id is None or pd.isna(raw_case_id):
        return None
    sx = str(raw_case_id).strip().upper()
    sx = sx.replace("CASE", "")
    sx = sx.replace("-", "")
    sx = sx.strip()
    if not sx:
        return None
    digits = re.sub(r"[^0-9]", "", sx)
    if not digits:
        return None
    return str(int(digits))


def find_first_present_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_original = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]
    return None


def to_numeric_safe(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def apply_alias_map(df: pd.DataFrame, target_to_aliases: dict[str, list[str]]) -> pd.DataFrame:
    lower_to_original = {c.lower(): c for c in df.columns}
    rename_map: dict[str, str] = {}
    for target, aliases in target_to_aliases.items():
        for alias in aliases:
            source = lower_to_original.get(alias.lower())
            if source:
                rename_map[source] = target
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df
