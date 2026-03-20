from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from app.mappings import ensure_co_prefix_iid, looks_like_sid_column
from app.utils import (
    apply_alias_map,
    ensure_unique_columns,
    find_first_present_column,
    normalize_case_id,
    normalize_columns,
    normalize_missing_values,
    read_csv_flexible,
)


def _clean_and_standardize(df: pd.DataFrame, null_like_values: list[str]) -> pd.DataFrame:
    df = normalize_columns(df)
    df = ensure_unique_columns(df)
    df = normalize_missing_values(df, null_like_values)
    return df


def _add_case_patient_norm(df: pd.DataFrame) -> pd.DataFrame:
    case_col = find_first_present_column(
        df,
        [
            "case_id",
            "fallnr",
            "coe2i222",
            "cocaseid",
            "encounter_id",
            "fall",
            "caseid",
            "id_cas",
            "patfal",
            "cas",
        ],
    )
    patient_col = find_first_present_column(df, ["patient_id", "copatientid", "patientid", "pid", "id_pat", "pat_id", "pat"])

    if case_col is not None:
        df["case_id"] = df[case_col].apply(normalize_case_id)
    if patient_col is not None:
        df["patient_id"] = (
            df[patient_col]
            .astype(str)
            .str.strip()
            .replace({"": pd.NA, "<NA>": pd.NA, "nan": pd.NA})
        )
    return df


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    duplicated_names = pd.Series(df.columns)
    duplicated_names = duplicated_names[duplicated_names.duplicated()].unique().tolist()
    if not duplicated_names:
        return df

    out = df.copy()
    for col_name in duplicated_names:
        subset = out.loc[:, out.columns == col_name]
        merged = subset.bfill(axis=1).iloc[:, 0]
        out = out.drop(columns=[col_name])
        out[col_name] = merged
    return out


def load_epa_data_3(
    path: str,
    null_like_values: list[str],
    item_name_to_iid: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_csv_flexible(path)
    df = _clean_and_standardize(df, null_like_values)

    lineage = []
    rename_map: dict[str, str] = {}

    if not df.empty:
        first_row = df.iloc[0].astype(str)
        iid_hits = first_row.str.contains(r"^e\d+_?i\d+$", case=False, regex=True).sum()
        if iid_hits >= max(5, int(0.15 * len(first_row))):
            for col in df.columns:
                raw_iid = str(df.iloc[0][col])
                if re.match(r"^e\d+_?i\d+$", raw_iid, re.IGNORECASE):
                    normalized_iid = raw_iid.replace("_", "").upper()
                    target_col = ensure_co_prefix_iid(normalized_iid)
                    rename_map[col] = target_col
                    lineage.append(("epa_data_3", col, target_col, "IID helper row mapping"))
            df = df.iloc[1:].reset_index(drop=True)

    df = _add_case_patient_norm(df)

    for col in df.columns:
        if col in rename_map:
            continue
        if re.match(r"^e\d+_?i\d+", col, re.IGNORECASE):
            normalized_iid = col.replace("_", "").upper()
            target_col = ensure_co_prefix_iid(normalized_iid)
            rename_map[col] = target_col
            lineage.append(("epa_data_3", col, target_col, "IID wide mapping"))
            continue

        iid_match = re.search(r"(e\d+_?i\d+)", col, re.IGNORECASE)
        if iid_match:
            normalized_iid = iid_match.group(1).replace("_", "").upper()
            target_col = ensure_co_prefix_iid(normalized_iid)
            rename_map[col] = target_col
            lineage.append(("epa_data_3", col, target_col, "IID token extracted from text header"))
            continue

        if item_name_to_iid:
            iid = item_name_to_iid.get(str(col).lower())
            if iid:
                target_col = ensure_co_prefix_iid(iid)
                rename_map[col] = target_col
                lineage.append(("epa_data_3", col, target_col, "Item-name -> IID mapping"))

    df = df.rename(columns=rename_map)
    df = _coalesce_duplicate_columns(df)
    df = ensure_unique_columns(df)
    lineage_df = pd.DataFrame(lineage, columns=["source_table", "source_field", "target_field", "rule"])
    lineage_df["source_file"] = Path(path).name
    df["source_file"] = Path(path).name
    return df, lineage_df


def _extract_epa2_iid_from_column(col_name: str, valid_iids: set[str]) -> str | None:
    text = str(col_name).lower()
    match = re.match(r"^epa(\d{3,4})(?:id|tx|an)?$", text)
    if not match:
        return None

    num = match.group(1)
    candidates = [f"E0I{num}"]
    num_trim = str(int(num)) if num.isdigit() else num.lstrip("0")
    if num_trim:
        candidates.append(f"E0I{num_trim}")

    for iid in candidates:
        if iid in valid_iids:
            return iid
    return None


def load_epa_data_2(path: str, sid_to_iid: dict[str, str], null_like_values: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_csv_flexible(path)
    df = _clean_and_standardize(df, null_like_values)
    df = _add_case_patient_norm(df)

    lineage = []
    rename_map: dict[str, str] = {}
    valid_iids = {str(iid).upper() for iid in sid_to_iid.values() if pd.notna(iid)}

    for col in df.columns:
        col_upper = col.upper()
        if looks_like_sid_column(col_upper):
            iid = sid_to_iid.get(col_upper)
            if iid:
                target_col = ensure_co_prefix_iid(iid)
                rename_map[col] = target_col
                lineage.append(("epa_data_2", col, target_col, "SID->IID mapping"))
                continue

        iid = _extract_epa2_iid_from_column(col, valid_iids)
        if iid:
            target_col = ensure_co_prefix_iid(iid)
            rename_map[col] = target_col
            lineage.append(("epa_data_2", col, target_col, "EPA header token -> IID mapping"))

    df = df.rename(columns=rename_map)
    df = _coalesce_duplicate_columns(df)
    df = ensure_unique_columns(df)
    lineage_df = pd.DataFrame(lineage, columns=["source_table", "source_field", "target_field", "rule"])
    lineage_df["source_file"] = Path(path).name
    df["source_file"] = Path(path).name
    return df, lineage_df


def load_epa_data_1(path: str, sid_to_iid: dict[str, str], null_like_values: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_csv_flexible(path)
    df = _clean_and_standardize(df, null_like_values)
    df = _add_case_patient_norm(df)

    iid_col = find_first_present_column(df, ["iid", "itmiid", "item_iid", "itemid"]) 
    sid_col = find_first_present_column(df, ["sid", "itmsid", "item_sid"])
    value_col = find_first_present_column(df, ["value", "wert", "score", "item_value", "raw_value"])

    if iid_col is None and sid_col is None:
        raise ValueError("epaAC-Data-1 requires IID or SID column in row-wise format")
    if value_col is None:
        fallback_candidates = [c for c in df.columns if c not in {iid_col, sid_col, "case_id", "patient_id"}]
        if fallback_candidates:
            value_col = fallback_candidates[-1]
        else:
            raise ValueError("epaAC-Data-1 cannot determine value column")

    if iid_col is None:
        df["_iid_clean"] = df[sid_col].astype(str).str.upper().map(sid_to_iid)
    else:
        df["_iid_clean"] = df[iid_col].astype(str).str.replace("_", "", regex=False).str.upper()

    df["_target_col"] = df["_iid_clean"].apply(lambda x: ensure_co_prefix_iid(str(x)) if pd.notna(x) else pd.NA)

    key_cols = [c for c in ["case_id", "patient_id"] if c in df.columns]
    other_keys = [c for c in ["assessment_date", "coe2i225", "assessment_type", "coe0i001", "station", "account"] if c in df.columns]
    group_keys = key_cols + other_keys
    if not group_keys:
        group_keys = ["_rowgroup"]
        df["_rowgroup"] = 1

    df["_seq"] = range(len(df))
    df = df.sort_values("_seq")
    deduped = df.dropna(subset=["_target_col"]).drop_duplicates(subset=group_keys + ["_target_col"], keep="last")

    wide = deduped.pivot_table(index=group_keys, columns="_target_col", values=value_col, aggfunc="last").reset_index()
    wide.columns.name = None

    lineage = (
        deduped[["_target_col"]]
        .drop_duplicates()
        .assign(source_table="epa_data_1", source_field="row_value", rule="row-wise IID/SID pivot")
        .rename(columns={"_target_col": "target_field"})
    )

    if "_rowgroup" in wide.columns:
        wide = wide.drop(columns=["_rowgroup"])

    wide["source_file"] = Path(path).name
    lineage = lineage[["source_table", "source_field", "target_field", "rule"]]
    lineage["source_file"] = Path(path).name
    return wide, lineage


def load_labs_data(path: str, null_like_values: list[str]) -> pd.DataFrame:
    raw = read_csv_flexible(path)
    if "clinic_3_labs" in Path(path).name.lower() and "case_id" not in [str(c).lower() for c in raw.columns]:
        raw = read_csv_flexible(path, header=None)
        raw.columns = [
            "specimen_datetime",
            "case_id",
            "patient_id",
            "age_years",
            "sex",
            "potassium_mmol_l",
            "potassium_flag",
            "potassium_ref_low",
            "potassium_ref_high",
            "sodium_mmol_l",
            "sodium_flag",
            "sodium_ref_low",
            "sodium_ref_high",
            "creatinine_mg_dl",
            "creatinine_flag",
            "creatinine_ref_low",
            "creatinine_ref_high",
        ] + [f"extra_{i}" for i in range(max(0, raw.shape[1] - 17))]
    df = _add_case_patient_norm(_clean_and_standardize(raw, null_like_values))
    df = apply_alias_map(
        df,
        {
            "case_id": ["caseid", "id_cas", "cas"],
            "patient_id": ["pid", "id_pat", "pat"],
            "specimen_datetime": ["specdt", "dat"],
            "sodium_mmol_l": ["na"],
            "sodium_flag": ["na_flag"],
            "sodium_ref_low": ["na_low"],
            "sodium_ref_high": ["na_high"],
            "potassium_mmol_l": ["k"],
            "potassium_flag": ["k_flag"],
            "potassium_ref_low": ["k_low"],
            "potassium_ref_high": ["k_high"],
            "creatinine_mg_dl": ["creat"],
            "creatinine_flag": ["creat_flag"],
            "creatinine_ref_low": ["creat_low"],
            "creatinine_ref_high": ["creat_high"],
            "glucose_mg_dl": ["glukose"],
            "lactate_mmol_l": ["laktat"],
        },
    )
    rename = {
        "specimen_datetime": "coSpecimen_datetime",
        "natrium": "coSodium_mmol_L",
        "natrium_flag": "coSodium_flag",
        "natrium_ref_low": "cosodium_ref_low",
        "natrium_ref_high": "cosodium_ref_high",
        "kalium": "coPotassium_mmol_L",
        "kalium_flag": "coPotassium_flag",
        "kalium_ref_low": "coPotassium_ref_low",
        "kalium_ref_high": "coPotassium_ref_high",
        "kreatinin": "coCreatinine_mg_dL",
        "kreatinin_flag": "coCreatinine_flag",
        "kreatinin_ref_low": "coCreatinine_ref_low",
        "kreatinin_ref_high": "coCreatinine_ref_high",
        "egfr": "coEgfr_mL_min_1_73m2",
        "egfr_flag": "coEgfr_flag",
        "egfr_ref_low": "coEgfr_ref_low",
        "egfr_ref_high": "coEgfr_ref_high",
        "glukose": "coGlucose_mg_dL",
        "glukose_flag": "coGlucose_flag",
        "glukose_ref_low": "coGlucose_ref_low",
        "glukose_ref_high": "coGlucose_ref_high",
        "hb": "coHemoglobin_g_dL",
        "hb_flag": "coHb_flag",
        "hb_ref_low": "coHb_ref_low",
        "hb_ref_high": "coHb_ref_high",
        "leukozyten": "coWbc_10e9_L",
        "leukozyten_flag": "coWbc_flag",
        "leukozyten_ref_low": "coWbc_ref_low",
        "leukozyten_ref_high": "coWbc_ref_high",
        "thrombozyten": "coPlatelets_10e9_L",
        "thrombozyten_flag": "coPlatelets_flag",
        "thrombozyten_ref_low": "coPlt_ref_low",
        "thrombozyten_ref_high": "coPlt_ref_high",
        "crp": "coCrp_mg_L",
        "crp_flag": "coCrp_flag",
        "crp_ref_low": "coCrp_ref_low",
        "crp_ref_high": "coCrp_ref_high",
        "alt": "coAlt_U_L",
        "alt_flag": "coAlt_flag",
        "alt_ref_low": "coAlt_ref_low",
        "alt_ref_high": "coAlt_ref_high",
        "ast": "coAst_U_L",
        "ast_flag": "coAst_flag",
        "ast_ref_low": "coAst_ref_low",
        "ast_ref_high": "coAst_ref_high",
        "bilirubin": "coBilirubin_mg_dL",
        "bilirubin_flag": "coBilirubin_flag",
        "bilirubin_ref_low": "coBili_ref_low",
        "bilirubin_ref_high": "coBili_ref_high",
        "albumin": "coAlbumin_g_dL",
        "albumin_flag": "coAlbumin_flag",
        "albumin_ref_low": "coAlbumin_ref_low",
        "albumin_ref_high": "coAlbumin_ref_high",
        "inr": "coInr",
        "inr_flag": "coInr_flag",
        "inr_ref_low": "coInr_ref_low",
        "inr_ref_high": "coInr_ref_high",
        "laktat": "coLactate_mmol_L",
        "laktat_flag": "coLactate_flag",
        "laktat_ref_low": "coLactate_ref_low",
        "laktat_ref_high": "coLactate_ref_high",
    }
    available_rename = {k: v for k, v in rename.items() if k in df.columns}
    mapped = df.rename(columns=available_rename)
    mapped = _coalesce_duplicate_columns(mapped)
    mapped = ensure_unique_columns(mapped)
    mapped["source_file"] = Path(path).name
    return mapped


def load_device_motion(path: str, null_like_values: list[str]) -> pd.DataFrame:
    raw = read_csv_flexible(path)
    if "clinic_3_device" in Path(path).name.lower() and "timestamp" not in [str(c).lower() for c in raw.columns]:
        raw = read_csv_flexible(path, header=None)
        raw.columns = [
            "timestamp",
            "patient_id",
            "fall_event_0_1",
            "movement_index_0_100",
            "micro_movements_count",
            "bed_exit_detected_0_1",
            "impact_magnitude_g",
            "post_fall_immobility_minutes",
        ]
    df = _add_case_patient_norm(_clean_and_standardize(raw, null_like_values))
    df = apply_alias_map(
        df,
        {
            "timestamp": ["date", "dat"],
            "movement_index_0_100": ["idx_mov"],
            "micro_movements_count": ["num_mov"],
            "bed_exit_detected_0_1": ["num_ex"],
            "fall_event_0_1": ["falle"],
            "impact_magnitude_g": ["mag_impact", "imp_mag"],
            "post_fall_immobility_minutes": ["min_immob", "post_fall_immobility"],
        },
    )
    rename = {
        "timestamp": "coTimestamp",
        "movement_index_0_100": "coMovement_index_0_100",
        "micro_movements_count": "coMicro_movements_count",
        "bed_exit_detected_0_1": "coBed_exit_detected_0_1",
        "fall_event_0_1": "coFall_event_0_1",
        "impact_magnitude_g": "coImpact_magnitude_g",
        "post_fall_immobility_minutes": "coPost_fall_immobility_minutes",
    }
    mapped = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    mapped["source_file"] = Path(path).name
    return mapped


def load_device_1hz(path: str, null_like_values: list[str]) -> pd.DataFrame:
    raw = read_csv_flexible(path)
    if "clinic_3_device_1hz" in Path(path).name.lower() and "timestamp" not in [str(c).lower() for c in raw.columns]:
        raw = read_csv_flexible(path, header=None)
        raw.columns = [
            "patient_id",
            "device_id",
            "timestamp",
            "bed_occupied_0_1",
            "movement_score_0_100",
            "pressure_zone1_0_100",
            "pressure_zone2_0_100",
            "pressure_zone3_0_100",
            "pressure_zone4_0_100",
            "accel_x_m_s2",
            "accel_y_m_s2",
            "accel_z_m_s2",
            "accel_magnitude_g",
            "bed_exit_event_0_1",
            "bed_return_event_0_1",
            "fall_event_0_1",
            "impact_magnitude_g",
            "event_id",
        ]
    df = _add_case_patient_norm(_clean_and_standardize(raw, null_like_values))
    df = apply_alias_map(
        df,
        {
            "patient_id": ["id_pat", "id_2", "patientid"],
            "device_id": ["id_dev", "id_1", "deviceid"],
            "timestamp": ["date"],
            "bed_occupied_0_1": ["occ", "bedoccupied"],
            "movement_score_0_100": ["mov", "movementscore"],
            "accel_x_m_s2": ["accx"],
            "accel_y_m_s2": ["accy"],
            "accel_z_m_s2": ["accz"],
            "accel_magnitude_g": ["acc_mag", "accelmag"],
            "pressure_zone1_0_100": ["zone1_pa", "pressz1"],
            "pressure_zone2_0_100": ["zone2_pa", "pressz2"],
            "pressure_zone3_0_100": ["zone3_pa", "pressz3"],
            "pressure_zone4_0_100": ["zone4_pa", "pressz4"],
            "bed_exit_event_0_1": ["exit", "bedexit"],
            "bed_return_event_0_1": ["return", "bedreturn"],
            "fall_event_0_1": ["falle"],
            "impact_magnitude_g": ["imp_mag"],
        },
    )
    rename = {
        "timestamp": "coTimestamp",
        "device_id": "coDevice_id",
        "bed_occupied_0_1": "coBed_occupied_0_1",
        "movement_score_0_100": "coMovement_score_0_100",
        "accel_x_m_s2": "coAccel_x_m_s2",
        "accel_y_m_s2": "coAccel_y_m_s2",
        "accel_z_m_s2": "coAccel_z_m_s2",
        "accel_magnitude_g": "coAccel_magnitude_g",
        "pressure_zone1_0_100": "coPressure_zone1_0_100",
        "pressure_zone2_0_100": "coPressure_zone2_0_100",
        "pressure_zone3_0_100": "coPressure_zone3_0_100",
        "pressure_zone4_0_100": "coPressure_zone4_0_100",
        "bed_exit_event_0_1": "coBed_exit_event_0_1",
        "bed_return_event_0_1": "coBed_return_event_0_1",
        "fall_event_0_1": "coFall_event_0_1",
        "impact_magnitude_g": "coImpact_magnitude_g",
        "event_id": "coEvent_id",
    }
    mapped = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    mapped["source_file"] = Path(path).name
    return mapped


def load_medication(path: str, null_like_values: list[str]) -> pd.DataFrame:
    raw = read_csv_flexible(path)
    if "clinic_3_medication" in Path(path).name.lower() and "record_type" not in [str(c).lower() for c in raw.columns]:
        raw = read_csv_flexible(path, header=None)
        raw.columns = [
            "order_id",
            "order_uuid",
            "record_type",
            "patient_id",
            "encounter_id",
            "ward",
            "admission_datetime",
            "discharge_datetime",
            "medication_code_atc",
            "medication_name",
            "route",
            "dose",
            "dose_unit",
            "frequency",
            "order_start_datetime",
            "order_stop_datetime",
            "is_prn_0_1",
            "indication",
            "prescriber_role",
            "order_status",
            "administration_datetime",
            "administered_dose",
            "administered_unit",
            "administration_status",
            "note",
        ]
    df = _add_case_patient_norm(_clean_and_standardize(raw, null_like_values))
    df = apply_alias_map(
        df,
        {
            "record_type": ["rec_type", "type"],
            "patient_id": ["pat_id", "id_2", "pid"],
            "encounter_id": ["enc_id", "id_1"],
            "ward": ["station", "war"],
            "admission_datetime": ["aufnahme_dt", "date_ad"],
            "discharge_datetime": ["entlassung_dt", "date_dis"],
            "order_uuid": ["uuid", "id_order2"],
            "medication_code_atc": ["atc_code", "atc"],
            "medication_name": ["medikament", "name"],
            "route": ["applikation", "via"],
            "dose": ["dosis", "amount"],
            "dose_unit": ["einheit", "unit"],
            "frequency": ["haeufigkeit", "freq"],
            "order_start_datetime": ["start_dt", "date_order1"],
            "order_stop_datetime": ["stop_dt", "date_order2"],
            "is_prn_0_1": ["prn"],
            "indication": ["indic"],
            "prescriber_role": ["role"],
        },
    )
    rename = {
        "record_type": "coRecord_type",
        "encounter_id": "coEncounter_id",
        "ward": "coWard",
        "admission_datetime": "coAdmission_datetime",
        "discharge_datetime": "coDischarge_datetime",
        "order_id": "coOrder_id",
        "order_uuid": "coOrder_uuid",
        "medication_code_atc": "coMedication_code_atc",
        "medication_name": "coMedication_name",
        "route": "coRoute",
        "dose": "coDose",
        "dose_unit": "coDose_unit",
        "frequency": "coFrequency",
        "order_start_datetime": "coOrder_start_datetime",
        "order_stop_datetime": "coOrder_stop_datetime",
        "is_prn_0_1": "coIs_prn_0_1",
        "indication": "coIndication",
        "prescriber_role": "prescriber_role",
        "order_status": "order_status",
        "administration_datetime": "administration_datetime",
        "administered_dose": "administered_dose",
        "administered_unit": "administered_unit",
        "administration_status": "administration_status",
        "note": "note",
    }
    mapped = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    mapped["source_file"] = Path(path).name
    return mapped


def load_nursing(path: str, null_like_values: list[str]) -> pd.DataFrame:
    raw = read_csv_flexible(path)
    if "clinic_3_nursing" in Path(path).name.lower() and "case_id" not in [str(c).lower() for c in raw.columns]:
        raw = read_csv_flexible(path, header=None)
        raw.columns = ["patient_id", "case_id", "report_date", "shift", "ward", "nursing_note_free_text"]
    df = _add_case_patient_norm(_clean_and_standardize(raw, null_like_values))
    df = apply_alias_map(
        df,
        {
            "case_id": ["caseid", "cas"],
            "patient_id": ["patientid", "pat", "pid"],
            "ward": ["war", "station"],
            "report_date": ["reportdate", "dat"],
            "shift": ["shf"],
            "nursing_note_free_text": ["nursingnote", "txt"],
        },
    )
    rename = {
        "ward": "coWard",
        "report_date": "coReport_date",
        "shift": "coShift",
        "nursing_note_free_text": "coNursing_note_free_text",
    }
    mapped = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    mapped["source_file"] = Path(path).name
    return mapped


def load_icd_ops(path: str, null_like_values: list[str]) -> pd.DataFrame:
    raw = read_csv_flexible(path)
    if "clinic_3_icd_ops" in Path(path).name.lower() and "case_id" not in [str(c).lower() for c in raw.columns]:
        raw = read_csv_flexible(path, header=None)
        raw.columns = [
            "case_id",
            "patient_id",
            "ops_codes",
            "ops_descriptions_en",
            "primary_icd10_code",
            "primary_icd10_description_en",
            "secondary_icd10_codes",
            "secondary_icd10_descriptions_en",
            "ward",
            "admission_date",
            "discharge_date",
            "length_of_stay_days",
        ]
    df = _add_case_patient_norm(_clean_and_standardize(raw, null_like_values))
    df = apply_alias_map(
        df,
        {
            "case_id": ["caseid", "id_cas", "fall", "patfal"],
            "patient_id": ["patientid", "id_pat", "pid"],
            "ward": ["station", "war"],
            "admission_date": ["aufnahmedatum", "date_ad", "d_m_str", "d_m"],
            "discharge_date": ["entlassungsdatum", "date_dis", "d_s_str", "d_s"],
            "length_of_stay_days": ["verweildauer_tage", "los"],
            "primary_icd10_code": ["icd10_haupt"],
            "primary_icd10_description_en": ["icd10_haupt_bezeichnung"],
            "secondary_icd10_codes": ["icd10_neben"],
            "secondary_icd10_descriptions_en": ["icd10_neben_bezeichnung", "proc_str"],
            "ops_codes": ["proc"],
        },
    )
    rename = {
        "ward": "coWard",
        "admission_date": "coAdmission_date",
        "discharge_date": "coDischarge_date",
        "length_of_stay_days": "coLength_of_stay_days",
        "primary_icd10_code": "coPrimary_icd10_code",
        "primary_icd10_description_en": "coPrimary_icd10_description_en",
        "secondary_icd10_codes": "coSecondary_icd10_codes",
        "secondary_icd10_descriptions_en": "cpSecondary_icd10_descriptions_en",
        "ops_codes": "coOps_codes",
        "ops_descriptions_en": "ops_descriptions_en",
    }
    mapped = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    mapped["source_file"] = Path(path).name
    return mapped


def drop_missing_mandatory(df: pd.DataFrame, required_fields: list[str], source_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not all(field in df.columns for field in required_fields):
        missing_fields = [f for f in required_fields if f not in df.columns]
        issue = pd.DataFrame(
            [{"source_table": source_name, "issue_type": "missing_required_column", "field": m, "count": len(df)} for m in missing_fields]
        )
        return df.iloc[0:0], issue

    before = len(df)
    valid_df = df.dropna(subset=required_fields)
    removed = before - len(valid_df)
    issue = pd.DataFrame(
        [{"source_table": source_name, "issue_type": "missing_required_value", "field": ",".join(required_fields), "count": removed}]
    )
    return valid_df, issue


def build_case_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for df in tables.values():
        if "case_id" in df.columns and "patient_id" in df.columns:
            rows.append(df[["case_id", "patient_id"]].dropna())

    if not rows:
        return pd.DataFrame(columns=["coId", "coE2I222", "coPatientId"])

    case_base = pd.concat(rows, ignore_index=True).drop_duplicates().reset_index(drop=True)
    case_base["coId"] = case_base.index + 1
    case_base["coE2I222"] = pd.to_numeric(case_base["case_id"], errors="coerce")
    case_base["coPatientId"] = pd.to_numeric(case_base["patient_id"], errors="coerce")
    return case_base[["coId", "coE2I222", "coPatientId", "case_id", "patient_id"]]


def attach_case_fk(df: pd.DataFrame, case_table: pd.DataFrame) -> pd.DataFrame:
    if "case_id" not in df.columns or "patient_id" not in df.columns:
        df["coCaseId"] = pd.NA
        return df

    merged = df.merge(case_table[["coId", "case_id", "patient_id"]], on=["case_id", "patient_id"], how="left")
    merged = merged.rename(columns={"coId": "coCaseId"})
    return merged


def merge_epa_sources(epa1: pd.DataFrame, epa2: pd.DataFrame, epa3: pd.DataFrame) -> pd.DataFrame:
    frames = [f for f in [epa1, epa2, epa3] if not f.empty]
    if not frames:
        return pd.DataFrame()

    normalized_frames = []
    for frame in frames:
        frame = frame.copy()
        frame = frame.loc[:, ~frame.columns.duplicated()]
        if "coe2i225" in frame.columns and "coE2I225" not in frame.columns:
            frame = frame.rename(columns={"coe2i225": "coE2I225"})
        if "coe0i001" in frame.columns and "coE0I001" not in frame.columns:
            frame = frame.rename(columns={"coe0i001": "coE0I001"})
        if "coe2i222" in frame.columns and "coE2I222" not in frame.columns:
            frame = frame.rename(columns={"coe2i222": "coE2I222"})
        normalized_frames.append(frame)

    merged = pd.concat(normalized_frames, ignore_index=True, sort=False)

    key_cols = [c for c in ["case_id", "patient_id", "coE2I225", "coE0I001"] if c in merged.columns]
    if not key_cols:
        key_cols = ["case_id", "patient_id"] if {"case_id", "patient_id"}.issubset(set(merged.columns)) else []

    if key_cols:
        merged["_seq"] = range(len(merged))
        merged = merged.sort_values("_seq").drop(columns=["_seq"]).drop_duplicates(subset=key_cols, keep="last")

    return merged


def standardize_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        if re.match(r"^coe\d+i\d+$", col, re.IGNORECASE):
            renamed[col] = col[:2] + col[2:].upper()
    if renamed:
        df = df.rename(columns=renamed)
    return df
