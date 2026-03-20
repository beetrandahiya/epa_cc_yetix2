from __future__ import annotations

from pathlib import Path

from app.utils import detect_csv_delimiter


def sniff_first_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return (f.readline() or "").strip().lower()


def classify_file(path: Path) -> str:
    name = path.name.lower()
    first = sniff_first_line(path)

    if "epaac-data-1" in name:
        return "epa1"
    if "epaac-data-2" in name:
        return "epa2"
    if "epaac-data-3" in name:
        return "epa3"
    if "epaac-data-5" in name:
        return "epa5"
    if "epaac-data" in name:
        if "sid;sid_value" in first:
            return "epa1"
        if "mandt;patgeb" in first:
            return "epa2"
        if "einschidfall" in first:
            return "epa3"

    if "1hz" in name and "device" in name:
        return "device_1hz"
    if "device" in name and "motion" in name:
        return "device_motion"
    if "_device.csv" in name or name.endswith("device.csv"):
        return "device_motion"
    if "medication" in name:
        return "medication"
    if "nursing" in name:
        return "nursing"
    if "icd" in name or "ops" in name:
        return "icd"
    if "lab" in name:
        return "labs"

    if "patient_id" in first and "movement_index" in first:
        return "device_motion"
    if "record_type" in first or "rec_type" in first:
        return "medication"
    if "nursing_note_free_text" in first or "nursingnote" in first:
        return "nursing"
    if "primary_icd10" in first or "icd10_haupt" in first:
        return "icd"

    return "unknown"


def discover_data_files(roots: list[str]) -> dict[str, list[str]]:
    discovered: dict[str, list[str]] = {
        "epa1": [],
        "epa2": [],
        "epa3": [],
        "epa5": [],
        "labs": [],
        "device_motion": [],
        "device_1hz": [],
        "medication": [],
        "nursing": [],
        "icd": [],
        "unknown": [],
    }

    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue

        for file_path in sorted(root_path.rglob("*.csv")):
            domain = classify_file(file_path)
            discovered[domain].append(str(file_path.resolve()))

    return discovered
