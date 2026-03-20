import os
from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    dotenv_path = BASE_DIR / ".env"
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_settings() -> dict[str, Any]:
    _load_dotenv()
    settings_path = BASE_DIR / "configs" / "settings.yaml"
    with settings_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    paths["raw_root"] = str((BASE_DIR / paths["raw_root"]).resolve())
    paths["processed_root"] = str((BASE_DIR / paths["processed_root"]).resolve())
    paths["duckdb_file"] = str((BASE_DIR / paths["duckdb_file"]).resolve())
    paths["iid_sid_map_file"] = str((BASE_DIR / paths["iid_sid_map_file"]).resolve())
    if "pdf_inbox_dir" in paths:
        paths["pdf_inbox_dir"] = str((BASE_DIR / paths["pdf_inbox_dir"]).resolve())

    input_files = config["input_files"]
    for key, rel_path in input_files.items():
        input_files[key] = str((BASE_DIR / rel_path).resolve())

    if "input_roots" in config:
        config["input_roots"] = [str((BASE_DIR / root).resolve()) for root in config["input_roots"]]

    return config
