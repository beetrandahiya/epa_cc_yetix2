# Smart Health Data Mapping App (On-Prem)

This project rebuilds the challenge system from zero as a working, local-first application:

- One-time preprocessing pipeline (cached by source file signature)
- Unified case-centric model (`tbCaseData` + import tables)
- Mapping across EPA formats 1/2/3 with IID/SID bridge
- Data quality checks, completeness metrics, anomaly detection
- Interactive dashboard with alerts, lineage, plots, manual correction export
- Optional Anthropic AI extraction for PDF and free text
- Enterprise add-ons:
  - Automated PDF inbox ingestion (incremental)
  - Deterministic nursing-note NLP normalization + optional AI enrichment

## Important Data Rules Implemented

- German providers context respected: **"Fall" means "case"**, not a physical fall.
- Required fields: `case_id`, `patient_id` (rows removed when missing)
- `case_id` normalization:
  - `CASE-0135` -> `135`
  - `0135` -> `135`
  - `135` -> `135`
- Null-like values mapped to NULL:
  - `NULL`, `Missing`, `unknow`, `NaN`, `N/A`, blank

## Install

```bash
pip install -r requirements.txt
```

## Run preprocessing once

```bash
python run_pipeline.py
```

Pipeline behavior:
- Writes tables to `data/processed/*.parquet` and `data/processed/health_mapping.duckdb`
- Skips reprocessing automatically when source files are unchanged
- Use force rebuild from dashboard sidebar when needed
- Includes enterprise tables:
  - `tbImportNursingNlpData`
  - `tbImportPdfClinicalData`

## Enterprise PDF ingestion

- Put clinical PDFs into `data/pdf_inbox`
- Run pipeline (`python run_pipeline.py`)
- Only new/changed PDFs are processed (manifest-based incremental ingestion)
- Extracted payload is persisted in `tbImportPdfClinicalData`

## Nursing NLP normalization

- Nursing free text is normalized into rule-based indicators (pain, fever, mobility issues, orientation issues, wounds, medication/monitoring actions, improvement)
- Output is persisted in `tbImportNursingNlpData`
- If Anthropic key is available, limited AI enrichment runs per pipeline execution (`max_ai_rows_per_run` in config)

## Run dashboard

```bash
python run_dashboard.py
```

## Anthropic AI (optional)

Set your key locally before using AI extraction tabs:

```powershell
$env:ANTHROPIC_API_KEY="your_key_here"
```

Default model in this project is:

```text
claude-haiku-4-5
```

You can also override both model and API key directly in the dashboard sidebar for the current session (without storing the key in project files).

No cloud storage is required. The app runs fully on-prem; only optional AI calls go to Anthropic when you enable them.
