# PEARL — Reproducibility Guide

This guide specifies how to verify and reproduce the PEARL v6 four-safeguard offline routing and audit pipeline.

## 1. Software Environment

Canonical analyses were executed in:

- **Python:** 3.12.2
- **scikit-learn:** 1.6.1
- **NumPy:** 1.26.4
- **pandas:** 2.2.2
- **SciPy:** 1.14.1
- **matplotlib:** 3.9+
- **Random seed:** fixed at `42` throughout (total real-data runtime across all cohorts and $B = 200$ end-to-end Bayesian bootstrap replicates: ~19 minutes on a 12-core Apple silicon workstation)

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Public Synthetic-Data Verification (Zero Patient Data)

No data-use agreement is required to verify the complete leak-free estimation, component separation, switched-patient support diagnostics, end-to-end Bayesian bootstrap inference, and matched-capacity decision-curve benchmarking:

```bash
python scripts/run_v6_synthetic_demo.py --n_patients 2500 --n_boot 20
```

Optional JSON output:

```bash
python scripts/run_v6_synthetic_demo.py --n_patients 2500 --n_boot 20 --out /tmp/pearl_synth_summary.json
```

## 3. Authorized Real-Data Replication (Requires DUA)

Patient-level Medicaid data cannot be shared publicly. Authorized users with local parquet extracts conforming to the schema documented in `data/extract_wpad.py` configure paths via environment variables:

```bash
export PEARL_DATA_PATH=/path/to/real_inputs
export PEARL_PROCESSED_PATH=/path/to/processed
export PEARL_SVI_PATH=/path/to/processed/svi_by_zip.parquet
export PEARL_OUTPUT_BASE=/path/to/outputs
export PEARL_FIGURES_DIR=/path/to/figures_png

# Step 1: Primary leak-free 5-cohort pipeline, action validation, onboarding 2x2, T7, and B=200 bootstrap
python scripts/pearl_v6_revision.py

# Step 2: Daily pre-action acute-care recency (7/14/30/90d + IP), strictly pre-index sensitivity, Camden analog
python scripts/pearl_v6_extended.py

# Step 3: Regenerate all main and supplementary figures strictly from canonical JSON/CSVs
python scripts/make_v6_figures.py
```

### Expected Input Schema (`PEARL_DATA_PATH` / `PEARL_PROCESSED_PATH`)

| File | Description |
|---|---|
| `member_attributes.parquet` | Baseline demographics, enrollment attributes, and rising-risk score |
| `member_goals.parquet` | Time-stamped care-plan goals (category and deletion flags) |
| `member_encounters.parquet` | Completed care-team encounters and clinician disciplines |
| `hospital_visits.parquet` | Daily inpatient admission and emergency department visit feed |
| `outcomes_monthly.parquet` | Strictly prior monthly utilization and spending summaries |
| `member_status_event.parquet` | Enrollment and staggered onboarding status transitions |
| `eligibility.parquet` | Health-plan state, race/ethnicity, ZIP code, and coverage spans |
| `member_patient_map.parquet` | De-identified internal linkage table |
| `signal_risk_latest.parquet` | Risk-model snapshot flags (omitted in strictly pre-index sensitivity) |
| `clinical_summary.parquet` | Claims comorbidity summaries (omitted in strictly pre-index sensitivity) |
| `pharmacy_summary.parquet` | Prior 90-day pharmacy fill counts and adherence indicators |
| `svi_by_zip.parquet` | Built from public CDC/ATSDR 2022 SVI + Census 2020 ZCTA-tract files via `scripts/build_svi.py` |

## 4. Public Governance Guarantees

- Zero patient-level data, individual records, or patient-fitted binary model files are committed to this repository or archived on Zenodo.
- `.gitignore` excludes all `.csv`, `.parquet`, `.jsonl`, `.joblib`, `.pkl`, and output directories.
- Pre-registration protocol: [https://doi.org/10.17605/OSF.IO/D69M2](https://doi.org/10.17605/OSF.IO/D69M2).
- Software archive (all versions): [https://doi.org/10.5281/zenodo.22236900](https://doi.org/10.5281/zenodo.22236900).
