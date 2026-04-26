# PEARL — Reproducibility Guide

This document specifies how the PEARL analysis is made reproducible for an
external auditor. Three levels of reproducibility are supported:

1. **Synthetic-data demonstration** — public, requires no DUA. Reproduces the
   full pipeline structure, falsification tests, IMI estimation, training, and
   evaluation on synthetic patients with known ground-truth IMI.
2. **CodeOcean compute capsule** — browser-runnable verification environment
   wrapping the synthetic-data demonstration. DOI to be assigned on acceptance.
3. **Real-data replication** — requires a data use agreement (DUA) with an ACO
   that has the schema described in `data/extract_wpad.py`. The DUA template is
   available from the corresponding author.

## Determinism

- All random seeds default to 42 (`configs/config.py` `seed`, plus per-method
  seeds in `data/extract_wpad.py`, `data/synthetic_generator.py`,
  `models/imi_estimator.py`, `evaluation/drope_evaluator.py`).
- Bootstrap iterations use independent seeds derived from the master seed.
- The 80/20 train/test split uses `random_state=42` in `train_test_split`.
- Given identical inputs and seed, the pipeline is bit-for-bit reproducible
  across runs on the same Python/NumPy/scikit-learn versions specified in
  `requirements.txt`.

## Output paths

All pipeline outputs default to `notebooks/pearl/outputs/` at the repository
root. Override via the `PEARL_OUTPUT_BASE` environment variable:

```bash
PEARL_OUTPUT_BASE=/tmp/pearl_run python scripts/run_pipeline.py --synthetic
```

Output structure:

```
notebooks/pearl/outputs/
├── results/
│   ├── main_results_table.csv       # 13-policy DR-OPE + IMI comparison
│   ├── sensitivity_results.csv      # 20 pre-specified sensitivity analyses
│   └── einstein_arena_results.json  # hyperparameter search results (if enabled)
├── checkpoints/
│   └── *.pkl                        # trained model artifacts (per seed)
└── figures/
    └── fig*.pdf                     # manuscript figures
```

## Verification commands

```bash
# (1) Synthetic-data demonstration (no DUA, ~2 minutes on CPU)
python scripts/run_pipeline.py --synthetic --n_patients 50000

# (2) Verify outputs land in notebooks/pearl/outputs/results/
ls notebooks/pearl/outputs/results/

# (3) Reproduce manuscript figures from the canonical CSVs
python scripts/generate_figures.py

# (4) Run all 20 pre-specified sensitivity analyses
bash scripts/run_sensitivity.sh
```

## Software environment

Pinned versions in `requirements.txt`:

- Python ≥ 3.10
- pandas ≥ 2.0
- numpy ≥ 1.24
- scikit-learn ≥ 1.3
- scipy ≥ 1.11
- statsmodels ≥ 0.14
- matplotlib ≥ 3.7
- xgboost ≥ 1.7
- lightgbm ≥ 4.0

A `Dockerfile` and `environment.yml` snapshot for the CodeOcean capsule will be
released on acceptance.

## CodeOcean capsule

Capsule contents (DOI to be assigned on acceptance):

- Pipeline code (this repository, archived at the version-pinned tag)
- Synthetic data generator (`data/synthetic_generator.py`)
- Pre-pinned environment (Python 3.12, all `requirements.txt` versions)
- Reproduction script: `scripts/run_pipeline.py --synthetic --n_patients 50000`
- Expected outputs (`notebooks/pearl/outputs/results/*.csv`) for byte-level diff
- Example figures (`notebooks/pearl/outputs/figures/*.pdf`)

The capsule does not include real ACO data. Reviewers can rerun the synthetic
pipeline in-browser and verify outputs match the expected files supplied with
the capsule.

## Real-data replication

The full pipeline expects the following parquet files in
`PEARL_DATA_PATH` (or `data/real_inputs/` by default):

| File | Purpose |
|---|---|
| `signal_risk_latest.parquet` | risk scores and condition flags |
| `clinical_summary.parquet` | Charlson index, diagnosis flags |
| `pharmacy_summary.parquet` | medication fill counts |
| `outcomes_monthly.parquet` | monthly utilization aggregates |
| `member_status_event.parquet` | enrollment and onboarding event log |
| `hospital_visits.parquet` | inpatient and ED visit records |
| `member_goals.parquet` | care management intervention assignments |
| `member_attributes.parquet` | demographics and risk score |
| `eligibility.parquet` | Medicaid coverage windows |
| `member_patient_map.parquet` | identifier crosswalk |

Schema definitions are in `data/extract_wpad.py`. The DUA template is available
from the corresponding author (sanjay.basu@ucsf.edu).

## Audit checklist

For an external auditor verifying that manuscript numbers match canonical pipeline output:

```bash
# Run the full real-data pipeline (requires DUA)
python scripts/run_pipeline.py --waymark --skip_arena

# Verify primary numbers match the manuscript
grep -E "Behavioral Policy|PEARL.MoE Router|BehavioralCloning SFT" \
  notebooks/pearl/outputs/results/main_results_table.csv

# Expected manuscript values:
#   Behavioral Policy IMI = 0.270 (manuscript: 27.0%)
#   PEARL (MoE Router) IMI = 0.160 (manuscript: 16.0%)
#   BehavioralCloning SFT (C4) IMI = 0.044 (manuscript: 4.4%)
#   Behavioral DR-OPE = 0.0396 (manuscript: 0.040)
#   PEARL (MoE Router) DR-OPE = 0.0413 (manuscript: 0.041)
#   PEARL ESS = 6995, Behavioral ESS = 2670

# Verify sensitivity analysis primary IMI
grep -E ",True,0\.[0-9]" notebooks/pearl/outputs/results/sensitivity_results.csv
# Expected: imi_estimate = 0.1603 across all primary-value rows (16.0%)
```
