# PEARL: Policy Evolution through Aligned Retrospective Learning

**Measuring and reducing intervention misalignment in next best action selection for care management: a within-patient natural experiment**

Basu S, Sheth P, Patel S. Manuscript under review, 2026.

The Intervention Misalignment Index (IMI) is the fraction of patients assigned a care management intervention type for which a better-matched alternative exists, identifiable from within-patient natural experiments created by staggered ACO onboarding logistics (Within-Patient Administrative Discontinuity, WPAD). Among 34,971 rising-risk Medicaid patients, IMI under behavioral routing is 27.0%; the within-patient causal-identification signal — not the preference-optimization architecture — drives intervention-matching improvement.

---

## What PEARL does

Current ACO care management programs route patients to interventions using risk scores — they identify *who* needs care but not *which* care. PEARL learns which intervention type produces better outcomes for each patient profile, using a causal identification strategy that does not require population-level unconfoundedness assumptions.

Results are reported in the accompanying manuscript (under review). This repository provides the complete code pipeline for reproducing the analysis on ACO care management data or running the synthetic demonstration.

---

## Method overview

### 1. WPAD causal identification

Staggered ACO onboarding schedules create within-patient natural experiments: patients enrolled early (ON-window) are compared to the same patients before enrollment (OFF-window). This Within-Patient Administrative Discontinuity (WPAD) design requires only administrative exogeneity — that onboarding order is driven by operational logistics, not patient health status — a weaker assumption than population-level unconfoundedness.

Six pre-specified falsification tests (T1–T6) assess administrative exogeneity. T1–T4 pass; T5 triggers restriction of the primary analysis to Type 1 (ACO onboarding) pairs.

### 2. IPTW-DPO preference optimization

PEARL trains a tabular preference model using IPTW-weighted Direct Preference Optimization on WPAD preference pairs. The DPO loss is equivalent to doubly robust off-policy optimization at the behavioral policy initialization:

- **Stage 1**: Demographic-stratified IPTW upsampling (equal group representation per batch)
- **Stage 2**: Per-group DPO loss with equal group weights (group-stratified fairness constraint)
- **Abstention**: PEARL defers to behavioral routing when the DPO log-ratio margin < τ

### 3. DR-OPE evaluation

Policy value estimated using the Marginalized DR estimator. Lower values = fewer predicted 90-day acute care events. Paired bootstrap (2,000 resamples) provides confidence intervals and hypothesis tests.

### 4. Intervention Misalignment Index (IMI)

IMI is the fraction of patients for whom the behavioral routing policy assigns a care management type for which a better-matched alternative exists by ≥ 2 percentage points in predicted acute care event probability:

$$\text{IMI}(\pi_b) = \mathbb{E}_i\!\left[\mathbf{1}\!\left(\exists\, a \neq A_i : \hat{\mu}_{DR}(X_i,\, a) < \hat{\mu}_{DR}(X_i,\, A_i) - \varepsilon\right)\right]$$

IMI is decomposed into marginal demographic-IMI and clinical-IMI components.

---

## Repository structure

```
pearl/
├── configs/
│   └── config.py                 # Hyperparameters and pipeline settings
├── data/
│   ├── extract_wpad.py           # WPAD pair construction (Algorithm 1; requires ACO data)
│   └── synthetic_generator.py   # Synthetic data generator (no DUA required)
├── evaluation/
│   ├── drope_evaluator.py        # DR-OPE and DM policy evaluation; paired bootstrap
│   └── falsification_tests.py   # T1–T6 WPAD administrative exogeneity tests
├── models/
│   ├── pearl_dpo.py              # Tabular IPTW-DPO training (Algorithm 2)
│   ├── imi_estimator.py          # IMI estimation, decomposition, E-value
│   └── comparators.py           # C1–C8 published comparator implementations
├── mixture_of_experts/
│   └── moe_router.py             # MoE Router: per-intervention expert + softmax gate
├── experiments/
│   └── einstein_arena.py         # Adversarial critique framework for design review
├── scripts/
│   ├── run_pipeline.py           # End-to-end pipeline (--synthetic or --waymark)
│   ├── generate_figures.py       # Reproduce manuscript figures from results CSV
│   ├── run_sensitivity.sh        # Run all 20 pre-specified sensitivity analyses
│   ├── build_pearl_pdf.py        # Build combined manuscript+supplement PDF (pandoc+xelatex)
│   └── build_pearl_docx.py      # Build DOCX submission files (pandoc+python-docx)
├── notebooks/
│   └── pearl_demo.ipynb          # Full pipeline demo on synthetic data (~2 min, CPU)
└── packaging/
    ├── README.md
    ├── requirements.txt
    └── setup.py
```

---

## Quickstart (synthetic data — no data use agreement required)

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline on synthetic data (~2 min on CPU)
python scripts/run_pipeline.py --synthetic --n_patients 50000

# Or open the notebook demo
jupyter notebook notebooks/pearl_demo.ipynb
```

The synthetic pipeline runs WPAD pair construction, falsification tests T1–T6, IMI estimation, IPTW-DPO training, DR-OPE evaluation of all 13 policies, and sensitivity analysis — producing the same output structure as the real-data run.

## Reproducibility

All pipeline outputs (CSVs, JSONs, checkpoints, figures) are written to `notebooks/pearl/outputs/` at the repository root by default. Override the output location via the `PEARL_OUTPUT_BASE` environment variable.

```bash
# Use a custom output directory
PEARL_OUTPUT_BASE=/tmp/pearl_run python scripts/run_pipeline.py --synthetic
```

The full pipeline is deterministic given a fixed seed (default 42 in `configs/config.py`). Bootstrap confidence intervals use a separate seed via `--seed` flag.

A CodeOcean compute capsule reproducing the synthetic-data pipeline end-to-end (DOI to be assigned on acceptance) provides a public, browser-runnable verification environment that requires no DUA.

---

## Running on ACO data

Requires a data use agreement. Set environment variable `WAYMARK_DATA_PATH` to the directory containing the required parquets, then:

```bash
python scripts/run_pipeline.py --waymark
```

Expected inputs (described in `data/extract_wpad.py`):
- `signal_risk_latest.parquet` — risk scores and condition flags
- `clinical_summary.parquet` — Charlson index, diagnoses
- `pharmacy_summary.parquet` — medication fill records
- `outcomes_monthly.parquet` — 90-day acute care events
- `member_status.parquet` — enrollment and onboarding dates
- `hospital_visits.parquet` — inpatient and ED visit records

---

## Comparators (pre-registered)

| ID | Method | Citation |
|----|--------|----------|
| C1 | LACE Index | van Walraven et al., *CMAJ* 2010 |
| C2 | HOSPITAL Score | Donzé et al., *JAMA Intern Med* 2013 |
| C3 | XGBoost readmission prediction | — |
| C4 | Behavioral Cloning SFT | Ablation: DPO vs. supervised imitation |
| C5 | Observational DPO | Ablation: WPAD identification vs. naive DPO |
| C6 | Causal Forest CATE | Wager & Athey, *J Am Stat Assoc* 2018 |
| C7 | Decision Transformer | Chen et al., *NeurIPS* 2021 |
| C8 | Conservative Q-Learning (CQL) | Kumar et al., *NeurIPS* 2020 |

---

## Sensitivity analyses

Twenty pre-specified analyses vary seven parameters (window length, IPTW clip, DPO β, outcome window, trajectory adjustment, WPAD type restriction, IMI threshold). Run all:

```bash
bash scripts/run_sensitivity.sh
```

Results of all 20 sensitivity analyses are reported in the accompanying manuscript. The ε = 0.01 IMI threshold analysis is pre-specified as a major sensitivity finding.

---

## Citation

```bibtex
@article{basu2026pearl,
  title   = {Measuring and reducing intervention misalignment in next best action
             selection for care management: a within-patient natural experiment},
  author  = {Basu, Sanjay and Sheth, Parth and Patel, Sadiq},
  year    = {2026},
  note    = {Manuscript under review}
}
```

---

## License

Apache License 2.0.

Proprietary ACO care management data used in the manuscript is not included. A data use agreement template for ACO replication is available from the corresponding author (sanjay.basu@ucsf.edu).
