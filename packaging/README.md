# PEARL: Policy Evolution through Aligned Retrospective Learning

Next best action selection for rising-risk ACO care management using
within-patient causal identification and tabular IPTW-DPO.

## Citation

Basu S, et al. PEARL: AI-Guided Next Best Action Selection for Rising-Risk
Medicaid ACO Care Management — A Within-Patient Natural Experiment.
*Lancet Digital Health*, 2026. [In submission]

## Overview

PEARL trains a care management policy on within-patient preference pairs
derived from staggered ACO onboarding schedules (Within-Patient
Administrative Discontinuity; WPAD). These natural experiments provide
plausibly exogenous variation in care management receipt, enabling
identification of which intervention type produces better outcomes for
each patient profile — without population-level unconfoundedness assumptions.

The trained policy is evaluated using doubly robust off-policy evaluation
(DR-OPE) and reduces the Intervention Misalignment Index (IMI), a formally
defined causal estimand measuring the fraction of patients receiving a
suboptimal care management intervention type.

## Repository Structure

```
pearl/
├── data/
│   ├── synthetic_generator.py   # Public reproducibility; no proprietary data
│   └── extract_wpad.py          # WPAD pair construction (requires ACO data)
├── models/
│   ├── pearl_dpo.py             # Tabular IPTW-DPO training (Algorithm 2)
│   ├── imi_estimator.py         # IMI estimation and decomposition
│   └── comparators.py           # Eight published comparator implementations
├── mixture_of_experts/
│   └── moe_router.py            # MoE Router for per-patient policy dispatch
├── evaluation/
│   ├── drope_evaluator.py       # DR-OPE policy evaluation + paired bootstrap
│   └── falsification_tests.py   # WPAD falsification tests T1–T5
├── scripts/
│   ├── run_pipeline.py          # End-to-end pipeline runner
│   ├── generate_figures.py      # Manuscript figures (requires results CSV)
│   └── run_sensitivity.sh       # Sensitivity analysis runner
├── notebooks/
│   └── pearl_demo.ipynb         # Public reproducibility demo (synthetic data)
└── configs/
    └── config.py                # Hyperparameters and paths
```

Manuscript, supplement, and submission figures are not included in this repository per the journal data use agreement.


## Quickstart (Synthetic Data)

No proprietary data required. Runs on CPU in ~2 minutes.

```bash
# 1. Install dependencies
pip install -r packaging/requirements.txt

# 2. Run pipeline in synthetic mode
python scripts/run_pipeline.py --synthetic --n-patients 50000

# 3. Run demo notebook
jupyter notebook notebooks/pearl_demo.ipynb
```

## Running with ACO Data

Requires a Waymark data use agreement. Data schema is described in
`data/extract_wpad.py`. To run on proprietary data:

```bash
python scripts/run_pipeline.py --waymark \
    --data-path /path/to/waymark/claims.parquet
```

## Key Methods

### WPAD Identification (Algorithm 1)

Staggered ACO onboarding schedules create within-patient comparisons:
patients enrolled early (ON-window) are compared to the same patients
before enrollment (OFF-window). Administrative exogeneity is assessed
via five pre-specified falsification tests (T1–T5).

### IPTW-DPO Training (Algorithm 2)

PEARL trains a tabular logistic regression policy (proxy for the full
Llama-3.1-8B LLM pipeline) using:
- Demographic-stratified IPTW upsampling (Hardt et al., NeurIPS 2016)
- Per-group DPO loss with equal group weights (Sagawa et al., ICLR 2020)
- Abstention when DPO log-ratio margin < τ

### DR-OPE Evaluation

Policy value estimated using the Marginalized DR estimator (Kallus &
Uehara, NeurIPS 2020). Paired bootstrap comparison (1,000 resamples)
provides confidence intervals and p-values for policy comparisons.

### IMI Estimation

The Intervention Misalignment Index estimates the fraction of patients
for whom a better-matched intervention exists under the counterfactual
distribution identified via WPAD. Decomposed into demographic-IMI and
clinical-IMI marginal components.

## Comparators

Eight published comparators are included:

| ID | Method | Citation |
|----|--------|----------|
| C1 | LACE Index | van Walraven et al., CMAJ 2010 |
| C2 | HOSPITAL Score | Donzé et al., JAMA Intern Med 2013 |
| C3 | XGBoost readmission prediction | Gradient boosting best practice |
| C4 | Behavioral Cloning SFT | InstructGPT / SFT literature |
| C5 | Observational DPO | Rafailov et al., NeurIPS 2023 |
| C6 | Causal Forest CATE | Wager & Athey, JASA 2018 |
| C7 | Decision Transformer | Chen et al., NeurIPS 2021 |
| C8 | Conservative Q-Learning | Kumar et al., NeurIPS 2020 |

## Falsification Tests

| Test | Description |
|------|-------------|
| T1 | Covariate balance (paired t-tests on pre-event features) |
| T2 | Placebo outcome (12-month pre-event outcomes) |
| T3 | Administrative predictors (health trajectory vs. churn timing) |
| T4 | Density continuity (non-care-management healthcare events at transition) |
| T5 | Heterogeneous churn (LATE comparison by churn type) |

## License

MIT License. See LICENSE for details.

Data use: The proprietary ACO care management data used in the manuscript
is available under a Waymark data use agreement. Contact
sanjaybasu@waymark.com for research access inquiries.

## Acknowledgments

This work was supported by Waymark. The corresponding author is employed
by Waymark; this relationship is disclosed in the manuscript's conflict
of interest statement.
