# PEARL: Policy Evolution through Aligned Retrospective Learning

**AI-guided next best action selection for rising-risk Medicaid ACO care management**

Basu S, Sheth P, Patel S. *The Lancet Digital Health* (under review), 2026.

---

## What PEARL does

Current ACO care management programs route patients to interventions using risk scores — they identify *who* needs care but not *which* care. PEARL learns which intervention type produces better outcomes for each patient profile, using a causal identification strategy that does not require population-level unconfoundedness assumptions.

**Key results (N = 34,971 rising-risk Medicaid patients):**
- Intervention Misalignment Index under behavioral routing: **10.0%** (95% CI 9.3–10.6%; E-value 11.31)
- PEARL reduces IMI to **2.0%** (Δ = 7.9 pp; p < 0.001) — 80% of the maximum achievable reduction
- PEARL ranks 3rd of 13 evaluated policies on doubly robust off-policy evaluation (DR-OPE = 0.039), outperforming all eight published comparators
- 2.0-fold socioeconomic deprivation gradient in IMI (6.8% ADI Q1 → 13.9% ADI Q5)

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
│   └── run_sensitivity.sh        # Run all 20 pre-specified sensitivity analyses
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
python scripts/run_pipeline.py --synthetic --n-patients 50000

# Or open the notebook demo
jupyter notebook notebooks/pearl_demo.ipynb
```

The synthetic pipeline runs WPAD pair construction, falsification tests T1–T6, IMI estimation, IPTW-DPO training, DR-OPE evaluation of all 13 policies, and sensitivity analysis — producing the same output structure as the real-data run.

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

DR-OPE direction is preserved across all 20 analyses. The IMI threshold sensitivity (ε = 0.01) is a pre-specified major finding: at a 1 pp threshold, PEARL IMI rises to 21.7%; the 2 pp primary threshold was pre-specified as the minimum clinically meaningful difference.

---

## Citation

```bibtex
@article{basu2026pearl,
  title   = {{PEARL}: {AI}-guided next best action selection for rising-risk
             {Medicaid} {ACO} care management --- a within-patient natural experiment},
  author  = {Basu, Sanjay and Sheth, Parth and Patel, Sadiq},
  journal = {The Lancet Digital Health},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

Apache License 2.0.

Proprietary ACO care management data used in the manuscript is not included. A data use agreement template for ACO replication is available from the corresponding author (sanjaybasu@waymark.com).
