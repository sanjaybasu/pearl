# PEARL: Policy Evolution through Aligned Retrospective Learning

**Measuring and evaluating intervention misalignment in next best action selection for care management: a retrospective cohort study**

Basu S, Sheth P, Patel S. Under review, 2026.

- **Pre-registration (OSF):** [https://doi.org/10.17605/OSF.IO/D69M2](https://doi.org/10.17605/OSF.IO/D69M2)
- **Software archive (Zenodo, all versions):** [https://doi.org/10.5281/zenodo.22236900](https://doi.org/10.5281/zenodo.22236900)
- **License:** MIT (code only; zero patient data)

---

## Overview

Care managers routinely choose one next best action for Medicaid patients with concurrent chronic, behavioral, medication, and social needs, yet standard risk scores indicate *who* needs care without indicating *which* intervention to prioritize. Offline evaluation of care-management routing from routinely collected data is vulnerable to four failure modes: proxy action labels inferred from clinical flags, acute-event referral confounding by indication, self-evaluation and extrapolation beyond observed support, and pre/post enrollment selection artifacts.

This repository implements **PEARL (Policy Evolution through Aligned Retrospective Learning)** together with a **four-safeguard offline routing and audit framework**:

1. **Action-ascertainment validation and covariate-overlap auditing:** Validates inferred clinical-flag heuristics against documented care-plan goals and completed-encounter clinician disciplines, and quantifies five-fold cross-validated multinomial logistic and nonlinear tree predictability to separate genuine treatment overlap from propensity-model smoothing.
2. **Daily pre-action acute-care recency adjustment and strictly pre-index time-stamped predictor sensitivity:** Adjusts for daily pre-index acute events (7, 14, 30, and 90 days plus prior 90-day inpatient stays) and verifies robustness when all cross-sectional snapshot flags are omitted.
3. **Explicit separation of policy safeguards:** Fits all propensity models, $M = 10$ without-replacement 80%-subsample inverse-probability-weighted gradient-boosted ensembles, nested 70/30 uncertainty-penalty ($\lambda$) tuning, and independent evaluators (200-tree random forest and L2-penalized logistic regression) strictly on training splits, isolating support restriction, the 2 percentage-point switching margin ($\varepsilon = 0.02$), uncertainty penalization ($\lambda \in \{0, 0.5, 1.0, 2.0\}$), and independent evaluation alongside switched-patient nonlinear propensity and effective sample size diagnostics.
4. **End-to-end Bayesian bootstrap uncertainty and matched-capacity case-review benchmarking:** Propagates estimation, selection, and sampling uncertainty across $B = 200$ Rubin Bayesian bootstrap replicates with Efron bias-corrected 95% confidence intervals, and benchmarks the misalignment flag against both existing rising-risk scores and calibrated baseline acute-care risk at exact matched review capacity.

---

## Repository Structure

```text
pearl/
├── configs/
│   └── config.py                        # Shared pipeline defaults and seeds
├── data/
│   ├── extract_wpad.py                  # Cohort builder and DUA-governed parquet schema
│   └── synthetic_generator.py           # Legacy synthetic data generator (zero PHI)
├── evaluation/
│   ├── drope_evaluator.py               # Direct-method and doubly robust policy evaluation
│   └── falsification_tests.py           # Pre-registered falsification tests (T1–T6)
├── models/
│   ├── imi_estimator.py                 # Misalignment estimator and Camden illustrative analog
│   ├── pearl_dpo.py                     # Pre-registered preference-optimization comparator
│   └── comparators.py                   # Baseline comparator implementations
├── scripts/
│   ├── run_v6_synthetic_demo.py         # Fast CPU synthetic reproduction of v6 leak-free pipeline
│   ├── pearl_v6_revision.py             # Main leak-free multi-cohort analysis and bootstrap pipeline
│   ├── pearl_v6_extended.py             # Daily acute-recency, strictly pre-index, and Camden analyses
│   ├── make_v6_figures.py               # Data-driven figure generation from canonical JSON/CSVs
│   ├── staffing_falsification_t7.py     # Staffing-capacity falsification test (T7)
│   ├── build_svi.py                     # Public CDC/ATSDR SVI 2022 ZCTA crosswalk builder
│   ├── revision_analyses.py             # Shared covariate, calibration, and eligibility utilities
│   └── run_pipeline.py                  # Legacy synthetic/real pipeline entrypoint
├── CITATION.cff                         # Machine-readable citation metadata
├── LICENSE                              # MIT License
├── REPRODUCIBILITY.md                   # Environment specification and verification guide
└── requirements.txt                     # Python package requirements
```

---

## Quickstart (Synthetic Data — Zero Patient Data Required)

```bash
pip install -r requirements.txt

# Run the v6 leak-free pipeline on synthetic 4-track care-management data (~15-25s on CPU)
python scripts/run_v6_synthetic_demo.py --n_patients 2500 --n_boot 20
```

The synthetic demonstration executes the exact `pearl_v6_revision.analyze_cohort` core (five-fold covariate determinism audit, training-only propensity and $M = 10$ subsample ensemble fitting, nested $\lambda$ selection, independent random forest and penalized logistic evaluation, component-separated ablations, switched-patient support diagnostics, end-to-end Bayesian bootstrap 95% CIs, and matched-capacity decision-curve benchmarking) without requiring any restricted health data.

---

## Running on Restricted Operational Data (Requires DUA)

Patient-level Medicaid data cannot be shared publicly under privacy and data-use restrictions. With authorized local extracts matching the schema in `data/extract_wpad.py`, set environment variables and run:

```bash
export PEARL_DATA_PATH=/path/to/real_inputs
export PEARL_PROCESSED_PATH=/path/to/processed
export PEARL_SVI_PATH=/path/to/processed/svi_by_zip.parquet
export PEARL_OUTPUT_BASE=/path/to/outputs
export PEARL_FIGURES_DIR=/path/to/figures_png

python scripts/pearl_v6_revision.py
python scripts/pearl_v6_extended.py
python scripts/make_v6_figures.py
```

See `REPRODUCIBILITY.md` for pinned package versions (`Python 3.12.2`, `scikit-learn 1.6.1`, `NumPy 1.26.4`, `pandas 2.2.2`, `SciPy 1.14.1`) and input schema details.

---

## Citation

```bibtex
@article{basu2026pearl,
  title   = {Measuring and evaluating intervention misalignment in next best action
             selection for care management: a retrospective cohort study},
  author  = {Basu, Sanjay and Sheth, Parth and Patel, Sadiq},
  year    = {2026},
  note    = {Under review}
}
```

## License and Data Governance

Code is released under the MIT License (`LICENSE`). This repository contains source code and synthetic demonstrations only; no patient data, model binaries fitted to patient data, or individual-level outputs are included. For data-use inquiries, contact the corresponding author (`sanjay.basu@ucsf.edu`).
