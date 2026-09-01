"""
Minimal reproduction check for the published primary intervention-misalignment
values. Builds the population, applies the published 80/20 split, fits the
primary estimator on the training set, and reports misalignment on the held-out
set for the behavioral policy and the mu-hat oracle.

Run under each candidate interpreter to test environment sensitivity:
  python3 scripts/check_repro.py
  /opt/anaconda3/bin/python3.12 scripts/check_repro.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data.extract_wpad import build_waymark_population
from models.imi_estimator import IMIEstimator, INTERVENTIONS

INTV_ALPHA = sorted(INTERVENTIONS)

print(f"python      {platform.python_version()}")
print(f"scikit-learn {sklearn.__version__}")
print(f"numpy        {np.__version__}")
print(f"pandas       {pd.__version__}")

pop = build_waymark_population(verbose=False)
rising = pop.patients.reset_index(drop=True)
train, test = train_test_split(rising, test_size=0.20, random_state=42,
                               stratify=rising["behavioral_intervention"])
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

est = IMIEstimator(outcome_col="y_behavioral",
                   intervention_col="behavioral_intervention",
                   threshold=0.02, n_bootstrap=0, seed=42).fit(train)
mu = est._predict_outcomes(est._get_feature_matrix(test))
le = LabelEncoder().fit(INTV_ALPHA)

a = le.transform(test["behavioral_intervention"].values)
own = mu[np.arange(len(mu)), a]
alt = mu.copy()
alt[np.arange(len(mu)), a] = np.inf
imi_behavioral = float((alt.min(axis=1) < own - 0.02).mean())

print(f"\nN train {len(train):,}  N test {len(test):,}")
print(f"IMI(behavioral policy), held-out test : {imi_behavioral*100:.2f}%")
print(f"  published value in Table 3          : 27.00%")
print(f"mean mu-hat (behavioral arm)          : {own.mean():.4f}")
print(f"mean between-arm spread (max - min)   : "
      f"{(mu.max(axis=1) - mu.min(axis=1)).mean():.4f}")
