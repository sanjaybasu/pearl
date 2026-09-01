"""
Feasibility check for the rebuilt analysis.

Question: if the assigned action is restricted to patients for whom it was
actually documented, and time-anchored to the index window, is the resulting
cohort large enough and well enough overlapped to estimate action-specific
contrasts? Reports cohort size, action distribution, per-action event counts,
propensity overlap, and contact volume.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

RAW = "/Users/sanjaybasu/waymark-local/data/real_inputs"

from data.extract_wpad import build_waymark_population, GOAL_MAP

pop = build_waymark_population(verbose=False)
rising = pop.patients.reset_index(drop=True)
print(f"rising-risk cohort: {len(rising):,}")

g = pd.read_parquet(f"{RAW}/member_goals.parquet")
g["goal_created_at"] = pd.to_datetime(g["goal_created_at"], errors="coerce", utc=True)
g = g[(g["category"] != "DEFAULT") & g["goal_created_at"].notna()].copy()
g["action"] = g["category"].map(GOAL_MAP)
g = g.dropna(subset=["action"])
print(f"dated non-DEFAULT goals mapping to an action: {len(g):,} "
      f"across {g['member_id'].nunique():,} members")

idx = rising[["member_id", "index_date", "y_behavioral"]].copy()
idx["index_date"] = pd.to_datetime(idx["index_date"], utc=True, errors="coerce")

m = g.merge(idx, on="member_id", how="inner")
m["days"] = (m["goal_created_at"] - m["index_date"]).dt.total_seconds() / 86400.0

for lo, hi, name in [(0, 90, "index to +90d"),
                     (-90, 90, "-90d to +90d"),
                     (-180, 90, "-180d to +90d"),
                     (-10000, 10000, "any time (current label)")]:
    w = m[(m["days"] >= lo) & (m["days"] <= hi)]
    n_pat = w["member_id"].nunique()
    print(f"\nwindow {name}: {len(w):,} goals, {n_pat:,} patients "
          f"({100*n_pat/len(rising):.1f}% of cohort)")
    if n_pat:
        modal = (w.groupby(["member_id", "action"]).size().reset_index(name="n")
                  .sort_values("n", ascending=False).drop_duplicates("member_id"))
        dist = modal["action"].value_counts()
        ev = modal.merge(idx[["member_id", "y_behavioral"]], on="member_id")
        evc = ev.groupby("action")["y_behavioral"].agg(["size", "sum"])
        print(f"  actions represented: {len(dist)}; "
              f"min per action {dist.min()}; median {int(dist.median())}")
        print(f"  overall event rate: {100*ev['y_behavioral'].mean():.2f}%")
        show = evc.sort_values("size", ascending=False)
        show.columns = ["n", "events"]
        print(show.to_string())

# contact volume, for reporting how many touches accompany an action
enc = pd.read_parquet(f"{RAW}/encounters.parquet")
enc = enc[enc["encounter_occurred"] == "YES"]
enc["created_at"] = pd.to_datetime(enc["created_at"], errors="coerce", utc=True)
pm = pd.read_parquet(f"{RAW}/member_patient_map.parquet")[["patient_id", "member_id"]]
enc = enc.merge(pm, on="patient_id", how="left")
e2 = enc.merge(idx, on="member_id", how="inner")
e2["days"] = (e2["created_at"] - e2["index_date"]).dt.total_seconds() / 86400.0
w = e2[(e2["days"] >= 0) & (e2["days"] <= 90)]
per = w.groupby("member_id").size()
print(f"\ncompleted encounters within 90 days of index: {len(w):,}")
print(f"  patients with >=1: {per.size:,} ({100*per.size/len(rising):.1f}%)")
if per.size:
    print(f"  per patient mean {per.mean():.1f}, median {per.median():.0f}, "
          f"p90 {per.quantile(0.9):.0f}")
