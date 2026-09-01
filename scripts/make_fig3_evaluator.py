"""
Figure 3: intervention misalignment for the same policies under four evaluation
schemes. Reads the canonical result CSVs written by revision_analyses.py; no
value is hard-coded.

Usage:
  PEARL_OUTPUT_BASE=<base> python scripts/make_fig3_evaluator.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
BASE = Path(os.environ.get("PEARL_OUTPUT_BASE",
                           str(_REPO / "notebooks" / "pearl" / "outputs")))
RESULTS = BASE / "results"
FIG_PNG = _REPO / "notebooks" / "pearl" / "figures_png"
FIG_PDF = _REPO / "notebooks" / "pearl" / "figures"
FIG_PNG.mkdir(parents=True, exist_ok=True)
FIG_PDF.mkdir(parents=True, exist_ok=True)

arm1 = pd.read_csv(RESULTS / "revision_arm1_independent_evaluator.csv")
arm2 = pd.read_csv(RESULTS / "revision_arm2_splitsample.csv")

df = arm1.merge(arm2[["policy", "imi_splitsample_rf"]], on="policy", how="left")

ORDER = [
    "Behavioral Policy",
    "PEARL (MoE Router)",
    "BehavioralCloning SFT (C4)",
    "PEARL (MoE Full)",
    "PEARL (base)",
    "CQL (C8)",
    "Observational DPO (C5)",
    "Oracle (mu-hat optimal)",
]
LABEL = {
    "Behavioral Policy": "Current routing",
    "PEARL (MoE Router)": "PEARL (primary)",
    "BehavioralCloning SFT (C4)": "Behavioral cloning (C4)",
    "PEARL (MoE Full)": "PEARL (MoE Full)",
    "PEARL (base)": "PEARL (base)",
    "CQL (C8)": "Conservative Q-learning (C8)",
    "Observational DPO (C5)": "Observational DPO (C5)",
    "Oracle (mu-hat optimal)": "Oracle (theoretical bound)",
}
SCHEMES = [
    ("imi_primary", "Pre-registered model\n(trained the policy)", "#B23A48"),
    ("imi_crossfit_rf", "Cross-fitted\nrandom forest", "#1F6F8B"),
    ("imi_crossfit_logit", "Cross-fitted\nlogistic", "#3D8168"),
    ("imi_splitsample_rf", "Full sample\nsplitting", "#7A5C8E"),
]

df = df.set_index("policy").reindex(ORDER).reset_index()

fig, ax = plt.subplots(figsize=(11, 6.0))
n_pol, n_sch = len(df), len(SCHEMES)
y = np.arange(n_pol)
h = 0.80 / n_sch

for k, (col, lab, color) in enumerate(SCHEMES):
    vals = df[col].values * 100.0
    off = (k - (n_sch - 1) / 2) * h
    ax.barh(y + off, vals, height=h * 0.92, color=color, label=lab,
            edgecolor="white", linewidth=0.5)
    for yi, v in zip(y + off, vals):
        if np.isfinite(v):
            ax.text(v + 1.0, yi, f"{v:.1f}", va="center", ha="left",
                    fontsize=7.5, color="#333333")

ax.set_yticks(y)
ax.set_yticklabels([LABEL[p] for p in df["policy"]], fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel("Intervention misalignment (%), held-out test set (N = 6,995)",
              fontsize=10)
ax.set_xlim(0, 104)
ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

ax.legend(fontsize=8.5, loc="lower right", frameon=True, framealpha=0.95,
          title="Model used to score the policy", title_fontsize=8.5)
ax.set_title(
    "The same policies, scored by four outcome models\n"
    "Only the first column was used to train the policies",
    fontsize=11.5, loc="left", pad=12)

fig.tight_layout()
for out in (FIG_PNG / "fig3_evaluator_dependence.png",
            FIG_PDF / "fig3_evaluator_dependence.pdf"):
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
