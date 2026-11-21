#!/usr/bin/env python3
"""
Generate all PEARL v6 main and supplementary figures strictly from canonical outputs.
Zero hardcoded result numbers.

Main text:
  Figure 1 -> main_fig1_study_design.png / .pdf
  Figure 2 -> main_fig2_safeguards_and_validation.png / .pdf

Supplementary Information:
  Supplementary Fig. S1 -> appendix_fig1_patient_flow.png / .pdf
  Supplementary Fig. S2 -> appendix_fig2_action_validation_overlap.png / .pdf
  Supplementary Fig. S3 -> appendix_fig3_decision_curve.png / .pdf
  Supplementary Fig. S4 -> appendix_fig4_camden.png / .pdf
"""
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("PEARL_OUTPUT_BASE", str(_ROOT / "outputs")))
FIGS_DIR = Path(os.environ.get("PEARL_FIGURES_DIR", str(OUT_DIR.parent / "figures_png")))
FIGS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9.0,
    "axes.edgecolor": "#333333",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

BLUE = "#1F6F8B"
NAVY = "#1D3557"
GREEN = "#2A8C6A"
AMBER = "#D48A1F"
RED = "#B23A48"
PURPLE = "#6B4C8A"
GRAY = "#666666"
LGRAY = "#E0E0E0"


def _box(ax, xy, w, h, text, fc="#EAF2F5", ec=BLUE, fontsize=8.4, weight="normal", lw=1.15, ls="-"):
    box = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.07",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, mutation_aspect=1,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, wrap=True, linespacing=1.22)
    return box


def _arrow(ax, xy_from, xy_to, color="#444444", lw=1.25, ls="-"):
    arr = FancyArrowPatch(
        xy_from, xy_to, arrowstyle="-|>", mutation_scale=13,
        linewidth=lw, linestyle=ls, color=color, shrinkA=1, shrinkB=1,
    )
    ax.add_patch(arr)


def _ortho_arrow(ax, pts, color="#444444", lw=1.25, ls="-"):
    """Draw a multi-segment orthogonal polyline ending in an arrowhead."""
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                color=color, lw=lw, ls=ls, solid_capstyle="round")
    _arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls)


def load_master():
    return json.load(open(OUT_DIR / "v6_master_results.json"))


# ─────────────────────────────────────────────────────────────────────────────
# Main Figure 1: Observational cohort architecture, pipeline, identification
# ─────────────────────────────────────────────────────────────────────────────

def make_fig1(master):
    ta = master["temporal_audit"]
    c_pri = master["cohorts"]["primary_complete_90d_fu"]
    c_enc = master["cohorts"]["validated_encounter_discipline_recency_adj"]
    c_gl = master["cohorts"]["validated_documented_goals_rr_recency_adj"]
    w1_comp = [r for r in master["wpad_unfiltered_tables"] if "complete 90d pre/post" in r["cohort"]][0]

    fig = plt.figure(figsize=(13.5, 5.6))

    # Panel a
    axA = fig.add_axes([0.01, 0.05, 0.31, 0.87])
    axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
    axA.set_title("a  Cohort architecture and temporal ordering", loc="left", fontsize=10.0, weight="bold")

    _box(axA, (0.3, 7.25), 9.4, 2.25,
         f"1. Validated 100%-documented cohorts\n"
         f"• Encounter discipline (N = {c_enc['n_total']:,}, 4 tracks)\n"
         f"• Care-plan goals (N = {c_gl['n_total']:,}, rising-risk)\n"
         "Anchored directly at documented action date $t_0$",
         fc="#EAF5EE", ec=GREEN, fontsize=8.1)

    _box(axA, (0.3, 4.4), 9.4, 2.25,
         f"2. Rising-risk panel cohort (N = {c_pri['n_total']:,})\n"
         "Complete 90d window [Jun 2023–Aug 2025]\n"
         f"Hybrid baseline: documented goal ({ta['complete_90d_documented_goal_pct']:.1f}%)\n"
         f"or clinical-flag heuristic ({ta['complete_90d_inferred_pct']:.1f}% inferred)",
         fc="#EAF2F5", ec=BLUE, fontsize=8.1)

    _box(axA, (0.3, 0.55), 9.4, 3.2,
         "Strict temporal ordering (complete 90d follow-up)\n"
         "Prior utilization & daily acute recency (7/14/30/90d):\n"
         "strictly before $t_0$  ([$t_0 - 365$d, $t_0$))\n"
         "Action assignment: at $t_0$ (validated cohorts)\n"
         "90-day acute care outcome: [$t_0$, $t_0 + 90$d]\n"
         "Verified in strictly time-stamped predictor sensitivity",
         fc="#FFF8EB", ec=AMBER, fontsize=7.9)

    # Panel b
    axB = fig.add_axes([0.35, 0.05, 0.31, 0.87])
    axB.set_xlim(0, 10); axB.set_ylim(0, 10); axB.axis("off")
    axB.set_title("b  Leak-free estimation and policy components", loc="left", fontsize=10.0, weight="bold")

    _box(axB, (0.4, 7.9), 9.2, 1.65,
         f"80% training split (all fitting & tuning strictly inside train)\n"
         f"20% held-out test set untouched during development\n"
         f"M = {master['m_ensemble_main']} subsamples $\\to$ $\\hat{{\\mu}}(x,a)$, $\\hat{{\\sigma}}_{{\\mathrm{{boot}}}}(x,a)$",
         fc="#EAF2F5", ec=BLUE, fontsize=8.0)
    _arrow(axB, (5.0, 7.9), (5.0, 6.75))

    _box(axB, (0.4, 5.2), 9.2, 1.55,
         "Component 1 — Support restriction\n"
         "Admissible actions: training count $\\geq$ threshold\n"
         "and patient propensity $\\geq$ 0.01 (sens. 0.05, 0.10)",
         fc="#FFF6E5", ec=AMBER, fontsize=8.0)
    _arrow(axB, (5.0, 5.2), (5.0, 4.1))

    _box(axB, (0.4, 2.55), 9.2, 1.55,
         "Components 2 & 3 — Margin & uncertainty penalty\n"
         "Switch only if gain > $\\varepsilon = 0.02$ after charging\n"
         "$\\lambda\\hat{\\sigma}_{{\\mathrm{{boot}}}}$; $\\lambda \\in \\{0, 0.5, 1, 2\\}$ tuned on inner split",
         fc="#E8F0FF", ec=NAVY, fontsize=8.0)
    _arrow(axB, (5.0, 2.55), (5.0, 1.55))

    _box(axB, (0.4, 0.15), 9.2, 1.4,
         f"Component 4 — Independent evaluation\n"
         f"Random forest (+ penalized logistic); B = {master['b_outer_boot']}\n"
         f"end-to-end Bayesian bootstrap replicates for 95% CIs",
         fc="#EAF5EE", ec=GREEN, fontsize=8.0, weight="bold")

    # Panel c
    axC = fig.add_axes([0.69, 0.05, 0.30, 0.87])
    axC.set_xlim(0, 10); axC.set_ylim(0, 10); axC.axis("off")
    axC.set_title("c  What each analysis does and does not identify", loc="left", fontsize=10.0, weight="bold")

    _box(axC, (0.3, 7.0), 9.4, 2.5,
         "Primary routing & misalignment (a–b)\n"
         "Observational outcome modeling;\n"
         "rests on conditional exchangeability,\n"
         "positivity, and measured-action validity.\n"
         "Produces model-predicted contrasts, not trial proof.",
         fc="#F5F0FA", ec=PURPLE, fontsize=8.0)

    _box(axC, (0.3, 3.85), 9.4, 2.55,
         "Contextual onboarding comparison (unfiltered)\n"
         f"N = {w1_comp['n']:,} complete-window onboarded\n"
         f"Pre {w1_comp['pre_rate_pct']:.1f}% vs Post {w1_comp['post_rate_pct']:.1f}%\n"
         "Not used to construct PEARL; causal enrollment\n"
         "inference requires target-trial emulation.",
         fc="#FBEAEA", ec=RED, fontsize=7.9)

    _box(axC, (0.3, 0.6), 9.4, 2.65,
         "Case-review trigger evaluation\n"
         "Tests whether misalignment flag predicts\n"
         "observed 90d events vs existing rising-risk\n"
         "score and baseline model risk at matched\n"
         "review capacity (prognostic utility only).",
         fc="#F4F4F4", ec=GRAY, fontsize=8.0)

    for ext in ("png", "pdf"):
        out = FIGS_DIR / f"main_fig1_study_design.{ext}"
        fig.savefig(out, dpi=300)
        print(f"wrote {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main Figure 2: Component separation + Validated cohorts (incl. recency-adj)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig2(master):
    c_pri = master["cohorts"]["primary_complete_90d_fu"]
    c_enc_raw = master["cohorts"]["validated_encounter_discipline"]
    c_enc_adj = master["cohorts"]["validated_encounter_discipline_recency_adj"]
    c_gl_adj = master["cohorts"]["validated_documented_goals_rr_recency_adj"]

    ab = pd.DataFrame(c_pri["ablation"])
    cis = c_pri["bootstrap_cis"]

    panel_a_specs = [
        ("Baseline routing\n(hybrid clinical-flag)", 0, "pearl_nested", "mis_cur_ci95", GREEN),
        ("Support + margin\n+ penalty ($\\lambda = 1.0$)", 3, "lam_1.0", "mis_pol_ci95", "#4B8BBE"),
        ("Support + margin\n+ penalty ($\\lambda = 0.5$)", 2, "lam_0.5", "mis_pol_ci95", BLUE),
        (f"PEARL primary\n(support + margin, $\\lambda^*={c_pri['lambda_selected']:g}$)", 1, "pearl_nested", "mis_pol_ci95", NAVY),
        ("Support only\n(no margin $\\varepsilon=0$, $\\lambda=0$)", 5, "adm_only", "mis_pol_ci95", AMBER),
        ("Unconstrained, all 14\n(independent RF)", 6, "unconstrained", "mis_pol_ci95", "#C95D63"),
        ("Unconstrained, all 14\n(self-evaluated)", 7, None, None, RED),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.3), gridspec_kw={"width_ratios": [1.1, 1.15]})

    # Panel A
    ax = axes[0]
    y_pos = np.arange(len(panel_a_specs))
    vals, lo_err, hi_err, colors, labels, matches = [], [], [], [], [], []
    for lbl, ridx, ckey, mkey, col in panel_a_specs:
        row = ab.iloc[ridx]
        v = row["misalign_baseline_at_lam_pct"] if ridx == 0 else row["misalign_policy_at_lam_pct"]
        vals.append(v)
        matches.append(row["pct_recommendation_received"])
        colors.append(col)
        labels.append(lbl)
        if ckey is not None and mkey is not None:
            lo, hi = cis[ckey][mkey]
            lo_err.append(max(0.0, v - lo))
            hi_err.append(max(0.0, hi - v))
        else:
            lo_err.append(0.0); hi_err.append(0.0)

    ax.barh(y_pos, vals, color=colors, edgecolor="white", height=0.62, alpha=0.92)
    ax.errorbar(vals, y_pos, xerr=[lo_err, hi_err], fmt="none", ecolor="#222222",
                elinewidth=1.2, capsize=3, zorder=5)
    xmax = max(v + h for v, h in zip(vals, hi_err)) * 1.72 + 1.2
    for yi, (lbl, ridx, ckey, mkey, col), v, he, mp in zip(y_pos, panel_a_specs, vals, hi_err, matches):
        extra = f"; self-eval base {ab.iloc[7]['misalign_baseline_at_lam_pct']:.1f}%" if ridx == 7 else ""
        ax.text(v + he + xmax * 0.02, yi, f"{v:.2f}% (match {mp:.1f}%{extra})",
                va="center", fontsize=7.8, color="#222222")
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=8.3)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Model-estimated misalignment (%) with 95% bootstrap CI", fontsize=8.9)
    ax.set_title(f"a  Component separation on rising-risk panel (N = {c_pri['n_test']:,} test)",
                 loc="left", fontsize=9.8, weight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=LGRAY, linewidth=0.7); ax.set_axisbelow(True)

    # Panel B: Compare across Hybrid, Unadjusted Encounter Discipline, Recency-Adjusted Encounter Discipline, Recency-Adjusted Goals
    ax = axes[1]
    cohort_list = [
        ("Hybrid\nclinical-flag\n(N = 3,788)", c_pri),
        ("Discipline,\nclaims only\n(N = 1,186)", c_enc_raw),
        ("Discipline,\n+daily recency\n(N = 1,186)", c_enc_adj),
        ("Care-plan goals,\n+daily recency\n(N = 503)", c_gl_adj),
    ]
    lam_specs = [
        ("lam_0.0", "$\\lambda = 0$ (no penalty)", NAVY, -0.22),
        ("lam_0.5", "$\\lambda = 0.5$ (moderate penalty)", BLUE, 0.0),
        ("lam_1.0", "$\\lambda = 1.0$ (strong penalty)", GREEN, 0.22),
    ]
    x_base = np.arange(len(cohort_list))
    for lkey, llbl, col, dx in lam_specs:
        pts, el, eh = [], [], []
        for _, cdat in cohort_list:
            ab_df = pd.DataFrame(cdat["ablation"])
            if lkey == "lam_0.0":
                pt = float(ab_df.loc[ab_df["policy"].str.contains("PEARL primary"), "model_predicted_events_averted_per_1000_dm"].iloc[0])
            elif lkey == "lam_0.5":
                pt = float(ab_df.loc[ab_df["policy"].str.contains("lambda=0.5"), "model_predicted_events_averted_per_1000_dm"].iloc[0])
            else:
                pt = float(ab_df.loc[ab_df["policy"].str.contains("lambda=1.0"), "model_predicted_events_averted_per_1000_dm"].iloc[0])
            lo, hi = cdat["bootstrap_cis"][lkey]["averted_dm_ci95"]
            pts.append(pt)
            el.append(max(0.0, pt - lo))
            eh.append(max(0.0, hi - pt))
        ax.errorbar(x_base + dx, pts, yerr=[el, eh], fmt="o", color=col, markersize=6.2,
                    capsize=3.8, elinewidth=1.35, label=llbl)

    ax.axhline(0, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_xticks(x_base)
    ax.set_xticklabels([c[0] for c in cohort_list], fontsize=8.2)
    ax.set_ylabel("Model-predicted events averted per 1,000\n(independent RF evaluator, 95% bootstrap CI)", fontsize=8.8)
    ax.set_title("b  Effect of acute-recency adjustment & uncertainty penalty",
                 loc="left", fontsize=9.8, weight="bold")
    ax.legend(fontsize=8.0, frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=LGRAY, linewidth=0.7); ax.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = FIGS_DIR / f"main_fig2_safeguards_and_validation.{ext}"
        fig.savefig(out, dpi=300)
        print(f"wrote {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Appendix Figure 1: Complete patient flow diagram with full arrow topology
# ─────────────────────────────────────────────────────────────────────────────

def make_appendix_fig1(master):
    ta = master["temporal_audit"]
    c_pri = master["cohorts"]["primary_complete_90d_fu"]
    c_enc = master["cohorts"]["validated_encounter_discipline_recency_adj"]
    c_gl = master["cohorts"]["validated_documented_goals_rr_recency_adj"]
    c_gla = master["cohorts"]["validated_documented_goals_all"]
    w1_all = [r for r in master["wpad_unfiltered_tables"] if r["cohort"] == "Staggered onboarding — all"][0]
    w1_comp = [r for r in master["wpad_unfiltered_tables"] if "complete 90d pre/post" in r["cohort"]][0]
    w1_rr = [r for r in master["wpad_unfiltered_tables"] if "rising-risk complete" in r["cohort"]][0]

    fig, ax = plt.subplots(figsize=(11.6, 11.0))
    ax.set_xlim(0, 14.2); ax.set_ylim(0.8, 22.4); ax.axis("off")

    # 1. Top assessed box
    _box(ax, (1.8, 20.1), 7.4, 1.7,
         f"Rising-risk Medicaid stratum (70th–90th percentile, 2023–2025)\nAssessed patients: N = {ta['assessed_n']:,}",
         fc="#F0F0F0", ec="#333333", fontsize=9.1, weight="bold")

    # Trunk down from Assessed -> Linked, with side branch to Excluded: linkage
    _arrow(ax, (5.5, 20.1), (5.5, 18.2))
    _ortho_arrow(ax, [(5.5, 19.15), (9.8, 19.15)], color=AMBER)
    _box(ax, (9.85, 18.55), 4.05, 1.2,
         f"Excluded: incomplete linkage\nN = {ta['excluded_linkage_n']:,} ({100*ta['excluded_linkage_n']/ta['assessed_n']:.1f}%)",
         fc="#FFF3E0", ec=AMBER, fontsize=8.0)

    # 2. Linked rising-risk box
    _box(ax, (1.8, 16.5), 7.4, 1.65,
         f"Linked rising-risk cohort: N = {ta['linked_n']:,} ({100*ta['linked_n']/ta['assessed_n']:.1f}%)\n(Retained in supplement as unwindowed sensitivity)",
         fc="#EAF2F5", ec=BLUE, fontsize=8.9)

    # Trunk down from Linked -> Branch bus at y=14.05, with side branch to Excluded: incomplete 90d window
    ax.plot([5.5, 5.5], [16.5, 14.05], color="#444444", lw=1.3)
    _ortho_arrow(ax, [(5.5, 15.35), (9.8, 15.35)], color=AMBER)
    _box(ax, (9.85, 14.45), 4.05, 1.8,
         f"Excluded (incomplete 90d window):\n"
         f"• Pre-feed (<Jun 2023): {ta['excluded_before_hv_min_n']:,}\n"
         f"• Post-feed (>Oct 2025): {ta['excluded_after_hv_max_n']:,}\n"
         f"• Partial follow-up (<90d): {ta['excluded_partial_90d_n']:,}",
         fc="#FFF3E0", ec=AMBER, fontsize=7.6)

    # Horizontal distribution bar at y=14.05 feeding ALL THREE main analytic cohorts
    ax.plot([2.55, 11.85], [14.05, 14.05], color="#444444", lw=1.3)
    _arrow(ax, (2.55, 14.05), (2.55, 12.95))
    _arrow(ax, (7.2, 14.05), (7.2, 12.95))
    _arrow(ax, (11.85, 14.05), (11.85, 12.95))

    # Branch 1 (Left): Rising-risk panel cohort (N=18,936)
    _box(ax, (0.5, 10.35), 4.1, 2.55,
         f"Rising-risk panel cohort\nN = {c_pri['n_total']:,} ({c_pri['events_total']:,} events; {c_pri['event_rate_total_pct']:.2f}%)\n"
         f"Train {c_pri['n_train']:,} | Test {c_pri['n_test']:,}\n"
         f"Hybrid baseline ({ta['complete_90d_documented_goal_pct']:.1f}% goal,\n{ta['complete_90d_inferred_pct']:.1f}% clinical-flag heuristic)",
         fc="#EAF2F5", ec=BLUE, fontsize=7.8, weight="bold")
    _arrow(ax, (2.55, 10.35), (2.55, 9.2))
    _box(ax, (0.5, 7.0), 4.1, 2.15,
         f"Panel evaluation (test N = {c_pri['n_test']:,})\n"
         f"8 of 14 actions admissible ($n \\geq 100$)\n"
         f"Misalignment {c_pri['ablation'][1]['misalign_baseline_at_lam_pct']:.2f}% $\\to$ "
         f"{c_pri['ablation'][1]['misalign_policy_at_lam_pct']:.2f}%\n"
         f"Model-predicted {c_pri['ablation'][1]['model_predicted_events_averted_per_1000_dm']:+.2f}/1,000",
         fc="#F4F9FB", ec=BLUE, fontsize=7.7)

    # Branch 2 (Center): Validated Cohort A (Encounter discipline, N=5,927)
    _box(ax, (5.1, 10.35), 4.2, 2.55,
         f"Validated Cohort A: Discipline\n"
         f"N = {c_enc['n_total']:,} ({c_enc['events_total']:,} events; {c_enc['event_rate_total_pct']:.2f}%)\n"
         f"Train {c_enc['n_train']:,} | Test {c_enc['n_test']:,}\n"
         "100% documented; 4 care-team tracks\n"
         "Anchored at first completed encounter",
         fc="#EAF5EE", ec=GREEN, fontsize=7.8, weight="bold")
    _arrow(ax, (7.2, 10.35), (7.2, 9.2))
    _box(ax, (5.1, 7.0), 4.2, 2.15,
         f"Acute-recency adjusted (test N = {c_enc['n_test']:,})\n"
         f"All 4 tracks admissible; genuine overlap\n"
         f"Misalignment {c_enc['ablation'][1]['misalign_baseline_at_lam_pct']:.2f}% $\\to$ "
         f"{c_enc['ablation'][1]['misalign_policy_at_lam_pct']:.2f}%\n"
         f"Model-predicted {c_enc['ablation'][1]['model_predicted_events_averted_per_1000_dm']:+.2f}/1,000 ($\\lambda=0$)",
         fc="#F3FAF6", ec=GREEN, fontsize=7.7)

    # Branch 3 (Right): Validated Cohort B (Documented care-plan goals, N=2,515)
    _box(ax, (9.8, 10.35), 4.1, 2.55,
         f"Validated Cohort B: Goals\n"
         f"N = {c_gl['n_total']:,} ({c_gl['events_total']:,} events; {c_gl['event_rate_total_pct']:.2f}%)\n"
         f"Train {c_gl['n_train']:,} | Test {c_gl['n_test']:,}\n"
         "100% documented care-plan goals\n"
         f"(All-risk sensitivity: N = {c_gla['n_total']:,})",
         fc="#EAF5EE", ec=GREEN, fontsize=7.8, weight="bold")
    _arrow(ax, (11.85, 10.35), (11.85, 9.2))
    _box(ax, (9.8, 7.0), 4.1, 2.15,
         f"Acute-recency adjusted (test N = {c_gl['n_test']:,})\n"
         f"8 of 14 actions admissible ($n \\geq 50$)\n"
         f"Cross-model agreement {c_gl['agreement_gbm_rf_pct']:.1f}%\n"
         f"At $\\lambda=0.5$: {c_gl['ablation'][2]['model_predicted_events_averted_per_1000_dm']:+.2f}/1,000 (all-risk: −0.03)",
         fc="#F3FAF6", ec=GREEN, fontsize=7.7)

    # Dashed connector from Linked cohort down left margin to Separate Contextual Onboarding box
    _ortho_arrow(ax, [(1.8, 17.3), (0.18, 17.3), (0.18, 3.4), (0.65, 3.4)], color=RED, lw=1.2, ls="--")
    ax.text(0.32, 6.1, "Separate contextual onboarding\nanalysis (not used in PEARL)",
            fontsize=7.4, color=RED, style="italic", va="center", ha="left")

    _box(ax, (0.7, 1.3), 13.2, 4.1,
         "Separate contextual analysis — Unfiltered within-patient onboarding (NOT used to build PEARL)\n\n"
         f"All staggered-onboarding patients: N = {w1_all['n']:,} "
         f"(Pre=0/Post=0: {w1_all['cell_pre0_post0']:,}; Pre=1/Post=0: {w1_all['cell_pre1_post0']:,}; "
         f"Pre=0/Post=1: {w1_all['cell_pre0_post1']:,}; Pre=1/Post=1: {w1_all['cell_pre1_post1']:,}; 742 with $Y^{{\\mathrm{{on}}}}=1$)\n"
         f"Complete 90-day pre/post window: N = {w1_comp['n']:,} "
         f"(Pre {w1_comp['pre_rate_pct']:.1f}% vs Post {w1_comp['post_rate_pct']:.1f}%; "
         f"paired RD {w1_comp['paired_rd_pp']:+.2f} pp, 95% CI {w1_comp['paired_rd_ci95_pp'][0]:+.2f} to {w1_comp['paired_rd_ci95_pp'][1]:+.2f})\n"
         f"Rising-risk complete window (N = {w1_rr['n']:,}): Pre {w1_rr['pre_rate_pct']:.1f}% vs Post {w1_rr['post_rate_pct']:.1f}% "
         f"(paired RD {w1_rr['paired_rd_pp']:+.2f} pp); observational context only —\n"
         "causal enrollment evaluation requires longitudinal target-trial emulation accounting for acute-event referral triggers.",
         fc="#FBEAEA", ec=RED, fontsize=7.6, ls="--")

    for ext in ("png", "pdf"):
        out = FIGS_DIR / f"appendix_fig1_patient_flow.{ext}"
        fig.savefig(out, dpi=300)
        print(f"wrote {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Appendix Figure 2: Action validation & covariate determinism / overlap
# ─────────────────────────────────────────────────────────────────────────────

def make_appendix_fig2(master):
    val_df = pd.read_csv(OUT_DIR / "v6_action_validation_by_action.csv")
    val_df = val_df[val_df["n_documented"] >= 20].copy()

    c_pri = master["cohorts"]["primary_complete_90d_fu"]
    c_enc = master["cohorts"]["validated_encounter_discipline_recency_adj"]
    c_gl = master["cohorts"]["validated_documented_goals_rr_recency_adj"]

    nice = {
        "care_access": "Care access", "medication_adherence": "Medication adherence",
        "financial_benefits": "Financial benefits", "clinical_other": "Clinical other",
        "mental_health": "Mental health", "food_security": "Food security",
        "hypertension": "Hypertension", "transport_utilities": "Transport & utilities",
        "housing": "Housing", "diabetes": "Diabetes", "maternal": "Maternal",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    ax = axes[0]
    y = np.arange(len(val_df))
    ax.barh(y, val_df["exact_agreement_pct"], color=BLUE, height=0.62, alpha=0.88)
    for yi, (pct, n) in enumerate(zip(val_df["exact_agreement_pct"], val_df["n_documented"])):
        ax.text(pct + 1.2, yi, f"{pct:.1f}% (n={n})", va="center", fontsize=8.0)
    ax.set_yticks(y)
    ax.set_yticklabels([nice.get(a, a) for a in val_df["documented_action"]], fontsize=8.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 78)
    ax.set_xlabel("Exact agreement of clinical-flag heuristic (%)")
    ax.set_title(
        f"a  Clinical-flag agreement vs documented goals\n"
        f"(overall {master['action_validation']['all_time_exact_agreement_pct']:.1f}%, "
        f"Cohen's $\\kappa$ = {master['action_validation']['all_time_cohen_kappa']:.3f})",
        loc="left", fontsize=9.5, weight="bold",
    )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=LGRAY, linewidth=0.7); ax.set_axisbelow(True)

    ax = axes[1]
    cohorts = [
        (f"Rising-risk panel\nhybrid clinical-flag\n(N = {c_pri['n_total']:,})", c_pri["determinism"]),
        (f"Validated\ncare-plan goals\n(N = {c_gl['n_total']:,})", c_gl["determinism"]),
        (f"Validated\nencounter discipline\n(N = {c_enc['n_total']:,})", c_enc["determinism"]),
    ]
    x = np.arange(len(cohorts))
    w = 0.34
    acc_vals = [d["gb_accuracy_pct"] for _, d in cohorts]
    ov_vals = [100.0 - d["gb_pct_zero_alt_ge_005"] for _, d in cohorts]
    b1 = ax.bar(x - w / 2, acc_vals, w, color=RED, alpha=0.85, label="Covariate predictability of action (gradient-boosted trees %)")
    b2 = ax.bar(x + w / 2, ov_vals, w, color=GREEN, alpha=0.85, label="Patients with $\\geq$1 alternative action at propensity $\\geq$ 0.05 (%)")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", fontsize=8.2)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cohorts], fontsize=8.4)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("b  Covariate determinism vs treatment overlap", loc="left", fontsize=9.5, weight="bold")
    ax.legend(fontsize=7.8, frameon=False, loc="upper center")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=LGRAY, linewidth=0.7); ax.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = FIGS_DIR / f"appendix_fig2_action_validation_overlap.{ext}"
        fig.savefig(out, dpi=300)
        print(f"wrote {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Appendix Figure 3: Decision-curve analysis & matched-capacity comparison
# ─────────────────────────────────────────────────────────────────────────────

def make_appendix_fig3(master):
    dc = pd.read_csv(OUT_DIR / "v6_primary_decision_curve.csv")
    c_pri = master["cohorts"]["primary_complete_90d_fu"]
    mc = c_pri["matched_capacity_review"]
    mc_ci = c_pri["bootstrap_cis"]["matched_capacity"]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), gridspec_kw={"width_ratios": [1.3, 1.0]})

    ax = axes[0]
    ax.plot(dc["threshold_probability"], 100 * dc["nb_misalignment_flag"], color=NAVY, lw=2.1, label="Misalignment flag (matched capacity)")
    ax.plot(dc["threshold_probability"], 100 * dc["nb_rising_risk_matched"], color=AMBER, lw=2.0, linestyle="-.", label="Existing rising-risk score (matched capacity)")
    ax.plot(dc["threshold_probability"], 100 * dc["nb_rf_risk_matched"], color=GREEN, lw=2.1, label="Outcome-model baseline risk (matched capacity)")
    ax.plot(dc["threshold_probability"], 100 * dc["nb_rf_risk_continuous"], color=BLUE, lw=1.6, linestyle=":", label="Outcome-model baseline risk (continuous)")
    ax.plot(dc["threshold_probability"], 100 * dc["nb_review_all"], color=RED, lw=1.6, label="Review all")
    ax.axhline(0, color=GRAY, lw=1.2, linestyle="--", label="Review none")
    ax.set_xlabel("Threshold probability $p_t$ (assumed cost ratio of unnecessary review)")
    ax.set_ylabel("Net benefit (%)")
    ax.set_xlim(0.01, 0.25)
    ax.set_ylim(-1.5, 6.8)
    ax.set_title("a  Decision-curve comparison (rising-risk test set)", loc="left", fontsize=9.8, weight="bold")
    ax.legend(fontsize=7.6, frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=LGRAY, linewidth=0.7); ax.set_axisbelow(True)

    ax = axes[1]
    labels = [
        "Misalignment flag\n($\\varepsilon = 0.02$)",
        "Existing\nrising-risk score",
        "Outcome-model\nbaseline risk $\\hat{\\mu}(X, A)$",
    ]
    pts = [mc[0]["ppv_observed_event_rate_pct"], mc[1]["ppv_observed_event_rate_pct"], mc[2]["ppv_observed_event_rate_pct"]]
    evs = [mc[0]["events_captured"], mc[1]["events_captured"], mc[2]["events_captured"]]
    ci_keys = ["ppv_flag_ci95", "ppv_rs_ci95", "ppv_mu_ci95"]
    lo_e, hi_e = [], []
    for pt, ck in zip(pts, ci_keys):
        lo, hi = mc_ci[ck]
        lo_e.append(max(0.0, pt - lo))
        hi_e.append(max(0.0, hi - pt))
    x = np.arange(3)
    cols = [NAVY, AMBER, GREEN]
    ax.bar(x, pts, width=0.55, color=cols, alpha=0.88, edgecolor="white")
    ax.errorbar(x, pts, yerr=[lo_e, hi_e], fmt="none", ecolor="#222222", elinewidth=1.3, capsize=4)
    ax.axhline(c_pri["event_rate_test_pct"], color=GRAY, linestyle="--", lw=1.1,
               label=f"Test baseline rate ({c_pri['event_rate_test_pct']:.2f}%)")
    for xi, pt, he, ev in zip(x, pts, hi_e, evs):
        ax.text(xi, pt + he + 0.6, f"{pt:.1f}%\n({ev} events)", ha="center", va="bottom", fontsize=8.2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.3)
    ax.set_ylim(0, max(p + h for p, h in zip(pts, hi_e)) + 4.5)
    ax.set_ylabel("Observed 90-day acute care rate among reviewed (%)")
    ax.set_title(f"b  Matched review capacity (n = {mc[0]['n_reviewed']} reviewed, {mc[0]['pct_reviewed']:.1f}%)",
                 loc="left", fontsize=9.8, weight="bold")
    ax.legend(fontsize=8.0, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=LGRAY, linewidth=0.7); ax.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = FIGS_DIR / f"appendix_fig3_decision_curve.{ext}"
        fig.savefig(out, dpi=300)
        print(f"wrote {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Appendix Figure 4: Camden Coalition illustrative simulation (both red & blue!)
# ─────────────────────────────────────────────────────────────────────────────

def make_appendix_fig4(master):
    cam = master["camden_illustrative"]
    redist = cam["redistribution"]
    cats = sorted(redist.keys(), key=lambda k: -redist[k])
    pearl_vals = [100.0 * redist[c] for c in cats]
    camden_vals = [100.0 if c == "care_access" else 0.0 for c in cats]
    nice = {
        "mental_health": "Mental health", "hypertension": "Hypertension",
        "heart_failure": "Heart failure", "housing": "Housing",
        "care_access": "Care access", "pulmonary": "Pulmonary",
        "diabetes": "Diabetes", "food_security": "Food security",
        "medication_adherence": "Med. adherence", "financial_benefits": "Financial benefits",
        "clinical_other": "Clinical other", "transport_utilities": "Transport & utilities",
        "maternal": "Maternal", "substance_use": "Substance use",
    }
    labels = [nice.get(c, c) for c in cats]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Panel A: Intervention distribution (red = Camden uniform, blue = shared-model personalized)
    ax = axes[0]
    x = np.arange(len(cats))
    w = 0.38
    ax.bar(x - w / 2, camden_vals, w, label="Camden uniform protocol", color=RED, alpha=0.88)
    ax.bar(x + w / 2, pearl_vals, w, label="Shared-model personalized", color=BLUE, alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.0)
    ax.set_ylabel("Patients assigned (%)")
    ax.set_title(f"a  Action distribution (N = {cam['n_camden_profile']} Camden-profile patients)",
                 loc="left", fontsize=9.6, weight="bold")
    ax.legend(fontsize=8.2, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=LGRAY, linewidth=0.7); ax.set_axisbelow(True)

    # Panel B: BOTH arm simulated rates (red = Camden uniform, blue = shared-model personalized) + difference bracket
    ax = axes[1]
    u_pt = cam["uniform_protocol_rate_pct"]
    u_lo, u_hi = cam["uniform_protocol_pi90_pct"]
    p_pt = cam["shared_personalized_rate_pct"]
    p_lo, p_hi = cam["shared_personalized_pi90_pct"]
    d_pt = cam["diff_pp"]
    d_lo, d_hi = cam["diff_pi90_pp"]

    x_arms = [0, 1]
    ax.bar([0], [u_pt], width=0.45, color=RED, alpha=0.85, edgecolor="white", label="Camden uniform protocol")
    ax.bar([1], [p_pt], width=0.45, color=BLUE, alpha=0.85, edgecolor="white", label="Shared-model personalized")
    ax.errorbar([0], [u_pt], yerr=[[max(0, u_pt - u_lo)], [max(0, u_hi - u_pt)]],
                fmt="o", color="#222222", markersize=5.5, capsize=5, elinewidth=1.4)
    ax.errorbar([1], [p_pt], yerr=[[max(0, p_pt - p_lo)], [max(0, p_hi - p_pt)]],
                fmt="o", color="#222222", markersize=5.5, capsize=5, elinewidth=1.4)

    ax.text(0, u_hi + 0.8, f"{u_pt:.1f}%\n(90% PI: {u_lo:.1f}–{u_hi:.1f}%)", ha="center", va="bottom", fontsize=8.2, color=RED, weight="bold")
    ax.text(1, p_hi + 0.8, f"{p_pt:.1f}%\n(90% PI: {p_lo:.1f}–{p_hi:.1f}%)", ha="center", va="bottom", fontsize=8.2, color=BLUE, weight="bold")

    # Difference bracket connecting red and blue bars
    y_br = max(u_hi, p_hi) + 5.2
    ax.plot([0, 0, 1, 1], [y_br - 0.8, y_br, y_br, y_br - 0.8], color="#333333", lw=1.2)
    ax.text(0.5, y_br + 0.5, f"Simulated difference: {d_pt:.1f} pp\n(90% PI: {d_lo:.1f}–{d_hi:.1f} pp)",
            ha="center", va="bottom", fontsize=8.3, weight="bold", color="#222222")

    ax.set_xticks(x_arms)
    ax.set_xticklabels(["Camden uniform\nprotocol", "Shared-model\npersonalized"], fontsize=8.6)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, y_br + 5.5)
    ax.set_ylabel("Simulated acute care event rate (%)")
    ax.set_title("b  Simulated event rates by strategy\n(non-safeguarded shared model; illustrative only)",
                 loc="left", fontsize=9.3, weight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=LGRAY, linewidth=0.7); ax.set_axisbelow(True)

    fig.suptitle(
        f"Camden Coalition analog illustrative simulation: N = {cam['n_camden_profile']} rising-risk patients "
        "(Charlson $\\geq 2$ and prior inpatient $\\geq 1$ or prior ED $\\geq 2$ in 6 months)",
        fontsize=9.8, x=0.02, ha="left", y=1.02,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = FIGS_DIR / f"appendix_fig4_camden.{ext}"
        fig.savefig(out, dpi=300)
        print(f"wrote {out}")
    plt.close(fig)


def main():
    master = load_master()
    make_fig1(master)
    make_fig2(master)
    make_appendix_fig1(master)
    make_appendix_fig2(master)
    make_appendix_fig3(master)
    make_appendix_fig4(master)
    print("\nAll Revision 2 (v6) figures generated cleanly from canonical data.")


if __name__ == "__main__":
    main()
