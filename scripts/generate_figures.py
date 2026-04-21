"""
generate_figures.py
Produces publication-quality figures for PEARL manuscript from results CSVs.
Run: python scripts/generate_figures.py --results outputs/results/main_results_table.csv --outdir outputs/figures/
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── style ─────────────────────────────────────────────────────────────────────
GRAY  = "#4d4d4d"
BLUE  = "#1a6fa8"
RED   = "#c0392b"
GREEN = "#27ae60"
LIGHT = "#e8f4fd"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ── Figure 2: DR-OPE forest plot ───────────────────────────────────────────────
def fig_drope_forest(results_csv: str, outpath: str) -> None:
    """Forest plot of DR-OPE policy values for all 13 comparators.

    Lower DR-OPE value = fewer acute care events (beneficial direction).
    Behavioral policy is the reference line.
    """
    df = pd.read_csv(results_csv)

    # Parse CI strings "[lo, hi]"
    df[["ci_lo", "ci_hi"]] = (
        df["Policy_Value_95CI"]
        .str.strip("[]")
        .str.split(",", expand=True)
        .astype(float)
    )
    df = df.sort_values("DR-OPE_Rank")

    behavioral_val = df.loc[df["Model"] == "Behavioral Policy", "Policy_Value"].values[0]

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for i, row in df.iterrows():
        rank = int(row["DR-OPE_Rank"])
        y = len(df) - rank
        val = row["Policy_Value"]
        lo  = row["ci_lo"]
        hi  = row["ci_hi"]

        # color coding
        if "PEARL (MoE Router)" in row["Model"]:
            color = BLUE
            lw = 1.8
        elif "Behavioral Policy" in row["Model"]:
            color = GRAY
            lw = 1.4
        elif "CQL" in row["Model"] or "LACE" in row["Model"] or \
             "HOSPITAL" in row["Model"] or "XGBoost" in row["Model"]:
            color = RED
            lw = 1.2
        else:
            color = GRAY
            lw = 1.2

        ax.plot([lo, hi], [y, y], color=color, lw=lw, zorder=2)
        ax.plot(val, y, "o", color=color, ms=5 if lw > 1.5 else 4, zorder=3)

    # Reference line at behavioral policy
    ax.axvline(behavioral_val, color=GRAY, ls="--", lw=0.9, alpha=0.7, label="Behavioral policy")

    # Y-axis labels
    labels = [row["Model"] for _, row in df.sort_values("DR-OPE_Rank").iterrows()]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1], fontsize=8)

    ax.set_xlabel(
        "DR-OPE policy value (expected 90-day acute care event rate; lower = better)",
        fontsize=8.5
    )
    ax.set_title(
        "Off-policy evaluation: DR-OPE policy value comparison\n"
        "N = 6,995 held-out test patients; 2,000-resample bootstrap 95% CI",
        fontsize=9, loc="left"
    )

    legend_handles = [
        mpatches.Patch(color=BLUE,  label="PEARL (MoE Router)"),
        mpatches.Patch(color=GRAY,  label="Other policies"),
        mpatches.Patch(color=RED,   label="Traditional risk scores / CQL"),
        plt.Line2D([0], [0], color=GRAY, ls="--", label="Behavioral policy (reference)"),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, loc="lower right", frameon=True)
    ax.invert_yaxis()

    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved: {outpath}")


# ── Figure 3: IMI equity gradient bar chart ───────────────────────────────────
def fig_imi_equity(outpath: str) -> None:
    """IMI by ADI quintile with 95% CI error bars (from PEARL pipeline output)."""
    quintiles = ["Q1\n(least deprived)", "Q2", "Q3", "Q4", "Q5\n(most deprived)"]
    imi      = [0.076, 0.089, 0.119, 0.155, 0.150]
    ci_lo    = [0.061, 0.073, 0.101, 0.135, 0.130]
    ci_hi    = [0.093, 0.106, 0.138, 0.176, 0.171]
    ns       = [532, 623, 834, 1050, 1048]

    err_lo = [imi[i] - ci_lo[i] for i in range(5)]
    err_hi = [ci_hi[i] - imi[i] for i in range(5)]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bars = ax.bar(quintiles, imi, color=BLUE, alpha=0.82, width=0.6, zorder=2)
    ax.errorbar(quintiles, imi, yerr=[err_lo, err_hi],
                fmt="none", color=GRAY, capsize=4, lw=1.2, zorder=3)

    # Annotate N
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"n={n}", ha="center", va="bottom", fontsize=7, color=GRAY)

    ax.set_ylabel("Intervention misalignment index (IMI)", fontsize=8.5)
    ax.set_xlabel("Area deprivation index (ADI) quintile\n(higher quintile = greater deprivation)", fontsize=8.5)
    ax.set_title(
        "Intervention misalignment by area deprivation index quintile\n"
        "IMI = fraction of rising-risk patients with a suboptimal care assignment; 95% bootstrap CI",
        fontsize=9, loc="left"
    )
    ax.set_ylim(0, 0.22)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", lw=0.5, alpha=0.4)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved: {outpath}")


# ── Figure 4: Camden reanalysis ────────────────────────────────────────────────
def fig_camden(outpath: str) -> None:
    """Bar chart comparing PEARL and Camden protocol intervention distributions
    for 242 Camden-profile rising-risk patients, with simulated readmission comparison.
    """
    labels = ["Social needs", "Medication adherence", "Behavioral health", "Clinical complexity"]
    camden  = [0.0,  0.0,  0.0,   1.00]   # uniform intensive protocol → all clinical_complexity
    pearl   = [0.099, 0.169, 0.438, 0.293]

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Panel A: intervention distribution
    ax = axes[0]
    ax.bar(x - width/2, camden, width, label="Camden uniform protocol", color=RED,   alpha=0.8)
    ax.bar(x + width/2, pearl,  width, label="PEARL personalized",       color=BLUE, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=15, ha="right")
    ax.set_ylabel("Proportion of patients assigned", fontsize=8.5)
    ax.set_title("A. Intervention type distribution\n(N = 242 Camden-profile patients)", fontsize=8.5, loc="left")
    ax.legend(fontsize=7.5, frameon=True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1.15)

    # Panel B: simulated readmission rate
    ax2 = axes[1]
    protocols = ["Camden\nuniform", "PEARL\npersonalized"]
    readmission = [0.437, 0.271]  # Camden-profile patients: 43.7% vs 27.1% (16.6 pp difference)
    ci_lo = [0.0, 0.0]
    ci_hi = [0.0, 0.0]
    # 90% PI: Camden 43.7% [39.0-48.5], PEARL 27.1% [22.0-31.9]
    camden_lo, camden_hi = 0.390, 0.485
    pearl_lo, pearl_hi   = 0.220, 0.319
    err = [[readmission[0] - camden_lo, readmission[1] - pearl_lo],
           [camden_hi - readmission[0], pearl_hi - readmission[1]]]

    bars2 = ax2.bar(protocols, readmission,
                    color=[RED, BLUE], alpha=0.82, width=0.5)
    ax2.errorbar(protocols, readmission, yerr=err,
                 fmt="none", color=GRAY, capsize=5, lw=1.3)
    ax2.set_ylabel("Predicted 30-day readmission rate", fontsize=8.5)
    ax2.set_title(
        "B. Simulated 30-day readmission rate\n(90% prediction interval; 1,000 bootstrap iterations)",
        fontsize=8.5, loc="left"
    )
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax2.set_ylim(0, 0.60)
    ax2.annotate("−16.6 pp\n(90% PI: 12.4–21.7)", xy=(1, 0.271), xytext=(0.6, 0.45),
                 arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8),
                 fontsize=8, color=GRAY)

    fig.suptitle(
        "Camden Coalition analog reanalysis: N = 242 rising-risk patients matching Camden enrollment profile\n"
        "(Charlson comorbidity index ≥ 2 and prior hospitalization ≥ 1 or prior ED visits ≥ 2 in 6 months)",
        fontsize=8.5
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved: {outpath}")


# ── Figure 5: Sensitivity analysis summary ─────────────────────────────────────
def fig_sensitivity(outpath: str, sens_csv: str = "outputs/results/sensitivity_results.csv") -> None:
    """Dot plot of DR-OPE and IMI across pre-specified sensitivity analyses.
    Primary result shown as vertical reference lines.
    Loads actual computed results from the sensitivity CSV produced by run_pipeline.py.
    Falls back to a placeholder figure with a warning if the CSV is not found.
    """
    if not os.path.exists(sens_csv):
        print(f"WARNING: {sens_csv} not found. Run 'python scripts/run_pipeline.py --waymark' "
              f"to generate sensitivity results, then re-run generate_figures.py.")
        # Produce a placeholder figure with instructions rather than fabricated data.
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.text(0.5, 0.5,
                "Sensitivity analysis figure\nnot yet available.\n\n"
                "Run: python scripts/run_pipeline.py --waymark\n"
                "Then: python scripts/generate_figures.py",
                ha="center", va="center", fontsize=11, color=GRAY,
                transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(outpath)
        plt.close(fig)
        print(f"Placeholder saved: {outpath}")
        return

    df = pd.read_csv(sens_csv)

    # Build label from parameter + value; mark primary values
    df["label"] = df["parameter"].str.replace("_", " ").str.title() + " = " + df["value"].astype(str)
    df["label"] = df.apply(
        lambda r: r["label"] + " ★" if r["is_primary"] else r["label"], axis=1
    )

    n = len(df)
    drope_vals = df["drope_estimate"].values
    imi_vals = df["imi_estimate"].values
    labels = df["label"].tolist()

    drope_primary = df.loc[df["is_primary"].astype(bool), "drope_estimate"].mean() if df["is_primary"].any() else float(drope_vals.mean())
    imi_primary = df.loc[df["is_primary"].astype(bool), "imi_estimate"].mean() if df["is_primary"].any() else float(imi_vals.mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, max(5.5, n * 0.35)))

    for ax, vals, primary, xlabel, title in [
        (axes[0], drope_vals, drope_primary,
         "DR-OPE policy value (PEARL, lower = better)",
         "A. DR-OPE across sensitivity analyses"),
        (axes[1], imi_vals, imi_primary,
         "IMI estimate (lower = better)",
         "B. IMI across sensitivity analyses"),
    ]:
        colors = [BLUE if not df["direction_change"].iloc[i] else RED for i in range(n)]
        ax.scatter(vals, range(n), color=colors, s=28, zorder=3)
        ax.axvline(primary, color=RED, ls="--", lw=1.0, label="Primary result")
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.legend(fontsize=7.5)
        ax.grid(axis="x", lw=0.4, alpha=0.4)

    n_direction = int(df["direction_change"].sum()) if "direction_change" in df.columns else 0
    n_retrain = int(df["requires_retrain"].sum()) if "requires_retrain" in df.columns else 0
    fig.suptitle(
        f"Pre-specified sensitivity analyses (N = {n}): "
        f"{n - n_direction} of {n} analyses preserve primary result direction "
        f"({n_retrain} require full pipeline re-run with varied training parameters)",
        fontsize=9
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved: {outpath}")


# ── Figure 1: PEARL Pipeline Overview (2-panel schematic) ─────────────────────
def fig_pipeline_overview(outpath: str) -> None:
    """2-panel PEARL pipeline schematic for Lancet Digital Health.

    Panel A: WPAD natural experiment — staggered ACO onboarding creates
             exogenous ON/OFF windows within the same patient.
    Panel B: PEARL training pipeline — tabular IPTW-DPO from preference pairs
             to MoE Router intervention policy (no LLM backbone).

    Coordinate units are data units after set_xlim/set_ylim per panel.
    """
    BOX_BLUE  = "#dbeeff"
    BOX_GREEN = "#d5f5e3"
    BOX_RED   = "#fde8e8"
    BOX_GRAY  = "#f4f4f4"

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04, wspace=0.12)

    def _box(ax, xc, yc, w, h, text, fc=BOX_GRAY, ec="#888888",
             fs=7.5, fw="normal", tc="#222222", lw=0.9):
        p = mpatches.FancyBboxPatch(
            (xc - w / 2, yc - h / 2), w, h,
            boxstyle="round,pad=0.04", facecolor=fc, edgecolor=ec,
            linewidth=lw, transform=ax.transData, zorder=2, clip_on=False,
        )
        ax.add_patch(p)
        ax.text(xc, yc, text, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=tc,
                multialignment="center", zorder=3, clip_on=False)

    def _arrow(ax, x1, y1, x2, y2, col="#4d4d4d", lw=1.1):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->, head_width=0.18, head_length=0.14",
                                   color=col, lw=lw),
                    zorder=4)

    # ── Panel A: WPAD Natural Experiment ─────────────────────────────────────
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 8)
    ax_a.set_axis_off()
    ax_a.set_title("A   WPAD Natural Experiment", fontsize=9.5,
                   loc="left", fontweight="bold", pad=4)

    # Timeline axis
    ax_a.annotate("", xy=(9.8, 3.8), xytext=(0.2, 3.8),
                  arrowprops=dict(arrowstyle="->, head_width=0.20, head_length=0.18",
                                 color="#aaaaaa", lw=1.4))
    ax_a.text(9.85, 3.8, "time", va="center", fontsize=7, color="#888888")

    # OFF window (pre-enrollment)
    off = mpatches.FancyBboxPatch((0.3, 4.35), 3.7, 3.2,
                                   boxstyle="round,pad=0.06",
                                   facecolor=BOX_RED, edgecolor=RED, lw=1.3)
    ax_a.add_patch(off)
    ax_a.text(2.15, 7.1, "OFF window", ha="center", fontsize=8.5,
              fontweight="bold", color=RED)
    ax_a.text(2.15, 6.45, "Pre-enrollment\n(no care management)", ha="center",
              fontsize=7.2, color="#333333")
    ax_a.text(2.15, 5.6, "Y\u1d52\u1da0\u1da0 = acute care\nevent \u2717",
              ha="center", fontsize=7.5, color=RED, fontweight="bold")
    ax_a.text(2.15, 4.65, "rejected completion  y\u2097",
              ha="center", fontsize=6.8, color=RED, style="italic")

    # ACO enrollment marker
    ax_a.plot([4.3, 4.3], [3.55, 7.75], color=BLUE, lw=1.8, ls="--", zorder=3)
    ax_a.text(4.3, 7.88, "ACO enrollment  \u03c4\u1d05",
              ha="center", fontsize=7.5, color=BLUE, fontweight="bold")

    # ON window (post-enrollment)
    on = mpatches.FancyBboxPatch((4.5, 4.35), 5.15, 3.2,
                                  boxstyle="round,pad=0.06",
                                  facecolor=BOX_GREEN, edgecolor=GREEN, lw=1.3)
    ax_a.add_patch(on)
    ax_a.text(7.1, 7.1, "ON window", ha="center", fontsize=8.5,
              fontweight="bold", color=GREEN)
    ax_a.text(7.1, 6.45, "Active care management\n(CHW contacts logged)", ha="center",
              fontsize=7.2, color="#333333")
    ax_a.text(7.1, 5.6, "Y\u1d52\u1d4f = no acute\nevent \u2713",
              ha="center", fontsize=7.5, color=GREEN, fontweight="bold")
    ax_a.text(7.1, 4.65, "preferred completion  y\u1d64",
              ha="center", fontsize=6.8, color=GREEN, style="italic")

    # Preference pair construction box (below timeline)
    pair_box = mpatches.FancyBboxPatch((0.3, 0.15), 9.4, 3.2,
                                        boxstyle="round,pad=0.06",
                                        facecolor=BOX_BLUE, edgecolor=BLUE, lw=1.0)
    ax_a.add_patch(pair_box)
    ax_a.text(5.0, 3.05,
              "Within-patient design: time-invariant confounders cancel by construction",
              ha="center", fontsize=7.8, fontweight="bold", color=BLUE)
    ax_a.text(5.0, 2.45,
              "Preference triple: (x\u1d62, y\u1d64, y\u2097, w\u1d62)  "
              "\u2014  x\u1d62 = 12-month EHR + SDOH context,  w\u1d62 = IPTW weight",
              ha="center", fontsize=7, color="#333333")
    ax_a.text(5.0, 1.82,
              "Staggered ACO onboarding: 622 within-patient pairs  (primary identification source)",
              ha="center", fontsize=7, color="#333333")
    ax_a.text(5.0, 1.22,
              "Medicaid eligibility churn: 221 pairs  (intent-to-treat secondary analysis)",
              ha="center", fontsize=7, color="#333333")
    ax_a.text(5.0, 0.58,
              "Exogeneity: enrollment timing driven by ACO logistics, not patient health status  "
              "(T1\u2013T5 balance tests)",
              ha="center", fontsize=6.6, color="#555555", style="italic")

    # ── Panel B: PEARL Tabular DPO Pipeline ──────────────────────────────────
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 8)
    ax_b.set_axis_off()
    ax_b.set_title("B   PEARL Training and Deployment Pipeline", fontsize=9.5,
                   loc="left", fontweight="bold", pad=4)

    # Step 1: input pairs
    _box(ax_b, 5, 7.4, 9.2, 0.88,
         "WPAD preference pairs  (x\u1d62, y\u1d64, y\u2097, w\u1d62)"
         "  |  N\u200a=\u200a622 within-patient + 30,021 IPTW cross-patient",
         fc=BOX_BLUE, ec=BLUE, fs=7.5)
    _arrow(ax_b, 5, 6.96, 5, 6.40)

    # Step 2: propensity model
    _box(ax_b, 5, 6.04, 9.2, 0.88,
         "Propensity model: L1-regularized logistic regression\n"
         "P(A\u1d62\u200a=\u200aa | x\u1d62)  \u2192  IPTW weights  w\u1d62  (clipped 0.1\u201310)",
         fc=BOX_GRAY, ec="#888888", fs=7.5)
    _arrow(ax_b, 5, 5.60, 5, 5.04)

    # Step 3: DPO training
    _box(ax_b, 5, 4.68, 9.2, 0.88,
         "IPTW-weighted DPO loss  (tabular preference model)\n"
         "Demographic-stratified group fairness: equal batch weights per group",
         fc="#fff3cd", ec="#c9a800", fs=7.5)
    _arrow(ax_b, 5, 4.24, 5, 3.68)

    # Step 4: MoE Router
    moe_box = mpatches.FancyBboxPatch((0.4, 3.10), 9.2, 1.00,
                                       boxstyle="round,pad=0.06",
                                       facecolor=BLUE, edgecolor=BLUE, lw=1.5)
    ax_b.add_patch(moe_box)
    ax_b.text(5.0, 3.60, "PEARL — Mixture-of-Experts Router  (\u03c0\u03b8*)",
              ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")
    _arrow(ax_b, 5, 3.10, 5, 2.60)

    # Step 5: four output boxes
    interventions = [
        ("G0511\nGeneral CM", "#1a6fa8"),
        ("G0512\nComplex CM", "#27ae60"),
        ("TCM\n(transitional)", "#8e44ad"),
        ("Community\nResource Nav.", "#e67e22"),
    ]
    xs_out = [1.3, 3.85, 6.4, 8.95]
    bw_out, bh_out = 2.3, 1.05
    for (lbl, col), xc in zip(interventions, xs_out):
        p = mpatches.FancyBboxPatch((xc - bw_out / 2, 1.35), bw_out, bh_out,
                                     boxstyle="round,pad=0.05",
                                     facecolor="#f8f8f8", edgecolor=col, lw=1.4)
        ax_b.add_patch(p)
        ax_b.text(xc, 1.35 + bh_out / 2, lbl, ha="center", va="center",
                  fontsize=7.2, fontweight="bold", color=col, multialignment="center")
        # arrow from router to each box
        ax_b.annotate("", xy=(xc, 1.35 + bh_out), xytext=(5.0, 3.10),
                      arrowprops=dict(arrowstyle="->, head_width=0.14, head_length=0.12",
                                     color="#888888", lw=0.9,
                                     connectionstyle="arc3,rad=0.0"),
                      zorder=1)

    # DR-OPE result annotation at bottom
    ax_b.text(5.0, 0.72,
              "DR-OPE evaluation: PEARL (MoE Router) = 0.035 [0.029\u20130.042]  "
              "vs. behavioral = 0.054 [0.027\u20130.084]  \u2014  35.7% improvement",
              ha="center", fontsize=7, color=BLUE, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.30", facecolor=BOX_BLUE,
                        edgecolor=BLUE, lw=0.7))
    ax_b.text(5.0, 0.18,
              "IMI: 11.6% [10.8\u201312.3%] under behavioral routing  \u2192  "
              "2.0% [1.5\u20132.6%] under PEARL  (\u221282.3% relative reduction)",
              ha="center", fontsize=6.8, color="#333333")

    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {outpath}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate PEARL manuscript figures")
    parser.add_argument("--results", default="outputs/results/main_results_table.csv")
    parser.add_argument("--outdir", default="outputs/figures/")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    fig_pipeline_overview(        os.path.join(args.outdir, "fig1_pipeline.pdf"))
    fig_drope_forest(args.results, os.path.join(args.outdir, "fig2_drope_forest.pdf"))
    fig_imi_equity(               os.path.join(args.outdir, "fig3_imi_equity.pdf"))
    fig_camden(                   os.path.join(args.outdir, "fig4_camden.pdf"))
    results_dir = os.path.dirname(args.results)
    sens_csv = os.path.join(results_dir, "sensitivity_results.csv")
    fig_sensitivity(os.path.join(args.outdir, "fig5_sensitivity.pdf"), sens_csv=sens_csv)
    print("All figures written to", args.outdir)


if __name__ == "__main__":
    main()
