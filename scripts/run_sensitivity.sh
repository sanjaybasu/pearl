#!/usr/bin/env bash
# run_sensitivity.sh
# Executes all 20 pre-specified sensitivity analyses for the PEARL manuscript.
# Each analysis re-runs the full pipeline with one parameter varied from its primary value.
# Results are combined into outputs/results/sensitivity_combined.csv.
#
# Usage:
#   bash scripts/run_sensitivity.sh [--synthetic]
#   bash scripts/run_sensitivity.sh          # default: --waymark (real data)
#
# The primary analysis values are:
#   t_min=60, iptw_clip=10, beta=0.10, outcome_window=30,
#   trajectory_adjustment=True, wpad_direction=all, camden_threshold=0.02

set -euo pipefail

MODE="${1:---waymark}"
OUTDIR="outputs/results/sensitivity"
mkdir -p "$OUTDIR"

RESULTS=()

run_variant() {
    local label="$1"
    shift
    local extra_args=("$@")
    local outfile="$OUTDIR/${label}.csv"

    echo "=== Sensitivity: ${label} ==="
    python scripts/run_pipeline.py "$MODE" "${extra_args[@]}" \
        --skip_arena --sensitivity_label "$label" \
        --sens_outfile "$outfile" 2>&1 | tail -20 || true
    RESULTS+=("$outfile")
}

# ── t_min variations ──────────────────────────────────────────────────────────
run_variant "t_min_30"  --t_min 30
run_variant "t_min_90"  --t_min 90

# ── iptw_clip variations ──────────────────────────────────────────────────────
run_variant "iptw_clip_5"  --iptw_clip 5
run_variant "iptw_clip_20" --iptw_clip 20

# ── beta variations ───────────────────────────────────────────────────────────
run_variant "beta_005" --beta 0.05
run_variant "beta_020" --beta 0.20

# ── outcome_window variations ─────────────────────────────────────────────────
run_variant "outcome_14d" --outcome_window 14
run_variant "outcome_60d" --outcome_window 60

# ── trajectory_adjustment ─────────────────────────────────────────────────────
run_variant "no_trajectory_adj" --no_trajectory_adjustment

# ── wpad_direction variations ─────────────────────────────────────────────────
run_variant "churn_only"   --wpad_direction churn_only
run_variant "waitlist_only" --wpad_direction waitlist_only

# ── camden_threshold variations ───────────────────────────────────────────────
run_variant "camden_eps_001" --camden_threshold 0.01
run_variant "camden_eps_005" --camden_threshold 0.05

# ── Cross-parameter interactions (7 additional analyses) ──────────────────────
run_variant "t_min30_eps001"   --t_min 30  --camden_threshold 0.01
run_variant "t_min90_eps005"   --t_min 90  --camden_threshold 0.05
run_variant "beta005_clip5"    --beta 0.05 --iptw_clip 5
run_variant "beta020_clip20"   --beta 0.20 --iptw_clip 20
run_variant "beta005_clip20"   --beta 0.05 --iptw_clip 20
run_variant "beta020_clip5"    --beta 0.20 --iptw_clip 5
run_variant "t_min30_beta005"  --t_min 30  --beta 0.05

# ── Combine all results ───────────────────────────────────────────────────────
echo ""
echo "=== Combining sensitivity results ==="
python - <<'EOF'
import pandas as pd
import glob
import os

files = sorted(glob.glob("outputs/results/sensitivity/*.csv"))
if not files:
    print("No sensitivity CSV files found.")
    raise SystemExit(1)

dfs = []
for f in files:
    label = os.path.splitext(os.path.basename(f))[0]
    try:
        df = pd.read_csv(f)
        df["sensitivity_label"] = label
        dfs.append(df)
    except Exception as e:
        print(f"Warning: could not read {f}: {e}")

combined = pd.concat(dfs, ignore_index=True)
combined.to_csv("outputs/results/sensitivity_combined.csv", index=False)
print(f"Combined {len(dfs)} sensitivity runs into outputs/results/sensitivity_combined.csv")
print(combined[["sensitivity_label","imi_estimate","drope_estimate","direction_change"]].to_string(index=False))
EOF

echo ""
echo "=== Sensitivity analysis complete ==="
echo "Combined results: outputs/results/sensitivity_combined.csv"
