#!/bin/bash
# Run old and new branches for 10 steps on CPU and compare checkpoints 1-10.
# Run from anywhere; no sbatch needed — completes in seconds.
# Usage: bash run_weight_match.sh

set -e

OLD_DIR="/tmp/old_debug/systematic_debug_report"
NEW_DIR="/tmp/new_debug/systematic_debug_report"
CFG="slurm_experiments/april23/weight_match_test/config.yaml"
VENV="/u/taddmao/venvs/orchard/bin/activate"
COMPARE="$NEW_DIR/compare_checkpoints.py"

source "$VENV"

echo "=== Running OLD branch (CPU, 10 steps) ==="
cd "$OLD_DIR"
python -m orchard.train --config "$CFG"
OLD_RUN=$(ls -dt runs/weight_match_test/*/ 2>/dev/null | head -1)
echo "Old run: $OLD_DIR/$OLD_RUN"

echo ""
echo "=== Running NEW branch (CPU, 10 steps) ==="
cd "$NEW_DIR"
python -m orchard.train --config "$CFG"
NEW_RUN=$(ls -dt runs/weight_match_test/*/ 2>/dev/null | head -1)
echo "New run: $NEW_DIR/$NEW_RUN"

echo ""
echo "=== Comparing checkpoints step 1-10 ==="
ALL_PASS=true
for i in $(seq 1 10); do
    OLD_PT="$OLD_DIR/${OLD_RUN}checkpoints/step_${i}.pt"
    NEW_PT="$NEW_DIR/${NEW_RUN}checkpoints/step_${i}.pt"
    echo "--- Step $i ---"
    if ! python "$COMPARE" "$OLD_PT" "$NEW_PT"; then
        ALL_PASS=false
    fi
done

echo ""
if $ALL_PASS; then
    echo "ALL 10 STEPS MATCH. Weights are identical — logic is equivalent."
    echo "Next: sbatch the 2M-step perf test."
    echo "  sbatch $NEW_DIR/slurm_experiments/april23/perf_test/submit.sh"
else
    echo "MISMATCH. Diff env_trace.csv to find first diverging row:"
    echo "  diff $OLD_DIR/${OLD_RUN}env_trace.csv $NEW_DIR/${NEW_RUN}env_trace.csv | head -40"
fi