#!/bin/bash
# =============================================================================
# Launch script — run from random_policy/
# Submits MC (in monte_carlo_script/), then training (in run_train/).
#
# Usage: cd random_policy && bash launch.sh
# =============================================================================

set -e

echo "Submitting MC jobs..."
MC_JOB=$(sbatch --parsable -D monte_carlo_script monte_carlo_script/run_mc.sh)
echo "MC job array: ${MC_JOB}"

echo "Submitting training sweep (waits for MC)..."
TRAIN_JOB=$(sbatch --parsable -D run_train --dependency=afterok:${MC_JOB} run_train/run_train.sh)
echo "Training job array: ${TRAIN_JOB}"

echo ""
echo "Pipeline:"
echo "  MC:    ${MC_JOB}  (array 0-9)  — ~30s each"
echo "  Train: ${TRAIN_JOB} (array 0-7) — starts after MC"
echo ""
echo "Monitor: squeue -u \$USER"
