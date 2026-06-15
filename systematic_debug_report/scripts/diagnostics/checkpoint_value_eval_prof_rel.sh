#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --job-name=ckpt-val-pr
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/u/kutyashev/orchard-action-market/systematic_debug_report/slurm_logs/%x-%A_%a.out
#SBATCH --error=/u/kutyashev/orchard-action-market/systematic_debug_report/slurm_logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/u/kutyashev/orchard-action-market}"
REPORT_ROOT="${PROJECT_ROOT}/systematic_debug_report"
EXP_DIR="${EXP_DIR:-${REPORT_ROOT}/slurm_experiments/prof_and_rel/11ag_prof_and_rel}"
OUT_DIR="${OUT_DIR:-${EXP_DIR}/checkpoint_value_eval}"

REL_VALUES=(5 4 3 2 1)
REL="${REL_VALUES[$SLURM_ARRAY_TASK_ID]}"
PROF="${PROF:-5}"

ROLLOUT_STEPS="${ROLLOUT_STEPS:-10000}"
DROP_TAIL_TRANSITIONS="${DROP_TAIL_TRANSITIONS:-1000}"
LR_LABEL="${LR_LABEL:-best}"
CHECKPOINT="${CHECKPOINT:-latest}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUT_DIR}" "${REPORT_ROOT}/slurm_logs"

if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
elif [ -f "${HOME}/venvs/orchard/bin/activate" ]; then
    source "${HOME}/venvs/orchard/bin/activate"
fi

cd "${REPORT_ROOT}"

echo "Node: $(hostname)"
echo "Experiment: ${EXP_DIR}"
echo "Relatedness: ${REL}, proficiency: ${PROF}, LR: ${LR_LABEL}"
echo "Rollout steps: ${ROLLOUT_STEPS}, drop tail transitions: ${DROP_TAIL_TRANSITIONS}"
echo "Device: ${DEVICE}"
python - <<'PYTORCH'
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PYTORCH

OUT="${OUT_DIR}/rel${REL}_prof${PROF}_checkpoint_value.csv"
python -m orchard.checkpoint_value_eval \
    --experiment-dir "${EXP_DIR}" \
    --mode dec \
    --rel "${REL}" \
    --prof "${PROF}" \
    --lr-label "${LR_LABEL}" \
    --checkpoint "${CHECKPOINT}" \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --drop-tail-transitions "${DROP_TAIL_TRANSITIONS}" \
    --device "${DEVICE}" \
    --output "${OUT}"

echo "Wrote ${OUT}"
