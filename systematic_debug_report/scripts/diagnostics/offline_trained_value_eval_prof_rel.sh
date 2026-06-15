#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --job-name=offline-val-pr
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/u/kutyashev/orchard-action-market/systematic_debug_report/slurm_logs/%x-%A_%a.out
#SBATCH --error=/u/kutyashev/orchard-action-market/systematic_debug_report/slurm_logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/u/kutyashev/orchard-action-market}"
REPORT_ROOT="${PROJECT_ROOT}/systematic_debug_report"
EXP_DIR="${EXP_DIR:-${REPORT_ROOT}/slurm_experiments/prof_and_rel/11ag_prof_and_rel}"
OUT_DIR="${OUT_DIR:-${EXP_DIR}/offline_trained_value_eval_lr3e-5}"

REL_VALUES=(5 4 3 2 1)
REL="${REL_VALUES[$SLURM_ARRAY_TASK_ID]}"
PROF="${PROF:-5}"

ROLLOUT_STEPS="${ROLLOUT_STEPS:-10000}"
DROP_TAIL_TRANSITIONS="${DROP_TAIL_TRANSITIONS:-1000}"
TRAIN_FRAC="${TRAIN_FRAC:-0.7}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SUP_LR="${SUP_LR:-3e-5}"
SEED="${SEED:-0}"
MODELS="${MODELS:-decentralized centralized}"
DEVICE="${DEVICE:-cuda}"
BEHAVIOR_DEVICE="${BEHAVIOR_DEVICE:-cpu}"
LR_LABEL="${LR_LABEL:-best}"
CHECKPOINT="${CHECKPOINT:-latest}"

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
echo "Models: ${MODELS}"
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY

OUT="${OUT_DIR}/rel${REL}_prof${PROF}_offline_value.csv"
HIST_OUT="${OUT_DIR}/rel${REL}_prof${PROF}_offline_value_history.csv"
python -m orchard.offline_trained_value_eval \
    --experiment-dir "${EXP_DIR}" \
    --mode dec \
    --rel "${REL}" \
    --prof "${PROF}" \
    --lr-label "${LR_LABEL}" \
    --checkpoint "${CHECKPOINT}" \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --drop-tail-transitions "${DROP_TAIL_TRANSITIONS}" \
    --train-frac "${TRAIN_FRAC}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${SUP_LR}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --behavior-device "${BEHAVIOR_DEVICE}" \
    --models ${MODELS} \
    --output "${OUT}" \
    --history-output "${HIST_OUT}"

echo "Wrote ${OUT}"
echo "Wrote ${HIST_OUT}"
