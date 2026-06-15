#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --job-name=reward-sup-pr
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
OUT_DIR="${OUT_DIR:-${EXP_DIR}/supervised_reward_eval}"

REL_VALUES=(5 4 3 2 1)
REL="${REL_VALUES[$SLURM_ARRAY_TASK_ID]}"
PROF="${PROF:-5}"

REWARD_LRS=(${REWARD_LRS:-3e-3 1e-3 3e-4 1e-4})
ROLLOUT_STEPS="${ROLLOUT_STEPS:-10000}"
DROP_TAIL_TRANSITIONS="${DROP_TAIL_TRANSITIONS:-0}"
TRAIN_FRAC="${TRAIN_FRAC:-0.7}"
TRAIN_STEPS="${TRAIN_STEPS:-25000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LOG_INTERVAL="${LOG_INTERVAL:-500}"
SEED="${SEED:-0}"
LR_LABEL="${LR_LABEL:-best}"
CHECKPOINT="${CHECKPOINT:-latest}"
DEVICE="${DEVICE:-cuda}"
BEHAVIOR_DEVICE="${BEHAVIOR_DEVICE:-cpu}"
ACTIVE_PERCENT="${ACTIVE_PERCENT:-25}"

mkdir -p "${OUT_DIR}" "${REPORT_ROOT}/slurm_logs"

if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
elif [ -f "${HOME}/venvs/orchard/bin/activate" ]; then
    source "${HOME}/venvs/orchard/bin/activate"
fi

cd "${REPORT_ROOT}"
export PYTHONPATH="${REPORT_ROOT}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Node: $(hostname)"
echo "Experiment: ${EXP_DIR}"
echo "Relatedness: ${REL}, proficiency: ${PROF}, behavior LR: ${LR_LABEL}"
echo "Reward LRs: ${REWARD_LRS[*]}"
echo "Rollout steps: ${ROLLOUT_STEPS}, train steps: ${TRAIN_STEPS}, batch size: ${BATCH_SIZE}"
echo "Device: ${DEVICE}, behavior device: ${BEHAVIOR_DEVICE}"
python - <<'PYTORCH'
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PYTORCH

if [ "${DEVICE}" = "cuda" ]; then
    if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
        echo "nvidia-cuda-mps-control not found on this node."
        exit 1
    fi
    TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
    export CUDA_MPS_PIPE_DIRECTORY="${TMPDIR:-/tmp}/mps_reward_${SLURM_JOB_ID}_${TASK_ID}"
    export CUDA_MPS_LOG_DIRECTORY="${TMPDIR:-/tmp}/mps_reward_${SLURM_JOB_ID}_${TASK_ID}"
    mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
    cleanup() {
        echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
        rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
    }
    trap cleanup EXIT
    nvidia-cuda-mps-control -d
    sleep 1
fi

PIDS=()
FAIL=0
for REWARD_LR in "${REWARD_LRS[@]}"; do
    LR_TAG="${REWARD_LR//./p}"
    LR_TAG="${LR_TAG//-/_}"
    OUT="${OUT_DIR}/rel${REL}_prof${PROF}_reward_lr${LR_TAG}.csv"
    HIST_OUT="${OUT_DIR}/rel${REL}_prof${PROF}_reward_lr${LR_TAG}_history.csv"
    LOG_OUT="${OUT_DIR}/rel${REL}_prof${PROF}_reward_lr${LR_TAG}.out"
    LOG_ERR="${OUT_DIR}/rel${REL}_prof${PROF}_reward_lr${LR_TAG}.err"
    echo "Launching rel=${REL} reward_lr=${REWARD_LR}"
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${ACTIVE_PERCENT}"     python -m orchard.supervised_reward_eval         --experiment-dir "${EXP_DIR}"         --mode dec         --rel "${REL}"         --prof "${PROF}"         --lr-label "${LR_LABEL}"         --checkpoint "${CHECKPOINT}"         --rollout-steps "${ROLLOUT_STEPS}"         --drop-tail-transitions "${DROP_TAIL_TRANSITIONS}"         --train-frac "${TRAIN_FRAC}"         --train-steps "${TRAIN_STEPS}"         --batch-size "${BATCH_SIZE}"         --lr "${REWARD_LR}"         --log-interval "${LOG_INTERVAL}"         --seed "${SEED}"         --device "${DEVICE}"         --behavior-device "${BEHAVIOR_DEVICE}"         --output "${OUT}"         --history-output "${HIST_OUT}"         > "${LOG_OUT}"         2> "${LOG_ERR}" &
    PIDS+=("$!")
done

for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        FAIL=1
    fi
done

exit "${FAIL}"
