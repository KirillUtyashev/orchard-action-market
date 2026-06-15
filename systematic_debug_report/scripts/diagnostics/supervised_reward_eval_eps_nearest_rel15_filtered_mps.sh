#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --job-name=rew-filt-rel15
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --exclude=gpunode16
#SBATCH --output=/u/kutyashev/orchard-action-market/systematic_debug_report/slurm_logs/%x-%A_%a.out
#SBATCH --error=/u/kutyashev/orchard-action-market/systematic_debug_report/slurm_logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/u/kutyashev/orchard-action-market}"
REPORT_ROOT="${PROJECT_ROOT}/systematic_debug_report"
EXP_DIR="${EXP_DIR:-${REPORT_ROOT}/slurm_experiments/prof_and_rel/11ag_prof_and_rel}"
CACHE_DIR="${CACHE_DIR:-${EXP_DIR}/supervised_reward_eval_eps_nearest_rel15}"
OUT_DIR="${OUT_DIR:-${EXP_DIR}/supervised_reward_eval_eps_nearest_rel15_filtered}"

REL_VALUES=(1)
PROF="${PROF:-5}"
POLICY="${POLICY:-eps_nearest}"
MODEL_TAG="${MODEL_TAG:-cnn32_mlp32_filtered}"
ENCODER="${ENCODER:-filtered_dec_cnn_grid}"
CONV_SPEC="${CONV_SPEC:-[[32,3],[32,3]]}"
MLP_DIMS="${MLP_DIMS:-[32]}"
REWARD_LRS=(${REWARD_LRS:-1e-3 2e-3 3e-3 4e-3 6e-3 1e-2})

ROLLOUT_STEPS="${ROLLOUT_STEPS:-10000}"
DROP_TAIL_TRANSITIONS="${DROP_TAIL_TRANSITIONS:-0}"
TRAIN_FRAC="${TRAIN_FRAC:-0.7}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LOG_INTERVAL="${LOG_INTERVAL:-250}"
SEED="${SEED:-0}"
LR_LABEL="${LR_LABEL:-best}"
CHECKPOINT="${CHECKPOINT:-latest}"
DEVICE="${DEVICE:-cuda}"
BEHAVIOR_DEVICE="${BEHAVIOR_DEVICE:-cpu}"
ACTIVE_PERCENT="${ACTIVE_PERCENT:-25}"
FORCE_CACHE="${FORCE_CACHE:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"

mkdir -p "${OUT_DIR}" "${CACHE_DIR}" "${REPORT_ROOT}/slurm_logs"

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

NUM_RELS=${#REL_VALUES[@]}
NUM_LRS=${#REWARD_LRS[@]}
TOTAL_RUNS=$((NUM_RELS * NUM_LRS))
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START_INDEX=$((TASK_ID * CHUNK_SIZE))
END_INDEX=$((START_INDEX + CHUNK_SIZE))
if [ "${END_INDEX}" -gt "${TOTAL_RUNS}" ]; then
    END_INDEX="${TOTAL_RUNS}"
fi
if [ "${START_INDEX}" -ge "${TOTAL_RUNS}" ]; then
    echo "No work for TASK_ID=${TASK_ID}; total runs=${TOTAL_RUNS}"
    exit 0
fi

echo "Node: $(hostname)"
echo "Experiment: ${EXP_DIR}"
echo "Policy: ${POLICY}, encoder: ${ENCODER}, model: ${MODEL_TAG}"
echo "Reward LRs: ${REWARD_LRS[*]}"
echo "Chunk: task=${TASK_ID}, indices [${START_INDEX}, ${END_INDEX}) / ${TOTAL_RUNS}, chunk size=${CHUNK_SIZE}"
echo "Train steps: ${TRAIN_STEPS}, batch size: ${BATCH_SIZE}, log interval: ${LOG_INTERVAL}"
echo "Device: ${DEVICE}, behavior device: ${BEHAVIOR_DEVICE}, active percent: ${ACTIVE_PERCENT}"
python - <<'PYTORCH'
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PYTORCH

ensure_cache() {
    local rel="$1"
    local cache="${CACHE_DIR}/rel${rel}_prof${PROF}_${POLICY}_rollout.pkl"
    local cache_args=()
    if [ "${FORCE_CACHE}" = "1" ]; then
        cache_args+=(--force-rollout-cache)
    fi
    if [ ! -f "${cache}" ] || [ "${FORCE_CACHE}" = "1" ]; then
        echo "Collecting rollout cache for rel=${rel}: ${cache}"
        python -m orchard.supervised_reward_eval \
            --experiment-dir "${EXP_DIR}" \
            --mode dec \
            --rel "${rel}" \
            --prof "${PROF}" \
            --lr-label "${LR_LABEL}" \
            --checkpoint "${CHECKPOINT}" \
            --policy "${POLICY}" \
            --rollout-steps "${ROLLOUT_STEPS}" \
            --drop-tail-transitions "${DROP_TAIL_TRANSITIONS}" \
            --train-frac "${TRAIN_FRAC}" \
            --seed "${SEED}" \
            --behavior-device "${BEHAVIOR_DEVICE}" \
            --rollout-cache "${cache}" \
            "${cache_args[@]}" \
            --collect-only
    else
        echo "Reusing rollout cache for rel=${rel}: ${cache}"
    fi
}

if [ "${DEVICE}" = "cuda" ]; then
    if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
        echo "nvidia-cuda-mps-control not found on this node."
        exit 1
    fi
    export CUDA_MPS_PIPE_DIRECTORY="${TMPDIR:-/tmp}/mps_reward_filtered_${SLURM_JOB_ID}_${TASK_ID}"
    export CUDA_MPS_LOG_DIRECTORY="${TMPDIR:-/tmp}/mps_reward_filtered_${SLURM_JOB_ID}_${TASK_ID}"
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
for ((IDX=START_INDEX; IDX<END_INDEX; IDX++)); do
    REL_INDEX=$((IDX / NUM_LRS))
    LR_INDEX=$((IDX % NUM_LRS))
    REL="${REL_VALUES[$REL_INDEX]}"
    REWARD_LR="${REWARD_LRS[$LR_INDEX]}"
    CACHE="${CACHE_DIR}/rel${REL}_prof${PROF}_${POLICY}_rollout.pkl"

    ensure_cache "${REL}"

    LR_TAG="${REWARD_LR//./p}"
    LR_TAG="${LR_TAG//-/_}"
    PREFIX="rel${REL}_prof${PROF}_${POLICY}_${MODEL_TAG}_lr${LR_TAG}"
    OUT="${OUT_DIR}/${PREFIX}.csv"
    HIST_OUT="${OUT_DIR}/${PREFIX}_history.csv"
    LOG_OUT="${OUT_DIR}/${PREFIX}.out"
    LOG_ERR="${OUT_DIR}/${PREFIX}.err"
    echo "Launching IDX=${IDX}: rel=${REL} encoder=${ENCODER} model=${MODEL_TAG} reward_lr=${REWARD_LR}"
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${ACTIVE_PERCENT}" \
    python -m orchard.supervised_reward_eval \
        --experiment-dir "${EXP_DIR}" \
        --mode dec \
        --rel "${REL}" \
        --prof "${PROF}" \
        --lr-label "${LR_LABEL}" \
        --checkpoint "${CHECKPOINT}" \
        --policy "${POLICY}" \
        --rollout-steps "${ROLLOUT_STEPS}" \
        --drop-tail-transitions "${DROP_TAIL_TRANSITIONS}" \
        --train-frac "${TRAIN_FRAC}" \
        --train-steps "${TRAIN_STEPS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${REWARD_LR}" \
        --model-tag "${MODEL_TAG}" \
        --encoder "${ENCODER}" \
        --conv-specs "${CONV_SPEC}" \
        --mlp-dims "${MLP_DIMS}" \
        --log-interval "${LOG_INTERVAL}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --behavior-device "${BEHAVIOR_DEVICE}" \
        --rollout-cache "${CACHE}" \
        --output "${OUT}" \
        --history-output "${HIST_OUT}" \
        > "${LOG_OUT}" \
        2> "${LOG_ERR}" &
    PIDS+=("$!")
done

for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        FAIL=1
    fi
done

exit "${FAIL}"
