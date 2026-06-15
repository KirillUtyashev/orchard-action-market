#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --job-name=rew-2a11t
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --exclude=gpunode16
#SBATCH --output=/u/kutyashev/orchard-action-market/systematic_debug_report/slurm_logs/%x-%A_%a.out
#SBATCH --error=/u/kutyashev/orchard-action-market/systematic_debug_report/slurm_logs/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/u/kutyashev/orchard-action-market}"
REPORT_ROOT="${PROJECT_ROOT}/systematic_debug_report"
CONFIG="${CONFIG:-${REPORT_ROOT}/orchard/configs/diagnostics/reward_pred_2agent_11task_4x4.yaml}"
EXP_DIR="${EXP_DIR:-${REPORT_ROOT}/slurm_experiments/reward_pred_2agent_11task_4x4}"
OUT_DIR="${OUT_DIR:-${EXP_DIR}/results}"
CACHE_DIR="${CACHE_DIR:-${EXP_DIR}/rollout_cache}"

POLICY="${POLICY:-eps_nearest}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-10000}"
DROP_TAIL_TRANSITIONS="${DROP_TAIL_TRANSITIONS:-0}"
TRAIN_FRAC="${TRAIN_FRAC:-0.7}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LOG_INTERVAL="${LOG_INTERVAL:-250}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
ACTIVE_PERCENT="${ACTIVE_PERCENT:-25}"
FORCE_CACHE="${FORCE_CACHE:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"
CONV_SPEC="${CONV_SPEC:-[[32,3],[32,3]]}"
MLP_DIMS="${MLP_DIMS:-[32]}"
REWARD_LRS=(${REWARD_LRS:-1e-3 2e-3 3e-3 4e-3})
EXTRA_OVERRIDES=(${EXTRA_OVERRIDES:-})

# setting fields are: rel:encoder:model_tag:setting_tag
SETTINGS=(
  "5:everything_cnn_grid:cnn32_mlp32_regular:rel5_regular"
  "1:everything_cnn_grid:cnn32_mlp32_regular:rel1_regular"
  "1:filtered_dec_cnn_grid:cnn32_mlp32_filtered:rel1_filtered"
)

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

NUM_SETTINGS=${#SETTINGS[@]}
NUM_LRS=${#REWARD_LRS[@]}
TOTAL_RUNS=$((NUM_SETTINGS * NUM_LRS))
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
echo "Config: ${CONFIG}"
echo "Experiment dir: ${EXP_DIR}"
echo "Policy: ${POLICY}"
echo "Reward LRs: ${REWARD_LRS[*]}"
echo "Settings: ${SETTINGS[*]}"
echo "Extra overrides: ${EXTRA_OVERRIDES[*]:-<none>}"
echo "Chunk: task=${TASK_ID}, indices [${START_INDEX}, ${END_INDEX}) / ${TOTAL_RUNS}"
echo "Train steps: ${TRAIN_STEPS}, rollout steps: ${ROLLOUT_STEPS}, batch size: ${BATCH_SIZE}"
echo "Device: ${DEVICE}, active percent: ${ACTIVE_PERCENT}"
python - <<'PYTORCH'
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PYTORCH

build_override_args() {
    local rel="$1"
    OVERRIDE_ARGS=(--override "env.relatedness_width=${rel}")
    local extra
    for extra in "${EXTRA_OVERRIDES[@]}"; do
        OVERRIDE_ARGS+=(--override "${extra}")
    done
}

ensure_cache() {
    local rel="$1"
    local encoder="$2"
    local tag="$3"
    local cache="${CACHE_DIR}/${tag}_${POLICY}_rollout.pkl"
    local cache_args=()
    build_override_args "${rel}"
    if [ "${FORCE_CACHE}" = "1" ]; then
        cache_args+=(--force-rollout-cache)
    fi
    if [ ! -f "${cache}" ] || [ "${FORCE_CACHE}" = "1" ]; then
        echo "Collecting rollout cache for ${tag}: ${cache}"
        python -m orchard.supervised_reward_eval_config             --config "${CONFIG}"             "${OVERRIDE_ARGS[@]}"             --policy "${POLICY}"             --rollout-steps "${ROLLOUT_STEPS}"             --drop-tail-transitions "${DROP_TAIL_TRANSITIONS}"             --train-frac "${TRAIN_FRAC}"             --seed "${SEED}"             --encoder "${encoder}"             --conv-specs "${CONV_SPEC}"             --mlp-dims "${MLP_DIMS}"             --rollout-cache "${cache}"             "${cache_args[@]}"             --collect-only
    else
        echo "Reusing rollout cache for ${tag}: ${cache}"
    fi
}

if [ "${DEVICE}" = "cuda" ]; then
    if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
        echo "nvidia-cuda-mps-control not found on this node."
        exit 1
    fi
    export CUDA_MPS_PIPE_DIRECTORY="${TMPDIR:-/tmp}/mps_reward_2a11t_${SLURM_JOB_ID}_${TASK_ID}"
    export CUDA_MPS_LOG_DIRECTORY="${TMPDIR:-/tmp}/mps_reward_2a11t_${SLURM_JOB_ID}_${TASK_ID}"
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
    SETTING_INDEX=$((IDX / NUM_LRS))
    LR_INDEX=$((IDX % NUM_LRS))
    IFS=':' read -r REL ENCODER MODEL_TAG SETTING_TAG <<< "${SETTINGS[$SETTING_INDEX]}"
    REWARD_LR="${REWARD_LRS[$LR_INDEX]}"
    CACHE="${CACHE_DIR}/${SETTING_TAG}_${POLICY}_rollout.pkl"

    ensure_cache "${REL}" "${ENCODER}" "${SETTING_TAG}"
    build_override_args "${REL}"

    LR_TAG="${REWARD_LR//./p}"
    LR_TAG="${LR_TAG//-/_}"
    PREFIX="${SETTING_TAG}_${MODEL_TAG}_lr${LR_TAG}"
    OUT="${OUT_DIR}/${PREFIX}.csv"
    HIST_OUT="${OUT_DIR}/${PREFIX}_history.csv"
    LOG_OUT="${OUT_DIR}/${PREFIX}.out"
    LOG_ERR="${OUT_DIR}/${PREFIX}.err"

    echo "Launching IDX=${IDX}: setting=${SETTING_TAG} rel=${REL} encoder=${ENCODER} lr=${REWARD_LR}"
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${ACTIVE_PERCENT}"     python -m orchard.supervised_reward_eval_config         --config "${CONFIG}"         "${OVERRIDE_ARGS[@]}"         --policy "${POLICY}"         --rollout-steps "${ROLLOUT_STEPS}"         --drop-tail-transitions "${DROP_TAIL_TRANSITIONS}"         --train-frac "${TRAIN_FRAC}"         --train-steps "${TRAIN_STEPS}"         --batch-size "${BATCH_SIZE}"         --lr "${REWARD_LR}"         --model-tag "${MODEL_TAG}"         --encoder "${ENCODER}"         --conv-specs "${CONV_SPEC}"         --mlp-dims "${MLP_DIMS}"         --log-interval "${LOG_INTERVAL}"         --seed "${SEED}"         --device "${DEVICE}"         --rollout-cache "${CACHE}"         --output "${OUT}"         --history-output "${HIST_OUT}"         > "${LOG_OUT}"         2> "${LOG_ERR}" &
    PIDS+=("$!")
done

for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        FAIL=1
    fi
done

exit "${FAIL}"
