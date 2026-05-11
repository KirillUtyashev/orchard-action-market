#!/bin/bash
#SBATCH --job-name=value-sigma
#SBATCH --partition=gpunodes
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-215
#SBATCH --exclude=gpunode16,gpunode13
#SBATCH --output=/u/kutyashev/orchard-action-market/systematic_debug_report/slurm_logs/%x-%A_%a.out
#SBATCH --error=/u/kutyashev/orchard-action-market/systematic_debug_report/slurm_logs/%x-%A_%a.err

set -euo pipefail

source /u/kutyashev/orchard-action-market/.venv/bin/activate
cd /u/kutyashev/orchard-action-market/systematic_debug_report
export PYTHONPATH="$PWD"

# Keep CPU-side math libraries from oversubscribing threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG="/u/kutyashev/orchard-action-market/systematic_debug_report/orchard/configs/reference.yaml"

LEARNING_TYPES=(decentralized centralized)
WIDTHS=(32)
LRS=(0.001 0.003 0.0003)
SIGMA_AS=(0 1 2 4)
SIGMA_BS=(0 1 2 4)

NUM_LEARNING_TYPES=${#LEARNING_TYPES[@]}
NUM_WIDTHS=${#WIDTHS[@]}
NUM_LRS=${#LRS[@]}
NUM_SIGMA_AS=${#SIGMA_AS[@]}
NUM_SIGMA_BS=${#SIGMA_BS[@]}
TOTAL_RUNS=$((NUM_LEARNING_TYPES * NUM_WIDTHS * NUM_LRS * NUM_SIGMA_AS * NUM_SIGMA_BS))

CHUNK_SIZE=4
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START_INDEX=$((TASK_ID * CHUNK_SIZE))
END_INDEX=$((START_INDEX + CHUNK_SIZE))
if [ "${END_INDEX}" -gt "${TOTAL_RUNS}" ]; then
  END_INDEX="${TOTAL_RUNS}"
fi

if [ "${START_INDEX}" -ge "${TOTAL_RUNS}" ]; then
  echo "No work for TASK_ID=${TASK_ID}"
  exit 0
fi

RUN_ROOT="/u/kutyashev/orchard-action-market/systematic_debug_report/slurm_experiments/value_learning_sigma_sweep"
mkdir -p "${RUN_ROOT}"

tag_value() {
  local value="$1"
  value="${value//./p}"
  echo "${value}"
}

learning_tag() {
  local learning_type="$1"
  if [ "${learning_type}" = "centralized" ]; then
    echo "c"
  else
    echo "dec"
  fi
}

if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "nvidia-cuda-mps-control not found on this node."
  exit 1
fi

export CUDA_MPS_PIPE_DIRECTORY="${TMPDIR:-/tmp}/mps_${SLURM_JOB_ID}_${TASK_ID}"
export CUDA_MPS_LOG_DIRECTORY="${TMPDIR:-/tmp}/mps_${SLURM_JOB_ID}_${TASK_ID}"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

cleanup() {
  echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
  rm -rf "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
}
trap cleanup EXIT

nvidia-cuda-mps-control -d
sleep 1

PIDS=()

for ((IDX=START_INDEX; IDX<END_INDEX; IDX++)); do
  LEARNING_TYPE_INDEX=$((IDX / (NUM_WIDTHS * NUM_LRS * NUM_SIGMA_AS * NUM_SIGMA_BS)))
  REMAINDER=$((IDX % (NUM_WIDTHS * NUM_LRS * NUM_SIGMA_AS * NUM_SIGMA_BS)))
  WIDTH_INDEX=$((REMAINDER / (NUM_LRS * NUM_SIGMA_AS * NUM_SIGMA_BS)))
  REMAINDER=$((REMAINDER % (NUM_LRS * NUM_SIGMA_AS * NUM_SIGMA_BS)))
  LR_INDEX=$((REMAINDER / (NUM_SIGMA_AS * NUM_SIGMA_BS)))
  REMAINDER=$((REMAINDER % (NUM_SIGMA_AS * NUM_SIGMA_BS)))
  SIGMA_A_INDEX=$((REMAINDER / NUM_SIGMA_BS))
  SIGMA_B_INDEX=$((REMAINDER % NUM_SIGMA_BS))

  LEARNING_TYPE="${LEARNING_TYPES[$LEARNING_TYPE_INDEX]}"
  WIDTH="${WIDTHS[$WIDTH_INDEX]}"
  LR="${LRS[$LR_INDEX]}"
  SIGMA_A="${SIGMA_AS[$SIGMA_A_INDEX]}"
  SIGMA_B="${SIGMA_BS[$SIGMA_B_INDEX]}"

  LEARNING_TAG="$(learning_tag "${LEARNING_TYPE}")"
  LR_TAG="$(tag_value "${LR}")"
  SIGMA_A_TAG="$(tag_value "${SIGMA_A}")"
  SIGMA_B_TAG="$(tag_value "${SIGMA_B}")"

  RUN_DIR="${RUN_ROOT}/runs/${LEARNING_TAG}/c99s99/w${WIDTH}/lr${LR_TAG}/sa${SIGMA_A_TAG}/sb${SIGMA_B_TAG}"
  mkdir -p "${RUN_DIR}"

  echo "Launching config IDX=${IDX}: learning_type=${LEARNING_TYPE}, width=${WIDTH}, lr=${LR}, sigma_a=${SIGMA_A}, sigma_b=${SIGMA_B}"

  python -m orchard.train \
    --config "${CONFIG}" \
    --override \
      logging.output_dir="${RUN_DIR}" \
      env.height=9 \
      env.width=9 \
      env.n_agents=12 \
      env.n_tasks=5 \
      env.gamma=0.99 \
      env.n_task_types=12 \
      env.clustering=99 \
      env.specialization=99 \
      env.max_tasks_per_type=5 \
      env.structure=id_distance \
      env.stochastic.spawn_prob=0.16 \
      env.stochastic.despawn_mode=probability \
      env.stochastic.despawn_prob=0.1 \
      env.stochastic.sigma_a="${SIGMA_A}" \
      env.stochastic.sigma_b="${SIGMA_B}" \
      env.stochastic.spawn_on_agent_cells=true \
      env.stochastic.spawn_at_round_end=true \
      train.algorithm.name=value \
      train.learning_type="${LEARNING_TYPE}" \
      train.total_steps=10000000 \
      train.seed=1234 \
      train.use_gpu=true \
      train.lr.start="${LR}" \
      train.lr.end="${LR}" \
      train.epsilon.start=0.05 \
      train.epsilon.end=0.05 \
      train.epsilon.schedule=none \
      train.td_lambda=0.3 \
      train.heuristic=nearest \
      train.stopping.condition=none \
      train.discount_method=round_steps \
      eval.eval_steps=500 \
      eval.n_test_states=50 \
      eval.checkpoint_freq=100000 \
      logging.main_csv_freq=10000 \
      logging.detail_csv_freq=50000 \
      model.encoder=everything_cnn_grid \
      model.mlp_dims="[]" \
      model.conv_specs="[[${WIDTH},3],[${WIDTH},3]]" \
      model.activation=relu \
      model.weight_init=default \
    > "${RUN_DIR}/launcher_stdout.log" \
    2> "${RUN_DIR}/launcher_stderr.log" &
  PIDS+=($!)
done

FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAIL=1
  fi
done

exit "${FAIL}"
