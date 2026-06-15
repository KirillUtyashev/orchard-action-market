#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --job-name=spawn-cal-pr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclude=gpunode16
#SBATCH --output=/u/kutyashev/orchard-action-market/systematic_debug_report/slurm_logs/%x-%A.out
#SBATCH --error=/u/kutyashev/orchard-action-market/systematic_debug_report/slurm_logs/%x-%A.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/u/kutyashev/orchard-action-market}"
REPORT_ROOT="${PROJECT_ROOT}/systematic_debug_report"
EXP_DIR="${EXP_DIR:-${REPORT_ROOT}/slurm_experiments/prof_and_rel/11ag_prof_and_rel}"
OUT_DIR="${OUT_DIR:-${EXP_DIR}/spawn_calibration}"

PROF="${PROF:-5}"
RELS=(${RELS:-1})
TARGET_REL="${TARGET_REL:-5}"
POLICY="${POLICY:-eps_nearest}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-10000}"
LR_LABEL="${LR_LABEL:-best}"
CHECKPOINT="${CHECKPOINT:-latest}"
DEVICE="${DEVICE:-cpu}"
SPAWN_MULTIPLIERS=(${SPAWN_MULTIPLIERS:-1 2 3 4 5})
MAX_TASK_MULTIPLIERS=(${MAX_TASK_MULTIPLIERS:-1 2 3 4})

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

OUT="${OUT_DIR}/rel${RELS[*]// /_}_targetrel${TARGET_REL}_${POLICY}_spawn_calibration.csv"

echo "Node: $(hostname)"
echo "Experiment: ${EXP_DIR}"
echo "Rels: ${RELS[*]}, target rel: ${TARGET_REL}, prof: ${PROF}"
echo "Policy: ${POLICY}, rollout steps: ${ROLLOUT_STEPS}, device: ${DEVICE}"
echo "Spawn multipliers: ${SPAWN_MULTIPLIERS[*]}"
echo "Max-task multipliers: ${MAX_TASK_MULTIPLIERS[*]}"
echo "Output: ${OUT}"

python -m orchard.spawn_calibration     --experiment-dir "${EXP_DIR}"     --mode dec     --rels "${RELS[@]}"     --target-rel "${TARGET_REL}"     --prof "${PROF}"     --lr-label "${LR_LABEL}"     --checkpoint "${CHECKPOINT}"     --policy "${POLICY}"     --rollout-steps "${ROLLOUT_STEPS}"     --device "${DEVICE}"     --spawn-multipliers "${SPAWN_MULTIPLIERS[@]}"     --max-task-multipliers "${MAX_TASK_MULTIPLIERS[@]}"     --output "${OUT}"

echo "Wrote ${OUT}"
