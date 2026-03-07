#!/bin/bash
#SBATCH --job-name=p2_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=cpunodes
#SBATCH --mem=4GB
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/sweep_%A_%a.out
#SBATCH --error=slurm_logs/sweep_%A_%a.err
#SBATCH --array=0-5
# =============================================================================
# Phase 2: TD(λ) Sweep — Fresh values, K=10, 50 MC states
#
#   0: TD(0) sanity check       λ=0,   w=1,   lr=0.01,  [64,64]
#   1: λ=0.4, medium window     λ=0.4, w=50,  lr=0.01,  [64,64]
#   2: λ=0.9, medium window     λ=0.9, w=50,  lr=0.01,  [64,64]
#   3: λ=0.9, lower lr          λ=0.9, w=50,  lr=0.001, [64,64]
#   4: λ=0.9, large window      λ=0.9, w=100, lr=0.01,  [64,64]
#   5: λ=0.4, bigger network    λ=0.4, w=50,  lr=0.01,  [128,128]
# =============================================================================
mkdir -p slurm_logs output_notebooks outputs
source "../../../bin/activate"
export PYTHONPATH="${PYTHONPATH}:$(cd .. && pwd)"

# --- Constants ---
R_PICKER=-1
LR_MIN=1e-6
LR_DECAY_FRAC=0.8
SEED=42
TRAIN_TICKS=500000
EVAL_FREQ=5000
PRINT_FREQ=50000
INPUT_NOTEBOOK="../train.ipynb"
MC_DIR="../monte_carlo_script/mc_data"

# --- Array Configurations ---
LAMBDAS=(0.0   0.4   0.9   0.9   0.9   0.4)
WINDOWS=(1     50    50    50    100   50)
LRS=(    0.01  0.01  0.01  0.001 0.01  0.01)
ARCHS=(
    "[64, 64]"
    "[64, 64]"
    "[64, 64]"
    "[64, 64]"
    "[64, 64]"
    "[128, 128]"
)

# Select current config
LAM=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}
WIN=${WINDOWS[$SLURM_ARRAY_TASK_ID]}
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
MLP="${ARCHS[$SLURM_ARRAY_TASK_ID]}"
MLP_TAG=$(echo "$MLP" | tr -d '[], ')
TAG="lam${LAM}_w${WIN}_lr${LR}_mlp${MLP_TAG}"
OUTPUT_NOTEBOOK="output_notebooks/p2_${TAG}.ipynb"

echo "=== Job ${SLURM_ARRAY_TASK_ID} ==="
echo "Lambda:      ${LAM}"
echo "Window:      ${WIN}"
echo "LR:          ${LR}"
echo "Network:     ${MLP}"
echo "Ticks:       ${TRAIN_TICKS}"
echo "Eval freq:   ${EVAL_FREQ}"
echo "Tag:         ${TAG}"

papermill "${INPUT_NOTEBOOK}" "${OUTPUT_NOTEBOOK}" -k python3 \
    -p R_PICKER ${R_PICKER} \
    -p LAMBDA ${LAM} \
    -p WINDOW_SIZE ${WIN} \
    -p TRAIN_TICKS ${TRAIN_TICKS} \
    -p LR ${LR} \
    -p LR_MIN ${LR_MIN} \
    -p LR_DECAY_FRAC ${LR_DECAY_FRAC} \
    -p SEED ${SEED} \
    -p MC_DIR "${MC_DIR}" \
    -p TAG "${TAG}" \
    -p EVAL_FREQ ${EVAL_FREQ} \
    -p PRINT_FREQ ${PRINT_FREQ} \
    -y "MLP_LAYERS: ${MLP}"

echo "=== Done: ${TAG} ==="