#!/bin/bash
#SBATCH --job-name=p1_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpunodes
#SBATCH --gpus-per-task=1
#SBATCH --exclude=gpunode16
#SBATCH --constraint="RTX_4090|RTX_A6000|RTX_A4500|RTX_A4000"
#SBATCH --mem=4GB
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/sweep_%A_%a.out
#SBATCH --error=slurm_logs/sweep_%A_%a.err
#SBATCH --array=0-7

# =============================================================================
# Phase 1: TD(λ) Sweep — Debug (gamma=0.9, 10 MC states)
#
#   0: TD(0), [16,16,16,16]          — your baseline
#   1: λ=0.4, [16,16,16,16]          — moderate λ
#   2: λ=0.9, [16,16,16,16]          — high λ, your architecture
#   3: λ=0.4, [32]                   — shallow
#   4: λ=0.4, [64,64]                — standard
#   5: λ=0.9, [64,64]                — *** best bet for <5% ***
#   6: λ=0.4, [128,128]              — wide
#   7: λ=0.4, [16,16,16,16,16,16]    — deep
# =============================================================================

mkdir -p slurm_logs output_notebooks outputs
source "../../../bin/activate"

# --- Constants ---
R_PICKER=-1
LR=0.001
LR_MIN=1e-6
LR_DECAY_FRAC=0.8
SEED=42
TRAIN_TICKS=500000
WINDOW_SIZE=100
INPUT_NOTEBOOK="../train.ipynb"
MC_DIR="../monte_carlo_script/mc_data"

# --- Array Configurations ---
LAMBDAS=(0.0 0.4 0.9 0.4 0.4 0.9 0.4 0.4)
ARCHS=(
    "[16, 16, 16, 16]"
    "[16, 16, 16, 16]"
    "[16, 16, 16, 16]"
    "[32]"
    "[64, 64]"
    "[64, 64]"
    "[128, 128]"
    "[16, 16, 16, 16, 16, 16]"
)

# Select current config
LAM=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}
MLP="${ARCHS[$SLURM_ARRAY_TASK_ID]}"

MLP_TAG=$(echo "$MLP" | tr -d '[], ')
TAG="lam${LAM}_mlp${MLP_TAG}"
OUTPUT_NOTEBOOK="output_notebooks/p1_${TAG}.ipynb"

echo "=== Job ${SLURM_ARRAY_TASK_ID} ==="
echo "Lambda:  ${LAM}"
echo "Network: ${MLP}"
echo "Ticks:   ${TRAIN_TICKS}"
echo "Tag:     ${TAG}"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

papermill "${INPUT_NOTEBOOK}" "${OUTPUT_NOTEBOOK}" -k python3 \
    -p R_PICKER ${R_PICKER} \
    -p LAMBDA ${LAM} \
    -p WINDOW_SIZE ${WINDOW_SIZE} \
    -p TRAIN_TICKS ${TRAIN_TICKS} \
    -p LR ${LR} \
    -p LR_MIN ${LR_MIN} \
    -p LR_DECAY_FRAC ${LR_DECAY_FRAC} \
    -p SEED ${SEED} \
    -p MC_DIR "${MC_DIR}" \
    -y "MLP_LAYERS: ${MLP}"

echo "=== Done: ${TAG} ==="
