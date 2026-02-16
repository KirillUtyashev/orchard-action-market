#!/bin/bash
#SBATCH --job-name=mc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpunodes
#SBATCH --gpus-per-task=1
#SBATCH --exclude=gpunode16
#SBATCH --constraint="RTX_4090|RTX_A6000|RTX_A4500|RTX_A4000"
#SBATCH --mem=2GB
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/mc-%A_%a.out
#SBATCH --error=slurm_logs/mc-%A_%a.err
#SBATCH --array=0-9

# =============================================================================
# MC Ground Truth — Debug (gamma=0.9)
#
# 10 states × 2000 trajectories × 500 ticks each.
# ~5 seconds per state. GPU not used, just grabbing a node.
# =============================================================================

mkdir -p slurm_logs mc_data
source "../../../bin/activate"

R_PICKER=-1
NUM_TRAJ=2000
TRAJ_LEN=500
SEED=42

echo "=== MC State ${SLURM_ARRAY_TASK_ID} ==="
echo "R_PICKER=${R_PICKER}, NUM_TRAJ=${NUM_TRAJ}, TRAJ_LEN=${TRAJ_LEN}"

python ../generate_mc.py \
    --state_index ${SLURM_ARRAY_TASK_ID} \
    --num_trajectories ${NUM_TRAJ} \
    --trajectory_length ${TRAJ_LEN} \
    --r_picker ${R_PICKER} \
    --seed ${SEED} \
    --output_dir mc_data

echo "=== Done: state ${SLURM_ARRAY_TASK_ID} ==="
