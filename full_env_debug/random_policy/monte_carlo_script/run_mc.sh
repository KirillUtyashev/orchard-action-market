#!/bin/bash
#SBATCH --job-name=mc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpunodes
#SBATCH --mem=2GB
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/mc-%A_%a.out
#SBATCH --error=slurm_logs/mc-%A_%a.err
#SBATCH --array=0-9

# =============================================================================
# MC Ground Truth — Debug (gamma=0.9)
# 10 states × 10000 trajectories × 500 ticks. ~25s per state.
# =============================================================================

mkdir -p slurm_logs mc_data
source "../../../bin/activate"

R_PICKER=-1
NUM_TRAJ=10000
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