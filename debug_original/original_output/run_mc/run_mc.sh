#!/bin/bash
#=========================================================================================
# SLURM DIRECTIVES
#=========================================================================================
#SBATCH --partition=cpunodes
#SBATCH --job-name=mc_gen
#SBATCH --array=0-99%50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/mc_gen_%A_%a.out
#SBATCH --error=slurm_logs/mc_gen_%A_%a.err
#=========================================================================================
# SETUP
#=========================================================================================
echo "Job started on $(hostname) at $(date)"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"

mkdir -p slurm_logs

# Activate virtual environment (adjust path as needed)
source ../../bin/activate


#=========================================================================================
# RUN
#=========================================================================================
python -u generate_mc.py \
    --state_index ${SLURM_ARRAY_TASK_ID} \
    --r_picker -1 \
    --seed 42069 \
    --mc_depth 1000 \
    --num_trajectories 5 \
    --num_rollouts 200 \
    --output_dir mc_cache

echo "Job finished at $(date)"
