#!/bin/bash
#SBATCH --job-name=val_old_cnn
#SBATCH --output=slurm_logs/val_old_cnn_%A_%a.out
#SBATCH --error=slurm_logs/val_old_cnn_%A_%a.err
#SBATCH --array=0-19  # This creates 20 tasks, indexed 0 through 19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint="RTX_4090|RTX_A6000|RTX_A4500|RTX_A4000|RTX_A2000"
#SBATCH --time=00:30:00 # Adjust if a single run takes longer
#SBATCH --gres=gpu:1
#SBATCH --partition=gpunodes # Or your specific GPU partition

# --- Setup ---
# Create necessary directories if they don't exist
mkdir -p slurm_logs output_notebooks

# Activate your conda/python environment
# This path should be correct for your setup
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate orchard

echo "--- JOB SETUP ---"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "-----------------"

# --- Parameter Definitions ---
# Define an array of seeds. The task ID will be used as an index.
SEEDS=(42 101 314 555 777 888 999 1234 2345 3456 4567 5678 6789 7890 8901 9012 1111 2222 3333 4444)
CURRENT_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# --- Define File Names ---
INPUT_NOTEBOOK="cnn_decentralized_p0.5_monte_carlo.ipynb"
RESULTS_FILE="old_code_results.csv"
OUTPUT_NOTEBOOK="output_notebooks/old_code_seed_${CURRENT_SEED}.ipynb"

# --- Create Header for CSV file (only for the first task to avoid race conditions) ---
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "seed,overall_mae,mae_neg1,mae_zero,mae_positive,mape_positive" > $RESULTS_FILE
    echo "CSV header written by task 0."
fi

# A small delay to ensure the header is written before other tasks try to append
sleep 2 

# --- Execute Papermill for the current task ---
echo "Executing Papermill for seed: $CURRENT_SEED"

papermill "$INPUT_NOTEBOOK" "$OUTPUT_NOTEBOOK" \
    -p SEED $CURRENT_SEED \
    -p OUTPUT_FILE $RESULTS_FILE

echo "Papermill process for seed $CURRENT_SEED has finished."