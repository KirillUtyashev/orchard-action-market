#!/bin/bash

#=========================================================================================
# SLURM-DIRECTIVES
#=========================================================================================
#SBATCH --partition=gpunodes
#SBATCH --job-name=orchard_array_map
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --constraint="RTX_4090|RTX_A6000|RTX_A4500|RTX_A4000|RTX_A2000"
#SBATCH --mem=8G
#SBATCH --time=20:00:00
#SBATCH --output=slurm_logs/slurm-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%A_%a.err
#SBATCH --array=0-161

#=========================================================================================
# JOB-SETUP
#=========================================================================================

echo "========================================================="
echo "Starting job array task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Job started at $(date)"
echo "========================================================="

#1p is actually the first line but slurm id is 0-indexed
PARAM_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" parameter_map.txt)

# assign variables algo, hidden, length, width, and num_agent based on the PARAM_LINE content
read -r algo hidden length width num_agents orchard_type alpha batch_size <<< "$PARAM_LINE"

echo "Parameters for this task (from line ${SLURM_ARRAY_TASK_ID} of parameter_map.txt):"
echo "  Algorithm:    ${algo}"
echo "  Hidden Dim:   ${hidden}"
echo "  Grid Size:    ${length}x${width}"
echo "  Num Agents:   ${num_agents}"
echo "  Orchard Type: ${orchard_type}"
echo "  Alpha:        ${alpha}"
echo "  Batch Size:   ${batch_size}"
echo "---------------------------------------------------------"

# --- Activate Virtual Environment ---
VENV_ACTIVATE="../bin/activate"
source "$VENV_ACTIVATE"
echo "Virtual environment activated."

#=========================================================================================
# RUN-EXPERIMENT
#=========================================================================================

# --- Fixed Parameters (same as before) ---
test_timesteps=1000000

# Run the python command using the parameters we just read from the file
  python -u ../run_experiments.py \
  --algorithm "${algo}" --width ${width} --length ${length} --num_agents ${num_agents} --seed 42069 \
  --timesteps 1000000 --apple_life 5 --s_target 0.16 --batch_size ${batch_size} --critic_vision 0 \
  --new_input 1 --alpha ${alpha} --hidden_dim ${hidden} --num_layers 4 --skip 0 --epsilon 0.1 --env_cls ${orchard_type} --new_dynamic 1

echo "--- Renaming log files ---"

# Construct the new, intuitive filename from our parameters.
# Example result: "Decentralized-256h-15x15-10a"
NEW_FILENAME="${algo}-${hidden}h-${length}x${width}-${num_agents}a-${orchard_type}-${alpha}lr-${batch_size}b-$(date +%F)"

# Slurm provides the default log file names in these environment variables.
# We must use them to find the original files.
DEFAULT_OUT_FILE="slurm_logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
DEFAULT_ERR_FILE="slurm_logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

# Define the new paths for the log files.
NEW_OUT_FILE="slurm_logs/${NEW_FILENAME}.out"
NEW_ERR_FILE="slurm_logs/${NEW_FILENAME}.err"

# Use the 'mv' command to rename the files.
# The sleep command is a small safety measure to ensure the file system has
# fully closed the log files before we try to rename them.
sleep 1
mv "$DEFAULT_OUT_FILE" "$NEW_OUT_FILE"
mv "$DEFAULT_ERR_FILE" "$NEW_ERR_FILE"

echo "========================================================="
echo "Job finished at $(date)"
echo "========================================================="
