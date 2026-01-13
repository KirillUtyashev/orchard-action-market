#!/bin/bash
#SBATCH --job-name=debug_step1_verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpunodes
#SBATCH --mem=1GB
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/debug_step1_verify-%j.out
#SBATCH --error=slurm_logs/debug_step1_verify-%j.err

#=========================================================================================
# SETUP
#=========================================================================================
mkdir -p slurm_logs output_notebooks outputs
source "../../../bin/activate"
export PYTHONPATH="${PYTHONPATH}:$(cd ../../../ && pwd)"

#=========================================================================================
# PARAMETERS
#=========================================================================================
INPUT_NOTEBOOK="../decentralized_debug_step1.ipynb"
OUTPUT_NOTEBOOK="output_notebooks/debug_step1_derived_results.ipynb"

LR=0.01
TRAIN_STEPS=200000
GAMMA=0.99
EVAL_FREQ=1000

#=========================================================================================
# EXECUTION
#=========================================================================================
echo "========================================"
echo "Starting Debug Step 1 Verification"
echo "Parameters: LR=${LR}, Steps=${TRAIN_STEPS}, Gamma=${GAMMA}"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

papermill "${INPUT_NOTEBOOK}" "${OUTPUT_NOTEBOOK}" -k python3 --log-output \
    -p LR ${LR} \
    -p TRAIN_STEPS ${TRAIN_STEPS} \
    -p GAMMA ${GAMMA} \
    -p PICKER_REWARD -1 \
    -p EVAL_FREQ ${EVAL_FREQ} \
    -p HIDDEN_DIM 64 \
    -p NUM_LAYERS 2 \
    -p NUM_TEST_CASES 1000 \
    -p HEIGHT 9 \
    -p WIDTH 9 \
    -p NUM_AGENTS 4 \
    -p SEED 22342

echo "========================================"
echo "Verification Complete"
echo "Output saved to: ${OUTPUT_NOTEBOOK}"
echo "========================================"