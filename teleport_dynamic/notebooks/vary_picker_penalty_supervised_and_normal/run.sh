#!/bin/bash
#=========================================================================================
# PICKER PENALTY SWEEP: TD-LAMBDA vs SUPERVISED
# Goal: Compare bias/variance patterns between TD-lambda and supervised learning
# Tests if supervised learning eliminates the bias/variance seen in TD methods
# 15 picker values × 2 methods = 30 experiments
#=========================================================================================
#SBATCH --job-name=picker_supervised_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpunodes
#SBATCH --gpus-per-task=1
#SBATCH --constraint="RTX_4090|RTX_A6000|RTX_A4500|RTX_A4000"
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpunode16
#SBATCH --output=slurm_logs/picker_supervised_sweep-%A_%a.out
#SBATCH --error=slurm_logs/picker_supervised_sweep-%A_%a.err
#SBATCH --array=0-14
#=========================================================================================
# SETUP
#=========================================================================================
mkdir -p slurm_logs output_notebooks outputs
source "../../../bin/activate"
export PYTHONPATH="${PYTHONPATH}:$(cd ../../../ && pwd)"

#=========================================================================================
# PICKER PENALTY VALUES (15 values)
#=========================================================================================
PICKER_VALS=(-2.0 -1.0 -0.7 -0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.25 0.5 1.0 1.5 2.0)
METHODS=("supervised" "td_lambda")

# Map array index to (picker_value, method)
# 0-14: td_lambda with each picker value
# 15-29: supervised with each picker value
PICKER_IDX=$((SLURM_ARRAY_TASK_ID % 15))
METHOD_IDX=$((SLURM_ARRAY_TASK_ID / 15))

PICKER_REWARD=${PICKER_VALS[$PICKER_IDX]}
LEARNING_METHOD=${METHODS[$METHOD_IDX]}

#=========================================================================================
# FIXED SETTINGS
#=========================================================================================
WIDTH=9
HEIGHT=9
NUM_AGENTS=4
NUM_APPLES=40
MAX_STEPS=500000
LR=0.0001
DISCOUNT=0.99
BATCH=128
LAMBDA=0.95
TRAJ=200
UPDATES=4

MODEL_TYPE="MLP"
MLP_H=64
MLP_L=2

# CNN params (not used but required by notebook)
CNN_CH="[8]"
CNN_K=3
CNN_HH=32
CNN_HL=1

#=========================================================================================
# EXECUTION
#=========================================================================================
EXPERIMENT_NAME="picker_sweep_${LEARNING_METHOD}"
INPUT_NOTEBOOK="../Value_Learning_Decentralized.ipynb"
OUTPUT_NOTEBOOK="output_notebooks/picker_${PICKER_REWARD}_${LEARNING_METHOD}.ipynb"
CSV_PATH="outputs/picker_supervised_sweep_results.csv"

echo "========================================"
echo "Picker Penalty Sweep: TD-Lambda vs Supervised"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "PICKER_REWARD: ${PICKER_REWARD}"
echo "LEARNING_METHOD: ${LEARNING_METHOD}"
echo "Others get: $(echo "scale=4; (1 - (${PICKER_REWARD})) / 3" | bc)"
echo "========================================"

papermill "${INPUT_NOTEBOOK}" "${OUTPUT_NOTEBOOK}" -k python3 --log-output \
    -p EXPERIMENT_NAME "${EXPERIMENT_NAME}" \
    -p WIDTH ${WIDTH} \
    -p HEIGHT ${HEIGHT} \
    -p NUM_AGENTS ${NUM_AGENTS} \
    -p NUM_APPLES ${NUM_APPLES} \
    -p REWARD_SCHEME "picker_penalty" \
    -p PICKER_REWARD ${PICKER_REWARD} \
    -p LEARNING_METHOD "${LEARNING_METHOD}" \
    -p LAMBDA ${LAMBDA} \
    -p TRAJECTORY_LENGTH ${TRAJ} \
    -p UPDATES_PER_TRIGGER ${UPDATES} \
    -p LR ${LR} \
    -p DISCOUNT ${DISCOUNT} \
    -p MAX_STEPS ${MAX_STEPS} \
    -p BATCH_SIZE ${BATCH} \
    -p EVAL_FREQ 1000 \
    -p TARGET_UPDATE_FREQ 1000 \
    -p MODEL_TYPE "${MODEL_TYPE}" \
    -p MLP_HIDDEN_DIM ${MLP_H} \
    -p MLP_NUM_LAYERS ${MLP_L} \
    -p CNN_CONV_CHANNELS "${CNN_CH}" \
    -p CNN_KERNEL_SIZE ${CNN_K} \
    -p CNN_HEAD_HIDDEN_DIM ${CNN_HH} \
    -p CNN_HEAD_NUM_LAYERS ${CNN_HL} \
    -p OUTPUT_DIR outputs \
    -p CSV_PATH "${CSV_PATH}" \
    -p SEED 42

echo "Picker penalty ${PICKER_REWARD} with ${LEARNING_METHOD} complete!"