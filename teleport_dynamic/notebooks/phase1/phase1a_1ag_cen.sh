#!/bin/bash
#=========================================================================================
# Phase 1A: 1 Agent Centralized (Reward + TD0)
# 16 model configs × 2 learning methods = 32 jobs
# Array IDs 0-15: Reward Learning
# Array IDs 16-31: Value Learning TD(0)
#=========================================================================================
#SBATCH --job-name=p1a_1ag_cen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpunodes
#SBATCH --gpus-per-task=1
#SBATCH --constraint="RTX_4090|RTX_A6000|RTX_A4500|RTX_A4000"
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/p1a-%A_%a.out
#SBATCH --error=slurm_logs/p1a-%A_%a.err
#SBATCH --array=0-31

#=========================================================================================
# SETUP
#=========================================================================================
mkdir -p slurm_logs output_notebooks outputs
source "../../../bin/activate"

IDX=$SLURM_ARRAY_TASK_ID

# Determine learning method
if [ $IDX -lt 16 ]; then
    LEARNING="reward"
    MODEL_IDX=$IDX
    INPUT_NOTEBOOK="../Reward_Learning_Centralized.ipynb"
else
    LEARNING="td0"
    MODEL_IDX=$((IDX - 16))
    INPUT_NOTEBOOK="../Value_Learning_Centralized.ipynb"
fi

#=========================================================================================
# MODEL CONFIGURATIONS (16 total)
# 0-6: MLP configs
# 7-15: CNN configs
#=========================================================================================

# Defaults
MODEL_TYPE="MLP"
MLP_LAYERS=0
MLP_DIM=0
CNN_CH="[1]"
KERNEL=1
HEAD_LAYERS=0
HEAD_DIM=0

case $MODEL_IDX in
    # MLP Configs
    0)  MODEL_TYPE="MLP"; MLP_LAYERS=0; MLP_DIM=0;   DESC="M0_linear" ;;
    1)  MODEL_TYPE="MLP"; MLP_LAYERS=1; MLP_DIM=8;   DESC="M1_1x8" ;;
    2)  MODEL_TYPE="MLP"; MLP_LAYERS=1; MLP_DIM=16;  DESC="M2_1x16" ;;
    3)  MODEL_TYPE="MLP"; MLP_LAYERS=1; MLP_DIM=32;  DESC="M3_1x32" ;;
    4)  MODEL_TYPE="MLP"; MLP_LAYERS=2; MLP_DIM=32;  DESC="M4_2x32" ;;
    5)  MODEL_TYPE="MLP"; MLP_LAYERS=2; MLP_DIM=64;  DESC="M5_2x64" ;;
    6)  MODEL_TYPE="MLP"; MLP_LAYERS=2; MLP_DIM=128; DESC="M6_2x128" ;;
    # CNN Configs
    7)  MODEL_TYPE="CNN"; CNN_CH="[1]";     KERNEL=1; HEAD_LAYERS=0; HEAD_DIM=0;  DESC="C0_floor" ;;
    8)  MODEL_TYPE="CNN"; CNN_CH="[4]";     KERNEL=1; HEAD_LAYERS=1; HEAD_DIM=16; DESC="C1_k1" ;;
    9)  MODEL_TYPE="CNN"; CNN_CH="[4]";     KERNEL=3; HEAD_LAYERS=1; HEAD_DIM=16; DESC="C1_k3" ;;
    10) MODEL_TYPE="CNN"; CNN_CH="[8]";     KERNEL=1; HEAD_LAYERS=1; HEAD_DIM=32; DESC="C2_k1" ;;
    11) MODEL_TYPE="CNN"; CNN_CH="[8]";     KERNEL=3; HEAD_LAYERS=1; HEAD_DIM=32; DESC="C2_k3" ;;
    12) MODEL_TYPE="CNN"; CNN_CH="[16]";    KERNEL=1; HEAD_LAYERS=2; HEAD_DIM=64; DESC="C3_k1" ;;
    13) MODEL_TYPE="CNN"; CNN_CH="[16]";    KERNEL=3; HEAD_LAYERS=2; HEAD_DIM=64; DESC="C3_k3" ;;
    14) MODEL_TYPE="CNN"; CNN_CH="[16,32]"; KERNEL=1; HEAD_LAYERS=2; HEAD_DIM=64; DESC="C4_k1" ;;
    15) MODEL_TYPE="CNN"; CNN_CH="[16,32]"; KERNEL=3; HEAD_LAYERS=2; HEAD_DIM=64; DESC="C4_k3" ;;
esac

#=========================================================================================
# EXECUTION
#=========================================================================================
EXPERIMENT_NAME="p1a_1ag_cen"
JOB_ID="${EXPERIMENT_NAME}_${LEARNING}_${DESC}"
OUTPUT_NOTEBOOK="output_notebooks/${JOB_ID}.ipynb"

echo "=== Phase 1A: 1 Agent Centralized ==="
echo "Job ID:     ${JOB_ID}"
echo "Learning:   ${LEARNING}"
echo "Model:      ${MODEL_TYPE} - ${DESC}"
echo "Notebook:   ${INPUT_NOTEBOOK}"

# Common parameters
COMMON_PARAMS="-p EXPERIMENT_NAME ${EXPERIMENT_NAME} \
    -p WIDTH 6 \
    -p HEIGHT 6 \
    -p NUM_AGENTS 1 \
    -p NUM_APPLES 5 \
    -p MODEL_TYPE ${MODEL_TYPE} \
    -p MLP_HIDDEN_DIM ${MLP_DIM} \
    -p MLP_NUM_LAYERS ${MLP_LAYERS} \
    -p CNN_CONV_CHANNELS ${CNN_CH} \
    -p CNN_KERNEL_SIZE ${KERNEL} \
    -p CNN_HEAD_HIDDEN_DIM ${HEAD_DIM} \
    -p CNN_HEAD_NUM_LAYERS ${HEAD_LAYERS} \
    -p OUTPUT_DIR outputs \
    -p CSV_PATH outputs/phase1_results.csv \
    -p SEED 42"

if [ "$LEARNING" == "reward" ]; then
    papermill "${INPUT_NOTEBOOK}" "${OUTPUT_NOTEBOOK}" -k python3 --log-output \
        $COMMON_PARAMS \
        -p LR 0.001 \
        -p MAX_STEPS 100000 \
        -p BATCH_SIZE 64 \
        -p EVAL_FREQ 1000
else
    # TD(0) value learning
    papermill "${INPUT_NOTEBOOK}" "${OUTPUT_NOTEBOOK}" -k python3 --log-output \
        $COMMON_PARAMS \
        -p LR 0.0001 \
        -p MAX_STEPS 500000 \
        -p BATCH_SIZE 64 \
        -p EVAL_FREQ 5000 \
        -p DISCOUNT 0.99 \
        -p LEARNING_METHOD "td0" \
        -p TARGET_UPDATE_FREQ 1000
fi

echo "Finished ${JOB_ID}"
