#!/bin/bash

#SBATCH --job-name=papermill_monitor_sweep
#SBATCH --output=slurm_logs/papermill_%A_%a.out
#SBATCH --error=slurm_logs/papermill_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-14


# --- Monitoring Function ---
# This function will run in the background. It takes one argument: the PID of the process to monitor.
monitor_usage() {
    local pid=$1
    local interval=30 # seconds

    echo "--- Starting resource monitoring for PID $pid every $interval seconds ---" >&2
    
    # Loop as long as the process with the given PID is running
    while ps -p $pid > /dev/null; do
        echo "[$(date)] --- GPU Status ---" >&2
        nvidia-smi >&2
        
        echo "[$(date)] --- CPU & Memory for Papermill PID $pid ---" >&2
        # %cpu: CPU percentage, %mem: RAM percentage, rss: Resident Set Size (actual RAM in KB)
        ps -p $pid -o pid,%cpu,%mem,rss,cmd --no-headers >&2
        echo "----------------------------------------------------" >&2
        
        sleep $interval
    done
    
    echo "--- Monitoring stopped. Process $pid has finished. ---" >&2
}

# --- Setup ---
source ../bin/activate
mkdir -p slurm_logs output_notebooks results_pkl

echo "Starting Slurm Array Job: $SLURM_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"

# --- Parameter Definitions ---
CONFIG_KEYS=("12x12_9agents" "9x9_4agents" "6x6_2agents")
CONFIG_VALUES=("12 12 9" "9 9 4" "6 6 2")
PROBABILITIES=(0.1 0.3 0.5 0.7 0.9)
NUM_PROBS=${#PROBABILITIES[@]}

# --- Map SLURM_ARRAY_TASK_ID to Parameters ---
config_index=$((SLURM_ARRAY_TASK_ID / NUM_PROBS))
prob_index=$((SLURM_ARRAY_TASK_ID % NUM_PROBS))
CONFIG_KEY=${CONFIG_KEYS[$config_index]}
read -r WIDTH HEIGHT AGENTS <<< "${CONFIG_VALUES[$config_index]}"
P_PICK=${PROBABILITIES[$prob_index]}

echo "Parameters: WIDTH=$WIDTH, HEIGHT=$HEIGHT, AGENTS=$AGENTS, P_PICK=$P_PICK"

# --- Define File Names ---
INPUT_NOTEBOOK="4dist_many_agent_random_apples.ipynb"
OUTPUT_NOTEBOOK="output_notebooks/output_${CONFIG_KEY}_${P_PICK}p.ipynb"
RESULTS_FOLDER="results_pkl"
RESULT_FILE_NAME="results_${WIDTH}x${HEIGHT}_${AGENTS}agents_${P_PICK}p_pick.pkl"

# --- Execute Papermill and Start Monitoring ---
papermill "$INPUT_NOTEBOOK" "$OUTPUT_NOTEBOOK" \
    -p WIDTH $WIDTH \
    -p HEIGHT $HEIGHT \
    -p NUM_AGENTS $AGENTS \
    -p PROBABILITY_AN_APPLE_IS_PICKED $P_PICK \
    & # The '&' runs this command in the background

# Get the Process ID (PID) of the papermill command we just launched
PAPPERMILL_PID=$!

# Start the monitor function in the background, passing it the papermill PID
monitor_usage $PAPPERMILL_PID &

# The 'wait' command is crucial. It tells the script to pause here and wait
# until the papermill process (with PID $PAPPERMILL_PID) has finished.
wait $PAPPERMILL_PID

echo "Papermill process has finished. Tidying up."

# --- Final Check and Cleanup ---
if [ -f "$RESULT_FILE_NAME" ]; then
    mv "$RESULT_FILE_NAME" "$RESULTS_FOLDER/"
    echo "Successfully moved result file to $RESULTS_FOLDER"
else
    echo "Error: Papermill execution did not produce the expected result file: $RESULT_FILE_NAME"
fi

echo "Task $SLURM_ARRAY_TASK_ID finished."