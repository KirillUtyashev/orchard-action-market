#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --constraint=RTX_4090
#SBATCH --job-name=new_perf_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=/tmp/new_debug/systematic_debug_report/slurm_experiments/april23/perf_test/slurm_logs/perf_%j.out
#SBATCH --error=/tmp/new_debug/systematic_debug_report/slurm_experiments/april23/perf_test/slurm_logs/perf_%j.err

EXP_DIR="/tmp/new_debug/systematic_debug_report"
CFG="${EXP_DIR}/slurm_experiments/april23/perf_test/config.yaml"

cd "$EXP_DIR" || exit 1
source "/u/taddmao/venvs/orchard/bin/activate"

echo "Node: $(hostname), GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m orchard.train --config "$CFG"