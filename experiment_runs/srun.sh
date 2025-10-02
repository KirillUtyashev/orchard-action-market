
srun \
    --partition=gpunodes \
    --job-name=interactive_gpu_job \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --gpus-per-task=1 \
    --constraint="RTX_4090|RTX_A6000|RTX_A4500|RTX_A4000|RTX_A2000" \
    --mem=64G \
    --time=20:00:00 \
    --pty bash
