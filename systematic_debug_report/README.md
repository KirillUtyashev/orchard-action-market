# Orchard RL

Multi-agent reinforcement learning in grid environments with after-state TD(0).

## Quick Start

```bash
# Single run
python3 -m orchard.train --config configs/2x2_value.yaml

# Override params from CLI (dot notation)
python3 -m orchard.train --config configs/2x2_value.yaml \
    --override train.lr.start=0.01 model.mlp_dims=[128,128]

# Run tests
pytest orchard/tests/ -v
```

## Config Reference

See `configs/reference.yaml` for every parameter with its type and meaning.

---

## SLURM Usage

### Single job

```bash
#!/bin/bash
#SBATCH --job-name=orchard-2x2
#SBATCH --output=logs/%j.out
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

python3 -m train.py --config configs/2x2_value.yaml
```

### Hyperparameter sweep with job arrays

Use `--override` with environment variables. SLURM sets `$SLURM_ARRAY_TASK_ID`.

**Example: sweep learning rates**

```bash
#!/bin/bash
#SBATCH --job-name=lr-sweep
#SBATCH --output=logs/%A_%a.out
#SBATCH --array=0-4
#SBATCH --time=02:00:00
#SBATCH --mem=4G

LRS=(0.01 0.005 0.001 0.0005 0.0001)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

python -m orchard.train \
    --config configs/2x2_value.yaml \
    --override train.lr.start=$LR train.seed=$SLURM_ARRAY_TASK_ID \
               logging.output_dir=runs/lr_sweep/
```

**Example: grid search over LR × hidden dims**

```bash
#!/bin/bash
#SBATCH --job-name=grid-search
#SBATCH --output=logs/%A_%a.out
#SBATCH --array=0-11
#SBATCH --time=04:00:00
#SBATCH --mem=4G

LRS=(0.001 0.0005 0.0001)
DIMS=("[32,32]" "[64,64]" "[128,128]" "[64,64,64]")

LR_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
DIM_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))

LR=${LRS[$LR_IDX]}
DIM=${DIMS[$DIM_IDX]}

python -m orchard.train \
    --config configs/2x2_value.yaml \
    --override train.lr.start=$LR model.mlp_dims=$DIM \
               train.seed=42 \
               logging.output_dir=runs/grid_search/
```

**Example: multi-seed runs for significance**

```bash
#!/bin/bash
#SBATCH --job-name=seeds
#SBATCH --output=logs/%A_%a.out
#SBATCH --array=0-9
#SBATCH --time=02:00:00
#SBATCH --mem=4G

python -m orchard.train \
    --config configs/2x2_value.yaml \
    --override train.seed=$SLURM_ARRAY_TASK_ID \
               logging.output_dir=runs/seed_sweep/
```

**Example: compare reward schemes**

```bash
#!/bin/bash
#SBATCH --job-name=rewards
#SBATCH --output=logs/%A_%a.out
#SBATCH --array=0-4
#SBATCH --time=02:00:00
#SBATCH --mem=4G

R_PICKERS=(-2.0 -1.0 0.0 1.0 2.0)
R=${R_PICKERS[$SLURM_ARRAY_TASK_ID]}

python -m orchard.train \
    --config configs/2x2_value.yaml \
    --override env.r_picker=$R \
               logging.output_dir=runs/reward_sweep/
```

### Output structure

Each run creates a timestamped folder:

```
runs/lr_sweep/
├── 2025-01-15_143022/
│   ├── metadata.yaml      # full config + timing
│   ├── metrics.csv        # main metrics (tail -f friendly)
│   └── details.csv        # weight/grad norms, LR, RAM
├── 2025-01-15_143025/
│   └── ...
```

### Monitoring live runs

```bash
# Watch metrics from a running job
tail -f runs/lr_sweep/2025-01-15_143022/metrics.csv

# Quick check across all runs in a sweep
for d in runs/lr_sweep/*/; do
    echo "=== $d ==="
    tail -1 "$d/metrics.csv"
done
```
