# Phase 1 Experiments

## Overview

**Total: 96 experiments** (32 per script)

| Script | Setting | Agents | Jobs |
|--------|---------|--------|------|
| `phase1a_1ag_cen.sh` | Centralized | 1 | 0-31 |
| `phase1b_2ag_cen.sh` | Centralized | 2 | 0-31 |
| `phase1c_2ag_decen.sh` | Decentralized (minus1_2) | 2 | 0-31 |

Each script runs:
- Jobs 0-15: Reward learning (16 model configs)
- Jobs 16-31: Value learning TD(0) (16 model configs)

## Model Configurations

| ID | Type | Config | Description |
|----|------|--------|-------------|
| 0 | MLP | 0 layers, 0 hidden | M0 - Linear (floor) |
| 1 | MLP | 1 layer, 8 hidden | M1 - Tiny |
| 2 | MLP | 1 layer, 16 hidden | M2 - Small |
| 3 | MLP | 1 layer, 32 hidden | M3 - Small-wide |
| 4 | MLP | 2 layers, 32 hidden | M4 - Medium-deep |
| 5 | MLP | 2 layers, 64 hidden | M5 - Medium |
| 6 | MLP | 2 layers, 128 hidden | M6 - Large (ceiling) |
| 7 | CNN | [1], k=1, 0×0 head | C0 - Floor |
| 8 | CNN | [4], k=1, 1×16 head | C1-k1 - Tiny |
| 9 | CNN | [4], k=3, 1×16 head | C1-k3 - Tiny |
| 10 | CNN | [8], k=1, 1×32 head | C2-k1 - Small |
| 11 | CNN | [8], k=3, 1×32 head | C2-k3 - Small |
| 12 | CNN | [16], k=1, 2×64 head | C3-k1 - Medium |
| 13 | CNN | [16], k=3, 2×64 head | C3-k3 - Medium |
| 14 | CNN | [16,32], k=1, 2×64 head | C4-k1 - Large (ceiling) |
| 15 | CNN | [16,32], k=3, 2×64 head | C4-k3 - Large (ceiling) |

## Directory Structure

```
experiments/phase1/
├── phase1a_1ag_cen.sh
├── phase1b_2ag_cen.sh
├── phase1c_2ag_decen.sh
├── slurm_logs/           # Auto-created
├── output_notebooks/     # Auto-created
└── outputs/
    └── phase1_results.csv
```

## Running

```bash
# Run all of Phase 1A
sbatch phase1a_1ag_cen.sh

# Run specific jobs (e.g., just reward learning)
sbatch --array=0-15 phase1a_1ag_cen.sh

# Run specific jobs (e.g., just value learning)
sbatch --array=16-31 phase1a_1ag_cen.sh

# Run a single job for testing
sbatch --array=0 phase1a_1ag_cen.sh
```

## Monitoring

```bash
# Check job status
squeue -u $USER

# Check specific array
squeue -u $USER -n p1a_1ag_cen

# Check outputs
tail -f slurm_logs/p1a-*.out

# Count completed experiments
wc -l outputs/phase1_results.csv
```

## After Phase 1

Results go to `outputs/phase1_results.csv`. Use this to:
1. Identify which configs converged
2. Find top 5 by wall time
3. Run Phase 2 (TD(λ) + other reward schemes) with top configs
