# Orchard RL

Multi-agent reinforcement learning in grid environments with after-state TD(0).

## Quick Start

```bash
python3 -m orchard.train --config configs/reference.yaml
```

Override config values via dot notation:

```bash
python3 -m orchard.train --config configs/reference.yaml \
    --override train.lr.start=0.01 model.mlp_dims=[128,128]
```

Run tests:

```bash
pytest orchard/tests/ -v
```

## Checkpoint / Resume Flags

| Flag | Description |
|------|-------------|
| `--resume <path>` | Load full checkpoint (actor + critic) |
| `--resume-critic-only <path>` | Load critic weights only; actors train from scratch |
| `--resume-actor-only <path>` | Load actor weights only; critics train from scratch |

Example:

```bash
# Resume full run
python3 -m orchard.train --config configs/reference.yaml --resume runs/my_run/checkpoints/final.pt

# Warm-start critics only
python3 -m orchard.train --config configs/reference.yaml --resume-critic-only runs/my_run/checkpoints/final.pt

# Warm-start actors only
python3 -m orchard.train --config configs/reference.yaml --resume-actor-only runs/my_run/checkpoints/final.pt
```

## Output Structure

Each run creates a timestamped folder under `logging.output_dir`:

```
runs/my_run/
├── metadata.yaml       # full config + timing
├── metrics.csv         # main metrics per eval step
├── details.csv         # weight/grad norms, LR, RAM
├── timing.csv          # per-section timing (if enabled)
└── checkpoints/
    ├── step_0.pt
    ├── step_1000.pt
    └── final.pt
```
