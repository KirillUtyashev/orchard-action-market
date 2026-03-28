# Orchard Viz — Trajectory Visualizer

Interactive HTML viewer for inspecting agent trajectories in the orchard environment.
Supports both legacy (homogeneous tasks) and task specialization (multi-type) modes.

## Quick Start

```bash
# Heuristic policy (no checkpoint needed)
python -m orchard.viz configs/my_config.yaml --steps 200

# Learned policy from checkpoint
python -m orchard.viz configs/my_config.yaml --checkpoint runs/exp1/checkpoints/final.pt

# Compare learned vs heuristic
python -m orchard.viz configs/my_config.yaml --checkpoint runs/exp1/checkpoints/final.pt --compare

# Fast sanity check (no rendering, just stats + CSV)
python -m orchard.viz configs/my_config.yaml --no-html --steps 500
```

## Policy Options

```
--policy nearest_task          Move toward nearest task (any type)
--policy nearest_correct_task  Move toward nearest task with τ ∈ G_actor
--policy random                Random actions (including pick actions in choice mode)
--policy learned               Greedy from checkpoint (requires --checkpoint)
```

**Default:** `learned` if `--checkpoint` is provided, otherwise auto-detects:
`nearest_correct_task` for `n_task_types > 1`, `nearest_task` for legacy.

**Backward compat:** `--policy nearest` still works (alias for `nearest_task`).

## Task Specialization Features

When the config has `n_task_types > 1`:

- **Color-coded tasks:** Each task type gets a distinct color (colorblind-friendly palette
  with up to 12 types). Task circles show the type number inside.
- **Correct/wrong pick borders:** When a pick occurs, the cell gets a green border
  (correct: τ ∈ G_actor) or red border (wrong: τ ∉ G_actor).
- **Legend:** Shows task type → color mapping and agent → assigned types mapping.
- **Info panel:** Shows actor's assignment G_i, pick type and correctness,
  per-type task counts, cumulative correct/wrong picks.
- **Stats summary:** Prints correct and wrong picks per step alongside total PPS.
- **Q-value tables** (with `--decisions`): Show Q-values for all actions including
  `pick(0)`, `pick(1)`, etc. in choice pick mode.

## All Options

```
positional arguments:
  config                Path to YAML config file (or metadata.yaml from a run)

optional arguments:
  --checkpoint PATH     Path to model checkpoint (.pt)
  --policy POLICY       Policy to visualize (see above)
  --compare             Side-by-side: learned (top) vs heuristic (bottom)
  --show-after-states   Show s_t and s_t^a per transition
  --steps N             Number of agent decisions (default: 200)
  --seed N              Override config seed
  --fps N               Autoplay FPS (default: 3)
  --output-dir DIR      Output directory (default: ./viz_output)
  --decisions           Show Q-values for all actions (requires --checkpoint)
  --values              Show per-agent V_i(s) (requires --checkpoint)
  --dpi N               PNG render DPI (default: 120)
  --no-html             Skip rendering and HTML (fast stats + CSV/JSON only)
```

## Output Files

- `trajectory.html` — Interactive HTML viewer with embedded frames
- `trajectory.csv` — One row per transition with actions, rewards, pick info
- `summary.json` — Aggregate statistics (PPS, correct/wrong picks, task counts)
- `trajectory_compare.csv` / `summary_compare.json` — Same for comparison policy (with `--compare`)

## Parameter Tuning Guide

Use viz to calibrate environment parameters before running full experiments:

### Spawn rate (`spawn_prob`)
```bash
# Try different spawn rates with --no-html for fast iteration
python -m orchard.viz config.yaml --no-html --steps 1000 \
    --override env.stochastic.spawn_prob=0.02

python -m orchard.viz config.yaml --no-html --steps 1000 \
    --override env.stochastic.spawn_prob=0.08
```

Look at `avg_tasks_last_100` in summary.json. You want agents to be neither
starved (too few tasks to pick) nor drowning (tasks everywhere, no challenge).

### Max tasks per type (`max_tasks_per_type`)
```bash
python -m orchard.viz config.yaml --no-html --steps 1000 \
    --override env.max_tasks_per_type=2

python -m orchard.viz config.yaml --no-html --steps 1000 \
    --override env.max_tasks_per_type=5
```

### Comparing heuristics
```bash
# See how nearest_correct_task performs
python -m orchard.viz config.yaml --policy nearest_correct_task --steps 500

# Compare with nearest_task (ignores type assignments)
python -m orchard.viz config.yaml --policy nearest_task --steps 500
```

The `--no-html` mode is fast (~1 second for 1000 steps) and prints PPS/RPS
directly, so you can iterate quickly over parameter sweeps.

### Visual inspection
Once you have good parameters, run with HTML to visually verify:
- Agents move toward their assigned task types
- Pick events are mostly correct (green borders)
- Task density looks right (not too sparse, not too dense)
- In choice mode: agents walk past wrong-type tasks and explicitly pick correct ones

```bash
python -m orchard.viz config.yaml --steps 100 --dpi 100
```

## Example Configs

### Legacy (homogeneous tasks)
```yaml
env:
  n_task_types: 1  # or omit entirely
  # ... standard config
model:
  input_type: cnn_grid
```

### Task specialization (forced pick)
```yaml
env:
  n_task_types: 4
  r_high: 1.0
  r_low: 0.0
  pick_mode: forced
  max_tasks_per_type: 3
  task_assignments: [[0], [1], [2], [3]]
  # ...
model:
  input_type: task_cnn_grid
```

### Task specialization (choice pick)
```yaml
env:
  n_task_types: 4
  r_high: 1.0
  r_low: -1.0
  pick_mode: choice
  max_tasks_per_type: 3
  rho: 0.25
  # ...
model:
  input_type: task_cnn_grid
```
