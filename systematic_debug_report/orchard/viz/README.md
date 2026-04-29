# Orchard Viz — Trajectory Visualizer

Interactive HTML viewer for inspecting agent trajectories in the orchard environment.
Supports both legacy (homogeneous tasks) and task specialization (multi-type) modes.

## Quick Start

```bash
# Heuristic policy (no checkpoint needed)
python -m orchard.viz configs/my_config.yaml --steps 200

# Learned policy from checkpoint
python -m orchard.viz configs/my_config.yaml --checkpoint runs/exp1/checkpoints/final.pt

# Compare learned vs heuristic (auto-selects heuristic)
python -m orchard.viz configs/my_config.yaml --checkpoint runs/exp1/checkpoints/final.pt --compare

# Compare learned vs a specific policy
python -m orchard.viz configs/my_config.yaml --checkpoint runs/exp1/checkpoints/final.pt --compare nearest_correct_task_stay_wrong

# Fast sanity check (no rendering, just stats + CSV)
python -m orchard.viz configs/my_config.yaml --no-html --steps 500

# Override config values on the fly (dot notation, same as train.py)
python -m orchard.viz metadata.yaml --checkpoint final.pt --override env.n_agents=8 env.height=11

# Apply a fixed-eval scenario (fixed spawn zones + agent start) for direct comparison with evaluate_checkpoint
python -m orchard.viz metadata.yaml --checkpoint final.pt \
    --scenario edge_zones_center_agents --compare nearest_correct_task_stay_wrong
```

## Policy Options

```
--policy nearest_task                       Move toward nearest task (any type)
--policy nearest_correct_task               Move toward nearest task with τ ∈ G_actor
--policy nearest_correct_task_stay_wrong    Move toward nearest correct; stay on wrong-type tasks
--policy random                             Random actions (including pick actions in choice mode)
--policy learned                            Greedy from checkpoint (requires --checkpoint)
```

## Scenarios

Scenarios patch the env config before creating the env, exactly mirroring what `evaluate_checkpoint`
does in `fixed_eval.py` — so what you see in viz is what gets measured in notebook plots.

```
--scenario frozen_zones               freeze spawn zones (eval_spawn_zone_move_interval=0), fixed seed
--scenario edge_zones_center_agents   spawn zones at grid edges (maximally spread around perimeter),
                                      all agents start at grid center — exposes centralized failures
```

The `--eval-seed` defaults to the scenario's seed (`42`) for reproducibility. Pass `--eval-seed N`
to see a different deterministic arrangement with the same zone layout.

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
- **Stats summary:** Prints correct and wrong picks per step alongside Team RPS.
- **Q-value tables** (with `--decisions`): Show Q-values for all actions including
  `pick(0)`, `pick(1)`, etc. in choice pick mode.

## All Options

```
positional arguments:
  config                    Path to YAML config file (or metadata.yaml from a run)

optional arguments:
  --checkpoint PATH         Path to model checkpoint (.pt)
  --policy POLICY           Policy to visualize (see above)
  --compare [POLICY]        Compare against another policy (default: auto-select heuristic).
                            Accepts same values as --policy.
  --show-after-states       Show s_t and s_t^a per transition
  --steps N                 Number of agent decisions (default: 200)
  --seed N                  Override config seed (affects env + training RNGs)
  --eval-seed N             Reseed env RNGs at eval start only (matches EvalConfig.eval_seed)
  --scenario NAME           Apply a fixed-eval scenario (see Scenarios section)
  --override key=val ...    Override config values using dot notation, e.g.:
                              env.n_agents=8
                              env.stochastic.spawn_zone_move_interval=0
                              train.learning_type=centralized
  --fps N                   Autoplay FPS (default: 3)
  --output-dir DIR          Output directory (default: ./viz_output)
  --decisions               Show Q-values for all actions (requires --checkpoint)
  --values                  Show per-agent V_i(s) (requires --checkpoint)
  --dpi N                   PNG render DPI (default: 120)
  --no-html                 Skip rendering and HTML (fast stats + CSV/JSON only)
```

## Output Files

- `trajectory.html` — Interactive HTML viewer with embedded frames
- `trajectory.csv` — One row per transition with actions, rewards, pick info
- `summary.json` — Aggregate statistics (Team RPS, correct/wrong picks, task counts)
- `trajectory_compare.csv` / `summary_compare.json` — Same for comparison policy (with `--compare`)

## Comparing Heuristics

```bash
# See how nearest_correct_task performs
python -m orchard.viz config.yaml --policy nearest_correct_task --steps 500

# Compare learned against the training heuristic
python -m orchard.viz metadata.yaml --checkpoint final.pt --compare nearest_correct_task_stay_wrong

# Compare with nearest_task (ignores type assignments)
python -m orchard.viz config.yaml --policy nearest_task --steps 500
```

The `--no-html` mode is fast (~1 second for 1000 steps) and prints Team RPS
directly, so you can iterate quickly.

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
  r_picker: 1.0
  r_low: 0.0
  # ... standard config
model:
  encoder: filtered_task_cnn_grid
```

### Task specialization (forced pick)
```yaml
env:
  n_task_types: 4
  r_picker: 1.0
  r_low: 0.0
  pick_mode: forced
  max_tasks_per_type: 3
  task_assignments: [[0], [1], [2], [3]]
  # ...
model:
  encoder: centralized_task_cnn_grid  # or filtered_task_cnn_grid for dec
```

### Task specialization (choice pick)
```yaml
env:
  n_task_types: 4
  r_picker: 1.0
  r_low: -1.0
  pick_mode: choice
  max_tasks_per_type: 3
  task_assignments: [[0], [1], [2], [3]]
  # ...
model:
  encoder: centralized_task_cnn_grid  # or filtered_task_cnn_grid for dec
```
