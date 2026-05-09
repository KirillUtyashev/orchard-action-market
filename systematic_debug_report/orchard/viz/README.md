# Orchard Viz — Trajectory Visualizer

Interactive HTML viewer for inspecting agent trajectories in the orchard environment.

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

# Override config values on the fly (dot notation)
python -m orchard.viz metadata.yaml --checkpoint final.pt --override env.n_agents=8 env.height=11

# Apply a fixed-eval scenario for direct comparison with evaluate_checkpoint
python -m orchard.viz metadata.yaml --checkpoint final.pt --scenario center_agents
```

## Policy Options

```
--policy nearest    Value-aware nearest heuristic: move toward task with highest
                    phi[actor,κ] * Σ_j R[actor,j] * r'[κ,j]; pick best eligible type
--policy random     Random actions (including pick actions)
--policy learned    Greedy from checkpoint (requires --checkpoint)
```

**Default:** `learned` if `--checkpoint` is provided, otherwise `nearest`.

## Scenarios

Scenarios mirror what `evaluate_checkpoint` does in `fixed_eval.py` — so what you see
in viz is what gets measured in evaluation.

```
--scenario center_agents   All agents start at grid center each time init_state() is called.
                           No spawn zone changes.
```

## φ/R Framework Display

The HTML viewer shows the φ/R reward structure in the legend panel:
- **φ matrix** (`phi[actor, κ]`): which task types each agent can profitably pick
- **R matrix** (`relatedness[actor, j]`): which agents share rewards with whom
- **r' matrix** (`category_rewards[κ, j]`): per-category per-agent reward values
- `C` (clustering) and `S` (specialization) parameters from the config

Pick events are annotated correct/wrong based on `phi[actor, κ] > 0`.

## All Options

```
positional arguments:
  config                    Path to YAML config file (or metadata.yaml from a run)

optional arguments:
  --checkpoint PATH         Path to model checkpoint (.pt)
  --policy POLICY           Policy to visualize: nearest, random, learned
  --compare [POLICY]        Compare against another policy (default: nearest).
                            Accepts same values as --policy.
  --show-after-states       Show s_t and s_t^a per transition
  --steps N                 Number of agent decisions (default: 200)
  --seed N                  Override config seed (affects env + training RNGs)
  --eval-seed N             Reseed env RNGs at eval start only (matches EvalConfig.eval_seed)
  --scenario NAME           Apply a fixed-eval scenario (see Scenarios section)
  --override key=val ...    Override config values using dot notation, e.g.:
                              env.n_agents=8
                              env.clustering=1
                              train.learning_type=centralized
  --rand-zone-seed N        Randomize initial spawn zone positions using this seed.
                            Use different values (0, 1, 2, ...) to sweep zone configs.
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

## Example Config

```yaml
env:
  height: 9
  width: 9
  n_agents: 4
  n_task_types: 2
  n_tasks: 10
  max_tasks_per_type: 10
  gamma: 0.99
  clustering: 0        # C: reward-sharing radius
  specialization: 0    # S: task-type eligibility radius
  stochastic:
    spawn_prob: 0.01
    despawn_prob: 0.0125
    despawn_mode: probability
    sigma_a: 0.0
    sigma_b: 0.0
model:
  encoder: general_dec_cnn_grid   # or general_cen_cnn_grid for centralized
  conv_specs: [[16, 3]]
  mlp_dims: [16]
```

## Tips

The `--no-html` mode is fast (~1 second for 1000 steps) and prints Team RPS
directly, so you can iterate quickly over configs.

Once you have good parameters, run with HTML to visually verify:
- Agents move toward tasks where `phi[actor, κ] > 0`
- Pick events are mostly correct (green borders)
- Task density looks right (not too sparse, not too dense)

```bash
python -m orchard.viz config.yaml --steps 100 --dpi 100
```
