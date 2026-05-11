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

# Inspect raw encoder inputs: show grid channels and scalars in the HTML viewer
python -m orchard.viz configs/my_config.yaml --checkpoint final.pt --show-encoding
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

## Encoding Inspector (`--show-encoding`)

Add `--show-encoding` to embed the raw encoder inputs for every frame in the HTML viewer.
An **Encoding** panel appears below the frame info panel and updates as you step through the trajectory.

### Controls

- **Agent dropdown** (`A0 … AN-1`) — shown whenever decentralized training is used (N networks),
  i.e. whenever each network gets its own encoding call. Switches between any agent's actual
  input view, independent of who the current actor is.
  For `general_dec_cnn_grid` each agent sees a structurally distinct grid (task-value channels
  weighted by `φ(i,κ)` and `R(i,j)`). For `everything_cnn_grid` with dec training the raw binary
  grids are identical across agents (the network learns structure from reward), but you can still
  verify this by switching agents.
  No dropdown when N=1 (centralized) — single shared encoding.
- **Channel dropdown** — `All channels` shows every channel as a compact heatmap in a scrollable
  grid. Select a specific channel to see it full-size with per-cell value annotations.

### Channel labels by encoder type

| Encoder | Channels | Scalars |
|---|---|---|
| `general_dec_cnn_grid` | `task-val κk` (×T), `self pos`, `teammates (R-wtd)`, `actor pos (R-wtd)` | `is_actor`, `R(actor→i)`, `pick_signal` |
| `general_cen_cnn_grid` | `opt-val κk` (×T), `agent j pos` (×N), `actor pos` | `actor=j` (×N), `pick_phase` |
| `everything_cnn_grid` | `task κk present` (×T), `agent j pos` (×N), `actor pos` | `actor=j` (×N), `pick_phase` |

### Heatmap color scale

All channels are normalized to `[0, 1]` per-channel per-frame (white = 0, orange = max value).
Non-zero cells are annotated with their raw value.

### Use cases

- Verify that task-value channels are non-zero only for tasks where `φ(i,κ) > 0` (decentralized)
- Confirm self-position and actor-position channels light up at the correct grid cell
- Check that teammates channels reflect the `R(i,j)` weights correctly
- Sanity-check that the pick_signal scalar flips to 1.0 at the right transitions

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
  --show-encoding           Show encoder grid channels and scalars in the HTML viewer
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
