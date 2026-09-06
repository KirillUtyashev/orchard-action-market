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
                    team reward phi[actor,κ] * Σ_j r'[κ,j]; pick best eligible type
                    (r'[κ] already carries the C^(κ) mask)
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
- **r' matrix** (`category_rewards[κ, j]`): per-task per-agent reward values, masked
  to the agents that care about task κ (`r'[κ,j] = 0` for `j ∉ C^(κ)`)
- `w_R` (relatedness_width) and `w_P` (proficiency_width) parameters from the config

Pick events are annotated correct/wrong based on `phi[actor, κ] > 0`.

## Encoding Inspector (`--show-encoding`)

Add `--show-encoding` to embed the raw encoder inputs for every frame in the HTML viewer.
An **Encoding** panel appears below the frame info panel and updates as you step through the trajectory.

### Controls

- **Agent dropdown** (`A0 … AN-1`) — shown whenever decentralized training is used (N networks),
  i.e. whenever each network gets its own encoding call. Switches between any agent's actual
  input view, independent of who the current actor is.
  Both encoders use raw binary positions only. For `everything_cnn_grid` the dec grids are
  identical across agents (full unmasked view). For `filtered_dec_cnn_grid` each agent sees a
  structurally distinct grid — masked to the tasks it cares about (`R_i`) and the agents it can
  reach (`W_i`) — so switching agents shows different channel contents.
  No dropdown when N=1 (centralized) — single shared encoding.
- **Channel dropdown** — `All channels` shows every channel as a compact heatmap in a scrollable
  grid. Select a specific channel to see it full-size with per-cell value annotations.

### Channel labels by encoder type

`KR = min(N, 2·w_R+1)`, `KW = min(N, 2·(w_R+w_P)+1)`.

| Encoder | Channels | Scalars |
|---|---|---|
| `everything_cnn_grid` | `task κk present` (×T), `agent j pos` (×N), `actor pos` | `actor=j` (×N), `pick_phase` |
| `filtered_dec_cnn_grid` | `task (R_i) t` (×KR), `agent (W_i) s` (×KW), `actor pos` | `actor-local s` (×KW), `pick_phase` |

### Heatmap color scale

All channels are normalized to `[0, 1]` per-channel per-frame (white = 0, orange = max value).
Non-zero cells are annotated with their raw value.

### Use cases

- Confirm agent-position and actor-position channels light up at the correct grid cell
- For `filtered_dec_cnn_grid`, verify a task type outside `R_i` (or an agent outside `W_i`) does
  not appear in agent `i`'s channels
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
  n_task_types: 4          # must equal n_agents (shared id space T=N)
  n_tasks: 10
  max_tasks_per_type: 10
  gamma: 0.99
  relatedness_width: 1     # w_R: reward-sharing radius
  proficiency_width: 1     # w_P: task-type eligibility radius
  stochastic:
    spawn_prob: 0.01
    despawn_prob: 0.0125
    despawn_mode: probability
    sigma_a: 0.0
    sigma_b: 0.0
model:
  encoder: everything_cnn_grid    # or filtered_dec_cnn_grid (decentralized, masked)
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
