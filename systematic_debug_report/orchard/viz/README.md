# orchard.viz — Trajectory Visualization

Interactive HTML viewer for inspecting orchard RL trajectories.
Produces a self-contained HTML file with a slider to scrub through
transitions, an info panel with precise math notation, and an apple-count sparkline.

## Transition Model

Each agent decision yields 1 or 2 transitions, matching training exactly:

**No pick:**
```
s_t -> (action, r_{t+1}=0, γ_{t+1}=γ) -> s_{t+1}
```

**Pick (agent lands on apple):**
```
s_t ─> (action, r_{t+1}=0, γ_{t+1}=γ) -> s_{t+1}  -> (PICK, r_{t+2}=pick_rewards, γ_{t+2}=1) -> s_{t+2}
```

The forced pick is a separate transition with γ=1, exactly as the training loop processes it.

## Usage

```bash
# Nearest-apple policy, 100 decisions
python -m orchard.viz configs/2x2/your_config.yaml --steps 100

# Trained model with decision introspection
python -m orchard.viz configs/2x2/your_config.yaml \
    --checkpoint runs/exp1/checkpoints/final.pt --decisions

# Side-by-side: learned vs nearest
python -m orchard.viz configs/2x2/your_config.yaml \
    --checkpoint runs/exp1/checkpoints/final.pt --compare

# Show after-states (s_t and s_{t+1} side by side per transition)
python -m orchard.viz configs/2x2/your_config.yaml --show-after-states

# Sample config included
python -m orchard.viz orchard/viz/sample_configs/2x2_stoch.yaml --steps 50

# Quick PPS sanity check (no rendering, fast)
python -m orchard.viz configs/9x9/your_config.yaml \
    --checkpoint runs/exp1/checkpoints/final.pt --compare --no-html --steps 1000
```

## Options

| Flag                  | Description                                         | Default    |
|-----------------------|-----------------------------------------------------|------------|
| `CONFIG_YAML`         | Path to YAML config (required)                      |            |
| `--checkpoint PATH`   | Model checkpoint (.pt)                              | None       |
| `--policy`            | `nearest`, `random`, or `learned`                   | auto       |
| `--compare`           | Vertical: learned (top) vs nearest (bottom)         | off        |
| `--show-after-states` | Show s_t → s_{t+1} per transition                   | off        |
| `--steps N`           | Number of agent decisions                           | 200        |
| `--seed S`            | Override config seed                                | config     |
| `--fps F`             | Autoplay FPS                                        | 3          |
| `--output-dir DIR`    | Output directory                                    | viz_output |
| `--decisions`         | Show Q-values per action (needs checkpoint)         | off        |
| `--values`            | Show V_i(s) per agent (needs checkpoint)            | off        |
| `--dpi N`             | PNG resolution                                      | 120        |
| `--no-html`           | Skip rendering/HTML, just print PPS + write CSV/JSON | off        |

## Output

- **`trajectory.html`** — slider viewer with ← → keys, Space play/pause
- **`trajectory.csv`** — per-transition: state_index, step, actor, action, rewards, discount, apple counts
- **`summary.json`** — total picks, picks/step, avg apples, per-agent stats

## Info Panel Notation

Each transition displays:
```
s_5 → s_6
A0 → RIGHT →
r_6 = [+2.00, -1.00]
γ_6 = 1.0
★ PICK!
```
Subscripts follow Sutton & Barto: from s_t, the transition produces r_{t+1} and γ_{t+1}.
