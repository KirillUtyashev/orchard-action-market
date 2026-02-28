# Orchard Visualization Module — Design

## Overview

A debugging and analysis tool for inspecting orchard RL trajectories.
Single entry point: `python -m orchard.viz`.

---

## Entry Point & CLI

```
python -m orchard.viz CONFIG_YAML [options]
```

### Required
- `CONFIG_YAML` — same YAML file used for training. Defines env, model architecture, encoder, etc.

### Optional
- `--checkpoint PATH` — load a trained model checkpoint. If provided, rolls out the **greedy** policy (argmax Q_team). The tool verifies the checkpoint is compatible with the config by loading state_dicts and failing loudly on shape mismatch.
- `--policy {nearest,random,learned}` — which policy to visualize. Default: `learned` if `--checkpoint` given, else `nearest`.
- `--compare` — side-by-side mode: runs the loaded model's greedy policy on the left, nearest-apple on the right, from the **same initial state**.
- `--steps N` — number of steps to roll out (default: 200).
- `--seed S` — override the config seed for this run.
- `--export {terminal,png,gif}` — output mode (default: `terminal`).
- `--fps F` — frames per second for GIF export (default: 3).
- `--output-dir DIR` — where to write exports (default: `./viz_output/`).
- `--decisions` — show Q-values for all actions at each step (requires `--checkpoint`). Shows why the agent chose its action.
- `--values` — show per-agent V_i(s) predictions at each step (requires `--checkpoint`).

### Examples
```bash
# Interactive: watch nearest-apple policy on your 2x2 stochastic env
python -m orchard.viz configs/2x2/stoch.yaml --steps 50

# Interactive: watch your trained model, with decision introspection
python -m orchard.viz configs/2x2/stoch.yaml --checkpoint runs/exp1/checkpoints/final.pt --decisions

# Side-by-side: learned vs nearest
python -m orchard.viz configs/2x2/stoch.yaml --checkpoint runs/exp1/checkpoints/final.pt --compare

# Export GIF of 500 steps
python -m orchard.viz configs/2x2/stoch.yaml --checkpoint runs/exp1/checkpoints/final.pt --steps 500 --export gif

# Export PNGs + CSV for notebook analysis
python -m orchard.viz configs/2x2/stoch.yaml --policy nearest --steps 1000 --export png
```

---

## Rollout Logic — Modifying eval.py (Option A)

### New generator in eval.py

```python
def rollout_trajectory(
    start_state: State,
    policy_fn: Callable[[State], Action],
    env: BaseEnv,
    n_steps: int,
) -> Iterator[Transition]:
    """Yield one Transition per step. Shared by eval and viz."""
    s = start_state
    for _ in range(n_steps):
        action = policy_fn(s)
        transition = env.step(s, action)
        yield transition
        s = transition.s_t_next
```

### Refactor existing eval functions to use it

`rollout_returns` and `picks_per_step` become thin wrappers around `rollout_trajectory`:

```python
def rollout_returns(start_state, policy_fn, env, rollout_len):
    rewards_history = []
    for t in rollout_trajectory(start_state, policy_fn, env, rollout_len):
        rewards_history.append(t.rewards)
    # ... same backward pass for discounted returns
```

This guarantees the viz sees **exactly** the same stepping logic as evaluation.

---

## Output Files (always produced)

Every run writes to `--output-dir`:

### `trajectory.csv`
One row per step:
```
step,actor,action,picked,reward_0,reward_1,...,n_apples,cum_picks
0,0,RIGHT,false,0.0,0.0,2,0
1,1,DOWN,true,2.0,-1.0,1,1
...
```

### `summary.json`
```json
{
  "policy": "learned",
  "config_path": "configs/2x2/stoch.yaml",
  "checkpoint_path": "runs/exp1/checkpoints/final.pt",
  "seed": 42,
  "total_steps": 200,
  "total_picks": 34,
  "picks_per_step": 0.170,
  "avg_apples_last_100": 1.42,
  "avg_apples_all": 1.38,
  "agent_pick_counts": [18, 16]
}
```

### `frames/` (only with `--export png` or `--export gif`)
Numbered PNGs: `frame_0000.png`, `frame_0001.png`, ...

### `trajectory.gif` (only with `--export gif`)
Animated GIF stitched from frames.

---

## Rendering

### Terminal Mode (default)

ANSI-colored text, interactive stepping (Enter = next, `q` = quit, `b` = back, number = jump to step).

```
      0       1
   ┌───────┬───────┐
 0 │   *   │ [0]   │
   ├───────┼───────┤
 1 │   1   │  * 1  │
   └───────┴───────┘

Step 5 [nearest]
  Actor: A0  Action: RIGHT →
  Rewards: [+2.00, -1.00]
  ★ PICK!

  Apples: 1    Picks/step: 0.0340    γ=0.9
  Positions: A0(0,1) A1(1,1)
```

Cell contents:
- `*` = apple (red colored)
- `0`, `1`, `2`, ... = agent (each has unique color)
- `[0]` = current actor (bracketed, bold)
- Multiple items concatenated with space: `* [0] 1` = apple + actor 0 + agent 1

Works for any number of agents/apples on a cell — just concatenates. Worst case `*[0]123` = 7 chars, fits in a ~9-char cell.

### PNG Mode

Matplotlib grid rendering:
- Cells are colored squares (light for empty, light red for apple)
- Agents are colored circles with number labels inside
- Actor circle has a thick border / highlight ring
- Apple is a small red circle in cell corner (so it doesn't overlap agent circles)
- Info text rendered below the grid

For 2x2, each frame renders in ~20ms. 1000 frames ≈ 20 seconds.

### Side-by-Side (--compare)

Terminal: two grids rendered next to each other with a vertical separator.
PNG: two subplots in one figure.

Both use the **same initial state and same RNG seed** for fair comparison.
Spawn/despawn randomness is synced by running two env copies with identical seed state.

---

## Frame Data Structure

```python
@dataclass
class Frame:
    # Core (always present)
    step: int
    state: State
    height: int
    width: int

    # Transition info
    actor: int | None
    action: Action | None
    rewards: tuple[float, ...] | None
    picked: bool
    gamma: float

    # Policy label
    policy_name: str

    # Running stats
    total_picks: int
    total_steps: int
    apples_on_grid: int

    # Extensible: decision introspection (--decisions flag)
    decisions: list[Decision] | None  # Q-values for all 5 actions

    # Extensible: per-agent values (--values flag)
    agent_values: dict[int, float] | None  # V_i(s)
```

The rollout module converts each `Transition` from `rollout_trajectory` into a `Frame`, enriching it with rendering metadata, running stats, and optionally Q-values.

---

## Module Structure

```
orchard/viz/
    __init__.py
    __main__.py       # CLI: parse args, load config/checkpoint, dispatch
    frame.py          # Frame and Decision dataclasses
    rollout.py        # Wraps rollout_trajectory → yields Frames (adds stats, Q-values)
    renderer.py       # Terminal ANSI renderer
    png_renderer.py   # Matplotlib PNG renderer
    export.py         # CSV/JSON/GIF export logic
```

Modified existing files:
- `eval.py` — add `rollout_trajectory` generator, refactor `rollout_returns` and `picks_per_step` to use it

---

## Checkpoint Loading & Validation

1. Load config → build env, init encoder, create networks (random weights).
2. Load checkpoint file → extract `networks` state_dicts.
3. Call `net.load_state_dict(ckpt_state_dict)` for each agent. This will raise if layer shapes don't match (e.g., wrong MLP dims, wrong encoder type), giving a clear error.
4. Set all networks to eval mode.

No custom validation needed — PyTorch's `load_state_dict` with `strict=True` (default) catches all mismatches.

---

## Compare Mode Sync

For `--compare`, we need two envs to see the same spawn/despawn randomness despite different agent positions (different policies → different movements → different occupied cells → different spawn candidates).

**Approach:** This is fundamentally impossible to make perfectly identical since spawn depends on occupied cells. Instead:
- Use the same seed and same init_state.
- Accept that trajectories diverge due to policy differences — that's the whole point.
- The interesting comparison is picks/step and qualitative behavior, not step-for-step apple positions.

The side-by-side view makes divergence visible, which is the goal.

---

## Steady-State Apple Analysis

Every run automatically computes and reports:
- Apple count at each step (in `trajectory.csv`)
- Rolling average over last N steps (in `summary.json`)
- Per-agent pick counts

For dedicated steady-state analysis without visualization:
```bash
python -m orchard.viz configs/stoch.yaml --policy nearest --steps 5000 --export png --no-render
```
The `--no-render` flag skips frame rendering but still writes the CSV and summary.

---

## Open Questions

1. **Apple ages:** Should we display apple age in the grid (e.g., `*2` = apple aged 2 turns)? Useful for lifetime despawn mode.
2. **History buffer for `b` (back):** Store all frames in memory, or re-roll from start? For ≤1000 steps, storing all frames is fine (~negligible memory).
3. **Future: step-through with editable actions?** i.e., override the policy's choice and see what happens. Would need a fork in the rollout.
