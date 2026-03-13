# compare_policies

Compare greedy policies across multiple trained orchard runs. For each state,
computes `Q_team(s, a)` for all actions under each model, picks the greedy
action, and flags disagreements. Outputs a self-contained HTML report with
grid renderings and per-action Q-value tables so you can see *why* models
disagree (near-tie vs real divergence).

## Usage
```bash
# Two runs
python -m orchard.compare_policies \
    runs/B_dec_mlp64_cnn64_lam0p3_lr1e-3/2026-*/ \
    runs/B_cen_mlp64_cnn64_lam0p3_lr1e-3/2026-*/ \
    --labels "DEC cnn64" "CEN cnn64"

# N runs at once
python -m orchard.compare_policies \
    runs/dec64/2026-*/ runs/cen64/2026-*/ runs/dec8/2026-*/ runs/cen12/2026-*/ \
    --labels "DEC 64" "CEN 64" "DEC 8" "CEN 12" \
    --n-states 200 --seed 42
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | `final.pt` | Checkpoint filename in `checkpoints/` |
| `--labels` | auto | Custom label per run |
| `--n-states` | 200 | Number of states to generate (ignored if `--match-training`) |
| `--seed` | 42 | Seed for state generation (ignored if `--match-training`) |
| `--match-training` | off | Use exact same 100 sample states as `train.py`, which uses random policy. Else uses nearest policy. |
| `--output-dir` | `./compare_policies_output/` | Where to write the report |
| `--dpi` | 100 | DPI for grid renderings |

## State generation modes

- **Default**: generates `--n-states` states by running nearest-apple policy
  rollouts with `--seed`. Good for large-N convergence checks.
- **`--match-training`**: reproduces the exact 100 states from `train.py`'s
  sample monitor (random actions, seed from config). The action distribution
  in the report should exactly match the final values in the `sample_values.csv`
  action fraction plots.
## Output

`policy_comparison.html` — self-contained, open in any browser. Contains:

- **Summary**: agreement rate, action distribution per model
- **Disagreement cards** (sorted by Q-gap, largest first): grid rendering,
  each model's chosen action, full Q-value table with chosen action highlighted
- **Agreement sample** (collapsed by default): confirms good runs pick the
  same action for the same reason

## How states are generated

Uses `collect_after_state_test_states` (or `collect_on_policy_test_states`
for pre-action TD target) with nearest-apple policy, same as eval code.
Deterministic given seed — all runs see identical states.

## Module structure
```
compare_policies/
├── __init__.py
├── __main__.py   # CLI entry point
├── compare.py    # state generation, Q-value computation, greedy comparison
├── report.py     # HTML report builder
└── README.md
```

Reuses `compare_values.loader.LoadedRun` for checkpoint loading and
`viz.renderer.render_state_png` for grid images — no duplicated code.