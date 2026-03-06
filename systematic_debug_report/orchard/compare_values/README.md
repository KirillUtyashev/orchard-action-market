# orchard.compare_values

Compare value predictions from two trained orchard runs side-by-side.
Produces a self-contained HTML report with plots, grid renderings, and a sortable state table.

## Usage

```bash
python -m orchard.compare_values RUN_DIR_A RUN_DIR_B [options]
```

Both run directories must contain `metadata.yaml` and `checkpoints/final.pt` (standard output from `train.py`).

### Requirements

The two runs **must** have identical env configs (grid size, n_agents, n_apples, gamma, r_picker, stochastic params, etc.) and the same `td_target`.

They **may** differ in: learning type (centralized vs decentralized), encoder, model architecture, hyperparameters, training duration.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint-a NAME` | `final.pt` | Checkpoint filename within `RUN_DIR_A/checkpoints/` |
| `--checkpoint-b NAME` | `final.pt` | Checkpoint filename within `RUN_DIR_B/checkpoints/` |
| `--n-states N` | `200` | Number of states to generate for comparison |
| `--seed S` | `42` | Seed for state generation |
| `--output-dir DIR` | `./compare_output/` | Where to write the HTML report |
| `--dpi N` | `100` | DPI for grid state renderings |
| `--label-a NAME` | `A` | Custom label for run A (used in plots and report) |
| `--label-b NAME` | `B` | Custom label for run B (used in plots and report) |

### Examples

```bash
# Compare a decentralized run vs a centralized run
python -m orchard.compare_values runs/2025-07-01_dec runs/2025-07-01_cen

# Compare specific checkpoints (not final)
python -m orchard.compare_values runs/exp1 runs/exp2 \
    --checkpoint-a step_50000.pt --checkpoint-b step_50000.pt

# More states, higher quality renderings
python -m orchard.compare_values runs/exp1 runs/exp2 \
    --n-states 500 --dpi 150 --output-dir ./analysis/

# Compare two checkpoints from the SAME run (learning progression)
python -m orchard.compare_values runs/exp1 runs/exp1 \
    --checkpoint-a step_10000.pt --checkpoint-b final.pt

# Custom labels for cleaner plots
python -m orchard.compare_values runs/dec_run runs/cen_run \
    --label-a "Decentralized" --label-b "Centralized"
```

## Output

A single file `compare_output/comparison.html` containing:

1. **Summary header** — run labels, env info, aggregate stats (mean |diff|, max |diff|, mean signed diff, std)
2. **V_team A vs B scatter plot** — each dot is a state, colored by |diff|, diagonal = perfect agreement
3. **Difference histogram** — distribution of (V_A - V_B) with mean line
4. **Sorted difference bar chart** — all states sorted by signed diff, red = A > B, green = B > A
5. **Per-agent scatter plots** — one per agent, only shown when both runs are decentralized
6. **Grid renderings** — top 5 largest disagreements and top 5 closest agreements, with values annotated
7. **Sortable state table** — click any column header to sort; columns include V_team, diff, per-agent values, positions

## How it works

1. Loads `metadata.yaml` from each run directory → reconstructs the exact `ExperimentConfig` used for training
2. Creates networks with matching architecture and loads checkpoint weights
3. Validates that env configs are identical (errors with a list of mismatches if not)
4. Generates states by running nearest-apple policy from a seeded initial state, collecting unique after-states (or pre-action states, depending on td_target)
5. For each state, computes team value from both models:
   - **Decentralized**: `V_team = Σ_i V_i(s)` using each agent's network + encoder
   - **Centralized**: `V_team = V_0(s)` (single network predicts team value directly)
6. Builds the HTML report with embedded matplotlib plots and grid PNGs

## Module structure

```
orchard/compare_values/
    __init__.py
    __main__.py     # CLI entry point
    loader.py       # load_run: run dir → (config, networks, encoder)
    compare.py      # state generation, value computation, comparison logic
    report.py       # HTML report builder with plots and renderings
    design.md       # detailed design document
    README.md       # this file
```

## Notes

- Each run gets its own independent encoder instance, so comparing runs with different encoder types (e.g. `cnn_grid` vs `centralized_cnn_grid`) works correctly.
- No changes to the existing codebase are required.
- State generation uses the same `collect_after_state_test_states` / `collect_on_policy_test_states` functions from `eval.py`.
