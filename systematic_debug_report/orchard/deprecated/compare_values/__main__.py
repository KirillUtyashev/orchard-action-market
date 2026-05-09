"""CLI entry point: python -m orchard.compare_values RUN_DIR_1 RUN_DIR_2 ... [options]"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from orchard.compare_values.loader import load_run
from orchard.compare_values.compare import (
    generate_states,
    run_comparison,
    validate_env_compatibility,
    validate_td_target_compatibility,
)
from orchard.compare_values.report import build_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare value predictions from N trained orchard runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m orchard.compare_values runs/dec_small runs/dec_large
  python -m orchard.compare_values runs/a runs/b runs/c --labels "Small" "Medium" "Large"
  python -m orchard.compare_values runs/exp1 runs/exp2 --n-states 500 --output-dir ./analysis/
""",
    )
    p.add_argument("runs", type=str, nargs="+", help="Paths to run directories (2 or more)")
    p.add_argument("--checkpoints", type=str, nargs="*", default=None,
                   help="Checkpoint filenames per run (default: final.pt for all)")
    p.add_argument("--labels", type=str, nargs="*", default=None,
                   help="Custom labels per run (default: auto-generated)")
    p.add_argument("--n-states", type=int, default=200,
                   help="Number of states to compare (default: 200)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for state generation (default: 42)")
    p.add_argument("--output-dir", type=str, default="./compare_output/",
                   help="Output directory (default: ./compare_output/)")
    p.add_argument("--dpi", type=int, default=100,
                   help="DPI for state renderings (default: 100)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_runs = len(args.runs)

    if n_runs < 2:
        raise ValueError("Need at least 2 run directories to compare.")

    # Resolve checkpoints
    if args.checkpoints:
        if len(args.checkpoints) != n_runs:
            raise ValueError(f"Got {len(args.checkpoints)} checkpoints but {n_runs} runs")
        checkpoints = args.checkpoints
    else:
        checkpoints = ["final.pt"] * n_runs

    # Resolve labels
    if args.labels:
        if len(args.labels) != n_runs:
            raise ValueError(f"Got {len(args.labels)} labels but {n_runs} runs")
        custom_labels = args.labels
    else:
        custom_labels = [None] * n_runs

    # --- Load all runs ---
    runs = []
    for i, (run_path, ckpt) in enumerate(zip(args.runs, checkpoints)):
        print(f"Loading run {i}: {run_path}")
        run = load_run(Path(run_path), ckpt)
        if custom_labels[i]:
            run.label = custom_labels[i]
        else:
            run.label = Path(run_path).name
        print(f"  → {run.label}")
        runs.append(run)

    # --- Validate compatibility ---
    print("Validating env compatibility...")
    validate_env_compatibility(runs)
    validate_td_target_compatibility(runs)
    print("  ✓ All env configs match")
    print(f"  TD target: {runs[0].cfg.train.td_target.name.lower()}")

    cen_runs = [r for r in runs if r.is_centralized]
    dec_runs = [r for r in runs if not r.is_centralized]
    if cen_runs and dec_runs:
        print(f"  Mix of centralized ({len(cen_runs)}) and decentralized ({len(dec_runs)}) runs")

    # --- Generate states ---
    ref = runs[0]
    print(f"Generating {args.n_states} states (seed={args.seed})...")
    t0 = time.time()
    states = generate_states(
        ref.cfg.env,
        ref.cfg.train.td_target,
        args.n_states,
        args.seed,
    )
    print(f"  Collected {len(states)} unique states in {time.time() - t0:.1f}s")

    # --- Compare ---
    print(f"Computing value predictions for {n_runs} runs...")
    t0 = time.time()
    comparisons = run_comparison(runs, states)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Quick summary
    import numpy as np
    labels = [r.label for r in runs]
    for i, li in enumerate(labels):
        mean_v = np.mean([abs(c.team_values[li]) for c in comparisons])
        print(f"  {li}: mean |V_team| = {mean_v:.4f}")

    per_state_stds = [np.std([c.team_values[l] for l in labels]) for c in comparisons]
    print(f"  Cross-model: mean σ = {np.mean(per_state_stds):.4f}, "
          f"max σ = {np.max(per_state_stds):.4f}")

    # --- Build report ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "comparison.html"

    print("Building HTML report...")
    t0 = time.time()
    build_report(comparisons, runs, report_path, dpi=args.dpi)
    size_mb = report_path.stat().st_size / 1024 / 1024
    print(f"  Wrote {report_path} ({size_mb:.1f} MB) in {time.time() - t0:.1f}s")

    print(f"\nDone! Open {report_path} in a browser to view.")


if __name__ == "__main__":
    main()
