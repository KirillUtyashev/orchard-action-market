"""CLI entry point: python -m orchard.compare_values RUN_DIR_A RUN_DIR_B [options]"""

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
        description="Compare value predictions from two trained orchard runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m orchard.compare_values runs/decentralized_001 runs/centralized_001
  python -m orchard.compare_values runs/exp1 runs/exp2 --checkpoint-a step_50000.pt
  python -m orchard.compare_values runs/exp1 runs/exp2 --n-states 500 --output-dir ./analysis/
  python -m orchard.compare_values runs/dec runs/cen --label-a "Decentralized" --label-b "Centralized"
""",
    )
    p.add_argument("run_a", type=str, help="Path to first run directory")
    p.add_argument("run_b", type=str, help="Path to second run directory")
    p.add_argument("--checkpoint-a", type=str, default="final.pt",
                   help="Checkpoint filename in run_a/checkpoints/ (default: final.pt)")
    p.add_argument("--checkpoint-b", type=str, default="final.pt",
                   help="Checkpoint filename in run_b/checkpoints/ (default: final.pt)")
    p.add_argument("--n-states", type=int, default=200,
                   help="Number of states to compare (default: 200)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for state generation (default: 42)")
    p.add_argument("--output-dir", type=str, default="./compare_output/",
                   help="Output directory (default: ./compare_output/)")
    p.add_argument("--dpi", type=int, default=100,
                   help="DPI for state renderings (default: 100)")
    p.add_argument("--label-a", type=str, default=None,
                   help="Custom label for run A (default: 'A')")
    p.add_argument("--label-b", type=str, default=None,
                   help="Custom label for run B (default: 'B')")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load both runs ---
    print(f"Loading run A: {args.run_a}")
    run_a = load_run(Path(args.run_a), args.checkpoint_a)
    print(f"  → {run_a.label}")

    print(f"Loading run B: {args.run_b}")
    run_b = load_run(Path(args.run_b), args.checkpoint_b)
    print(f"  → {run_b.label}")

    # Apply labels: "A (details)" / "B (details)" or custom
    prefix_a = args.label_a if args.label_a else "A"
    prefix_b = args.label_b if args.label_b else "B"
    run_a.label = f"{prefix_a} ({run_a.label})"
    run_b.label = f"{prefix_b} ({run_b.label})"

    # --- Validate compatibility ---
    print("Validating env compatibility...")
    validate_env_compatibility(run_a.cfg.env, run_b.cfg.env)
    validate_td_target_compatibility(run_a, run_b)
    print("  ✓ Env configs match")
    print(f"  TD target: {run_a.cfg.train.td_target.name.lower()}")

    # Print what differs
    diffs = []
    if run_a.is_centralized != run_b.is_centralized:
        diffs.append(f"learning type: {'centralized' if run_a.is_centralized else 'decentralized'} vs {'centralized' if run_b.is_centralized else 'decentralized'}")
    if run_a.cfg.model.input_type != run_b.cfg.model.input_type:
        diffs.append(f"encoder: {run_a.cfg.model.input_type.name} vs {run_b.cfg.model.input_type.name}")
    if run_a.cfg.model.model_type != run_b.cfg.model.model_type:
        diffs.append(f"model: {run_a.cfg.model.model_type.name} vs {run_b.cfg.model.model_type.name}")
    if diffs:
        print(f"  Differences: {'; '.join(diffs)}")

    # --- Generate states ---
    print(f"Generating {args.n_states} states (seed={args.seed})...")
    t0 = time.time()
    states = generate_states(
        run_a.cfg.env,
        run_a.cfg.train.td_target,
        args.n_states,
        args.seed,
    )
    print(f"  Collected {len(states)} unique states in {time.time() - t0:.1f}s")

    # --- Compare ---
    print("Computing value predictions...")
    t0 = time.time()
    comparisons = run_comparison(run_a, run_b, states)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Quick summary
    abs_diffs = [c.team_abs_diff for c in comparisons]
    mean_abs = sum(abs_diffs) / len(abs_diffs)
    max_abs = max(abs_diffs)
    mean_diff = sum(c.team_diff for c in comparisons) / len(comparisons)
    mean_mag_a = sum(abs(c.team_value_a) for c in comparisons) / len(comparisons)
    mean_mag_b = sum(abs(c.team_value_b) for c in comparisons) / len(comparisons)
    mean_mag_avg = (mean_mag_a + mean_mag_b) / 2
    pct = (mean_abs / mean_mag_avg * 100) if mean_mag_avg > 1e-8 else float("inf")
    print(f"  Mean |V_team|: A={mean_mag_a:.4f}, B={mean_mag_b:.4f}")
    print(f"  Mean |diff| = {mean_abs:.4f} ({pct:.1f}%), max |diff| = {max_abs:.4f}, mean diff = {mean_diff:+.4f}")

    # --- Build report ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "comparison.html"

    print(f"Building HTML report...")
    t0 = time.time()
    build_report(comparisons, run_a, run_b, report_path, dpi=args.dpi)
    size_mb = report_path.stat().st_size / 1024 / 1024
    print(f"  Wrote {report_path} ({size_mb:.1f} MB) in {time.time() - t0:.1f}s")

    print(f"\nDone! Open {report_path} in a browser to view.")


if __name__ == "__main__":
    main()
