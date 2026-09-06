"""CLI entry point: python -m orchard.compare_policies RUN_A RUN_B [RUN_C ...] [options]"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from orchard.compare_values.loader import load_run
from orchard.compare_policies.compare import (
    generate_states, run_comparison, add_heuristic_policy, add_nearest_policy,
)
from orchard.compare_policies.report import build_report
from orchard.compare_values.compare import validate_env_compatibility, validate_td_target_compatibility
from orchard.enums import Heuristic


_HEURISTIC_MAP = {
    "nearest_task": Heuristic.NEAREST_TASK,
    "nearest_correct_task": Heuristic.NEAREST_CORRECT_TASK,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare greedy policies from multiple trained orchard runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m orchard.compare_policies runs/dec_cnn64_lr1e-3 runs/cen_cnn64_lr1e-3
  python -m orchard.compare_policies runs/a runs/b runs/c --labels "Dec 64" "Cen 64" "Dec 8"
  python -m orchard.compare_policies runs/a runs/b --checkpoint step_500000.pt --n-states 200
  python -m orchard.compare_policies runs/a runs/b --heuristic nearest_correct_task
""",
    )
    p.add_argument("runs", nargs="+", type=str,
                   help="Paths to run directories (2 or more)")
    p.add_argument("--checkpoint", type=str, default="final.pt",
                   help="Checkpoint filename in each run's checkpoints/ dir (default: final.pt)")
    p.add_argument("--labels", nargs="+", type=str, default=None,
                   help="Custom labels for each run (must match number of runs)")
    p.add_argument("--n-states", type=int, default=200,
                   help="Number of states to compare (default: 200)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for state generation (default: 42)")
    p.add_argument("--output-dir", type=str, default="./compare_policies_output/",
                   help="Output directory (default: ./compare_policies_output/)")
    p.add_argument("--dpi", type=int, default=100,
                   help="DPI for state renderings (default: 100)")
    p.add_argument("--match-training", action="store_true",
                   help="Use same 100 random-action sample states as train.py (seed from config)")
    p.add_argument("--nearest", action="store_true",
                   help="Include nearest-task heuristic policy as an extra comparison column")
    p.add_argument("--heuristic", type=str, default=None,
                   choices=list(_HEURISTIC_MAP.keys()),
                   help="Include heuristic policy as extra column (auto-detected from config if not set)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    min_runs = 1 if (args.nearest or args.heuristic) else 2
    if len(args.runs) < min_runs:
        print(f"Error: need at least {min_runs} runs to compare"
              + (" (or pass --nearest/--heuristic with 1 run)" if not (args.nearest or args.heuristic) else ""))
        return

    if args.labels and len(args.labels) != len(args.runs):
        print(f"Error: {len(args.labels)} labels but {len(args.runs)} runs")
        return

    # --- Load runs ---
    loaded = []
    for i, run_path in enumerate(args.runs):
        print(f"Loading run {i}: {run_path}")
        rp = Path(run_path)
        if not (rp / "metadata.yaml").exists():
            subdirs = sorted([d for d in rp.iterdir() if d.is_dir()])
            if subdirs:
                rp = subdirs[-1]
        run = load_run(rp, args.checkpoint)
        loaded.append(run)
        print(f"  → {run.label}")

    # --- Labels ---
    if args.labels:
        labels = list(args.labels)
    else:
        labels = []
        for i, run in enumerate(loaded):
            lt = "cen" if run.is_centralized else "dec"
            labels.append(f"{lt} (step {run.checkpoint_step})")

    # --- Validate compatibility ---
    if len(loaded) >= 2:
        print("Validating env compatibility...")
        validate_env_compatibility(loaded)
        validate_td_target_compatibility(loaded)
        print("  ✓ All envs compatible")

    ref = loaded[0]

    # --- Generate states ---
    if args.match_training:
        from orchard.compare_policies.compare import generate_training_sample_states
        print(f"Generating training sample states (seed={ref.cfg.train.seed}, random actions)...")
        t0 = time.time()
        states = generate_training_sample_states(ref, n_sample=100)
    else:
        print(f"Generating {args.n_states} states (seed={args.seed})...")
        t0 = time.time()
        states = generate_states(ref, args.n_states, args.seed)
    print(f"  Collected {len(states)} states in {time.time() - t0:.1f}s")

    # --- Compare ---
    print(f"Computing greedy actions for {len(loaded)} runs × {len(states)} states...")
    t0 = time.time()
    comparisons = run_comparison(loaded, states)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # --- Optionally add heuristic policy ---
    if args.heuristic:
        h = _HEURISTIC_MAP[args.heuristic]
        hname = args.heuristic.replace("_", " ").title()
        print(f"Adding {hname} heuristic policy...")
        add_heuristic_policy(comparisons, ref.cfg.env, h)
        labels.append(hname)
    elif args.nearest:
        # Auto-detect: use nearest_correct_task for multi-type
        if ref.cfg.env.n_task_types > 1:
            print("Adding nearest_correct_task heuristic policy...")
            add_heuristic_policy(comparisons, ref.cfg.env, Heuristic.NEAREST_CORRECT_TASK)
            labels.append("Nearest Correct Task")
        else:
            print("Adding nearest-task heuristic policy...")
            add_nearest_policy(comparisons, ref.cfg.env)
            labels.append("Nearest Task")

    # Summary
    n_agree = sum(1 for c in comparisons if c.agrees)
    n_disagree = sum(1 for c in comparisons if not c.agrees)
    pct = n_agree / len(comparisons) * 100
    print(f"  Agreement: {n_agree}/{len(comparisons)} ({pct:.1f}%)")
    print(f"  Disagreements: {n_disagree}")

    if n_disagree > 0:
        disagree = [c for c in comparisons if not c.agrees]
        gaps = [c.q_gap for c in disagree]
        print(f"  Q-gap on disagreements: mean={sum(gaps)/len(gaps):.6f}, "
              f"max={max(gaps):.6f}, min={min(gaps):.6f}")

    # --- Build report ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "policy_comparison.html"

    print(f"Building HTML report...")
    t0 = time.time()
    build_report(comparisons, loaded, labels, report_path, dpi=args.dpi)
    size_mb = report_path.stat().st_size / 1024 / 1024
    print(f"  Wrote {report_path} ({size_mb:.1f} MB) in {time.time() - t0:.1f}s")

    print(f"\nDone! Open {report_path} in a browser.")


if __name__ == "__main__":
    main()
