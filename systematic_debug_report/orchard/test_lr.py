"""Short-run diagnostic tool: test arbitrary hyperparameter combos quickly.

Each "job" is a tag (becomes folder name + plot label) plus config overrides.
Runs train() sequentially, logs sample_values.csv / details.csv / metrics.csv,
then the diagnostics notebook reads the output.

Usage:
    python -m orchard.test_lr --config base.yaml --steps 3000 \
        --output-dir /path/to/output \
        --jobs \
        "lam0.3_lr0.01   train.td_lambda=0.3  train.lr.start=0.01" \
        "lam0.8_lr3e-3   train.td_lambda=0.8  train.lr.start=3e-3" \
        "lam0.95_lr8e-4  train.td_lambda=0.95 train.lr.start=8.5e-4"

    # Or: auto-generate a grid from dimensions
    python -m orchard.test_lr --config base.yaml --steps 3000 \
        --output-dir /path/to/output \
        --grid \
        "train.td_lambda=0.3,0.8,0.95" \
        "train.lr.start=1e-3,3e-3,1e-2"

    # Replot without rerunning (prints summary, then use notebook):
    python -m orchard.test_lr --plot-only --output-dir /path/to/output

The diagnostics.ipynb notebook reads the same output directory.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path


def _parse_job(job_str: str) -> tuple[str, list[str]]:
    """Parse 'tag key=val key=val ...' into (tag, [overrides])."""
    parts = job_str.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Job must have at least a tag and one override: '{job_str}'")
    tag = parts[0]
    overrides = parts[1:]
    for o in overrides:
        if "=" not in o:
            raise ValueError(f"Override must be key=value, got: '{o}' in job '{tag}'")
    return tag, overrides


def _parse_grid(grid_specs: list[str]) -> list[tuple[str, list[str]]]:
    """Parse grid specs like 'train.td_lambda=0.3,0.8' into list of (tag, overrides).

    Each spec is 'key=val1,val2,...'. Grid is the cartesian product of all specs.
    """
    dimensions: list[list[tuple[str, str]]] = []
    dim_names: list[str] = []

    for spec in grid_specs:
        if "=" not in spec:
            raise ValueError(f"Grid spec must be key=val1,val2,...  Got: '{spec}'")
        key, vals_str = spec.split("=", 1)
        vals = [v.strip() for v in vals_str.split(",")]
        dimensions.append([(key, v) for v in vals])
        dim_names.append(key.split(".")[-1])

    jobs: list[tuple[str, list[str]]] = []
    for combo in itertools.product(*dimensions):
        tag_parts = []
        overrides = []
        for (key, val), dim_name in zip(combo, dim_names):
            tag_parts.append(f"{dim_name}{val}")
            overrides.append(f"{key}={val}")
        tag = "_".join(tag_parts)
        jobs.append((tag, overrides))

    return jobs


def _run_single(
    config_path: str,
    tag: str,
    steps: int,
    output_parent: Path,
    overrides: list[str],
) -> Path:
    """Run train() with given overrides, return the run directory."""
    from orchard.config import load_config
    from orchard.train import train

    tag_dir = output_parent / tag
    log_freq = max(steps // 10, 100)

    # Defaults for diagnostic mode (user overrides come last and win)
    all_overrides = [
        "train.lr.end=0",
        "train.lr.schedule=none",
        f"train.total_steps={steps}",
        f"logging.main_csv_freq={log_freq}",
        f"logging.detail_csv_freq={log_freq}",
        f"logging.output_dir={tag_dir}",
        "train.stopping_condition=none",
    ] + overrides

    cfg = load_config(config_path, all_overrides)
    train(cfg)

    subdirs = sorted([d for d in tag_dir.iterdir() if d.is_dir()])
    if not subdirs:
        raise FileNotFoundError(f"No run directory created in {tag_dir}")
    return subdirs[-1]


def _print_summary(results: dict[str, Path]) -> None:
    """Quick text summary."""
    import csv

    def read_csv(path):
        data = {}
        if not path.exists():
            return data
        with open(path) as f:
            for row in csv.DictReader(f):
                for k, v in row.items():
                    if k not in data:
                        data[k] = []
                    try:
                        data[k].append(float(v))
                    except (ValueError, TypeError):
                        data[k].append(float("nan"))
        return data

    print("\n" + "=" * 90)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 90)
    print(f"{'Tag':<30} {'V range start':<14} {'V range end':<14} {'W norm ratio':<14} {'TD loss end':<14}")
    print("-" * 90)

    for tag in sorted(results.keys()):
        run_dir = results[tag]
        d = read_csv(run_dir / "details.csv")
        s = read_csv(run_dir / "sample_values.csv")
        m = read_csv(run_dir / "metrics.csv")

        vr_start = vr_end = ""
        if s and "value_range" in s and s["value_range"]:
            vr_start = f"{s['value_range'][0]:.4f}"
            vr_end = f"{s['value_range'][-1]:.4f}"

        wnorm_ratio = ""
        if d:
            wk = [k for k in d if k.startswith("weight_norm_agent_0")]
            if wk:
                first = sum(d[k][0] for k in wk)
                last = sum(d[k][-1] for k in wk)
                wnorm_ratio = f"{last/first:.3f}" if first > 0 else "inf"

        td_end = ""
        if m and "td_loss_avg" in m and m["td_loss_avg"]:
            td_end = f"{m['td_loss_avg'][-1]:.6f}"

        print(f"{tag:<30} {vr_start:<14} {vr_end:<14} {wnorm_ratio:<14} {td_end:<14}")

    print("=" * 90)
    print("SIGNALS:")
    print("  V range growing         -> network differentiating states (good)")
    print("  V range flat            -> LR too low or traces too short")
    print("  W norm ratio >> 2       -> LR too high (weights exploding)")
    print("  W norm ratio ~ 1        -> LR may be too low")
    print("  See notebook for sample value traces and action distribution")
    print("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Short-run hyperparameter diagnostic for orchard RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explicit jobs:
  %(prog)s --config base.yaml --steps 3000 --output-dir ./test \\
      --jobs "lam0.3_lr0.01 train.td_lambda=0.3 train.lr.start=0.01" \\
             "lam0.8_lr3e-3 train.td_lambda=0.8 train.lr.start=3e-3"

  # Grid sweep (cartesian product):
  %(prog)s --config base.yaml --steps 3000 --output-dir ./test \\
      --grid "train.td_lambda=0.3,0.8,0.95" "train.lr.start=1e-3,3e-3,1e-2"

  # Print summary of existing results:
  %(prog)s --plot-only --output-dir ./test
        """,
    )
    parser.add_argument("--config", help="Base YAML config")
    parser.add_argument("--steps", type=int, default=3000, help="Steps per job (default: 3000)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--jobs", nargs="+", default=None,
        help="Explicit jobs: 'tag key=val key=val ...'",
    )
    parser.add_argument(
        "--grid", nargs="+", default=None,
        help="Grid specs: 'key=val1,val2,...' (cartesian product)",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Extra overrides applied to ALL jobs",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip training, print summary of existing results",
    )
    args = parser.parse_args()

    output_parent = Path(args.output_dir)
    output_parent.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        results = {}
        for subdir in sorted(output_parent.iterdir()):
            if not subdir.is_dir():
                continue
            for d in sorted(subdir.iterdir()):
                if d.is_dir() and (d / "metrics.csv").exists():
                    results[subdir.name] = d
                    break
        if results:
            _print_summary(results)
        else:
            print(f"No completed runs found in {output_parent}")
        return

    if args.config is None:
        parser.error("--config is required when not using --plot-only")

    # Parse jobs
    jobs: list[tuple[str, list[str]]] = []
    if args.jobs:
        jobs = [_parse_job(j) for j in args.jobs]
    elif args.grid:
        jobs = _parse_grid(args.grid)
    else:
        parser.error("Must specify either --jobs or --grid")

    print(f"Running {len(jobs)} diagnostic job(s), {args.steps} steps each:")
    for tag, overrides in jobs:
        print(f"  {tag}: {' '.join(overrides)}")
    print()

    results: dict[str, Path] = {}
    t0 = time.time()
    for i, (tag, overrides) in enumerate(jobs):
        all_overrides = overrides + args.override
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(jobs)}] {tag}")
        print(f"  Overrides: {' '.join(all_overrides)}")
        print(f"{'='*60}")

        # Skip if already completed
        tag_dir = output_parent / tag
        existing = None
        if tag_dir.exists():
            for d in sorted(tag_dir.iterdir()):
                if d.is_dir() and (d / "metrics.csv").exists():
                    existing = d
                    break
        if existing:
            print(f"  Already completed, skipping. Run dir: {existing}")
            results[tag] = existing
            continue

        run_dir = _run_single(args.config, tag, args.steps, output_parent, all_overrides)
        results[tag] = run_dir

    elapsed = time.time() - t0
    print(f"\nAll done in {elapsed:.0f}s")
    _print_summary(results)
    print(f"\nOpen diagnostics.ipynb with ROOT_DIR = \"{output_parent}\"")


if __name__ == "__main__":
    main()
