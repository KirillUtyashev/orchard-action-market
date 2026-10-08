"""Summarize learned-actor runs: mean of the final 10 evaluations per run.

Usage: python summarize_learned_actor.py output/learned_actor_sanity

Expects <root>/<group>/seed<k>/<timestamp>/metrics.csv (the layout produced by
run_learned_actor_sanity.sh). For each group (e.g. dec, cen) it reports the mean
and standard deviation across seeds of the per-run mean over the last 10 rows,
for greedy, sampled and nearest-heuristic team reward per step.
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

COLUMNS = ["greedy_team_rps", "sampled_team_rps", "nearest_team_rps"]
LAST_N = 10


def run_value(metrics: Path, column: str) -> float | None:
    with open(metrics) as f:
        rows = [r for r in csv.DictReader(f) if r.get(column, "") != ""]
    if not rows:
        return None
    tail = rows[-LAST_N:]
    return sum(float(r[column]) for r in tail) / len(tail)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "output/learned_actor_sanity")
    groups: dict[str, list[Path]] = {}
    for metrics in sorted(root.glob("*/seed*/*/metrics.csv")):
        groups.setdefault(metrics.parents[2].name, []).append(metrics)
    if not groups:
        sys.exit(f"No metrics.csv found under {root}")
    for group, files in groups.items():
        print(f"{group}: {len(files)} run(s)")
        for col in COLUMNS:
            vals = [v for v in (run_value(m, col) for m in files) if v is not None]
            if not vals:
                continue
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"  {col:18s} mean={statistics.mean(vals):.4f} sd={sd:.4f} (n={len(vals)})")


if __name__ == "__main__":
    main()
