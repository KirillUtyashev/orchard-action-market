"""Evaluate existing checkpoint value functions on trained-policy rollouts.

This is the checkpoint-backed companion to ``offline_trained_value_eval``. It
loads the current learned decentralized critics, rolls out the learned greedy
policy, computes Monte Carlo return-to-go targets, and reports raw/normalized
value-prediction error without training any new supervised model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from orchard.offline_trained_value_eval import discover_run_dir
from orchard.value_reward_diagnostics import (
    evaluate_value_reward_diagnostics,
    write_value_reward_csv,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate learned checkpoint value accuracy.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["dec"], default="dec")
    parser.add_argument("--rel", type=int, default=None)
    parser.add_argument("--prof", type=int, default=5)
    parser.add_argument("--lr-label", default="best")
    parser.add_argument("--final-score-evals", type=int, default=10)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--policy", choices=["greedy"], default="greedy")
    parser.add_argument("--rollout-steps", type=int, default=10000)
    parser.add_argument("--drop-tail-transitions", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.run_dir is None:
        if args.experiment_dir is None or args.rel is None:
            raise ValueError("Provide either --run-dir or both --experiment-dir and --rel")
        run_dir, lr_label = discover_run_dir(
            args.experiment_dir,
            mode=args.mode,
            rel=args.rel,
            prof=args.prof,
            lr_label=args.lr_label,
            final_score_evals=args.final_score_evals,
        )
    else:
        run_dir = args.run_dir
        lr_label = args.lr_label if args.lr_label != "best" else run_dir.parent.name

    df = evaluate_value_reward_diagnostics(
        run_dir,
        checkpoint=args.checkpoint,
        policy=args.policy,
        rollout_steps=args.rollout_steps,
        drop_tail_transitions=args.drop_tail_transitions,
        eval_seed=args.eval_seed,
        device=args.device,
    )
    df["lr_label"] = lr_label
    write_value_reward_csv(args.output, df)

    summary = df[df["agent"].astype(str) == "mean"]
    rows = summary if not summary.empty else df.head(1)
    for _, row in rows.iterrows():
        print(
            f"{row['model_type']} agent={row['agent']}: "
            f"norm_rmse={float(row['normalized_rmse']):.4f}, "
            f"raw_rmse={float(row['raw_rmse']):.4f}, "
            f"r2={float(row['r2']):.4f}, "
            f"reward_nonzero_freq={float(row['reward_nonzero_freq']):.4f}"
        )
    print(f"Run dir: {run_dir}")
    print(f"LR label: {lr_label}")
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
