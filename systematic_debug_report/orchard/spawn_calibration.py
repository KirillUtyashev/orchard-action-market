"""Calibrate spawn/inventory settings against reward-event frequency.

This diagnostic varies stochastic spawn settings for an existing decentralized run
and measures how often agents receive nonzero immediate rewards under a fixed
behavior policy. It is meant to separate representation difficulty from reward
signal sparsity when changing relatedness width.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from orchard.offline_trained_value_eval import discover_run_dir
from orchard.supervised_reward_eval import collect_reward_trace
from orchard.value_reward_diagnostics import build_loaded_value_trainer


FIELDS = [
    "run_dir",
    "checkpoint_path",
    "checkpoint_step",
    "rel",
    "prof",
    "lr_label",
    "policy",
    "eval_seed",
    "rollout_steps",
    "recorded_transitions",
    "base_spawn_prob",
    "spawn_multiplier",
    "spawn_prob",
    "base_max_tasks_per_type",
    "max_tasks_multiplier",
    "max_tasks_per_type",
    "initial_tasks_per_type",
    "despawn_prob",
    "spawn_at_round_end",
    "reward_nonzero_freq_mean",
    "reward_nonzero_freq_std",
    "reward_nonzero_freq_min",
    "reward_nonzero_freq_max",
    "reward_nonzero_freq_by_agent",
    "reward_entry_nonzero_freq",
    "any_reward_transition_freq",
    "nonzero_reward_abs_mean",
    "reward_abs_mean",
    "task_inventory_mean",
    "task_inventory_std",
    "task_inventory_p95",
    "task_inventory_max",
    "mean_tasks_per_type",
    "max_mean_tasks_per_type",
    "frac_type_states_at_cap",
]


def _fmt_list(values: Sequence[float]) -> str:
    return ";".join(f"{float(v):.8g}" for v in values)


def _task_inventory_stats(states, n_task_types: int, max_tasks_per_type: int) -> dict[str, float]:
    totals = np.asarray([len(s.task_positions) for s in states], dtype=np.float64)
    per_type = np.zeros((len(states), n_task_types), dtype=np.float64)
    for idx, state in enumerate(states):
        if state.task_types is None:
            continue
        for tau in state.task_types:
            per_type[idx, int(tau)] += 1.0

    return {
        "task_inventory_mean": float(totals.mean()) if len(totals) else np.nan,
        "task_inventory_std": float(totals.std()) if len(totals) else np.nan,
        "task_inventory_p95": float(np.percentile(totals, 95)) if len(totals) else np.nan,
        "task_inventory_max": float(totals.max()) if len(totals) else np.nan,
        "mean_tasks_per_type": float(per_type.mean()) if per_type.size else np.nan,
        "max_mean_tasks_per_type": float(per_type.mean(axis=0).max()) if per_type.size else np.nan,
        "frac_type_states_at_cap": float((per_type >= max_tasks_per_type).mean()) if per_type.size else np.nan,
    }


def evaluate_setting(
    *,
    run_dir: Path,
    lr_label: str,
    checkpoint: str,
    device: str,
    policy: str,
    random_policy_prob: float,
    rollout_steps: int,
    eval_seed: int | None,
    spawn_multiplier: float,
    max_tasks_multiplier: float,
    base_spawn_prob: float | None,
    base_max_tasks_per_type: int | None,
) -> dict[str, object]:
    base_cfg, _base_env, _base_trainer, _base_ckpt, _base_step = build_loaded_value_trainer(
        run_dir,
        checkpoint=checkpoint,
        device=device,
    )
    base_spawn = float(base_cfg.env.stochastic.spawn_prob if base_spawn_prob is None else base_spawn_prob)
    base_max = int(base_cfg.env.max_tasks_per_type if base_max_tasks_per_type is None else base_max_tasks_per_type)
    spawn_prob = min(max(base_spawn * float(spawn_multiplier), 0.0), 1.0)
    max_tasks = max(1, int(math.ceil(base_max * float(max_tasks_multiplier))))

    overrides = [
        f"env.stochastic.spawn_prob={spawn_prob}",
        f"env.max_tasks_per_type={max_tasks}",
    ]
    cfg, env, trainer, ckpt_path, loaded_step = build_loaded_value_trainer(
        run_dir,
        checkpoint=checkpoint,
        device=device,
        overrides=overrides,
    )
    seed = eval_seed if eval_seed is not None else cfg.eval.eval_seed
    env.set_eval_mode(True, seed=seed)
    try:
        start = env.init_state()
        states, rewards = collect_reward_trace(
            start,
            trainer,
            env,
            cfg,
            rollout_steps=rollout_steps,
            policy=policy,
            random_policy_prob=random_policy_prob,
        )
    finally:
        env.set_eval_mode(False)

    rewards = np.asarray(rewards, dtype=np.float64)
    nonzero = np.abs(rewards) > 1e-12
    per_agent_freq = nonzero.mean(axis=0) if rewards.size else np.full(cfg.env.n_agents, np.nan)
    any_reward = nonzero.any(axis=1) if rewards.size else np.asarray([], dtype=bool)
    nonzero_abs = np.abs(rewards[nonzero]) if bool(nonzero.any()) else np.asarray([], dtype=np.float64)
    inventory = _task_inventory_stats(states, cfg.env.n_task_types, cfg.env.max_tasks_per_type)

    return {
        "run_dir": str(run_dir),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_step": loaded_step,
        "rel": cfg.env.relatedness_width,
        "prof": cfg.env.proficiency_width,
        "lr_label": lr_label,
        "policy": policy,
        "eval_seed": seed,
        "rollout_steps": int(rollout_steps),
        "recorded_transitions": len(states),
        "base_spawn_prob": base_spawn,
        "spawn_multiplier": float(spawn_multiplier),
        "spawn_prob": cfg.env.stochastic.spawn_prob,
        "base_max_tasks_per_type": base_max,
        "max_tasks_multiplier": float(max_tasks_multiplier),
        "max_tasks_per_type": cfg.env.max_tasks_per_type,
        "initial_tasks_per_type": cfg.env.n_tasks,
        "despawn_prob": cfg.env.stochastic.despawn_prob,
        "spawn_at_round_end": cfg.env.stochastic.spawn_at_round_end,
        "reward_nonzero_freq_mean": float(np.nanmean(per_agent_freq)),
        "reward_nonzero_freq_std": float(np.nanstd(per_agent_freq)),
        "reward_nonzero_freq_min": float(np.nanmin(per_agent_freq)),
        "reward_nonzero_freq_max": float(np.nanmax(per_agent_freq)),
        "reward_nonzero_freq_by_agent": _fmt_list(per_agent_freq),
        "reward_entry_nonzero_freq": float(nonzero.mean()) if rewards.size else np.nan,
        "any_reward_transition_freq": float(any_reward.mean()) if len(any_reward) else np.nan,
        "nonzero_reward_abs_mean": float(nonzero_abs.mean()) if len(nonzero_abs) else np.nan,
        "reward_abs_mean": float(np.abs(rewards).mean()) if rewards.size else np.nan,
        **inventory,
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep spawn_prob/max_tasks_per_type and measure reward event frequency.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["dec"], default="dec")
    parser.add_argument("--rels", type=int, nargs="+", default=[1])
    parser.add_argument("--target-rel", type=int, default=5, help="Also measure this rel at base spawn/max as a reference; set <0 to disable.")
    parser.add_argument("--prof", type=int, default=5)
    parser.add_argument("--lr-label", default="best")
    parser.add_argument("--final-score-evals", type=int, default=10)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--policy", choices=["greedy", "trained_greedy", "nearest", "eps_nearest", "nearest_rewarding_task", "eps_nearest_rewarding_task", "nearest_task", "random", "nearest_or_random"], default="eps_nearest")
    parser.add_argument("--random-policy-prob", type=float, default=0.5)
    parser.add_argument("--rollout-steps", type=int, default=10000)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--spawn-multipliers", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0])
    parser.add_argument("--max-task-multipliers", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0])
    parser.add_argument("--base-spawn-prob", type=float, default=None)
    parser.add_argument("--base-max-tasks-per-type", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows: list[dict[str, object]] = []

    if args.target_rel is not None and args.target_rel >= 0:
        target_run, target_lr = discover_run_dir(
            args.experiment_dir,
            mode=args.mode,
            rel=args.target_rel,
            prof=args.prof,
            lr_label=args.lr_label,
            final_score_evals=args.final_score_evals,
        )
        print(f"Measuring target baseline rel={args.target_rel} run={target_run}", flush=True)
        rows.append(evaluate_setting(
            run_dir=target_run,
            lr_label=target_lr,
            checkpoint=args.checkpoint,
            device=args.device,
            policy=args.policy,
            random_policy_prob=args.random_policy_prob,
            rollout_steps=args.rollout_steps,
            eval_seed=args.eval_seed,
            spawn_multiplier=1.0,
            max_tasks_multiplier=1.0,
            base_spawn_prob=args.base_spawn_prob,
            base_max_tasks_per_type=args.base_max_tasks_per_type,
        ))
        write_csv(args.output, rows)

    for rel in args.rels:
        run_dir, lr_label = discover_run_dir(
            args.experiment_dir,
            mode=args.mode,
            rel=rel,
            prof=args.prof,
            lr_label=args.lr_label,
            final_score_evals=args.final_score_evals,
        )
        for spawn_mult in args.spawn_multipliers:
            for max_mult in args.max_task_multipliers:
                print(
                    f"Measuring rel={rel} spawn_mult={spawn_mult:g} max_mult={max_mult:g} run={run_dir}",
                    flush=True,
                )
                rows.append(evaluate_setting(
                    run_dir=run_dir,
                    lr_label=lr_label,
                    checkpoint=args.checkpoint,
                    device=args.device,
                    policy=args.policy,
                    random_policy_prob=args.random_policy_prob,
                    rollout_steps=args.rollout_steps,
                    eval_seed=args.eval_seed,
                    spawn_multiplier=spawn_mult,
                    max_tasks_multiplier=max_mult,
                    base_spawn_prob=args.base_spawn_prob,
                    base_max_tasks_per_type=args.base_max_tasks_per_type,
                ))
                write_csv(args.output, rows)

    print(f"Wrote {len(rows)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
