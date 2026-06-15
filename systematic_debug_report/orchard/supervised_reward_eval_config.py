"""Config-driven supervised immediate-reward prediction diagnostic.

Unlike orchard.supervised_reward_eval, this entrypoint does not require a trained
checkpoint/run directory. It builds an environment from a YAML config, collects a
rollout under a fixed heuristic/random policy, and trains decentralized reward
predictors on the resulting immediate rewards.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

import orchard.encoding as encoding
from orchard.config import load_config
from orchard.env import create_env
from orchard.seed import set_all_seeds
from orchard.supervised_reward_eval import (
    REWARD_HISTORY_FIELDS,
    REWARD_METRIC_FIELDS,
    RewardRollout,
    _load_cached_rollout,
    _save_cached_rollout,
    collect_reward_trace,
    encoder_name,
    metric_rows,
    parse_conv_specs_arg,
    parse_encoder_arg,
    parse_mlp_dims_arg,
    predict_decentralized_rewards,
    reward_stats_by_agent,
    split_contiguous,
    train_decentralized_reward_models,
)
from orchard.offline_trained_value_eval import normalize_targets


def prepare_config_rollout(
    *,
    config: Path,
    overrides: Sequence[str],
    rollout_steps: int,
    drop_tail_transitions: int,
    eval_seed: int | None,
    policy: str,
    random_policy_prob: float,
    cache_path: Path | None,
    force_cache: bool,
) -> tuple[object, object, RewardRollout]:
    if policy in {"greedy", "trained_greedy"}:
        raise ValueError("Config-driven reward eval does not load a checkpoint; use a heuristic/random policy.")

    cfg = load_config(config, overrides=list(overrides))
    set_all_seeds(cfg.train.seed if eval_seed is None else eval_seed)
    env = create_env(cfg.env)

    if cache_path is not None and cache_path.exists() and not force_cache:
        return cfg, env, _load_cached_rollout(cache_path)

    seed = eval_seed if eval_seed is not None else cfg.eval.eval_seed
    env.set_eval_mode(True, seed=seed)
    try:
        start = env.init_state()
        states, rewards, picked_task_types = collect_reward_trace(
            start,
            None,
            env,
            cfg,
            rollout_steps=rollout_steps,
            policy=policy,
            random_policy_prob=random_policy_prob,
        )
    finally:
        env.set_eval_mode(False)

    recorded = len(states)
    keep_end = max(recorded - max(int(drop_tail_transitions), 0), 0)
    if keep_end <= 1:
        raise ValueError(
            f"Not enough transitions after tail drop: recorded={recorded}, drop_tail={drop_tail_transitions}"
        )
    rollout = RewardRollout(
        states=states[:keep_end],
        rewards=np.asarray(rewards[:keep_end], dtype=np.float64),
        picked_task_types=picked_task_types[:keep_end],
        recorded_transitions=recorded,
        kept_transitions=keep_end,
    )
    if cache_path is not None:
        _save_cached_rollout(cache_path, rollout, {
            "config": str(config),
            "overrides": list(overrides),
            "rel": cfg.env.relatedness_width,
            "prof": cfg.env.proficiency_width,
            "rollout_steps": int(rollout_steps),
            "drop_tail_transitions": int(drop_tail_transitions),
            "eval_seed": seed,
            "policy": policy,
            "random_policy_prob": random_policy_prob,
        })
    return cfg, env, rollout


def evaluate_config(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cfg, env, rollout = prepare_config_rollout(
        config=args.config,
        overrides=args.override,
        rollout_steps=args.rollout_steps,
        drop_tail_transitions=args.drop_tail_transitions,
        eval_seed=args.eval_seed,
        policy=args.policy,
        random_policy_prob=args.random_policy_prob,
        cache_path=args.rollout_cache,
        force_cache=args.force_rollout_cache,
    )
    split = split_contiguous(rollout, train_frac=args.train_frac)
    train_norm, mean, std = normalize_targets(split.train_rewards, split.train_rewards)
    test_norm, _mean2, _std2 = normalize_targets(split.train_rewards, split.test_rewards)

    model_cfg = replace(
        cfg.model,
        encoder=parse_encoder_arg(args.encoder, cfg.model.encoder),
        conv_specs=parse_conv_specs_arg(args.conv_specs, cfg.model.conv_specs),
        mlp_dims=parse_mlp_dims_arg(args.mlp_dims, cfg.model.mlp_dims),
    )
    encoding.init_encoder(model_cfg.encoder, env, n_networks=cfg.env.n_agents)

    device = torch.device(args.device)
    nets, history = train_decentralized_reward_models(
        split.train_states,
        train_norm,
        model_cfg=model_cfg,
        env_cfg=cfg.env,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
        val_states=split.test_states,
        val_targets_norm=test_norm,
        log_interval=args.log_interval,
    )
    pred_norm = predict_decentralized_rewards(nets, split.test_states, batch_size=args.batch_size, device=device)
    pred_raw = pred_norm * std + mean
    train_reward_freq, _train_reward_nonzero_abs, _train_reward_abs = reward_stats_by_agent(split.train_rewards)
    reward_freq, reward_nonzero_abs, reward_abs = reward_stats_by_agent(split.test_rewards)
    val_pickup_mask = np.asarray([state.pick_phase for state in split.test_states], dtype=bool)

    common = {
        "run_dir": str(args.config),
        "checkpoint_path": "",
        "checkpoint_step": "",
        "rel": cfg.env.relatedness_width,
        "prof": cfg.env.proficiency_width,
        "lr_label": args.lr_label,
        "reward_lr": args.lr,
        "model_tag": args.model_tag,
        "encoder": encoder_name(model_cfg.encoder),
        "rollout_policy": args.policy,
        "rollout_steps": int(args.rollout_steps),
        "recorded_transitions": rollout.recorded_transitions,
        "kept_transitions": rollout.kept_transitions,
        "drop_tail_transitions": int(args.drop_tail_transitions),
        "train_transitions": len(split.train_states),
        "test_transitions": len(split.test_states),
        "train_frac": args.train_frac,
        "policy": args.policy,
        "eval_seed": args.eval_seed if args.eval_seed is not None else cfg.eval.eval_seed,
        "train_steps": int(args.train_steps),
        "batch_size": int(args.batch_size),
        "log_interval": int(args.log_interval),
        "device": args.device,
    }
    rows = metric_rows(
        common=common,
        raw_pred=pred_raw,
        norm_pred=pred_norm,
        raw_target=split.test_rewards,
        norm_target=test_norm,
        target_mean=mean,
        target_std=std,
        reward_freq=reward_freq,
        reward_nonzero_abs=reward_nonzero_abs,
        reward_abs=reward_abs,
        train_reward_freq=train_reward_freq,
        val_pickup_mask=val_pickup_mask,
        val_picked_task_types=split.test_picked_task_types,
        n_task_types=cfg.env.n_task_types,
    )
    history_rows = []
    for item in history:
        history_rows.append({
            "run_dir": str(args.config),
            "rel": cfg.env.relatedness_width,
            "prof": cfg.env.proficiency_width,
            "lr_label": args.lr_label,
            "reward_lr": args.lr,
            "model_tag": args.model_tag,
            "encoder": encoder_name(model_cfg.encoder),
            "rollout_policy": args.policy,
            "model_type": item["model_type"],
            "step": item["step"],
            "train_loss": item["train_loss"],
            "val_loss": item["val_loss"],
            "train_transitions": len(split.train_states),
            "test_transitions": len(split.test_states),
            "batch_size": int(args.batch_size),
            "device": args.device,
        })
    return rows, history_rows


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: Sequence[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven supervised immediate reward prediction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--lr-label", default="config")
    parser.add_argument("--policy", choices=["nearest", "eps_nearest", "nearest_rewarding_task", "eps_nearest_rewarding_task", "nearest_task", "random", "nearest_or_random"], default="eps_nearest")
    parser.add_argument("--random-policy-prob", type=float, default=0.5)
    parser.add_argument("--rollout-steps", type=int, default=10000)
    parser.add_argument("--drop-tail-transitions", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--model-tag", default="config_model")
    parser.add_argument("--encoder", choices=["everything_cnn_grid", "filtered_dec_cnn_grid"], default=None)
    parser.add_argument("--conv-specs", default=None)
    parser.add_argument("--mlp-dims", default=None)
    parser.add_argument("--log-interval", type=int, default=250)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--rollout-cache", type=Path, default=None)
    parser.add_argument("--force-rollout-cache", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--history-output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.collect_only:
        if args.rollout_cache is None:
            raise ValueError("--collect-only requires --rollout-cache")
        _cfg, _env, rollout = prepare_config_rollout(
            config=args.config,
            overrides=args.override,
            rollout_steps=args.rollout_steps,
            drop_tail_transitions=args.drop_tail_transitions,
            eval_seed=args.eval_seed,
            policy=args.policy,
            random_policy_prob=args.random_policy_prob,
            cache_path=args.rollout_cache,
            force_cache=args.force_rollout_cache,
        )
        print(f"Cached rollout with {rollout.kept_transitions} transitions at {args.rollout_cache}", flush=True)
        return
    if args.output is None:
        raise ValueError("--output is required unless --collect-only is set")
    rows, history_rows = evaluate_config(args)
    history_output = args.history_output or args.output.with_name(args.output.stem + "_history.csv")
    write_csv(args.output, rows, REWARD_METRIC_FIELDS)
    write_csv(history_output, history_rows, REWARD_HISTORY_FIELDS)
    for row in rows:
        if str(row.get("agent")) == "mean" and row.get("model_type") == "decentralized_reward_mean":
            print(
                f"{row['model_type']} lr={args.lr:g}: norm_rmse={float(row['normalized_rmse']):.4f}, "
                f"raw_rmse={float(row['raw_rmse']):.4f}, r2={float(row['r2']):.4f}, "
                f"reward_freq={float(row['reward_nonzero_freq']):.4f}",
                flush=True,
            )
    print(f"Wrote {len(rows)} rows to {args.output}", flush=True)
    print(f"Wrote {len(history_rows)} history rows to {history_output}", flush=True)


if __name__ == "__main__":
    main()
