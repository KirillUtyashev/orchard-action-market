"""Supervised immediate-reward prediction diagnostic.

Loads a trained decentralized checkpoint as a fixed behavior policy, collects the
same decomposed critic-state transition stream used by value learning, and trains
fresh supervised regressors to predict immediate per-agent rewards from each
agent's decentralized observation.
"""

from __future__ import annotations

import argparse
import ast
import csv
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

import orchard.encoding as encoding
from orchard.datatypes import State
from orchard.enums import EncoderType
from orchard.model import ValueNetwork
from orchard.offline_trained_value_eval import (
    batch_encode_all_agents,
    discover_run_dir,
    normalize_targets,
)
from orchard.offline_value_eval import choose_behavior_action
from orchard.value_reward_diagnostics import build_loaded_value_trainer


REWARD_METRIC_FIELDS = [
    "run_dir",
    "checkpoint_path",
    "checkpoint_step",
    "rel",
    "prof",
    "lr_label",
    "reward_lr",
    "model_tag",
    "encoder",
    "rollout_policy",
    "model_type",
    "agent",
    "task_type",
    "n_val_cases",
    "raw_mse",
    "raw_rmse",
    "raw_mae",
    "raw_bias",
    "normalized_mse",
    "normalized_rmse",
    "normalized_mae",
    "normalized_bias",
    "val_mse_zero_cases",
    "val_mse_nonzero_cases",
    "val_mse_pickup_cases",
    "fraction_nonzero_train",
    "fraction_nonzero_val",
    "r2",
    "target_mean",
    "target_std",
    "reward_nonzero_freq",
    "reward_nonzero_abs_mean",
    "reward_abs_mean",
    "rollout_steps",
    "recorded_transitions",
    "kept_transitions",
    "drop_tail_transitions",
    "train_transitions",
    "test_transitions",
    "train_frac",
    "policy",
    "eval_seed",
    "train_steps",
    "batch_size",
    "log_interval",
    "device",
]

REWARD_HISTORY_FIELDS = [
    "run_dir",
    "rel",
    "prof",
    "lr_label",
    "reward_lr",
    "model_tag",
    "encoder",
    "rollout_policy",
    "model_type",
    "step",
    "train_loss",
    "val_loss",
    "train_transitions",
    "test_transitions",
    "batch_size",
    "device",
]


@dataclass(frozen=True)
class RewardRollout:
    states: list[State]
    rewards: np.ndarray
    picked_task_types: np.ndarray
    recorded_transitions: int
    kept_transitions: int


@dataclass(frozen=True)
class RewardSplit:
    train_states: list[State]
    test_states: list[State]
    train_rewards: np.ndarray
    test_rewards: np.ndarray
    train_picked_task_types: np.ndarray
    test_picked_task_types: np.ndarray


def _select_rollout_action(state: State, env, trainer, *, policy: str, random_policy_prob: float):
    if policy in {"greedy", "trained_greedy"}:
        return trainer._greedy_action(state)
    return choose_behavior_action(
        state,
        env,
        policy=policy,
        random_policy_prob=random_policy_prob,
    )


def collect_reward_trace(
    start: State,
    trainer,
    env,
    cfg,
    *,
    rollout_steps: int,
    policy: str,
    random_policy_prob: float,
) -> tuple[list[State], np.ndarray, np.ndarray]:
    """Collect value-learning critic states and immediate rewards under a behavior policy."""
    state = start
    prev_critic_state: State | None = None
    critic_states: list[State] = []
    rewards: list[tuple[float, ...]] = []
    picked_task_types: list[int] = []

    for _ in range(max(int(rollout_steps), 0)):
        move_action = _select_rollout_action(
            state,
            env,
            trainer,
            policy=policy,
            random_policy_prob=random_policy_prob,
        )
        if not move_action.is_move():
            raise AssertionError(f"Move phase returned non-move action: {move_action}")

        moved = env.apply_action(state, move_action)
        actor = moved.actor
        eligible_types = env.proficiency_positive_types[actor]
        on_task = moved.is_agent_on_task(actor, eligible_types)
        move_critic_state = moved.with_pick_phase() if on_task else moved

        if prev_critic_state is not None:
            critic_states.append(prev_critic_state)
            rewards.append(tuple(0.0 for _ in range(cfg.env.n_agents)))
            picked_task_types.append(-1)

        if on_task:
            pick_action = _select_rollout_action(
                move_critic_state,
                env,
                trainer,
                policy=policy,
                random_policy_prob=random_policy_prob,
            )
            requested_pick_type = pick_action.pick_type() if pick_action.is_pick() else None
            picked, pick_rewards = env.resolve_pick(
                moved,
                pick_type=requested_pick_type,
            )
            picked_task_type = (
                int(requested_pick_type)
                if requested_pick_type is not None and len(picked.task_positions) < len(moved.task_positions)
                else -1
            )
            critic_states.append(move_critic_state)
            rewards.append(pick_rewards)
            picked_task_types.append(picked_task_type)
            prev_critic_state = picked
            post_pick = picked
        else:
            prev_critic_state = move_critic_state
            post_pick = moved

        state = env.advance_actor(env.spawn_and_despawn(post_pick))

    return (
        critic_states,
        np.asarray(rewards, dtype=np.float64),
        np.asarray(picked_task_types, dtype=np.int64),
    )


def _load_cached_rollout(cache_path: Path) -> RewardRollout:
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    return RewardRollout(
        states=payload["states"],
        rewards=np.asarray(payload["rewards"], dtype=np.float64),
        picked_task_types=np.asarray(
            payload.get("picked_task_types", np.full(len(payload["states"]), -1, dtype=np.int64)),
            dtype=np.int64,
        ),
        recorded_transitions=int(payload["recorded_transitions"]),
        kept_transitions=int(payload["kept_transitions"]),
    )


def _save_cached_rollout(cache_path: Path, rollout: RewardRollout, metadata: dict[str, object]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **metadata,
        "states": rollout.states,
        "rewards": rollout.rewards,
        "picked_task_types": rollout.picked_task_types,
        "recorded_transitions": rollout.recorded_transitions,
        "kept_transitions": rollout.kept_transitions,
    }
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_path)


def prepare_reward_rollout(
    run_dir: Path,
    *,
    checkpoint: str,
    rollout_steps: int,
    drop_tail_transitions: int,
    eval_seed: int | None,
    behavior_device: str,
    policy: str,
    random_policy_prob: float,
    cache_path: Path | None = None,
    force_cache: bool = False,
) -> tuple[object, object, object, Path, int | None, RewardRollout]:
    cfg, env, behavior_trainer, ckpt_path, loaded_step = build_loaded_value_trainer(
        run_dir,
        checkpoint=checkpoint,
        device=behavior_device,
    )
    if cache_path is not None and cache_path.exists() and not force_cache:
        rollout = _load_cached_rollout(cache_path)
        return cfg, env, behavior_trainer, ckpt_path, loaded_step, rollout

    seed = eval_seed if eval_seed is not None else cfg.eval.eval_seed
    env.set_eval_mode(True, seed=seed)
    try:
        start = env.init_state()
        states, rewards, picked_task_types = collect_reward_trace(
            start,
            behavior_trainer,
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
        rewards=rewards[:keep_end],
        picked_task_types=picked_task_types[:keep_end],
        recorded_transitions=recorded,
        kept_transitions=keep_end,
    )
    if cache_path is not None:
        _save_cached_rollout(cache_path, rollout, {
            "run_dir": str(run_dir),
            "checkpoint_path": str(ckpt_path),
            "checkpoint_step": loaded_step,
            "rel": cfg.env.relatedness_width,
            "prof": cfg.env.proficiency_width,
            "rollout_steps": int(rollout_steps),
            "drop_tail_transitions": int(drop_tail_transitions),
            "eval_seed": seed,
            "policy": policy,
            "random_policy_prob": random_policy_prob,
        })
    return cfg, env, behavior_trainer, ckpt_path, loaded_step, rollout

def split_contiguous(rollout: RewardRollout, *, train_frac: float) -> RewardSplit:
    n = len(rollout.states)
    if n < 2:
        raise ValueError("Need at least two transitions for train/test split")
    n_train = int(np.floor(n * train_frac))
    n_train = min(max(n_train, 1), n - 1)
    return RewardSplit(
        train_states=rollout.states[:n_train],
        test_states=rollout.states[n_train:],
        train_rewards=rollout.rewards[:n_train],
        test_rewards=rollout.rewards[n_train:],
        train_picked_task_types=rollout.picked_task_types[:n_train],
        test_picked_task_types=rollout.picked_task_types[n_train:],
    )


def predict_decentralized_rewards(
    nets: Sequence[ValueNetwork],
    states: Sequence[State],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    preds = np.zeros((len(states), len(nets)), dtype=np.float64)
    with torch.no_grad():
        indices = np.arange(len(states))
        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = indices[start:start + batch_size]
            grids, scalars = batch_encode_all_agents(states, batch_idx, device=device)
            for agent, net in enumerate(nets):
                p = net.forward_raw(grids[:, agent], scalars[:, agent]).detach().cpu().numpy()
                preds[start:start + len(batch_idx), agent] = p
    return preds


def train_decentralized_reward_models(
    states: Sequence[State],
    targets_norm: np.ndarray,
    *,
    model_cfg,
    env_cfg,
    train_steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    val_states: Sequence[State],
    val_targets_norm: np.ndarray,
    log_interval: int,
) -> tuple[list[ValueNetwork], list[dict[str, float | int | str]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    nets = [ValueNetwork(model_cfg, env_cfg, td_lambda=0.0).to(device) for _ in range(env_cfg.n_agents)]
    opt = torch.optim.Adam([p for net in nets for p in net.parameters()], lr=lr)
    history: list[dict[str, float | int | str]] = []
    train_losses: list[float] = []
    n = len(states)
    log_every = max(int(log_interval), 1)
    for net in nets:
        net.train()

    for step in range(1, max(int(train_steps), 0) + 1):
        batch_idx = rng.integers(0, n, size=max(int(batch_size), 1))
        grids, scalars = batch_encode_all_agents(states, batch_idx, device=device)
        y = torch.as_tensor(targets_norm[batch_idx], dtype=torch.float32, device=device)
        losses = []
        for agent, net in enumerate(nets):
            pred = net.forward_raw(grids[:, agent], scalars[:, agent])
            losses.append(torch.mean((pred - y[:, agent]) ** 2))
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train_losses.append(float(loss.detach().cpu().item()))

        if step == 1 or step % log_every == 0 or step == train_steps:
            for net in nets:
                net.eval()
            val_pred = predict_decentralized_rewards(nets, val_states, batch_size=batch_size, device=device)
            val_loss = float(np.mean((val_pred - val_targets_norm) ** 2))
            history.append({
                "model_type": "decentralized_reward",
                "step": step,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
            })
            print(
                f"step={step} lr={lr:g} train_loss={float(np.mean(train_losses)):.6g} "
                f"val_loss={val_loss:.6g}",
                flush=True,
            )
            train_losses.clear()
            for net in nets:
                net.train()

    for net in nets:
        net.eval()
    return nets, history


def reward_stats_by_agent(rewards: np.ndarray, *, zero_tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nonzero = np.abs(rewards) > zero_tol
    freq = nonzero.mean(axis=0)
    abs_rewards = np.abs(rewards)
    abs_mean = abs_rewards.mean(axis=0)
    nonzero_abs_mean = np.full(rewards.shape[1], np.nan, dtype=np.float64)
    for agent in range(rewards.shape[1]):
        mask = nonzero[:, agent]
        if bool(mask.any()):
            nonzero_abs_mean[agent] = float(abs_rewards[mask, agent].mean())
    return freq, nonzero_abs_mean, abs_mean


def metric_rows(
    *,
    common: dict[str, object],
    raw_pred: np.ndarray,
    norm_pred: np.ndarray,
    raw_target: np.ndarray,
    norm_target: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    reward_freq: np.ndarray,
    reward_nonzero_abs: np.ndarray,
    reward_abs: np.ndarray,
    train_reward_freq: np.ndarray | None = None,
    val_pickup_mask: np.ndarray | None = None,
    val_picked_task_types: np.ndarray | None = None,
    n_task_types: int | None = None,
    zero_tol: float = 1e-12,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    per_agent_raw_mse = []
    per_agent_norm_mse = []
    if train_reward_freq is None:
        train_reward_freq = np.full(raw_target.shape[1], np.nan, dtype=np.float64)
    if val_pickup_mask is None:
        val_pickup_mask = np.zeros(raw_target.shape[0], dtype=bool)
    else:
        val_pickup_mask = np.asarray(val_pickup_mask, dtype=bool)
    if val_picked_task_types is None:
        val_picked_task_types = np.full(raw_target.shape[0], -1, dtype=np.int64)
    else:
        val_picked_task_types = np.asarray(val_picked_task_types, dtype=np.int64)
    if n_task_types is None:
        n_task_types = int(val_picked_task_types[val_picked_task_types >= 0].max() + 1) if bool((val_picked_task_types >= 0).any()) else 0

    def masked_mse(values: np.ndarray, mask: np.ndarray) -> float:
        if not bool(mask.any()):
            return np.nan
        return float(np.mean(values[mask] ** 2))

    def safe_nanmean(values: Iterable[object]) -> float:
        arr = np.asarray(list(values), dtype=np.float64)
        if arr.size == 0 or bool(np.all(np.isnan(arr))):
            return np.nan
        return float(np.nanmean(arr))

    for agent in range(raw_target.shape[1]):
        err = raw_pred[:, agent] - raw_target[:, agent]
        norm_err = norm_pred[:, agent] - norm_target[:, agent]
        raw_mse = float(np.mean(err ** 2))
        norm_mse = float(np.mean(norm_err ** 2))
        zero_mask = np.abs(raw_target[:, agent]) <= zero_tol
        nonzero_mask = ~zero_mask
        denom = float(np.sum((raw_target[:, agent] - raw_target[:, agent].mean()) ** 2))
        per_agent_raw_mse.append(raw_mse)
        per_agent_norm_mse.append(norm_mse)
        rows.append({
            **common,
            "model_type": "decentralized_reward",
            "agent": agent,
            "task_type": "",
            "n_val_cases": raw_target.shape[0],
            "raw_mse": raw_mse,
            "raw_rmse": float(np.sqrt(raw_mse)),
            "raw_mae": float(np.mean(np.abs(err))),
            "raw_bias": float(np.mean(err)),
            "normalized_mse": norm_mse,
            "normalized_rmse": float(np.sqrt(norm_mse)),
            "normalized_mae": float(np.mean(np.abs(norm_err))),
            "normalized_bias": float(np.mean(norm_err)),
            "val_mse_zero_cases": masked_mse(norm_err, zero_mask),
            "val_mse_nonzero_cases": masked_mse(norm_err, nonzero_mask),
            "val_mse_pickup_cases": masked_mse(norm_err, val_pickup_mask),
            "fraction_nonzero_train": float(train_reward_freq[agent]),
            "fraction_nonzero_val": float(reward_freq[agent]),
            "r2": float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else np.nan,
            "target_mean": float(target_mean[agent]),
            "target_std": float(target_std[agent]),
            "reward_nonzero_freq": float(reward_freq[agent]),
            "reward_nonzero_abs_mean": float(reward_nonzero_abs[agent]) if np.isfinite(reward_nonzero_abs[agent]) else np.nan,
            "reward_abs_mean": float(reward_abs[agent]),
        })
    rows.insert(0, {
        **common,
        "model_type": "decentralized_reward_mean",
        "agent": "mean",
        "task_type": "",
        "n_val_cases": raw_target.shape[0],
        "raw_mse": float(np.mean(per_agent_raw_mse)),
        "raw_rmse": float(np.sqrt(np.mean(per_agent_raw_mse))),
        "raw_mae": float(np.mean([row["raw_mae"] for row in rows])),
        "raw_bias": float(np.mean([row["raw_bias"] for row in rows])),
        "normalized_mse": float(np.mean(per_agent_norm_mse)),
        "normalized_rmse": float(np.sqrt(np.mean(per_agent_norm_mse))),
        "normalized_mae": float(np.mean([row["normalized_mae"] for row in rows])),
        "normalized_bias": float(np.mean([row["normalized_bias"] for row in rows])),
        "val_mse_zero_cases": float(np.nanmean([row["val_mse_zero_cases"] for row in rows])),
        "val_mse_nonzero_cases": float(np.nanmean([row["val_mse_nonzero_cases"] for row in rows])),
        "val_mse_pickup_cases": float(np.nanmean([row["val_mse_pickup_cases"] for row in rows])),
        "fraction_nonzero_train": float(np.nanmean(train_reward_freq)),
        "fraction_nonzero_val": float(np.mean(reward_freq)),
        "r2": float(np.nanmean([row["r2"] for row in rows])),
        "target_mean": float(np.mean(target_mean)),
        "target_std": float(np.mean(target_std)),
        "reward_nonzero_freq": float(np.mean(reward_freq)),
        "reward_nonzero_abs_mean": float(np.nanmean(reward_nonzero_abs)),
        "reward_abs_mean": float(np.mean(reward_abs)),
    })

    for task_type in range(int(n_task_types)):
        task_mask = val_pickup_mask & (val_picked_task_types == task_type)
        task_agent_rows: list[dict[str, object]] = []
        for agent in range(raw_target.shape[1]):
            err = raw_pred[:, agent] - raw_target[:, agent]
            norm_err = norm_pred[:, agent] - norm_target[:, agent]
            n_cases = int(task_mask.sum())
            if n_cases > 0:
                raw_mse = masked_mse(err, task_mask)
                norm_mse = masked_mse(norm_err, task_mask)
                raw_mae = float(np.mean(np.abs(err[task_mask])))
                norm_mae = float(np.mean(np.abs(norm_err[task_mask])))
                raw_bias = float(np.mean(err[task_mask]))
                norm_bias = float(np.mean(norm_err[task_mask]))
                task_targets = raw_target[task_mask, agent]
                denom = float(np.sum((task_targets - task_targets.mean()) ** 2))
                zero_mask = task_mask & (np.abs(raw_target[:, agent]) <= zero_tol)
                nonzero_mask = task_mask & (np.abs(raw_target[:, agent]) > zero_tol)
                nonzero_freq = float(np.mean(np.abs(task_targets) > zero_tol))
                nonzero_abs = (
                    float(np.mean(np.abs(task_targets[np.abs(task_targets) > zero_tol])))
                    if bool((np.abs(task_targets) > zero_tol).any())
                    else np.nan
                )
                reward_abs_task = float(np.mean(np.abs(task_targets)))
                target_mean_task = float(np.mean(task_targets))
                target_std_task = float(np.std(task_targets))
                r2 = float(1.0 - np.sum(err[task_mask] ** 2) / denom) if denom > 0 else np.nan
            else:
                raw_mse = norm_mse = raw_mae = norm_mae = raw_bias = norm_bias = np.nan
                zero_mask = nonzero_mask = task_mask
                nonzero_freq = nonzero_abs = reward_abs_task = np.nan
                target_mean_task = target_std_task = r2 = np.nan

            row = {
                **common,
                "model_type": "decentralized_reward_by_picked_task",
                "agent": agent,
                "task_type": task_type,
                "n_val_cases": n_cases,
                "raw_mse": raw_mse,
                "raw_rmse": float(np.sqrt(raw_mse)) if np.isfinite(raw_mse) else np.nan,
                "raw_mae": raw_mae,
                "raw_bias": raw_bias,
                "normalized_mse": norm_mse,
                "normalized_rmse": float(np.sqrt(norm_mse)) if np.isfinite(norm_mse) else np.nan,
                "normalized_mae": norm_mae,
                "normalized_bias": norm_bias,
                "val_mse_zero_cases": masked_mse(norm_err, zero_mask),
                "val_mse_nonzero_cases": masked_mse(norm_err, nonzero_mask),
                "val_mse_pickup_cases": norm_mse,
                "fraction_nonzero_train": np.nan,
                "fraction_nonzero_val": nonzero_freq,
                "r2": r2,
                "target_mean": target_mean_task,
                "target_std": target_std_task,
                "reward_nonzero_freq": nonzero_freq,
                "reward_nonzero_abs_mean": nonzero_abs,
                "reward_abs_mean": reward_abs_task,
            }
            rows.append(row)
            task_agent_rows.append(row)

        rows.append({
            **common,
            "model_type": "decentralized_reward_by_picked_task_mean",
            "agent": "mean",
            "task_type": task_type,
            "n_val_cases": int(task_mask.sum()),
            "raw_mse": safe_nanmean(row["raw_mse"] for row in task_agent_rows),
            "raw_rmse": safe_nanmean(row["raw_rmse"] for row in task_agent_rows),
            "raw_mae": safe_nanmean(row["raw_mae"] for row in task_agent_rows),
            "raw_bias": safe_nanmean(row["raw_bias"] for row in task_agent_rows),
            "normalized_mse": safe_nanmean(row["normalized_mse"] for row in task_agent_rows),
            "normalized_rmse": safe_nanmean(row["normalized_rmse"] for row in task_agent_rows),
            "normalized_mae": safe_nanmean(row["normalized_mae"] for row in task_agent_rows),
            "normalized_bias": safe_nanmean(row["normalized_bias"] for row in task_agent_rows),
            "val_mse_zero_cases": safe_nanmean(row["val_mse_zero_cases"] for row in task_agent_rows),
            "val_mse_nonzero_cases": safe_nanmean(row["val_mse_nonzero_cases"] for row in task_agent_rows),
            "val_mse_pickup_cases": safe_nanmean(row["val_mse_pickup_cases"] for row in task_agent_rows),
            "fraction_nonzero_train": np.nan,
            "fraction_nonzero_val": safe_nanmean(row["fraction_nonzero_val"] for row in task_agent_rows),
            "r2": safe_nanmean(row["r2"] for row in task_agent_rows),
            "target_mean": safe_nanmean(row["target_mean"] for row in task_agent_rows),
            "target_std": safe_nanmean(row["target_std"] for row in task_agent_rows),
            "reward_nonzero_freq": safe_nanmean(row["reward_nonzero_freq"] for row in task_agent_rows),
            "reward_nonzero_abs_mean": safe_nanmean(row["reward_nonzero_abs_mean"] for row in task_agent_rows),
            "reward_abs_mean": safe_nanmean(row["reward_abs_mean"] for row in task_agent_rows),
        })

    return rows




def parse_encoder_arg(value: str | None, default: EncoderType) -> EncoderType:
    if value is None:
        return default
    key = value.lower().strip()
    mapping = {
        "everything_cnn_grid": EncoderType.EVERYTHING_CNN_GRID,
        "filtered_dec_cnn_grid": EncoderType.FILTERED_DEC_CNN_GRID,
    }
    if key not in mapping:
        raise ValueError(f"Unknown encoder {value!r}; expected one of {sorted(mapping)}")
    return mapping[key]


def encoder_name(encoder: EncoderType) -> str:
    if encoder == EncoderType.EVERYTHING_CNN_GRID:
        return "everything_cnn_grid"
    if encoder == EncoderType.FILTERED_DEC_CNN_GRID:
        return "filtered_dec_cnn_grid"
    return str(encoder)


def parse_conv_specs_arg(value: str | None, default):
    if value is None:
        return default
    parsed = ast.literal_eval(value)
    if parsed is None:
        return None
    return tuple(tuple(int(x) for x in pair) for pair in parsed)


def parse_mlp_dims_arg(value: str | None, default):
    if value is None:
        return default
    parsed = ast.literal_eval(value)
    return tuple(int(x) for x in parsed)


def evaluate_run(args: argparse.Namespace, run_dir: Path, lr_label: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cfg, _env, _behavior_trainer, ckpt_path, loaded_step, rollout = prepare_reward_rollout(
        run_dir,
        checkpoint=args.checkpoint,
        rollout_steps=args.rollout_steps,
        drop_tail_transitions=args.drop_tail_transitions,
        eval_seed=args.eval_seed,
        behavior_device=args.behavior_device,
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
    encoding.init_encoder(model_cfg.encoder, _env, n_networks=cfg.env.n_agents)

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
    pred_norm = predict_decentralized_rewards(
        nets,
        split.test_states,
        batch_size=args.batch_size,
        device=device,
    )
    pred_raw = pred_norm * std + mean
    train_reward_freq, _train_reward_nonzero_abs, _train_reward_abs = reward_stats_by_agent(split.train_rewards)
    reward_freq, reward_nonzero_abs, reward_abs = reward_stats_by_agent(split.test_rewards)
    val_pickup_mask = np.asarray([state.pick_phase for state in split.test_states], dtype=bool)
    common = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_step": loaded_step,
        "rel": cfg.env.relatedness_width,
        "prof": cfg.env.proficiency_width,
        "lr_label": lr_label,
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
            "run_dir": str(run_dir),
            "rel": cfg.env.relatedness_width,
            "prof": cfg.env.proficiency_width,
            "lr_label": lr_label,
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
    parser = argparse.ArgumentParser(description="Train supervised immediate reward predictors on rollout states.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["dec"], default="dec")
    parser.add_argument("--rel", type=int, default=None)
    parser.add_argument("--prof", type=int, default=5)
    parser.add_argument("--lr-label", default="best")
    parser.add_argument("--final-score-evals", type=int, default=10)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--policy", choices=["greedy", "trained_greedy", "nearest", "eps_nearest", "nearest_rewarding_task", "eps_nearest_rewarding_task", "nearest_task", "random", "nearest_or_random"], default="greedy")
    parser.add_argument("--random-policy-prob", type=float, default=0.5)
    parser.add_argument("--rollout-steps", type=int, default=10000)
    parser.add_argument("--drop-tail-transitions", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-tag", default="config_model")
    parser.add_argument("--encoder", choices=["everything_cnn_grid", "filtered_dec_cnn_grid"], default=None)
    parser.add_argument("--conv-specs", default=None, help="Python literal, e.g. '[[32,3],[32,3]]'.")
    parser.add_argument("--mlp-dims", default=None, help="Python literal, e.g. '[32]' or '[]'.")
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--behavior-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--rollout-cache", type=Path, default=None)
    parser.add_argument("--force-rollout-cache", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--history-output", type=Path, default=None)
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

    if args.collect_only:
        if args.rollout_cache is None:
            raise ValueError("--collect-only requires --rollout-cache")
        _cfg, _env, _trainer, _ckpt_path, _loaded_step, rollout = prepare_reward_rollout(
            run_dir,
            checkpoint=args.checkpoint,
            rollout_steps=args.rollout_steps,
            drop_tail_transitions=args.drop_tail_transitions,
            eval_seed=args.eval_seed,
            behavior_device=args.behavior_device,
            policy=args.policy,
            random_policy_prob=args.random_policy_prob,
            cache_path=args.rollout_cache,
            force_cache=args.force_rollout_cache,
        )
        print(f"Cached rollout with {rollout.kept_transitions} transitions at {args.rollout_cache}", flush=True)
        return
    if args.output is None:
        raise ValueError("--output is required unless --collect-only is set")

    rows, history_rows = evaluate_run(args, run_dir, lr_label)
    history_output = args.history_output or args.output.with_name(args.output.stem + "_history.csv")
    write_csv(args.output, rows, REWARD_METRIC_FIELDS)
    write_csv(history_output, history_rows, REWARD_HISTORY_FIELDS)
    summary = [row for row in rows if str(row.get("agent")) == "mean"]
    for row in summary:
        if row.get("model_type") != "decentralized_reward_mean":
            continue
        print(
            f"{row['model_type']} lr={args.lr:g}: norm_rmse={float(row['normalized_rmse']):.4f}, "
            f"raw_rmse={float(row['raw_rmse']):.4f}, r2={float(row['r2']):.4f}, "
            f"reward_freq={float(row['reward_nonzero_freq']):.4f}",
            flush=True,
        )
    print(f"Run dir: {run_dir}", flush=True)
    print(f"LR label: {lr_label}", flush=True)
    print(f"Wrote {len(rows)} rows to {args.output}", flush=True)
    print(f"Wrote {len(history_rows)} history rows to {history_output}", flush=True)


if __name__ == "__main__":
    main()
