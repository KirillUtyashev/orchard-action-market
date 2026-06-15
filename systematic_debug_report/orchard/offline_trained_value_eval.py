"""Offline supervised value fitting on trained-policy rollouts.

This diagnostic loads an existing decentralized value-learning checkpoint as the
fixed behavior policy, collects one long greedy rollout, computes Monte Carlo
return-to-go targets, and trains fresh supervised value regressors. It is meant
to separate representation/value-prediction difficulty from online TD dynamics.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

import orchard.encoding as encoding
from orchard.datatypes import EncoderOutput, ModelConfig, State
from orchard.enums import EncoderType
from orchard.interest_diagnostics import _load_config_or_metadata
from orchard.model import ValueNetwork
from orchard.offline_value_eval import CentralizedMultiheadValueNetwork
from orchard.value_reward_diagnostics import (
    build_loaded_value_trainer,
    collect_value_reward_trace,
    discounted_returns,
)


OFFLINE_TRAINED_HISTORY_CSV_FIELDS = [
    "run_dir",
    "rel",
    "prof",
    "lr_label",
    "model_type",
    "epoch",
    "train_loss",
    "val_loss",
    "train_transitions",
    "val_transitions",
    "supervised_lr",
    "supervised_batch_size",
    "device",
]


OFFLINE_TRAINED_CSV_FIELDS = [
    "run_dir",
    "checkpoint_path",
    "checkpoint_step",
    "rel",
    "prof",
    "lr_label",
    "model_type",
    "agent",
    "raw_mse",
    "raw_rmse",
    "normalized_mse",
    "normalized_rmse",
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
    "supervised_epochs",
    "supervised_batch_size",
    "supervised_lr",
    "device",
]


@dataclass(frozen=True)
class PreparedRollout:
    states: list[State]
    returns: np.ndarray
    team_returns: np.ndarray
    rewards: np.ndarray
    recorded_transitions: int
    kept_transitions: int


@dataclass(frozen=True)
class SplitData:
    train_states: list[State]
    test_states: list[State]
    train_returns: np.ndarray
    test_returns: np.ndarray
    train_team_returns: np.ndarray
    test_team_returns: np.ndarray
    test_rewards: np.ndarray


def final_score(metrics_path: Path, n: int) -> float:
    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        return float("nan")
    if df.empty or "greedy_team_rps" not in df or "step" not in df:
        return float("nan")
    df = df.copy()
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["greedy_team_rps"] = pd.to_numeric(df["greedy_team_rps"], errors="coerce")
    df = df.dropna(subset=["step", "greedy_team_rps"])
    if df.empty:
        return float("nan")
    return float(df.nlargest(max(int(n), 1), "step")["greedy_team_rps"].mean())


def latest_run_for_lr(point_dir: Path, lr_label: str) -> Path | None:
    base = point_dir / lr_label
    if not base.exists():
        return None
    runs = sorted(
        p for p in base.glob("*")
        if p.is_dir() and (p / "metadata.yaml").exists() and (p / "checkpoints").exists()
    )
    return runs[-1] if runs else None


def discover_run_dir(
    experiment_dir: Path,
    *,
    mode: str,
    rel: int,
    prof: int,
    lr_label: str,
    final_score_evals: int,
) -> tuple[Path, str]:
    point_dir = experiment_dir / "runs" / mode / f"rel{rel}_prof{prof}"
    if not point_dir.exists():
        raise FileNotFoundError(f"Point directory not found: {point_dir}")

    if lr_label != "best":
        run_dir = latest_run_for_lr(point_dir, lr_label)
        if run_dir is None:
            raise FileNotFoundError(f"No run found for {point_dir / lr_label}")
        return run_dir, lr_label

    best: tuple[float, str, Path] | None = None
    for lr_dir in sorted(p for p in point_dir.iterdir() if p.is_dir() and p.name.startswith("lr_")):
        run_dir = latest_run_for_lr(point_dir, lr_dir.name)
        if run_dir is None:
            continue
        score = final_score(run_dir / "metrics.csv", final_score_evals)
        if np.isfinite(score) and (best is None or score > best[0]):
            best = (score, lr_dir.name, run_dir)
    if best is None:
        raise FileNotFoundError(f"No scored runs found in {point_dir}")
    return best[2], best[1]


def prepare_rollout(
    run_dir: Path,
    *,
    checkpoint: str,
    rollout_steps: int,
    drop_tail_transitions: int,
    eval_seed: int | None,
    device: str,
) -> tuple[object, object, object, Path, int | None, PreparedRollout]:
    cfg, env, behavior_trainer, ckpt_path, loaded_step = build_loaded_value_trainer(
        run_dir,
        checkpoint=checkpoint,
        device=device,
    )
    seed = eval_seed if eval_seed is not None else cfg.eval.eval_seed
    env.set_eval_mode(True, seed=seed)
    try:
        start = env.init_state()
        states, rewards, discounts = collect_value_reward_trace(
            start,
            behavior_trainer,
            env,
            cfg,
            rollout_steps=rollout_steps,
            policy="greedy",
        )
    finally:
        env.set_eval_mode(False)

    recorded = len(states)
    keep_end = max(recorded - max(int(drop_tail_transitions), 0), 0)
    if keep_end <= 1:
        raise ValueError(
            f"Not enough transitions after tail drop: recorded={recorded}, drop_tail={drop_tail_transitions}"
        )
    returns = discounted_returns(rewards, discounts)
    kept_states = states[:keep_end]
    kept_rewards = rewards[:keep_end]
    kept_returns = returns[:keep_end]
    return cfg, env, behavior_trainer, ckpt_path, loaded_step, PreparedRollout(
        states=kept_states,
        returns=kept_returns,
        team_returns=kept_returns.sum(axis=1),
        rewards=kept_rewards,
        recorded_transitions=recorded,
        kept_transitions=keep_end,
    )


def split_contiguous(rollout: PreparedRollout, *, train_frac: float) -> SplitData:
    n = len(rollout.states)
    if n < 2:
        raise ValueError("Need at least two kept transitions for train/test split")
    n_train = int(np.floor(n * train_frac))
    n_train = min(max(n_train, 1), n - 1)
    return SplitData(
        train_states=rollout.states[:n_train],
        test_states=rollout.states[n_train:],
        train_returns=rollout.returns[:n_train],
        test_returns=rollout.returns[n_train:],
        train_team_returns=rollout.team_returns[:n_train],
        test_team_returns=rollout.team_returns[n_train:],
        test_rewards=rollout.rewards[n_train:],
    )


def normalize_targets(train_targets: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train_targets, axis=0)
    std = np.std(train_targets, axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return (targets - mean) / std, mean, std


def batch_encode_states(states: Sequence[State], indices: np.ndarray, *, device: torch.device, agent_idx: int) -> EncoderOutput:
    encoded = [encoding.encode(states[int(idx)], agent_idx) for idx in indices]
    grids = torch.stack([enc.grid for enc in encoded]).to(device)
    scalars = torch.stack([enc.scalar for enc in encoded]).to(device)
    return EncoderOutput(grid=grids, scalar=scalars)


def batch_encode_all_agents(states: Sequence[State], indices: np.ndarray, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    grids = []
    scalars = []
    for idx in indices:
        g, s = encoding.encode_all_agents(states[int(idx)])
        grids.append(g)
        scalars.append(s)
    return torch.stack(grids).to(device), torch.stack(scalars).to(device)


def train_decentralized_supervised(
    states: Sequence[State],
    targets_norm: np.ndarray,
    *,
    model_cfg: ModelConfig,
    env_cfg,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    val_states: Sequence[State] | None = None,
    val_targets_norm: np.ndarray | None = None,
) -> tuple[list[ValueNetwork], list[dict[str, float | int | str]]]:
    torch.manual_seed(seed)
    nets = [ValueNetwork(model_cfg, env_cfg, td_lambda=0.0).to(device) for _ in range(env_cfg.n_agents)]
    opt = torch.optim.Adam([p for net in nets for p in net.parameters()], lr=lr)
    indices = np.arange(len(states))
    history: list[dict[str, float | int | str]] = []
    for net in nets:
        net.train()
    for epoch in range(max(int(epochs), 0)):
        np.random.default_rng(seed + epoch).shuffle(indices)
        train_loss_sum = 0.0
        train_count = 0
        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = indices[start:start + batch_size]
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
            train_loss_sum += float(loss.detach().cpu().item()) * len(batch_idx)
            train_count += len(batch_idx)
        for net in nets:
            net.eval()
        if val_states is not None and val_targets_norm is not None and len(val_states):
            val_pred = predict_decentralized(nets, val_states, batch_size=batch_size, device=device)
            val_loss = float(np.mean((val_pred - val_targets_norm) ** 2))
        else:
            val_loss = np.nan
        history.append({
            "model_type": "decentralized",
            "epoch": epoch + 1,
            "train_loss": train_loss_sum / max(train_count, 1),
            "val_loss": val_loss,
        })
        for net in nets:
            net.train()
    for net in nets:
        net.eval()
    return nets, history


def predict_decentralized(nets: Sequence[ValueNetwork], states: Sequence[State], *, batch_size: int, device: torch.device) -> np.ndarray:
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


def train_centralized_supervised(
    states: Sequence[State],
    targets_norm: np.ndarray,
    *,
    model_cfg: ModelConfig,
    env_cfg,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    val_states: Sequence[State] | None = None,
    val_targets_norm: np.ndarray | None = None,
) -> tuple[ValueNetwork, list[dict[str, float | int | str]]]:
    torch.manual_seed(seed)
    net = ValueNetwork(model_cfg, env_cfg, td_lambda=0.0).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    indices = np.arange(len(states))
    history: list[dict[str, float | int | str]] = []
    net.train()
    for epoch in range(max(int(epochs), 0)):
        np.random.default_rng(seed + epoch).shuffle(indices)
        train_loss_sum = 0.0
        train_count = 0
        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = indices[start:start + batch_size]
            enc = batch_encode_states(states, batch_idx, device=device, agent_idx=0)
            y = torch.as_tensor(targets_norm[batch_idx], dtype=torch.float32, device=device)
            pred = net(enc)
            loss = torch.mean((pred - y) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss_sum += float(loss.detach().cpu().item()) * len(batch_idx)
            train_count += len(batch_idx)
        net.eval()
        if val_states is not None and val_targets_norm is not None and len(val_states):
            val_pred = predict_centralized(net, val_states, batch_size=batch_size, device=device)
            val_loss = float(np.mean((val_pred - val_targets_norm) ** 2))
        else:
            val_loss = np.nan
        history.append({
            "model_type": "centralized",
            "epoch": epoch + 1,
            "train_loss": train_loss_sum / max(train_count, 1),
            "val_loss": val_loss,
        })
        net.train()
    net.eval()
    return net, history


def predict_centralized(net: ValueNetwork, states: Sequence[State], *, batch_size: int, device: torch.device) -> np.ndarray:
    preds = np.zeros(len(states), dtype=np.float64)
    with torch.no_grad():
        indices = np.arange(len(states))
        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = indices[start:start + batch_size]
            enc = batch_encode_states(states, batch_idx, device=device, agent_idx=0)
            p = net(enc).detach().cpu().numpy()
            preds[start:start + len(batch_idx)] = p
    return preds


def train_multihead_supervised(
    states: Sequence[State],
    targets_norm: np.ndarray,
    *,
    model_cfg: ModelConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    val_states: Sequence[State] | None = None,
    val_targets_norm: np.ndarray | None = None,
) -> tuple[CentralizedMultiheadValueNetwork, list[dict[str, float | int | str]]]:
    torch.manual_seed(seed)
    net = CentralizedMultiheadValueNetwork(model_cfg, n_outputs=targets_norm.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    indices = np.arange(len(states))
    history: list[dict[str, float | int | str]] = []
    net.train()
    for epoch in range(max(int(epochs), 0)):
        np.random.default_rng(seed + epoch).shuffle(indices)
        train_loss_sum = 0.0
        train_count = 0
        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = indices[start:start + batch_size]
            enc = batch_encode_states(states, batch_idx, device=device, agent_idx=0)
            y = torch.as_tensor(targets_norm[batch_idx], dtype=torch.float32, device=device)
            pred = net.forward_raw(enc.grid, enc.scalar)
            loss = torch.mean((pred - y) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss_sum += float(loss.detach().cpu().item()) * len(batch_idx)
            train_count += len(batch_idx)
        net.eval()
        if val_states is not None and val_targets_norm is not None and len(val_states):
            val_pred = predict_multihead(net, val_states, batch_size=batch_size, device=device)
            val_loss = float(np.mean((val_pred - val_targets_norm) ** 2))
        else:
            val_loss = np.nan
        history.append({
            "model_type": "centralized_multihead",
            "epoch": epoch + 1,
            "train_loss": train_loss_sum / max(train_count, 1),
            "val_loss": val_loss,
        })
        net.train()
    net.eval()
    return net, history


def predict_multihead(net: CentralizedMultiheadValueNetwork, states: Sequence[State], *, batch_size: int, device: torch.device) -> np.ndarray:
    preds = np.zeros((len(states), net.net[-1].out_features), dtype=np.float64)
    with torch.no_grad():
        indices = np.arange(len(states))
        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = indices[start:start + batch_size]
            enc = batch_encode_states(states, batch_idx, device=device, agent_idx=0)
            p = net.forward_raw(enc.grid, enc.scalar).detach().cpu().numpy()
            preds[start:start + len(batch_idx)] = p
    return preds


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
    model_type: str,
    raw_pred: np.ndarray,
    norm_pred: np.ndarray,
    raw_target: np.ndarray,
    norm_target: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    reward_freq: np.ndarray,
    reward_nonzero_abs: np.ndarray,
    reward_abs: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if raw_target.ndim == 1:
        err = raw_pred - raw_target
        norm_err = norm_pred - norm_target
        denom = float(np.sum((raw_target - raw_target.mean()) ** 2))
        rows.append({
            **common,
            "model_type": model_type,
            "agent": "all",
            "raw_mse": float(np.mean(err ** 2)),
            "raw_rmse": float(np.sqrt(np.mean(err ** 2))),
            "normalized_mse": float(np.mean(norm_err ** 2)),
            "normalized_rmse": float(np.sqrt(np.mean(norm_err ** 2))),
            "r2": float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else np.nan,
            "target_mean": float(np.asarray(target_mean).reshape(-1)[0]),
            "target_std": float(np.asarray(target_std).reshape(-1)[0]),
            "reward_nonzero_freq": float(np.mean(reward_freq)),
            "reward_nonzero_abs_mean": float(np.nanmean(reward_nonzero_abs)),
            "reward_abs_mean": float(np.mean(reward_abs)),
        })
        return rows

    per_agent_mse = []
    per_agent_norm_mse = []
    for agent in range(raw_target.shape[1]):
        err = raw_pred[:, agent] - raw_target[:, agent]
        norm_err = norm_pred[:, agent] - norm_target[:, agent]
        denom = float(np.sum((raw_target[:, agent] - raw_target[:, agent].mean()) ** 2))
        raw_mse = float(np.mean(err ** 2))
        norm_mse = float(np.mean(norm_err ** 2))
        per_agent_mse.append(raw_mse)
        per_agent_norm_mse.append(norm_mse)
        rows.append({
            **common,
            "model_type": model_type,
            "agent": agent,
            "raw_mse": raw_mse,
            "raw_rmse": float(np.sqrt(raw_mse)),
            "normalized_mse": norm_mse,
            "normalized_rmse": float(np.sqrt(norm_mse)),
            "r2": float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else np.nan,
            "target_mean": float(target_mean[agent]),
            "target_std": float(target_std[agent]),
            "reward_nonzero_freq": float(reward_freq[agent]),
            "reward_nonzero_abs_mean": float(reward_nonzero_abs[agent]) if np.isfinite(reward_nonzero_abs[agent]) else np.nan,
            "reward_abs_mean": float(reward_abs[agent]),
        })
    rows.insert(0, {
        **common,
        "model_type": f"{model_type}_mean",
        "agent": "mean",
        "raw_mse": float(np.mean(per_agent_mse)),
        "raw_rmse": float(np.sqrt(np.mean(per_agent_mse))),
        "normalized_mse": float(np.mean(per_agent_norm_mse)),
        "normalized_rmse": float(np.sqrt(np.mean(per_agent_norm_mse))),
        "r2": float(np.nanmean([r["r2"] for r in rows])),
        "target_mean": float(np.mean(target_mean)),
        "target_std": float(np.mean(target_std)),
        "reward_nonzero_freq": float(np.mean(reward_freq)),
        "reward_nonzero_abs_mean": float(np.nanmean(reward_nonzero_abs)),
        "reward_abs_mean": float(np.mean(reward_abs)),
    })
    return rows


def evaluate_run(args: argparse.Namespace, run_dir: Path, lr_label: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cfg, env, _behavior_trainer, ckpt_path, loaded_step, rollout = prepare_rollout(
        run_dir,
        checkpoint=args.checkpoint,
        rollout_steps=args.rollout_steps,
        drop_tail_transitions=args.drop_tail_transitions,
        eval_seed=args.eval_seed,
        device=args.behavior_device,
    )
    split = split_contiguous(rollout, train_frac=args.train_frac)
    reward_freq, reward_nonzero_abs, reward_abs = reward_stats_by_agent(split.test_rewards)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is not available")

    common = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_step": loaded_step,
        "rel": cfg.env.relatedness_width,
        "prof": cfg.env.proficiency_width,
        "lr_label": lr_label,
        "rollout_steps": args.rollout_steps,
        "recorded_transitions": rollout.recorded_transitions,
        "kept_transitions": rollout.kept_transitions,
        "drop_tail_transitions": args.drop_tail_transitions,
        "train_transitions": len(split.train_states),
        "test_transitions": len(split.test_states),
        "train_frac": args.train_frac,
        "policy": "trained_greedy",
        "eval_seed": args.eval_seed if args.eval_seed is not None else cfg.eval.eval_seed,
        "supervised_epochs": args.epochs,
        "supervised_batch_size": args.batch_size,
        "supervised_lr": args.lr,
        "device": args.device,
    }

    rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    models = set(args.models)

    def add_history(history: list[dict[str, object]]) -> None:
        for item in history:
            history_rows.append({
                "run_dir": str(run_dir),
                "rel": cfg.env.relatedness_width,
                "prof": cfg.env.proficiency_width,
                "lr_label": lr_label,
                "train_transitions": len(split.train_states),
                "val_transitions": len(split.test_states),
                "supervised_lr": args.lr,
                "supervised_batch_size": args.batch_size,
                "device": args.device,
                **item,
            })

    if "decentralized" in models:
        dec_cfg = cfg.model
        encoding.init_encoder(dec_cfg.encoder, env, n_networks=cfg.env.n_agents)
        train_norm, dec_mean, dec_std = normalize_targets(split.train_returns, split.train_returns)
        test_norm = (split.test_returns - dec_mean) / dec_std
        nets, history = train_decentralized_supervised(
            split.train_states,
            train_norm,
            model_cfg=dec_cfg,
            env_cfg=cfg.env,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed,
            val_states=split.test_states,
            val_targets_norm=test_norm,
        )
        add_history(history)
        pred_norm = predict_decentralized(nets, split.test_states, batch_size=args.batch_size, device=device)
        pred_raw = pred_norm * dec_std + dec_mean
        rows.extend(metric_rows(
            common=common,
            model_type="decentralized",
            raw_pred=pred_raw,
            norm_pred=pred_norm,
            raw_target=split.test_returns,
            norm_target=test_norm,
            target_mean=dec_mean,
            target_std=dec_std,
            reward_freq=reward_freq,
            reward_nonzero_abs=reward_nonzero_abs,
            reward_abs=reward_abs,
        ))

    if "centralized" in models:
        central_cfg = ModelConfig(
            encoder=EncoderType.EVERYTHING_CNN_GRID,
            mlp_dims=cfg.model.mlp_dims,
            conv_specs=cfg.model.conv_specs,
            activation=cfg.model.activation,
            weight_init=cfg.model.weight_init,
        )
        encoding.init_encoder(EncoderType.EVERYTHING_CNN_GRID, env, n_networks=1)
        train_norm, team_mean, team_std = normalize_targets(split.train_team_returns, split.train_team_returns)
        test_norm = (split.test_team_returns - team_mean) / team_std
        net, history = train_centralized_supervised(
            split.train_states,
            train_norm,
            model_cfg=central_cfg,
            env_cfg=cfg.env,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed + 1000,
            val_states=split.test_states,
            val_targets_norm=test_norm,
        )
        add_history(history)
        pred_norm = predict_centralized(net, split.test_states, batch_size=args.batch_size, device=device)
        pred_raw = pred_norm * team_std + team_mean
        rows.extend(metric_rows(
            common=common,
            model_type="centralized",
            raw_pred=pred_raw,
            norm_pred=pred_norm,
            raw_target=split.test_team_returns,
            norm_target=test_norm,
            target_mean=np.asarray(team_mean),
            target_std=np.asarray(team_std),
            reward_freq=reward_freq,
            reward_nonzero_abs=reward_nonzero_abs,
            reward_abs=reward_abs,
        ))

    if "centralized_multihead" in models:
        central_cfg = ModelConfig(
            encoder=EncoderType.EVERYTHING_CNN_GRID,
            mlp_dims=cfg.model.mlp_dims,
            conv_specs=cfg.model.conv_specs,
            activation=cfg.model.activation,
            weight_init=cfg.model.weight_init,
        )
        encoding.init_encoder(EncoderType.EVERYTHING_CNN_GRID, env, n_networks=1)
        train_norm, mh_mean, mh_std = normalize_targets(split.train_returns, split.train_returns)
        test_norm = (split.test_returns - mh_mean) / mh_std
        net, history = train_multihead_supervised(
            split.train_states,
            train_norm,
            model_cfg=central_cfg,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed + 2000,
            val_states=split.test_states,
            val_targets_norm=test_norm,
        )
        add_history(history)
        pred_norm = predict_multihead(net, split.test_states, batch_size=args.batch_size, device=device)
        pred_raw = pred_norm * mh_std + mh_mean
        rows.extend(metric_rows(
            common=common,
            model_type="centralized_multihead",
            raw_pred=pred_raw,
            norm_pred=pred_norm,
            raw_target=split.test_returns,
            norm_target=test_norm,
            target_mean=mh_mean,
            target_std=mh_std,
            reward_freq=reward_freq,
            reward_nonzero_abs=reward_nonzero_abs,
            reward_abs=reward_abs,
        ))

    return rows, history_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline supervised value fitting on trained-policy rollouts.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["dec"], default="dec")
    parser.add_argument("--rel", type=int, default=None)
    parser.add_argument("--prof", type=int, default=5)
    parser.add_argument("--lr-label", default="best")
    parser.add_argument("--final-score-evals", type=int, default=10)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--rollout-steps", type=int, default=10000)
    parser.add_argument("--drop-tail-transitions", type=int, default=1000)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="+", choices=["decentralized", "centralized", "centralized_multihead"], default=["decentralized", "centralized", "centralized_multihead"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--behavior-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-output", type=Path, default=None)
    return parser.parse_args(argv)


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OFFLINE_TRAINED_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OFFLINE_TRAINED_CSV_FIELDS})


def write_history_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OFFLINE_TRAINED_HISTORY_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OFFLINE_TRAINED_HISTORY_CSV_FIELDS})


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
        lr_label = args.lr_label
        if lr_label == "best":
            lr_label = run_dir.parent.name

    rows, history_rows = evaluate_run(args, run_dir, lr_label)
    write_csv(args.output, rows)
    history_output = args.history_output or args.output.with_name(args.output.stem + "_history.csv")
    write_history_csv(history_output, history_rows)
    summary = [row for row in rows if str(row.get("agent")) in ("mean", "all")]
    for row in summary:
        print(
            f"{row['model_type']} agent={row['agent']}: "
            f"norm_rmse={float(row['normalized_rmse']):.4f}, "
            f"raw_rmse={float(row['raw_rmse']):.4f}, r2={float(row['r2']):.4f}"
        )
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Wrote {len(history_rows)} history rows to {history_output}")


if __name__ == "__main__":
    main()
