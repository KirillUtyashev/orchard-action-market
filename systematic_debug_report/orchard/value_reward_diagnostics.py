"""Reward sparsity and decentralized value-error diagnostics.

This diagnostic evaluates decentralized value checkpoints on one long greedy
rollout. It records value predictions on the same decomposed critic-state stream
used by value learning: move-after states, optional pick-phase states, and
post-pick states. Monte Carlo return-to-go targets are computed after the
rollout, with a configurable tail dropped to reduce truncation bias.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

import orchard.encoding as encoding
from orchard.datatypes import EncoderOutput, ExperimentConfig, State
from orchard.env import create_env
from orchard.enums import LearningType
from orchard.interest_diagnostics import _load_config_or_metadata, find_checkpoint
from orchard.seed import set_all_seeds
from orchard.trainer import create_trainer


VALUE_REWARD_CSV_FIELDS = [
    "run_dir",
    "checkpoint_path",
    "checkpoint_step",
    "lr_label",
    "policy",
    "rel",
    "prof",
    "rollout_steps",
    "recorded_transitions",
    "kept_transitions",
    "drop_tail_transitions",
    "eval_seed",
    "discount_method",
    "model_type",
    "agent",
    "reward_nonzero_freq",
    "reward_nonzero_abs_mean",
    "reward_abs_mean",
    "raw_mse",
    "raw_rmse",
    "raw_mae",
    "raw_bias",
    "normalized_mse",
    "normalized_rmse",
    "normalized_mae",
    "normalized_bias",
    "r2",
    "target_mean",
    "target_std",
    "value_mse",
    "value_rmse",
    "value_mae",
    "value_bias",
    "mean_value_pred",
    "mean_return",
]


def build_loaded_value_trainer(
    run_dir: str | Path,
    *,
    checkpoint: str = "latest",
    device: str = "cpu",
    overrides: Sequence[str] | None = None,
):
    """Load a decentralized value-learning trainer and checkpoint."""
    run_dir = Path(run_dir)
    cfg = _load_config_or_metadata(run_dir, overrides=overrides)
    cfg = dataclasses.replace(
        cfg,
        train=dataclasses.replace(cfg.train, use_gpu=(device == "cuda")),
    )
    if cfg.train.learning_type != LearningType.DECENTRALIZED:
        raise ValueError("value/reward diagnostics are defined for decentralized runs only")

    set_all_seeds(cfg.train.seed)
    env = create_env(cfg.env)
    encoding.init_encoder(cfg.model.encoder, env, n_networks=cfg.env.n_agents)
    trainer = create_trainer(cfg, env)
    ckpt_path = find_checkpoint(run_dir, checkpoint)
    loaded_step = trainer.load_checkpoint(ckpt_path)
    trainer.sync_to_cpu()
    for net in trainer.critic_networks:
        net.eval()
    return cfg, env, trainer, ckpt_path, loaded_step


def predict_decentralized_values_for_states(
    trainer,
    states: Sequence[State],
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Return array [len(states), n_agents] of V_i(state) predictions.

    The rollout is sequential because action choice changes the environment, but
    value diagnostics can be computed afterward in batches.
    """
    n_states = len(states)
    n_agents = len(trainer.critic_networks)
    out = np.zeros((n_states, n_agents), dtype=np.float64)
    if n_states == 0:
        return out

    with torch.no_grad():
        for start in range(0, n_states, max(int(batch_size), 1)):
            chunk = states[start:start + batch_size]
            encoded = [trainer._encode_all(state) for state in chunk]

            if isinstance(encoded[0], list):
                for agent, net in enumerate(trainer.critic_networks):
                    device = next(net.parameters()).device
                    grids = torch.stack([enc[agent].grid for enc in encoded]).to(device)
                    scalars = torch.stack([enc[agent].scalar for enc in encoded]).to(device)
                    vals = net(EncoderOutput(grid=grids, scalar=scalars)).detach().cpu().numpy()
                    out[start:start + len(chunk), agent] = vals.astype(np.float64)
            else:
                for agent, net in enumerate(trainer.critic_networks):
                    device = next(net.parameters()).device
                    grids = torch.stack([enc[0][agent] for enc in encoded]).to(device)
                    scalars = torch.stack([enc[1][agent] for enc in encoded]).to(device)
                    vals = net.forward_raw(grids, scalars).detach().cpu().numpy()
                    out[start:start + len(chunk), agent] = vals.astype(np.float64)
    return out


def predict_decentralized_values(trainer, state: State) -> np.ndarray:
    """Return V_i(state) for each decentralized critic i."""
    return predict_decentralized_values_for_states(trainer, [state])[0]


def _move_discount(cfg: ExperimentConfig, round_step_count: int) -> tuple[float, int]:
    if cfg.train.discount_method == "round_steps":
        discount = cfg.env.gamma if round_step_count == 0 else 1.0
        round_step_count += 1
        if round_step_count >= cfg.env.n_agents:
            round_step_count = 0
        return float(discount), round_step_count
    return float(cfg.env.gamma), round_step_count


def collect_value_reward_trace(
    start: State,
    trainer,
    env,
    cfg: ExperimentConfig,
    *,
    rollout_steps: int,
    policy: str = "greedy",
) -> tuple[list[State], np.ndarray, np.ndarray]:
    """Return (critic_states, rewards, discounts) for one decomposed rollout.

    rollout_steps counts outer actor turns. The returned arrays are over the
    value-learning transition stream, which may be longer because pick decisions
    are represented as their own reward-bearing transition.
    """
    if policy != "greedy":
        raise ValueError(f"Unknown policy {policy!r}; currently only 'greedy' is supported")

    state = start
    prev_critic_state: State | None = None
    round_step_count = 0
    critic_states: list[State] = []
    rewards: list[tuple[float, ...]] = []
    discounts: list[float] = []

    for _ in range(max(rollout_steps, 0)):
        move_discount, round_step_count = _move_discount(cfg, round_step_count)
        move_action = trainer._greedy_action(state)
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
            discounts.append(move_discount)

        if on_task:
            pick_action = trainer._greedy_action(move_critic_state)
            picked, pick_rewards = env.resolve_pick(
                moved,
                pick_type=pick_action.pick_type() if pick_action.is_pick() else None,
            )
            critic_states.append(move_critic_state)
            rewards.append(pick_rewards)
            discounts.append(1.0)
            prev_critic_state = picked
            post_pick = picked
        else:
            prev_critic_state = move_critic_state
            post_pick = moved

        state = env.advance_actor(env.spawn_and_despawn(post_pick))

    return (
        critic_states,
        np.asarray(rewards, dtype=np.float64),
        np.asarray(discounts, dtype=np.float64),
    )


def discounted_returns(rewards: np.ndarray, discounts: np.ndarray) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float64)
    running = np.zeros(rewards.shape[1], dtype=np.float64)
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + discounts[t] * running
        returns[t] = running
    return returns


def summarize_value_reward_trace(
    value_preds: np.ndarray,
    rewards: np.ndarray,
    discounts: np.ndarray,
    *,
    drop_tail_transitions: int,
    zero_tol: float = 1e-12,
) -> tuple[pd.DataFrame, int, int]:
    recorded = int(len(rewards))
    drop = max(int(drop_tail_transitions), 0)
    keep_end = max(recorded - drop, 0)
    if keep_end <= 0:
        raise ValueError(
            f"No transitions left after dropping tail: recorded={recorded}, drop_tail={drop}"
        )

    returns = discounted_returns(rewards, discounts)
    vp = value_preds[:keep_end]
    rw = rewards[:keep_end]
    rt = returns[:keep_end]
    errors = vp - rt
    nonzero = np.abs(rw) > zero_tol

    rows = []
    per_agent_mse = []
    per_agent_norm_mse = []
    for agent in range(rw.shape[1]):
        nz = nonzero[:, agent]
        abs_rewards = np.abs(rw[:, agent])
        agent_errors = errors[:, agent]
        target = rt[:, agent]
        target_mean = float(np.mean(target))
        target_std = float(np.std(target))
        safe_std = target_std if target_std > 1e-8 else 1.0
        norm_errors = agent_errors / safe_std
        denom = float(np.sum((target - target_mean) ** 2))
        raw_mse = float(np.mean(agent_errors ** 2))
        norm_mse = float(np.mean(norm_errors ** 2))
        per_agent_mse.append(raw_mse)
        per_agent_norm_mse.append(norm_mse)
        rows.append({
            "model_type": "checkpoint_decentralized",
            "agent": agent,
            "reward_nonzero_freq": float(nz.mean()),
            "reward_nonzero_abs_mean": float(abs_rewards[nz].mean()) if bool(nz.any()) else np.nan,
            "reward_abs_mean": float(abs_rewards.mean()),
            "raw_mse": raw_mse,
            "raw_rmse": float(np.sqrt(raw_mse)),
            "raw_mae": float(np.mean(np.abs(agent_errors))),
            "raw_bias": float(np.mean(agent_errors)),
            "normalized_mse": norm_mse,
            "normalized_rmse": float(np.sqrt(norm_mse)),
            "normalized_mae": float(np.mean(np.abs(norm_errors))),
            "normalized_bias": float(np.mean(norm_errors)),
            "r2": float(1.0 - np.sum(agent_errors ** 2) / denom) if denom > 0 else np.nan,
            "target_mean": target_mean,
            "target_std": target_std,
            "value_mse": raw_mse,
            "value_rmse": float(np.sqrt(raw_mse)),
            "value_mae": float(np.mean(np.abs(agent_errors))),
            "value_bias": float(np.mean(agent_errors)),
            "mean_value_pred": float(np.mean(vp[:, agent])),
            "mean_return": target_mean,
        })
    if rows:
        rows.insert(0, {
            "model_type": "checkpoint_decentralized_mean",
            "agent": "mean",
            "reward_nonzero_freq": float(np.mean([row["reward_nonzero_freq"] for row in rows])),
            "reward_nonzero_abs_mean": float(np.nanmean([row["reward_nonzero_abs_mean"] for row in rows])),
            "reward_abs_mean": float(np.mean([row["reward_abs_mean"] for row in rows])),
            "raw_mse": float(np.mean(per_agent_mse)),
            "raw_rmse": float(np.sqrt(np.mean(per_agent_mse))),
            "raw_mae": float(np.mean([row["raw_mae"] for row in rows])),
            "raw_bias": float(np.mean([row["raw_bias"] for row in rows])),
            "normalized_mse": float(np.mean(per_agent_norm_mse)),
            "normalized_rmse": float(np.sqrt(np.mean(per_agent_norm_mse))),
            "normalized_mae": float(np.mean([row["normalized_mae"] for row in rows])),
            "normalized_bias": float(np.mean([row["normalized_bias"] for row in rows])),
            "r2": float(np.nanmean([row["r2"] for row in rows])),
            "target_mean": float(np.mean([row["target_mean"] for row in rows])),
            "target_std": float(np.mean([row["target_std"] for row in rows])),
            "value_mse": float(np.mean(per_agent_mse)),
            "value_rmse": float(np.sqrt(np.mean(per_agent_mse))),
            "value_mae": float(np.mean([row["value_mae"] for row in rows])),
            "value_bias": float(np.mean([row["value_bias"] for row in rows])),
            "mean_value_pred": float(np.mean([row["mean_value_pred"] for row in rows])),
            "mean_return": float(np.mean([row["mean_return"] for row in rows])),
        })
    return pd.DataFrame(rows), recorded, keep_end


def evaluate_value_reward_diagnostics(
    run_dir: str | Path,
    *,
    checkpoint: str = "latest",
    policy: str = "greedy",
    rollout_steps: int = 10000,
    drop_tail_transitions: int = 1000,
    eval_seed: int | None = None,
    device: str = "cpu",
    overrides: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Evaluate one decentralized run and return one row per agent."""
    run_dir = Path(run_dir)
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
        critic_states, rewards, discounts = collect_value_reward_trace(
            start,
            trainer,
            env,
            cfg,
            rollout_steps=rollout_steps,
            policy=policy,
        )
        value_preds = predict_decentralized_values_for_states(trainer, critic_states)
    finally:
        env.set_eval_mode(False)

    metrics, recorded, kept = summarize_value_reward_trace(
        value_preds,
        rewards,
        discounts,
        drop_tail_transitions=drop_tail_transitions,
    )
    metrics.insert(0, "discount_method", cfg.train.discount_method)
    metrics.insert(0, "eval_seed", seed)
    metrics.insert(0, "drop_tail_transitions", int(drop_tail_transitions))
    metrics.insert(0, "kept_transitions", kept)
    metrics.insert(0, "recorded_transitions", recorded)
    metrics.insert(0, "rollout_steps", int(rollout_steps))
    metrics.insert(0, "prof", cfg.env.proficiency_width)
    metrics.insert(0, "rel", cfg.env.relatedness_width)
    metrics.insert(0, "policy", policy)
    metrics.insert(0, "checkpoint_step", loaded_step)
    metrics.insert(0, "checkpoint_path", str(ckpt_path))
    metrics.insert(0, "lr_label", run_dir.parent.name)
    metrics.insert(0, "run_dir", str(run_dir))
    return metrics


def write_value_reward_csv(path: str | Path, rows: pd.DataFrame | Iterable[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        ordered = rows.reindex(columns=VALUE_REWARD_CSV_FIELDS)
        ordered.to_csv(output, index=False)
        return
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VALUE_REWARD_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in VALUE_REWARD_CSV_FIELDS})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reward sparsity and decentralized value error.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--policy", choices=["greedy"], default="greedy")
    parser.add_argument("--rollout-steps", type=int, default=10000)
    parser.add_argument("--drop-tail-transitions", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("value_reward_diagnostics.csv"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    frames = []
    for run_dir in args.run_dirs:
        df = evaluate_value_reward_diagnostics(
            run_dir,
            checkpoint=args.checkpoint,
            policy=args.policy,
            rollout_steps=args.rollout_steps,
            drop_tail_transitions=args.drop_tail_transitions,
            eval_seed=args.eval_seed,
            device=args.device,
        )
        frames.append(df)
        summary = df[df["agent"].astype(str) == "mean"]
        row = summary.iloc[0] if not summary.empty else df.iloc[0]
        print(
            f"{run_dir}: reward_nonzero_freq={float(row['reward_nonzero_freq']):.4f}, "
            f"value_rmse={float(row['value_rmse']):.4f}, "
            f"norm_rmse={float(row['normalized_rmse']):.4f}, r2={float(row['r2']):.4f}"
        )
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=VALUE_REWARD_CSV_FIELDS)
    write_value_reward_csv(args.output, out)
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
