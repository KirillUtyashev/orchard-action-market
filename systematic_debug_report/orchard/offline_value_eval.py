"""Offline supervised value-prediction diagnostic.

This module freezes a behavior policy, collects full episodes, computes
empirical return-to-go targets, and trains value networks as supervised
regressors. It is intentionally separate from online RL training.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

import orchard.encoding as encoding
from orchard.datatypes import EncoderOutput, EnvConfig, ModelConfig, State, StochasticConfig
from orchard.env import create_env
from orchard.enums import ACTION_PRIORITY, Activation, DespawnMode, EncoderType, WeightInit
from orchard.model import ValueNetwork
from orchard.policy import (
    get_all_actions,
    get_phase2_actions,
    nearest_action,
    eps_nearest_action,
    nearest_rewarding_task_action,
    nearest_task_action,
)
from orchard.seed import rng, set_all_seeds


OFFLINE_VALUE_CSV_FIELDS = [
    "rr",
    "seed",
    "model_type",
    "agent",
    "test_mse",
    "nonzero_reward_freq",
    "mean_abs_nonzero_reward",
    "n_train_episodes",
    "n_test_episodes",
    "n_train_transitions",
    "n_test_transitions",
    "policy",
    "discount_mode",
    "gamma",
    "proficiency_radius",
    "sigma_a",
    "sigma_b",
]


@dataclass(frozen=True)
class EpisodeData:
    states: list[State]
    rewards: np.ndarray
    discounts: np.ndarray
    returns: np.ndarray
    team_returns: np.ndarray


class CentralizedMultiheadValueNetwork(nn.Module):
    """Centralized value regressor with one output head per agent."""

    def __init__(self, model_cfg: ModelConfig, n_outputs: int) -> None:
        super().__init__()
        channels = encoding.get_grid_channels()
        scalar_extra = encoding.get_scalar_dim()
        self.conv, conv_flat_dim = ValueNetwork._build_conv(
            channels,
            model_cfg.conv_specs,
            encoding.get_grid_height(),
            encoding.get_grid_width(),
            model_cfg.activation,
        )
        self.flatten = nn.Flatten()
        self.net = self._build_mlp(
            conv_flat_dim + scalar_extra,
            model_cfg.mlp_dims,
            n_outputs,
            model_cfg.activation,
        )

        if model_cfg.weight_init == WeightInit.ZERO_BIAS:
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        activation: Activation,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        d = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(d, hidden_dim))
            if activation == Activation.LEAKY_RELU:
                layers.append(nn.LeakyReLU())
            elif activation == Activation.RELU:
                layers.append(nn.ReLU())
            d = hidden_dim
        layers.append(nn.Linear(d, output_dim))
        return nn.Sequential(*layers)

    def forward_raw(self, grid: torch.Tensor, scalar: torch.Tensor | None = None) -> torch.Tensor:
        x = grid
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.flatten(self.conv(x))
        if scalar is not None:
            s = scalar
            if s.dim() == 1:
                s = s.unsqueeze(0)
            x = torch.cat([x, s], dim=-1)
        return self.net(x)


def discounted_returns(rewards: np.ndarray, discounts: np.ndarray) -> np.ndarray:
    """Compute return-to-go for one episode using per-transition discounts."""
    returns = np.zeros_like(rewards, dtype=np.float64)
    running = np.zeros(rewards.shape[1], dtype=np.float64)
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + discounts[t] * running
        returns[t] = running
    return returns


def choose_behavior_action(
    state: State,
    env,
    *,
    policy: str,
    random_policy_prob: float,
) -> object:
    """Select a move or pick action from the frozen behavior policy."""
    normalized = "mixed" if policy == "nearest_or_random" else policy
    use_random = normalized == "random" or (
        normalized == "mixed" and rng.random() < random_policy_prob
    )
    if use_random:
        actions = get_phase2_actions(state, env) if state.pick_phase else get_all_actions(env.cfg)
        return rng.choice(actions) if actions else ACTION_PRIORITY[-1]
    if normalized == "nearest":
        return nearest_action(state, env)
    if normalized == "eps_nearest":
        return eps_nearest_action(state, env)
    if normalized == "nearest_rewarding_task":
        return nearest_rewarding_task_action(state, env)
    if normalized == "eps_nearest_rewarding_task":
        return nearest_rewarding_task_action(state, env, epsilon=0.2)
    if normalized == "nearest_task":
        return nearest_task_action(state, env)
    if normalized == "mixed":
        return nearest_action(state, env)
    raise ValueError(f"Unknown behavior policy: {policy!r}")


def collect_episode(
    env,
    *,
    episode_steps: int,
    policy: str,
    random_policy_prob: float,
    discount_mode: str,
) -> EpisodeData:
    """Collect one episode as a decomposed move/pick transition stream."""
    states: list[State] = []
    rewards: list[tuple[float, ...]] = []
    discounts: list[float] = []
    state = env.init_state()
    zero_rewards = tuple(0.0 for _ in range(env.cfg.n_agents))

    for _ in range(max(int(episode_steps), 0)):
        move_action = choose_behavior_action(
            state,
            env,
            policy=policy,
            random_policy_prob=random_policy_prob,
        )
        if not move_action.is_move():
            raise AssertionError(f"Move phase returned non-move action: {move_action}")

        moved = env.apply_action(state, move_action)
        states.append(state)
        rewards.append(zero_rewards)
        discounts.append(float(env.cfg.gamma))

        actor = moved.actor
        eligible_types = env.proficiency_positive_types[actor]
        on_task = moved.is_agent_on_task(actor, eligible_types)
        post_pick = moved

        if on_task:
            pick_state = moved.with_pick_phase()
            pick_action = choose_behavior_action(
                pick_state,
                env,
                policy=policy,
                random_policy_prob=random_policy_prob,
            )
            picked, pick_rewards = env.resolve_pick(
                moved,
                pick_type=pick_action.pick_type() if pick_action.is_pick() else None,
            )
            states.append(pick_state)
            rewards.append(pick_rewards)
            discounts.append(1.0 if discount_mode == "rl" else float(env.cfg.gamma))
            post_pick = picked

        state = env.advance_actor(env.spawn_and_despawn(post_pick))

    reward_arr = np.asarray(rewards, dtype=np.float64)
    discount_arr = np.asarray(discounts, dtype=np.float64)
    returns = discounted_returns(reward_arr, discount_arr)
    return EpisodeData(
        states=states,
        rewards=reward_arr,
        discounts=discount_arr,
        returns=returns,
        team_returns=returns.sum(axis=1),
    )


def collect_episodes(
    env,
    *,
    num_episodes: int,
    episode_steps: int,
    policy: str,
    random_policy_prob: float,
    discount_mode: str,
) -> list[EpisodeData]:
    return [
        collect_episode(
            env,
            episode_steps=episode_steps,
            policy=policy,
            random_policy_prob=random_policy_prob,
            discount_mode=discount_mode,
        )
        for _ in range(max(int(num_episodes), 0))
    ]


def split_episodes(
    episodes: Sequence[EpisodeData],
    *,
    train_frac: float,
    seed: int,
) -> tuple[list[EpisodeData], list[EpisodeData]]:
    if not episodes:
        raise ValueError("No episodes were collected")
    if len(episodes) < 2:
        raise ValueError("At least two episodes are required for a held-out episode split")
    indices = np.arange(len(episodes))
    np.random.default_rng(seed).shuffle(indices)
    n_train = int(round(len(indices) * train_frac))
    n_train = min(max(n_train, 1), len(indices) - 1) if len(indices) > 1 else len(indices)
    train_ids = set(indices[:n_train].tolist())
    train = [ep for i, ep in enumerate(episodes) if i in train_ids]
    test = [ep for i, ep in enumerate(episodes) if i not in train_ids]
    if not test:
        test = train
    return train, test


def flatten_states(episodes: Sequence[EpisodeData]) -> list[State]:
    return [state for ep in episodes for state in ep.states]


def flatten_returns(episodes: Sequence[EpisodeData]) -> np.ndarray:
    return np.concatenate([ep.returns for ep in episodes], axis=0)


def flatten_team_returns(episodes: Sequence[EpisodeData]) -> np.ndarray:
    return np.concatenate([ep.team_returns for ep in episodes], axis=0)


def flatten_rewards(episodes: Sequence[EpisodeData]) -> np.ndarray:
    return np.concatenate([ep.rewards for ep in episodes], axis=0)


def count_transitions(episodes: Sequence[EpisodeData]) -> int:
    return int(sum(len(ep.states) for ep in episodes))


def batch_encode_states(
    states: Sequence[State],
    indices: np.ndarray,
    *,
    device: torch.device,
    agent_idx: int = 0,
) -> EncoderOutput:
    encoded = [encoding.encode(states[int(idx)], agent_idx) for idx in indices]
    if any(enc.grid is None or enc.scalar is None for enc in encoded):
        raise ValueError("Offline value evaluation currently requires grid and scalar encodings")
    grids = torch.stack([enc.grid for enc in encoded]).to(device)
    scalars = torch.stack([enc.scalar for enc in encoded]).to(device)
    return EncoderOutput(grid=grids, scalar=scalars)


def batch_encode_all_agents(
    states: Sequence[State],
    indices: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    grids = []
    scalars = []
    for idx in indices:
        g, s = encoding.encode_all_agents(states[int(idx)])
        grids.append(g)
        scalars.append(s)
    return torch.stack(grids).to(device), torch.stack(scalars).to(device)


def train_decentralized(
    states: Sequence[State],
    targets: np.ndarray,
    *,
    model_cfg: ModelConfig,
    env_cfg: EnvConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> list[ValueNetwork]:
    torch.manual_seed(seed)
    nets = [ValueNetwork(model_cfg, env_cfg, td_lambda=0.0).to(device) for _ in range(env_cfg.n_agents)]
    opts = [torch.optim.Adam(net.parameters(), lr=lr) for net in nets]
    indices = np.arange(len(states))

    for net in nets:
        net.train()
    for epoch in range(max(int(epochs), 0)):
        np.random.default_rng(seed + epoch).shuffle(indices)
        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = indices[start:start + batch_size]
            grids, scalars = batch_encode_all_agents(states, batch_idx, device=device)
            y = torch.as_tensor(targets[batch_idx], dtype=torch.float32, device=device)
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            losses = []
            for agent, net in enumerate(nets):
                pred = net.forward_raw(grids[:, agent], scalars[:, agent])
                losses.append(torch.mean((pred - y[:, agent]) ** 2))
            loss = torch.stack(losses).mean()
            loss.backward()
            for opt in opts:
                opt.step()
    for net in nets:
        net.eval()
    return nets


def train_centralized(
    states: Sequence[State],
    targets: np.ndarray,
    *,
    model_cfg: ModelConfig,
    env_cfg: EnvConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> ValueNetwork:
    torch.manual_seed(seed)
    net = ValueNetwork(model_cfg, env_cfg, td_lambda=0.0).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    indices = np.arange(len(states))
    net.train()
    for epoch in range(max(int(epochs), 0)):
        np.random.default_rng(seed + epoch).shuffle(indices)
        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = indices[start:start + batch_size]
            enc = batch_encode_states(states, batch_idx, device=device)
            y = torch.as_tensor(targets[batch_idx], dtype=torch.float32, device=device)
            pred = net(enc)
            loss = torch.mean((pred - y) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    return net


def train_centralized_multihead(
    states: Sequence[State],
    targets: np.ndarray,
    *,
    model_cfg: ModelConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> CentralizedMultiheadValueNetwork:
    torch.manual_seed(seed)
    net = CentralizedMultiheadValueNetwork(model_cfg, n_outputs=targets.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    indices = np.arange(len(states))
    net.train()
    for epoch in range(max(int(epochs), 0)):
        np.random.default_rng(seed + epoch).shuffle(indices)
        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = indices[start:start + batch_size]
            enc = batch_encode_states(states, batch_idx, device=device)
            y = torch.as_tensor(targets[batch_idx], dtype=torch.float32, device=device)
            pred = net.forward_raw(enc.grid, enc.scalar)
            loss = torch.mean((pred - y) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    return net


def evaluate_decentralized(
    nets: Sequence[ValueNetwork],
    states: Sequence[State],
    targets: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    sqerr_sum = np.zeros(targets.shape[1], dtype=np.float64)
    n = 0
    with torch.no_grad():
        indices = np.arange(len(states))
        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = indices[start:start + batch_size]
            grids, scalars = batch_encode_all_agents(states, batch_idx, device=device)
            y = targets[batch_idx]
            preds = []
            for agent, net in enumerate(nets):
                pred = net.forward_raw(grids[:, agent], scalars[:, agent])
                preds.append(pred.detach().cpu().numpy())
            pred_arr = np.stack(preds, axis=1)
            sqerr_sum += np.sum((pred_arr - y) ** 2, axis=0)
            n += len(batch_idx)
    return sqerr_sum / max(n, 1)


def evaluate_centralized(
    net: ValueNetwork,
    states: Sequence[State],
    targets: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    sqerr_sum = 0.0
    n = 0
    with torch.no_grad():
        indices = np.arange(len(states))
        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = indices[start:start + batch_size]
            enc = batch_encode_states(states, batch_idx, device=device)
            y = targets[batch_idx]
            pred = net(enc).detach().cpu().numpy()
            sqerr_sum += float(np.sum((pred - y) ** 2))
            n += len(batch_idx)
    return sqerr_sum / max(n, 1)


def evaluate_centralized_multihead(
    net: CentralizedMultiheadValueNetwork,
    states: Sequence[State],
    targets: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    sqerr_sum = np.zeros(targets.shape[1], dtype=np.float64)
    n = 0
    with torch.no_grad():
        indices = np.arange(len(states))
        for start in range(0, len(indices), max(int(batch_size), 1)):
            batch_idx = indices[start:start + batch_size]
            enc = batch_encode_states(states, batch_idx, device=device)
            pred = net.forward_raw(enc.grid, enc.scalar).detach().cpu().numpy()
            sqerr_sum += np.sum((pred - targets[batch_idx]) ** 2, axis=0)
            n += len(batch_idx)
    return sqerr_sum / max(n, 1)


def reward_stats_by_agent(rewards: np.ndarray, *, zero_tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    nonzero = np.abs(rewards) > zero_tol
    freq = nonzero.mean(axis=0)
    mean_abs = np.full(rewards.shape[1], np.nan, dtype=np.float64)
    abs_rewards = np.abs(rewards)
    for agent in range(rewards.shape[1]):
        mask = nonzero[:, agent]
        if bool(mask.any()):
            mean_abs[agent] = float(abs_rewards[mask, agent].mean())
    return freq, mean_abs


def aggregate_reward_stats(freq: np.ndarray, mean_abs: np.ndarray) -> tuple[float, float]:
    return float(np.mean(freq)), float(np.nanmean(mean_abs)) if not np.all(np.isnan(mean_abs)) else np.nan


def make_rows(
    *,
    rr: int,
    seed: int,
    model_type: str,
    agent_mse: np.ndarray | None,
    scalar_mse: float | None,
    reward_freq: np.ndarray,
    reward_abs: np.ndarray,
    train_episodes: Sequence[EpisodeData],
    test_episodes: Sequence[EpisodeData],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    freq_mean, abs_mean = aggregate_reward_stats(reward_freq, reward_abs)
    common = {
        "rr": rr,
        "seed": seed,
        "n_train_episodes": len(train_episodes),
        "n_test_episodes": len(test_episodes),
        "n_train_transitions": count_transitions(train_episodes),
        "n_test_transitions": count_transitions(test_episodes),
        "policy": args.policy,
        "discount_mode": args.discount_mode,
        "gamma": args.gamma,
        "proficiency_radius": args.proficiency_radius,
        "sigma_a": args.sigma_a,
        "sigma_b": args.sigma_b,
    }

    if scalar_mse is not None:
        rows.append({
            **common,
            "model_type": model_type,
            "agent": "all",
            "test_mse": scalar_mse,
            "nonzero_reward_freq": freq_mean,
            "mean_abs_nonzero_reward": abs_mean,
        })
        return rows

    assert agent_mse is not None
    rows.append({
        **common,
        "model_type": f"{model_type}_mean",
        "agent": "mean",
        "test_mse": float(np.mean(agent_mse)),
        "nonzero_reward_freq": freq_mean,
        "mean_abs_nonzero_reward": abs_mean,
    })
    for agent, mse in enumerate(agent_mse):
        rows.append({
            **common,
            "model_type": model_type,
            "agent": agent,
            "test_mse": float(mse),
            "nonzero_reward_freq": float(reward_freq[agent]),
            "mean_abs_nonzero_reward": float(reward_abs[agent]) if not np.isnan(reward_abs[agent]) else np.nan,
        })
    return rows


def build_env_config(args: argparse.Namespace, rr: int) -> EnvConfig:
    grid_size = args.grid_size
    if len(grid_size) == 1:
        height = width = grid_size[0]
    elif len(grid_size) == 2:
        height, width = grid_size
    else:
        raise ValueError("--grid-size expects one value (square) or two values (height width)")

    return EnvConfig(
        height=height,
        width=width,
        n_agents=args.num_agents,
        n_tasks=args.num_initial_tasks,
        gamma=args.gamma,
        n_task_types=args.num_agents,
        relatedness_width=rr,
        proficiency_width=args.proficiency_radius,
        max_tasks_per_type=args.max_tasks_per_type,
        stochastic=StochasticConfig(
            spawn_prob=args.spawn_prob,
            despawn_mode=DespawnMode.PROBABILITY if args.despawn_mode == "probability" else DespawnMode.NONE,
            despawn_prob=args.despawn_prob,
            sigma_a=args.sigma_a,
            sigma_b=args.sigma_b,
            spawn_on_agent_cells=args.spawn_on_agent_cells,
            spawn_at_round_end=args.spawn_at_round_end,
        ),
    )


def build_model_config(args: argparse.Namespace, encoder: EncoderType) -> ModelConfig:
    return ModelConfig(
        encoder=encoder,
        mlp_dims=tuple(args.mlp_dims),
        conv_specs=tuple((int(ch), int(ks)) for ch, ks in args.conv_specs),
        activation=Activation.LEAKY_RELU if args.activation == "leaky_relu" else Activation.RELU,
        weight_init=WeightInit.ZERO_BIAS,
    )


def evaluate_one_setting(args: argparse.Namespace, *, rr: int, seed: int) -> list[dict[str, object]]:
    set_all_seeds(seed)
    env_cfg = build_env_config(args, rr)
    env = create_env(env_cfg)
    env.set_eval_mode(True, seed=seed)
    try:
        episodes = collect_episodes(
            env,
            num_episodes=args.num_episodes,
            episode_steps=args.episode_steps,
            policy=args.policy,
            random_policy_prob=args.random_policy_prob,
            discount_mode=args.discount_mode,
        )
    finally:
        env.set_eval_mode(False)

    train_episodes, test_episodes = split_episodes(
        episodes,
        train_frac=args.train_frac,
        seed=seed + 17,
    )
    train_states = flatten_states(train_episodes)
    test_states = flatten_states(test_episodes)
    train_returns = flatten_returns(train_episodes)
    test_returns = flatten_returns(test_episodes)
    train_team_returns = flatten_team_returns(train_episodes)
    test_team_returns = flatten_team_returns(test_episodes)
    test_rewards = flatten_rewards(test_episodes)
    reward_freq, reward_abs = reward_stats_by_agent(test_rewards)

    device = torch.device(args.device)
    rows: list[dict[str, object]] = []

    dec_encoder = (
        EncoderType.FILTERED_DEC_CNN_GRID
        if args.dec_encoder == "filtered_dec_cnn_grid"
        else EncoderType.EVERYTHING_CNN_GRID
    )
    dec_model_cfg = build_model_config(args, dec_encoder)
    encoding.init_encoder(dec_encoder, env, n_networks=env_cfg.n_agents)
    dec_nets = train_decentralized(
        train_states,
        train_returns,
        model_cfg=dec_model_cfg,
        env_cfg=env_cfg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=seed + 1000,
    )
    dec_mse = evaluate_decentralized(
        dec_nets,
        test_states,
        test_returns,
        batch_size=args.batch_size,
        device=device,
    )
    rows.extend(make_rows(
        rr=rr,
        seed=seed,
        model_type="decentralized",
        agent_mse=dec_mse,
        scalar_mse=None,
        reward_freq=reward_freq,
        reward_abs=reward_abs,
        train_episodes=train_episodes,
        test_episodes=test_episodes,
        args=args,
    ))

    central_encoder = EncoderType.EVERYTHING_CNN_GRID
    central_model_cfg = build_model_config(args, central_encoder)
    encoding.init_encoder(central_encoder, env, n_networks=1)
    central_net = train_centralized(
        train_states,
        train_team_returns,
        model_cfg=central_model_cfg,
        env_cfg=env_cfg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=seed + 2000,
    )
    central_mse = evaluate_centralized(
        central_net,
        test_states,
        test_team_returns,
        batch_size=args.batch_size,
        device=device,
    )
    rows.extend(make_rows(
        rr=rr,
        seed=seed,
        model_type="centralized",
        agent_mse=None,
        scalar_mse=central_mse,
        reward_freq=reward_freq,
        reward_abs=reward_abs,
        train_episodes=train_episodes,
        test_episodes=test_episodes,
        args=args,
    ))

    if args.include_multihead:
        multihead_net = train_centralized_multihead(
            train_states,
            train_returns,
            model_cfg=central_model_cfg,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=seed + 3000,
        )
        multihead_mse = evaluate_centralized_multihead(
            multihead_net,
            test_states,
            test_returns,
            batch_size=args.batch_size,
            device=device,
        )
        rows.extend(make_rows(
            rr=rr,
            seed=seed,
            model_type="centralized_multihead",
            agent_mse=multihead_mse,
            scalar_mse=None,
            reward_freq=reward_freq,
            reward_abs=reward_abs,
            train_episodes=train_episodes,
            test_episodes=test_episodes,
            args=args,
        ))

    return rows


def parse_conv_specs(raw: str) -> tuple[tuple[int, int], ...]:
    specs = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError("conv specs must look like '16:3,32:3'")
        channels, kernel = item.split(":", 1)
        specs.append((int(channels), int(kernel)))
    if not specs:
        raise argparse.ArgumentTypeError("at least one conv spec is required")
    return tuple(specs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train supervised value regressors on fixed-policy offline Orchard rollouts.",
    )
    parser.add_argument("--num-agents", type=int, default=11)
    parser.add_argument("--grid-size", type=int, nargs="+", default=[9])
    parser.add_argument("--proficiency-radius", type=int, default=5)
    parser.add_argument("--relatedness-radii", type=int, nargs="+", required=True)
    parser.add_argument("--sigma-a", type=float, default=2.0)
    parser.add_argument("--sigma-b", type=float, default=0.0)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--episode-steps", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--policy",
        choices=[
            "nearest",
            "eps_nearest",
            "nearest_rewarding_task",
            "eps_nearest_rewarding_task",
            "nearest_task",
            "random",
            "mixed",
            "nearest_or_random",
        ],
        default="nearest",
    )
    parser.add_argument(
        "--random-policy-prob",
        type=float,
        default=0.5,
        help="Probability of taking a random action when --policy=mixed.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--discount-mode", choices=["rl", "constant"], default="rl")

    parser.add_argument("--num-initial-tasks", type=int, default=3)
    parser.add_argument("--max-tasks-per-type", type=int, default=3)
    parser.add_argument("--spawn-prob", type=float, default=0.01)
    parser.add_argument("--despawn-prob", type=float, default=0.0125)
    parser.add_argument("--despawn-mode", choices=["probability", "none"], default="probability")
    parser.add_argument("--spawn-on-agent-cells", action="store_true")
    parser.add_argument("--spawn-at-round-end", action="store_true")

    parser.add_argument(
        "--dec-encoder",
        choices=["filtered_dec_cnn_grid", "everything_cnn_grid"],
        default="filtered_dec_cnn_grid",
    )
    parser.add_argument("--conv-specs", type=parse_conv_specs, default=((16, 3),))
    parser.add_argument("--mlp-dims", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--activation", choices=["leaky_relu", "relu"], default="leaky_relu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--include-multihead", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("offline_value_eval.csv"))
    return parser.parse_args(argv)


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OFFLINE_VALUE_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OFFLINE_VALUE_CSV_FIELDS})


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested, but CUDA is not available")
    if not 0.0 < args.train_frac <= 1.0:
        raise ValueError("--train-frac must be in (0, 1]")
    if not 0.0 <= args.random_policy_prob <= 1.0:
        raise ValueError("--random-policy-prob must be in [0, 1]")

    rows: list[dict[str, object]] = []
    for rr in args.relatedness_radii:
        for seed in args.seeds:
            setting_rows = evaluate_one_setting(args, rr=rr, seed=seed)
            rows.extend(setting_rows)
            dec_mean = [
                row for row in setting_rows
                if row["model_type"] == "decentralized_mean"
            ][0]
            print(
                f"rr={rr} seed={seed}: "
                f"decentralized_mean_mse={float(dec_mean['test_mse']):.6g}"
            )

    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
