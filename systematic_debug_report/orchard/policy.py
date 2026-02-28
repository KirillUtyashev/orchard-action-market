"""Policy functions: Q_team, greedy, nearest_apple, epsilon_greedy."""

from __future__ import annotations

import torch

import orchard.encoding as encoding
from orchard.enums import NUM_ACTIONS, Action, ACTION_PRIORITY
from orchard.env.base import BaseEnv
from orchard.model import ValueNetwork
from orchard.seed import rng
from orchard.datatypes import EnvConfig, State


def Q_team(
    state: State,
    action: Action,
    networks: list[ValueNetwork],
    env: BaseEnv,
) -> float:
    """Team Q-value: sum of all agents' values of the after-state.

    Q(s, a) = Σ_i V_i(after_state(s, a))
    """
    s_after = env.apply_action(state, action)
    total: float = 0.0
    with torch.no_grad():
        for i, network in enumerate(networks):
            total += network(encoding.encode(s_after, i)).item()
    return total


def argmax_a_Q_team(
    state: State,
    networks: list[ValueNetwork],
    env: BaseEnv,
) -> Action:
    """Greedy action: argmax_a Q_team(s, a). Tie-break via ACTION_PRIORITY."""
    best_value: float | None = None
    best_action: Action = ACTION_PRIORITY[0]

    for action in ACTION_PRIORITY:
        q = Q_team(state, action, networks, env)
        if best_value is None or q > best_value:
            best_value = q
            best_action = action

    return best_action


def nearest_apple_action(state: State, env_cfg: EnvConfig) -> Action:
    """Move actor toward nearest apple (Manhattan distance).

    Deterministic tie-break via ACTION_PRIORITY order.
    """
    if not state.apple_positions:
        return Action.STAY
    
    actor = state.actor
    ar, ac = state.agent_positions[actor]
    best_dist = float("inf")
    best_action = Action.STAY

    for action in ACTION_PRIORITY:
        dr, dc = action.delta
        nr = max(0, min(env_cfg.height - 1, ar + dr))
        nc = max(0, min(env_cfg.width - 1, ac + dc))
        min_d = min(
            abs(nr - ap.row) + abs(nc - ap.col)
            for ap in state.apple_positions
        )
        if min_d < best_dist:
            best_dist = min_d
            best_action = action

    return best_action


def epsilon_greedy(
    state: State,
    networks: list[ValueNetwork],
    env: BaseEnv,
    epsilon: float,
) -> Action:
    """With probability epsilon choose random, else greedy.

    Uses module-level RNG (seeded at startup via set_all_seeds).
    """
    if rng.random() < epsilon:
        return Action(rng.randint(0, NUM_ACTIONS - 1))
    return argmax_a_Q_team(state, networks, env)
