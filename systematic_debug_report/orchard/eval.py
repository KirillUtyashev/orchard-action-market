"""Evaluation: value metrics, rollout returns, picks-per-step, ground truth."""

from __future__ import annotations

from typing import Callable, Iterator

import torch

import orchard.encoding as encoding
from orchard.enums import Action, TDTarget, TrainMode
from orchard.env.base import BaseEnv
from orchard.model import ValueNetwork
from orchard.policy import argmax_a_Q_team, nearest_apple_action
from orchard.seed import rng
from orchard.datatypes import EnvConfig, EvalConfig, State, Transition


def rollout_trajectory(
    start_state: State,
    policy_fn: Callable[[State], Action],
    env: BaseEnv,
    n_steps: int,
) -> Iterator[Transition]:
    """Yield Transitions for n_steps agent decisions.

    A pick step yields two transitions:
      1. Move transition (γ=gamma, r=0, action=chosen action)
      2. Pick transition (γ=1.0, r=pick_rewards, action=PICK)
    A non-pick step yields one transition (γ=gamma, r=0).
    """
    s = start_state
    gamma = env.cfg.gamma
    zero_rewards = tuple(0.0 for _ in range(env.cfg.n_agents))

    for _ in range(n_steps):
        action = policy_fn(s)
        s_moved = env.apply_action(s, action)

        if env.cfg.force_pick and s_moved.is_agent_on_apple(s_moved.actor):
            # Transition 1: move (agent lands on apple)
            yield Transition(
                s_t=s,
                action=action,
                s_t_after=s_moved,
                s_t_next=s_moved,
                rewards=zero_rewards,
                discount=gamma,
            )

            # Transition 2: forced pick
            s_picked, pick_rewards = env.resolve_pick(s_moved)
            s_next = env.advance_actor(env.spawn_and_despawn(s_picked))
            yield Transition(
                s_t=s_moved,
                action=Action.PICK,
                s_t_after=s_picked,
                s_t_next=s_next,
                rewards=pick_rewards,
                discount=1.0,
            )
            s = s_next
        else:
            # Single transition: move (no pick)
            s_next = env.advance_actor(env.spawn_and_despawn(s_moved))
            yield Transition(
                s_t=s,
                action=action,
                s_t_after=s_moved,
                s_t_next=s_next,
                rewards=zero_rewards,
                discount=gamma,
            )
            s = s_next

# ---------------------------------------------------------------------------
# Rollout returns (ground truth)
# ---------------------------------------------------------------------------
def rollout_returns(
    start_state: State,
    policy_fn: Callable[[State], Action],
    env: BaseEnv,
    rollout_len: int,
) -> list[float]:
    """Roll out a policy, return discounted return per agent."""
    transitions: list[Transition] = []
    for t in rollout_trajectory(start_state, policy_fn, env, rollout_len):
        transitions.append(t)

    n_agents = env.cfg.n_agents
    G = [0.0] * n_agents
    for t in reversed(transitions):
        for i in range(n_agents):
            G[i] = t.discount * (t.rewards[i] + G[i])
    return G

def precompute_ground_truth(
    test_states: list[State],
    env: BaseEnv,
    eval_cfg: EvalConfig,
    td_target: TDTarget = TDTarget.PRE_ACTION
) -> list[list[float]]:
    """True V_i(s) for all test states under nearest-apple policy.

    Returns: result[k][i] = V_i(test_states[k]).
    """
    def policy(s: State) -> Action:
        return nearest_apple_action(s, env.cfg)

    results = []
    for s in test_states:
        if td_target == TDTarget.AFTER_STATE:
            s_pre = env.advance_actor(env.spawn_and_despawn(s))
            results.append(rollout_returns(s_pre, policy, env, eval_cfg.rollout_len))
        else:
            results.append(rollout_returns(s, policy, env, eval_cfg.rollout_len))
    return results

# ---------------------------------------------------------------------------
# Test state collection
# ---------------------------------------------------------------------------
def collect_on_policy_test_states(
    env: BaseEnv,
    n_test_states: int,
    n_collection_steps: int | None = None,
) -> list[State]:
    """Collect unique states by running nearest-apple policy.

    Runs for up to n_collection_steps (default: n_test_states * 20) and
    samples every state, keeping only unique ones, up to n_test_states.
    """
    if n_collection_steps is None:
        n_collection_steps = n_test_states * 20

    seen: set[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], int] = set()
    states: list[State] = []
    s = env.init_state()

    for _ in range(n_collection_steps):
        if len(states) >= n_test_states:
            break

        key = (s.agent_positions, s.apple_positions, s.actor)
        if key not in seen:
            seen.add(key)
            states.append(s)

        action = nearest_apple_action(s, env.cfg)
        transition = env.step(s, action)
        s = transition.s_t_next

    return states


def collect_after_state_test_states(
    env: BaseEnv,
    n_test_states: int,
    n_collection_steps: int | None = None,
) -> list[State]:
    """Collect unique after-states by running nearest-apple policy.

    After-states collected are:
    - movement after-states (agent moved, no pick)
    - pick after-states (apple removed, pre-spawn)
    NOT the intermediate movement-onto-apple state.
    """
    if n_collection_steps is None:
        n_collection_steps = n_test_states * 20

    seen: set = set()
    states: list[State] = []
    s = env.init_state()

    for _ in range(n_collection_steps):
        if len(states) >= n_test_states:
            break

        action = nearest_apple_action(s, env.cfg)
        s_moved = env.apply_action(s, action)
        on_apple = s_moved.is_agent_on_apple(s_moved.actor)

        if on_apple:
            s_picked, _ = env.resolve_pick(s_moved)
            # Store pick after-state (pre-spawn)
            key = (s_picked.agent_positions, s_picked.apple_positions, s_picked.actor)
            if key not in seen:
                seen.add(key)
                states.append(s_picked)
            s = env.advance_actor(env.spawn_and_despawn(s_picked))
        else:
            # Store movement after-state
            key = (s_moved.agent_positions, s_moved.apple_positions, s_moved.actor)
            if key not in seen:
                seen.add(key)
                states.append(s_moved)
            s = env.advance_actor(env.spawn_and_despawn(s_moved))

    return states

# ---------------------------------------------------------------------------
# Picks per step
# ---------------------------------------------------------------------------
def picks_per_step(
    start_state: State,
    policy_fn: Callable[[State], Action],
    env: BaseEnv,
    n_steps: int,
) -> float:
    """Fraction of steps that result in an apple pick."""
    picks = 0
    for t in rollout_trajectory(start_state, policy_fn, env, n_steps):
        if any(r != 0.0 for r in t.rewards):
            picks += 1
    return picks / n_steps if n_steps > 0 else 0.0


# ---------------------------------------------------------------------------
# Value learning evaluation
# ---------------------------------------------------------------------------
def evaluate_value_learning(
    networks: list[ValueNetwork],
    env_cfg: EnvConfig,
    test_states: list[State],
    ground_truth: list[list[float]],
) -> dict[str, float]:
    """Compare V predictions to precomputed ground truth.

    Returns dict with per-agent and average: mae, pct_error, mape, bias.
    """
    n_agents = env_cfg.n_agents

    # Per-agent accumulators
    sum_ae: list[float] = [0.0] * n_agents       # absolute error
    sum_se: list[float] = [0.0] * n_agents       # signed error (for bias)
    sum_ape: list[float] = [0.0] * n_agents      # absolute percentage error
    sum_abs_true: list[float] = [0.0] * n_agents  # for pct_error normalization
    count = len(test_states)

    for net in networks:
        net.eval()

    with torch.no_grad():
        for k, s in enumerate(test_states):
            for i in range(len(networks)):
                pred = networks[i](encoding.encode(s, i)).item()
                true_v = ground_truth[k][i]
                err = pred - true_v
                ae = abs(err)

                sum_ae[i] += ae
                sum_se[i] += err
                sum_abs_true[i] += abs(true_v)
                if abs(true_v) > 1e-8:
                    sum_ape[i] += ae / abs(true_v)

    for net in networks:
        net.train()

    result: dict[str, float] = {}
    for i in range(len(networks)):
        mae = sum_ae[i] / max(count, 1)
        bias = sum_se[i] / max(count, 1)
        mean_abs_true = sum_abs_true[i] / max(count, 1)
        pct_error = mae / mean_abs_true if mean_abs_true > 1e-8 else 0.0
        mape = sum_ape[i] / max(count, 1)

        result[f"mae_agent_{i}"] = mae
        result[f"pct_error_agent_{i}"] = pct_error
        result[f"mape_agent_{i}"] = mape
        result[f"bias_agent_{i}"] = bias

    # Averages
    result["mae_avg"] = sum(result[f"mae_agent_{i}"] for i in range(len(networks))) / len(networks)
    result["pct_error_avg"] = sum(result[f"pct_error_agent_{i}"] for i in range(len(networks))) / len(networks)
    result["mape_avg"] = sum(result[f"mape_agent_{i}"] for i in range(len(networks))) / len(networks)
    result["bias_avg"] = sum(result[f"bias_agent_{i}"] for i in range(len(networks))) / len(networks)

    return result


# ---------------------------------------------------------------------------
# Policy learning evaluation
# ---------------------------------------------------------------------------
def evaluate_policy_learning(
    networks: list[ValueNetwork],
    env: BaseEnv,
    eval_steps: int,
) -> dict[str, float]:
    """Evaluate greedy vs nearest-apple picks-per-step."""
    eval_start = env.init_state()

    def greedy_policy(s: State) -> Action:
        return argmax_a_Q_team(s, networks, env)

    def nearest_policy(s: State) -> Action:
        return nearest_apple_action(s, env.cfg)

    return {
        "greedy_pps": picks_per_step(eval_start, greedy_policy, env, eval_steps),
        "nearest_pps": picks_per_step(eval_start, nearest_policy, env, eval_steps),
    }
