"""Evaluation: value metrics, rollout returns, picks-per-step, ground truth."""

from __future__ import annotations

from typing import Callable, Iterator

import torch

import orchard.encoding as encoding
from orchard.enums import Action, Heuristic, PickMode, TDTarget, TrainMode
from orchard.env.base import BaseEnv
from orchard.model import ValueNetwork
from orchard.policy import (
    argmax_a_Q_team, argmax_a_Q_team_batched,
    nearest_task_action, heuristic_action,
    nearest_apple_action,
)
from orchard.seed import rng
from orchard.datatypes import EnvConfig, EvalConfig, State, Transition


# ---------------------------------------------------------------------------
# Rollout trajectory
# ---------------------------------------------------------------------------
def rollout_trajectory(
    start_state: State,
    policy_fn: "Callable[[State, bool], Action]",
    env: BaseEnv,
    n_steps: int,
) -> "Iterator[Transition]":
    """Yield Transitions for n_steps agent decisions using unified two-phase structure.

    Every decision consists of:
      Phase 1: move action (always). Yields one Transition at discount gamma.
      Phase 2: only if actor lands on a task cell.
               FORCED: auto-picks (policy_fn called with phase2=True always returns pick(τ)).
               CHOICE: policy_fn called with phase2=True returns pick(τ) or STAY.
               Yields one Transition at discount 1.0.

    n_steps counts phase-1 decisions (agent turns), not transitions.
    """
    s = start_state
    gamma = env.cfg.gamma
    zero_rewards = tuple(0.0 for _ in range(env.cfg.n_agents))

    for _ in range(n_steps):
        # --- Phase 1: move ---
        move_action = policy_fn(s, False)
        assert move_action.is_move(), f"Phase 1 must be a move action, got {move_action}"
        s_moved = env.apply_action(s, move_action)

        if s_moved.is_agent_on_task(s_moved.actor):
            # Yield phase-1 transition
            yield Transition(
                s_t=s, action=move_action, s_t_after=s_moved, s_t_next=s_moved,
                rewards=zero_rewards, discount=gamma,
            )
            # --- Phase 2: pick decision ---
            if env.cfg.pick_mode == PickMode.FORCED:
                # Auto-pick: find the task type at this cell and issue pick(τ)
                from orchard.enums import make_pick_action
                tau = s_moved.task_type_at(s_moved.agent_positions[s_moved.actor])
                pick_action = make_pick_action(tau)
            else:
                # CHOICE: ask policy for pick(τ) or STAY
                pick_action = policy_fn(s_moved, True)

            s_picked, pick_rewards = env.resolve_pick(
                s_moved,
                pick_type=pick_action.pick_type() if pick_action.is_pick() else None,
            )
            s_next = env.advance_actor(env.spawn_and_despawn(s_picked))
            yield Transition(
                s_t=s_moved, action=pick_action, s_t_after=s_picked,
                s_t_next=s_next, rewards=pick_rewards, discount=1.0,
            )
            s = s_next
        else:
            # No task — single transition, no phase 2
            s_next = env.advance_actor(env.spawn_and_despawn(s_moved))
            yield Transition(
                s_t=s, action=move_action, s_t_after=s_moved,
                s_t_next=s_next, rewards=zero_rewards, discount=gamma,
            )
            s = s_next


# ---------------------------------------------------------------------------
# Rollout returns (ground truth)
# ---------------------------------------------------------------------------
def rollout_returns(
    start_state: State,
    policy_fn: "Callable[[State, bool], Action]",
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
    td_target: TDTarget = TDTarget.PRE_ACTION,
    heuristic: Heuristic = Heuristic.NEAREST_TASK,
) -> list[list[float]]:
    """True V_i(s) for all test states under heuristic policy."""
    def policy(s: State, phase2: bool = False) -> Action:
        return heuristic_action(s, env.cfg, heuristic, phase2=phase2)

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
    heuristic: Heuristic = Heuristic.NEAREST_TASK,
) -> list[State]:
    """Collect unique states by running heuristic policy."""
    if n_collection_steps is None:
        n_collection_steps = n_test_states * 20

    seen: set = set()
    states: list[State] = []
    s = env.init_state()

    for _ in range(n_collection_steps):
        if len(states) >= n_test_states:
            break

        key = (s.agent_positions, s.task_positions, s.actor)
        if key not in seen:
            seen.add(key)
            states.append(s)

        move_action = heuristic_action(s, env.cfg, heuristic, phase2=False)
        s_moved = env.apply_action(s, move_action)
        if s_moved.is_agent_on_task(s_moved.actor):
            from orchard.enums import make_pick_action as _mpa
            if env.cfg.pick_mode == PickMode.FORCED:
                tau = s_moved.task_type_at(s_moved.agent_positions[s_moved.actor])
                pick_action = _mpa(tau)
            else:
                pick_action = heuristic_action(s_moved, env.cfg, heuristic, phase2=True)
            s_picked, _ = env.resolve_pick(s_moved, pick_type=pick_action.pick_type() if pick_action.is_pick() else None)
            s = env.advance_actor(env.spawn_and_despawn(s_picked))
        else:
            s = env.advance_actor(env.spawn_and_despawn(s_moved))

    return states


def collect_after_state_test_states(
    env: BaseEnv,
    n_test_states: int,
    n_collection_steps: int | None = None,
    heuristic: Heuristic = Heuristic.NEAREST_TASK,
) -> list[State]:
    """Collect unique after-states by running heuristic policy."""
    if n_collection_steps is None:
        n_collection_steps = n_test_states * 20

    seen: set = set()
    states: list[State] = []
    s = env.init_state()

    for _ in range(n_collection_steps):
        if len(states) >= n_test_states:
            break

        move_action = heuristic_action(s, env.cfg, heuristic, phase2=False)
        s_moved = env.apply_action(s, move_action)

        if s_moved.is_agent_on_task(s_moved.actor):
            from orchard.enums import make_pick_action as _mpa
            if env.cfg.pick_mode == PickMode.FORCED:
                tau = s_moved.task_type_at(s_moved.agent_positions[s_moved.actor])
                pick_action = _mpa(tau)
            else:
                pick_action = heuristic_action(s_moved, env.cfg, heuristic, phase2=True)
            s_picked, _ = env.resolve_pick(s_moved, pick_type=pick_action.pick_type() if pick_action.is_pick() else None)
            key = (s_picked.agent_positions, s_picked.task_positions, s_picked.actor)
            if key not in seen:
                seen.add(key)
                states.append(s_picked)
            s = env.advance_actor(env.spawn_and_despawn(s_picked))
        else:
            key = (s_moved.agent_positions, s_moved.task_positions, s_moved.actor)
            if key not in seen:
                seen.add(key)
                states.append(s_moved)
            s = env.advance_actor(env.spawn_and_despawn(s_moved))

    return states


def collect_reward_test_states(
    env: BaseEnv,
    n_test_states: int,
    n_collection_steps: int | None = None,
) -> list[State]:
    """Collect movement after-states under random policy for reward learning eval."""
    from orchard.policy import get_all_actions as _gaa

    if n_collection_steps is None:
        n_collection_steps = n_test_states * 20

    seen: set = set()
    states: list[State] = []
    s = env.init_state()

    for _ in range(n_collection_steps):
        if len(states) >= n_test_states:
            break

        # Phase 1: random move action
        move_actions = _gaa(env.cfg)
        action = move_actions[rng.randint(0, len(move_actions) - 1)]
        s_moved = env.apply_action(s, action)

        key = (s_moved.agent_positions, s_moved.task_positions, s_moved.actor)
        if key not in seen:
            seen.add(key)
            states.append(s_moved)

        # Advance env (two-phase)
        if s_moved.is_agent_on_task(s_moved.actor):
            from orchard.enums import make_pick_action as _mpa
            if env.cfg.pick_mode == PickMode.FORCED:
                tau = s_moved.task_type_at(s_moved.agent_positions[s_moved.actor])
                s_picked, _ = env.resolve_pick(s_moved)
            else:
                # Random phase-2 action
                from orchard.policy import get_phase2_actions as _gp2
                p2_acts = _gp2(s_moved, env.cfg)
                p2_act = p2_acts[rng.randint(0, len(p2_acts) - 1)]
                s_picked, _ = env.resolve_pick(s_moved, pick_type=p2_act.pick_type() if p2_act.is_pick() else None)
            s = env.advance_actor(env.spawn_and_despawn(s_picked))
        else:
            s = env.advance_actor(env.spawn_and_despawn(s_moved))

    return states


# ---------------------------------------------------------------------------
# Picks per step (legacy)
# ---------------------------------------------------------------------------
def picks_per_step(
    start_state: State,
    policy_fn: Callable[[State], Action],
    env: BaseEnv,
    n_steps: int,
) -> float:
    """Fraction of steps that result in a task pick."""
    picks = 0
    for t in rollout_trajectory(start_state, policy_fn, env, n_steps):
        if any(r != 0.0 for r in t.rewards):
            picks += 1
    return picks / n_steps if n_steps > 0 else 0.0


# ---------------------------------------------------------------------------
# Reward per step + correct/wrong picks per step (n_task_types > 1)
# ---------------------------------------------------------------------------
def evaluate_policy_metrics(
    start_state: State,
    policy_fn: Callable[[State], Action],
    env: BaseEnv,
    n_steps: int,
) -> dict[str, float]:
    """Compute rps, correct_pps, wrong_pps over a rollout.

    For n_task_types > 1 only. Uses task_assignments to classify picks.
    """
    total_reward = 0.0
    correct_picks = 0
    wrong_picks = 0

    for t in rollout_trajectory(start_state, policy_fn, env, n_steps):
        actor = t.s_t.actor
        actor_reward = t.rewards[actor]
        total_reward += actor_reward

        if env.cfg.task_assignments is not None and t.action.is_pick():
            tau = t.action.pick_type()
            g_actor = set(env.cfg.task_assignments[actor])
            if tau in g_actor:
                correct_picks += 1
            else:
                wrong_picks += 1
        elif env.cfg.task_assignments is None and actor_reward != 0.0:
            # Legacy path: no task types, detect picks by nonzero reward
            if actor_reward > 0:
                correct_picks += 1
            else:
                wrong_picks += 1

    return {
        "rps": total_reward / n_steps if n_steps > 0 else 0.0,
        "correct_pps": correct_picks / n_steps if n_steps > 0 else 0.0,
        "wrong_pps": wrong_picks / n_steps if n_steps > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Value learning evaluation
# ---------------------------------------------------------------------------
def evaluate_value_learning(
    networks: list[ValueNetwork],
    env_cfg: EnvConfig,
    test_states: list[State],
    ground_truth: list[list[float]],
) -> dict[str, float]:
    """Compare V predictions to precomputed ground truth."""
    n_agents = env_cfg.n_agents

    sum_ae: list[float] = [0.0] * n_agents
    sum_se: list[float] = [0.0] * n_agents
    sum_ape: list[float] = [0.0] * n_agents
    sum_abs_true: list[float] = [0.0] * n_agents
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
    batch_actions: bool = False,
    heuristic: Heuristic = Heuristic.NEAREST_TASK,
) -> dict[str, float]:
    """Evaluate greedy vs heuristic policy."""
    eval_start = env.init_state()

    def greedy_policy(s: State, phase2: bool = False) -> Action:
        if batch_actions:
            return argmax_a_Q_team_batched(s, networks, env, phase2=phase2)
        return argmax_a_Q_team(s, networks, env, phase2=phase2)

    def baseline_policy(s: State, phase2: bool = False) -> Action:
        return heuristic_action(s, env.cfg, heuristic, phase2=phase2)

    heuristic_name = heuristic.name.lower()

    if env.cfg.n_task_types > 1:
        greedy_metrics = evaluate_policy_metrics(eval_start, greedy_policy, env, eval_steps)
        baseline_metrics = evaluate_policy_metrics(eval_start, baseline_policy, env, eval_steps)
        return {
            "greedy_rps": greedy_metrics["rps"],
            "greedy_correct_pps": greedy_metrics["correct_pps"],
            "greedy_wrong_pps": greedy_metrics["wrong_pps"],
            f"{heuristic_name}_rps": baseline_metrics["rps"],
            f"{heuristic_name}_correct_pps": baseline_metrics["correct_pps"],
            f"{heuristic_name}_wrong_pps": baseline_metrics["wrong_pps"],
        }
    else:
        greedy_metrics = evaluate_policy_metrics(eval_start, greedy_policy, env, eval_steps)
        baseline_metrics = evaluate_policy_metrics(eval_start, baseline_policy, env, eval_steps)
        return {
            "greedy_rps": greedy_metrics["rps"],
            f"{heuristic_name}_rps": baseline_metrics["rps"],
        }


# ---------------------------------------------------------------------------
# Reward learning
# ---------------------------------------------------------------------------
def compute_reward_ground_truth(
    test_states: list[State],
    env_cfg: EnvConfig,
    n_networks: int,
    centralized: bool,
) -> tuple[list[list[float]], list[list[str]]]:
    """Deterministic reward ground truth for each (state, network).

    For n_task_types == 1: categories are "zero" | "picker" | "other" (dec) or "zero" | "pick" (cen)
    For n_task_types > 1: categories are "no_pick" | "my_task" | "other_task" (dec) or "no_pick" | "pick" (cen)
    """
    ground_truth: list[list[float]] = []
    categories: list[list[str]] = []

    if env_cfg.n_task_types == 1:
        # --- Legacy path ---
        r_picker = env_cfg.r_picker
        n_agents = env_cfg.n_agents
        r_other = (1.0 - r_picker) / (n_agents - 1) if n_agents > 1 else 0.0
        r_sum = r_picker + r_other * (n_agents - 1)

        for s in test_states:
            actor = s.actor
            on_task = s.is_agent_on_task(actor)
            gt_row: list[float] = []
            cat_row: list[str] = []

            if centralized:
                if on_task:
                    gt_row.append(r_sum)
                    cat_row.append("pick")
                else:
                    gt_row.append(0.0)
                    cat_row.append("zero")
            else:
                for i in range(n_networks):
                    if not on_task:
                        gt_row.append(0.0)
                        cat_row.append("zero")
                    elif i == actor:
                        gt_row.append(r_picker)
                        cat_row.append("picker")
                    else:
                        gt_row.append(r_other)
                        cat_row.append("other")

            ground_truth.append(gt_row)
            categories.append(cat_row)
    else:
        # --- Task specialization path ---
        for s in test_states:
            actor = s.actor
            on_task = s.is_agent_on_task(actor)
            gt_row = []
            cat_row = []

            if centralized:
                if not on_task:
                    gt_row.append(0.0)
                    cat_row.append("no_pick")
                else:
                    # What type is at actor's position?
                    tau = s.task_type_at(s.agent_positions[actor])
                    g_actor = set(env_cfg.task_assignments[actor])
                    reward = env_cfg.r_high if tau in g_actor else env_cfg.r_low
                    gt_row.append(reward)
                    cat_row.append("pick")
            else:
                for i in range(n_networks):
                    if not on_task:
                        gt_row.append(0.0)
                        cat_row.append("no_pick")
                    elif i == actor:
                        tau = s.task_type_at(s.agent_positions[actor])
                        g_actor = set(env_cfg.task_assignments[actor])
                        if tau in g_actor:
                            gt_row.append(env_cfg.r_high)
                            cat_row.append("my_task")
                        else:
                            gt_row.append(env_cfg.r_low)
                            cat_row.append("other_task")
                    else:
                        gt_row.append(0.0)
                        cat_row.append("no_pick")

            ground_truth.append(gt_row)
            categories.append(cat_row)

    return ground_truth, categories


def evaluate_reward_learning(
    networks: list[ValueNetwork],
    env_cfg: EnvConfig,
    test_states: list[State],
    ground_truth: list[list[float]],
    categories: list[list[str]],
    centralized: bool,
) -> dict[str, float]:
    """MAE per agent, per category, averaged."""
    n_networks = len(networks)

    sum_ae: dict[tuple[int, str], float] = {}
    counts: dict[tuple[int, str], int] = {}
    total_ae: list[float] = [0.0] * n_networks
    total_count: list[int] = [0] * n_networks

    for net in networks:
        net.eval()

    with torch.no_grad():
        for k, s in enumerate(test_states):
            for i in range(n_networks):
                pred = networks[i](encoding.encode(s, i)).item()
                true_v = ground_truth[k][i]
                cat = categories[k][i]
                ae = abs(pred - true_v)

                total_ae[i] += ae
                total_count[i] += 1

                key = (i, cat)
                sum_ae[key] = sum_ae.get(key, 0.0) + ae
                counts[key] = counts.get(key, 0) + 1

    for net in networks:
        net.train()

    result: dict[str, float] = {}

    for i in range(n_networks):
        result[f"mae_agent_{i}"] = total_ae[i] / max(total_count[i], 1)
    result["mae_avg"] = sum(total_ae) / max(sum(total_count), 1)

    # Determine categories based on mode
    if env_cfg.n_task_types == 1:
        all_cats = ["zero", "pick"] if centralized else ["zero", "picker", "other"]
    else:
        all_cats = ["no_pick", "pick"] if centralized else ["no_pick", "my_task", "other_task"]

    for cat in all_cats:
        cat_ae_sum = 0.0
        cat_count_sum = 0
        for i in range(n_networks):
            key = (i, cat)
            c = counts.get(key, 0)
            if c > 0:
                result[f"mae_{cat}_agent_{i}"] = sum_ae[key] / c
            else:
                result[f"mae_{cat}_agent_{i}"] = 0.0
            cat_ae_sum += sum_ae.get(key, 0.0)
            cat_count_sum += c
        result[f"mae_{cat}_avg"] = cat_ae_sum / max(cat_count_sum, 1)

    return result
