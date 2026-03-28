"""Generate states and compare greedy policies across multiple trained runs."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from orchard.compare_values.loader import LoadedRun
from orchard.datatypes import State
from orchard.enums import Action, ACTION_PRIORITY, Heuristic, PickMode, TDTarget, num_actions
from orchard.env import create_env
from orchard.eval import collect_after_state_test_states, collect_on_policy_test_states
from orchard.policy import get_all_actions, heuristic_action
from orchard.seed import rng, set_all_seeds


@dataclass
class PolicyComparison:
    """Result of comparing greedy policies on one state."""
    state_index: int
    state: State

    # Per-run results (parallel lists, one entry per LoadedRun)
    actions: list[Action]                    # greedy action chosen
    q_values: list[dict[Action, float]]      # Q_team(s, a) for all a

    @property
    def agrees(self) -> bool:
        return len(set(self.actions)) == 1

    @property
    def n_distinct_actions(self) -> int:
        return len(set(self.actions))

    @property
    def q_gap(self) -> float:
        """Mean gap between best and second-best Q across runs."""
        gaps = []
        for qv in self.q_values:
            sorted_q = sorted(qv.values(), reverse=True)
            if len(sorted_q) >= 2 and not math.isnan(sorted_q[0]):
                gaps.append(sorted_q[0] - sorted_q[1])
        return sum(gaps) / len(gaps) if gaps else 0.0


def generate_states(
    run: LoadedRun,
    n_states: int,
    seed: int,
) -> list[State]:
    """Generate comparison states using heuristic policy."""
    set_all_seeds(seed)
    env = create_env(run.cfg.env)
    heuristic = run.cfg.train.heuristic

    if run.cfg.train.td_target == TDTarget.AFTER_STATE:
        return collect_after_state_test_states(env, n_states, heuristic=heuristic)
    else:
        return collect_on_policy_test_states(env, n_states, heuristic=heuristic)


def compute_q_values(
    state: State,
    run: LoadedRun,
) -> dict[Action, float]:
    """Compute Q_team(s, a) for all valid actions."""
    env = create_env(run.cfg.env)
    all_actions = get_all_actions(run.cfg.env)
    q: dict[Action, float] = {}

    with torch.no_grad():
        for action in all_actions:
            if action.is_pick():
                s_after, _ = env.resolve_pick(state, pick_type=action.pick_type())
            else:
                s_after = env.apply_action(state, action)
            total = 0.0
            for i, net in enumerate(run.networks):
                total += net(run.encoder.encode(s_after, i)).item()
            q[action] = total

    return q


def greedy_action(q: dict[Action, float]) -> Action:
    """Pick best action from Q dict. Movement actions tie-break by ACTION_PRIORITY,
    then pick actions in order."""
    best_val: float | None = None
    best_act: Action | None = None

    # Iterate movement actions in priority order first, then pick actions
    ordered = list(ACTION_PRIORITY)
    for act in sorted(q.keys(), key=lambda a: a.value):
        if act not in ordered:
            ordered.append(act)

    for act in ordered:
        if act in q:
            if best_val is None or q[act] > best_val:
                best_val = q[act]
                best_act = act

    return best_act if best_act is not None else ACTION_PRIORITY[0]


def generate_training_sample_states(
    run: LoadedRun,
    n_sample: int = 100,
) -> list[State]:
    """Reproduce the exact sample states from train.py (random actions, fixed seed 9999)."""
    set_all_seeds(9999)
    env = create_env(run.cfg.env)
    pick_mode = run.cfg.env.pick_mode
    n_act = num_actions(pick_mode, run.cfg.env.n_task_types)

    states: list[State] = []
    s_tmp = env.init_state()

    for _ in range(n_sample * 3):
        a_tmp = Action(rng.randint(0, n_act - 1))

        if run.cfg.train.td_target == TDTarget.AFTER_STATE:
            if a_tmp.is_pick():
                s_after, _ = env.resolve_pick(s_tmp, pick_type=a_tmp.pick_type())
                if len(states) < n_sample:
                    states.append(s_after)
                s_tmp = env.advance_actor(env.spawn_and_despawn(s_after))
            else:
                s_moved = env.apply_action(s_tmp, a_tmp)
                if pick_mode == PickMode.FORCED and s_moved.is_agent_on_task(s_moved.actor):
                    s_after, _ = env.resolve_pick(s_moved)
                    if len(states) < n_sample:
                        states.append(s_after)
                    s_tmp = env.advance_actor(env.spawn_and_despawn(s_after))
                else:
                    s_after = s_moved
                    if len(states) < n_sample:
                        states.append(s_after)
                    s_tmp = env.advance_actor(env.spawn_and_despawn(s_moved))
        else:
            if len(states) < n_sample:
                states.append(s_tmp)
            tr = env.step(s_tmp, a_tmp)
            s_tmp = tr.s_t_next

    return states


def add_heuristic_policy(
    comparisons: list[PolicyComparison],
    env_cfg,
    heuristic: Heuristic = Heuristic.NEAREST_TASK,
) -> None:
    """Append heuristic action to each comparison in-place.

    Q-values are set to NaN (heuristic is not Q-based).
    """
    all_actions = get_all_actions(env_cfg)
    nan_q = {a: float("nan") for a in all_actions}
    for comp in comparisons:
        a = heuristic_action(comp.state, env_cfg, heuristic)
        comp.actions.append(a)
        comp.q_values.append(dict(nan_q))


# Backward compat alias
def add_nearest_policy(comparisons, env_cfg):
    add_heuristic_policy(comparisons, env_cfg, Heuristic.NEAREST_TASK)


def run_comparison(
    runs: list[LoadedRun],
    states: list[State],
) -> list[PolicyComparison]:
    """Compare greedy policies of all runs on each state."""
    results: list[PolicyComparison] = []

    for idx, state in enumerate(states):
        actions = []
        q_vals = []
        for run in runs:
            q = compute_q_values(state, run)
            a = greedy_action(q)
            actions.append(a)
            q_vals.append(q)

        results.append(PolicyComparison(
            state_index=idx,
            state=state,
            actions=actions,
            q_values=q_vals,
        ))

    return results
