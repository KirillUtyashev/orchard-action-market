"""Generate states and compare greedy policies across multiple trained runs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from orchard.compare_values.loader import LoadedRun
from orchard.datatypes import State
from orchard.enums import Action, ACTION_PRIORITY, TDTarget
from orchard.env import create_env
from orchard.eval import collect_after_state_test_states, collect_on_policy_test_states
from orchard.seed import set_all_seeds


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
        """Mean gap between best and second-best Q across runs.
        Small gap = the disagreement is over near-ties."""
        gaps = []
        for qv in self.q_values:
            sorted_q = sorted(qv.values(), reverse=True)
            if len(sorted_q) >= 2:
                gaps.append(sorted_q[0] - sorted_q[1])
        return sum(gaps) / len(gaps) if gaps else 0.0


def generate_states(
    run: LoadedRun,
    n_states: int,
    seed: int,
) -> list[State]:
    """Generate comparison states using nearest-apple policy."""
    set_all_seeds(seed)
    env = create_env(run.cfg.env)

    if run.cfg.train.td_target == TDTarget.AFTER_STATE:
        return collect_after_state_test_states(env, n_states)
    else:
        return collect_on_policy_test_states(env, n_states)


def compute_q_values(
    state: State,
    run: LoadedRun,
) -> dict[Action, float]:
    """Compute Q_team(s, a) for all actions in ACTION_PRIORITY."""
    env = create_env(run.cfg.env)
    q: dict[Action, float] = {}
    with torch.no_grad():
        for action in ACTION_PRIORITY:
            s_after = env.apply_action(state, action)
            total = 0.0
            for i, net in enumerate(run.networks):
                total += net(run.encoder.encode(s_after, i)).item()
            q[action] = total
    return q


def greedy_action(q: dict[Action, float]) -> Action:
    """Pick best action from Q dict, tie-breaking by ACTION_PRIORITY."""
    best_val: float | None = None
    best_act = ACTION_PRIORITY[0]
    for act in ACTION_PRIORITY:
        if best_val is None or q[act] > best_val:
            best_val = q[act]
            best_act = act
    return best_act

def generate_training_sample_states(
    run: LoadedRun,
    n_sample: int = 100,
) -> list[State]:
    """Reproduce the exact sample states from train.py (random actions, fixed seed 9999)."""
    from orchard.enums import Action, NUM_ACTIONS, TDTarget
    from orchard.seed import rng
    set_all_seeds(9999)
    env = create_env(run.cfg.env)

    states: list[State] = []
    s_tmp = env.init_state()

    for _ in range(n_sample * 3):
        a_tmp = Action(rng.randint(0, NUM_ACTIONS - 1))

        if run.cfg.train.td_target == TDTarget.AFTER_STATE:
            s_moved = env.apply_action(s_tmp, a_tmp)
            if s_moved.is_agent_on_apple(s_moved.actor):
                s_after, _ = env.resolve_pick(s_moved)
            else:
                s_after = s_moved
            if len(states) < n_sample:
                states.append(s_after)
            s_picked, _ = env.resolve_pick(s_moved)
            s_tmp = env.advance_actor(env.spawn_and_despawn(s_picked))
        else:
            if len(states) < n_sample:
                states.append(s_tmp)
            s_moved = env.apply_action(s_tmp, a_tmp)
            s_picked, _ = env.resolve_pick(s_moved)
            s_tmp = env.advance_actor(env.spawn_and_despawn(s_picked))

    return states

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
