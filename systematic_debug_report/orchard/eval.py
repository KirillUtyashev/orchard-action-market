"""Evaluation: rollout trajectories and policy metrics."""

from __future__ import annotations

from typing import Callable, Iterator

from orchard.enums import Action, PickMode, make_pick_action
from orchard.env.base import BaseEnv
from orchard.datatypes import State, Transition


# ---------------------------------------------------------------------------
# Rollout trajectory
# ---------------------------------------------------------------------------
def rollout_trajectory(
    start_state: State,
    policy_fn: Callable[[State, bool], Action],
    env: BaseEnv,
    n_steps: int,
) -> Iterator[Transition]:
    """Yield Transitions for n_steps agent turns.

    Each turn:
      Phase 1: move action (always). Yields one Transition at discount γ.
      Phase 2: only if actor lands on task cell.
               FORCED: auto-picks. CHOICE: asks policy_fn.
               Yields one Transition at discount 1.0.

    n_steps counts phase-1 decisions (agent turns), not transitions.
    """
    s = start_state
    gamma = env.cfg.gamma
    zero_rewards = tuple(0.0 for _ in range(env.cfg.n_agents))

    for _ in range(n_steps):
        move_action = policy_fn(s, False)
        assert move_action.is_move(), f"Phase 1 must be a move action, got {move_action}"
        s_moved = env.apply_action(s, move_action)

        if s_moved.is_agent_on_task(s_moved.actor):
            yield Transition(
                s_t=s, action=move_action, s_t_after=s_moved, s_t_next=s_moved,
                rewards=zero_rewards, discount=gamma,
            )
            if env.cfg.pick_mode == PickMode.FORCED:
                tau = s_moved.task_type_at(s_moved.agent_positions[s_moved.actor])
                pick_action = make_pick_action(tau)
            else:
                pick_action = policy_fn(s_moved.with_pick_phase(), True)

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
            s_next = env.advance_actor(env.spawn_and_despawn(s_moved))
            yield Transition(
                s_t=s, action=move_action, s_t_after=s_moved,
                s_t_next=s_next, rewards=zero_rewards, discount=gamma,
            )
            s = s_next


# ---------------------------------------------------------------------------
# Policy metrics
# ---------------------------------------------------------------------------
def evaluate_policy_metrics(
    start_state: State,
    policy_fn: Callable[[State, bool], Action],
    env: BaseEnv,
    n_steps: int,
) -> dict[str, float]:
    """Compute rps, team_rps, correct_pps, wrong_pps over a rollout."""
    total_reward = 0.0
    total_team_reward = 0.0
    correct_picks = 0
    wrong_picks = 0

    for t in rollout_trajectory(start_state, policy_fn, env, n_steps):
        actor = t.s_t.actor
        total_reward += t.rewards[actor]
        total_team_reward += sum(t.rewards)

        if t.action.is_pick():
            tau = t.action.pick_type()
            g_actor = set(env.cfg.task_assignments[actor])
            if tau in g_actor:
                correct_picks += 1
            else:
                wrong_picks += 1

    return {
        "rps": total_reward / n_steps if n_steps > 0 else 0.0,
        "team_rps": total_team_reward / n_steps if n_steps > 0 else 0.0,
        "correct_pps": correct_picks / n_steps if n_steps > 0 else 0.0,
        "wrong_pps": wrong_picks / n_steps if n_steps > 0 else 0.0,
    }
