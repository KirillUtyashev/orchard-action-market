"""Heuristic policies and action space helpers."""

from __future__ import annotations

from orchard.enums import (
    Action, ACTION_PRIORITY, Heuristic, make_pick_action,
)
from orchard.datatypes import EnvConfig, State


# ---------------------------------------------------------------------------
# Action space helpers
# ---------------------------------------------------------------------------
def get_all_actions(env_cfg: EnvConfig) -> list[Action]:
    """Phase-1 actions: always the 5 move actions."""
    return list(ACTION_PRIORITY)


def get_phase2_actions(state: State, env) -> list[Action]:
    """Phase-2 actions after a move landed on a task cell.

    Offers STAY + pick(κ) for each task type κ present at the actor's cell
    where phi(actor, κ) > 0 (actor has non-zero proficiency).

    env must be a BaseEnv with phi_positive_types attribute.
    """
    actor = state.actor
    actor_pos = state.agent_positions[actor]
    tasks_here = state.tasks_at(actor_pos)
    if not tasks_here:
        return []

    eligible_types = env.phi_positive_types[actor]
    types_here = sorted({tau for _, tau in tasks_here if tau in eligible_types})
    if not types_here:
        return [Action.STAY]
    actions: list[Action] = [Action.STAY]
    for tau in types_here:
        actions.append(make_pick_action(tau))
    return actions


# ---------------------------------------------------------------------------
# Value-aware nearest heuristic
# ---------------------------------------------------------------------------
def nearest_action(state: State, env) -> Action:
    """Document's greedy-optimal heuristic.

    Phase 1: move toward (q*, κ*) = argmax_{tasks} φ(i,κ)·Σ_j R(i,j)·r'_j
             Ties broken by Manhattan distance, then ACTION_PRIORITY.
    Phase 2: pick argmax-value eligible type present; STAY if none eligible.

    env must be a BaseEnv with phi, relatedness, category_rewards, phi_positive_types.
    """
    actor = state.actor

    if state.pick_phase:
        # Phase 2: pick the highest-value eligible task type at this cell
        actor_pos = state.agent_positions[actor]
        tasks_here = state.tasks_at(actor_pos)
        eligible = env.phi_positive_types[actor]
        best_tau = None
        best_val = -float("inf")
        for _, tau in tasks_here:
            if tau not in eligible:
                continue
            val = float(env.phi[actor, tau]) * float(
                (env.relatedness[actor] * env.category_rewards[tau]).sum()
            )
            if val > best_val:
                best_val = val
                best_tau = tau
        if best_tau is not None:
            return make_pick_action(best_tau)
        return Action.STAY

    # Phase 1: compute per-task value and move toward best
    if not state.task_positions or state.task_types is None:
        return Action.STAY

    # task_val[k] = phi[actor, tau_k] * sum_j R[actor,j] * r'[tau_k, j]
    task_vals = []
    for pos, tau in zip(state.task_positions, state.task_types):
        phi_val = float(env.phi[actor, tau])
        r_prime = env.category_rewards[tau]           # (N,)
        rel_row = env.relatedness[actor]              # (N,)
        val = phi_val * float((rel_row * r_prime).sum())
        task_vals.append((val, pos))

    # Find the best target: argmax value, then min distance
    ar, ac = state.agent_positions[actor]
    best_action = Action.STAY
    best_val = -float("inf")
    best_dist = float("inf")

    for action in ACTION_PRIORITY:
        dr, dc = action.delta
        nr = max(0, min(env.cfg.height - 1, ar + dr))
        nc = max(0, min(env.cfg.width - 1, ac + dc))
        # Find closest task and its value from this candidate position
        candidate_best_val = -float("inf")
        candidate_best_dist = float("inf")
        for val, pos in task_vals:
            d = abs(nr - pos.row) + abs(nc - pos.col)
            # Primary: maximize value; secondary: minimize distance
            if (val > candidate_best_val or
                    (val == candidate_best_val and d < candidate_best_dist)):
                candidate_best_val = val
                candidate_best_dist = d
        # Compare this action against current best
        if (candidate_best_val > best_val or
                (candidate_best_val == best_val and candidate_best_dist < best_dist)):
            best_val = candidate_best_val
            best_dist = candidate_best_dist
            best_action = action

    return best_action


def heuristic_action(state: State, env, heuristic: Heuristic) -> Action:
    """Dispatch to the configured heuristic policy."""
    if heuristic == Heuristic.NEAREST:
        return nearest_action(state, env)
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")
