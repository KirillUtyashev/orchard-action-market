"""Heuristic policies and action space helpers."""

from __future__ import annotations

from orchard.enums import (
    Action, ACTION_PRIORITY, Heuristic, PickMode, make_pick_action,
)
from orchard.datatypes import EnvConfig, State


# ---------------------------------------------------------------------------
# Action space helpers
# ---------------------------------------------------------------------------
def get_all_actions(env_cfg: EnvConfig) -> list[Action]:
    """Phase-1 actions: always the 5 move actions."""
    return list(ACTION_PRIORITY)


def get_phase2_actions(state: State, env_cfg: EnvConfig) -> list[Action]:
    """Phase-2 actions after a move landed on a task cell.

    Returns {STAY, pick(τ) for each τ present at actor's cell}.
    Empty list if actor is not on any task cell.
    """
    actor_pos = state.agent_positions[state.actor]
    tasks_here = state.tasks_at(actor_pos)
    if not tasks_here:
        return []
    types_here = sorted({tau for _, tau in tasks_here})
    actions: list[Action] = [Action.STAY]
    for tau in types_here:
        actions.append(make_pick_action(tau))
    return actions


# ---------------------------------------------------------------------------
# Heuristic policies
# ---------------------------------------------------------------------------
def nearest_task_action(state: State, env_cfg: EnvConfig) -> Action:
    """Move actor toward nearest task (Manhattan distance)."""
    actor = state.actor

    if state.pick_phase:
        tasks_here = state.tasks_at(state.agent_positions[actor])
        if tasks_here:
            _, tau = tasks_here[0]
            return make_pick_action(tau)
        return Action.STAY

    if not state.task_positions:
        return Action.STAY

    ar, ac = state.agent_positions[actor]
    best_dist = float("inf")
    best_action = Action.STAY

    for action in ACTION_PRIORITY:
        dr, dc = action.delta
        nr = max(0, min(env_cfg.height - 1, ar + dr))
        nc = max(0, min(env_cfg.width - 1, ac + dc))
        min_d = min(
            abs(nr - tp.row) + abs(nc - tp.col)
            for tp in state.task_positions
        )
        if min_d < best_dist:
            best_dist = min_d
            best_action = action

    return best_action


def nearest_correct_task_action(state: State, env_cfg: EnvConfig) -> Action:
    """Move toward nearest correct-type task (τ ∈ G_actor).

    Phase 2: always picks whatever task is present (mirrors forced-pick behavior).
    """
    actor = state.actor
    good_types = set(env_cfg.task_assignments[actor]) if env_cfg.task_assignments else set()

    if state.pick_phase:
        tasks_here = state.tasks_at(state.agent_positions[actor])
        if tasks_here:
            _, tau = tasks_here[0]
            return make_pick_action(tau)
        return Action.STAY

    if state.task_types is not None:
        good_tasks = [
            (pos, typ) for pos, typ in zip(state.task_positions, state.task_types)
            if typ in good_types
        ]
    else:
        good_tasks = [(pos, 0) for pos in state.task_positions]

    if not good_tasks:
        return Action.STAY

    ar, ac = state.agent_positions[actor]
    best_dist = float("inf")
    best_action = Action.STAY

    for action in ACTION_PRIORITY:
        dr, dc = action.delta
        nr = max(0, min(env_cfg.height - 1, ar + dr))
        nc = max(0, min(env_cfg.width - 1, ac + dc))
        min_d = min(
            abs(nr - pos.row) + abs(nc - pos.col)
            for pos, _ in good_tasks
        )
        if min_d < best_dist:
            best_dist = min_d
            best_action = action

    return best_action


def nearest_correct_task_stay_wrong_action(
    state: State, env_cfg: EnvConfig,
) -> Action:
    """Move toward nearest correct-type task; phase 2 picks correct, STAYs on wrong."""
    actor = state.actor
    good_types = set(env_cfg.task_assignments[actor]) if env_cfg.task_assignments else set()

    if state.pick_phase:
        tasks_here = state.tasks_at(state.agent_positions[actor])
        for _, tau in tasks_here:
            if tau in good_types:
                return make_pick_action(tau)
        return Action.STAY

    return nearest_correct_task_action(state, env_cfg)


def heuristic_action(
    state: State, env_cfg: EnvConfig, heuristic: Heuristic,
) -> Action:
    """Dispatch to the configured heuristic policy."""
    if heuristic == Heuristic.NEAREST_TASK:
        return nearest_task_action(state, env_cfg)
    elif heuristic == Heuristic.NEAREST_CORRECT_TASK:
        return nearest_correct_task_action(state, env_cfg)
    elif heuristic == Heuristic.NEAREST_CORRECT_TASK_STAY_WRONG:
        return nearest_correct_task_stay_wrong_action(state, env_cfg)
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")
