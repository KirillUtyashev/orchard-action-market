"""Row builders for actor policy diagnostics."""

from __future__ import annotations

import numpy as np

from orchard.actor_critic.action_space import full_action_head_dim
from orchard.actor_critic.policy_eval_states import serialize_state
from orchard.datatypes import EnvConfig, State
from orchard.enums import Action, make_pick_action


def _probability_columns(probs, env_cfg: EnvConfig) -> dict[str, float]:
    probs_arr = np.asarray(probs, dtype=float).reshape(-1)
    expected = full_action_head_dim(env_cfg)
    if probs_arr.shape != (expected,):
        raise ValueError(f"Expected probability vector shape {(expected,)}, got {probs_arr.shape}.")

    row = {
        "prob_up": float(probs_arr[Action.UP.value]),
        "prob_down": float(probs_arr[Action.DOWN.value]),
        "prob_left": float(probs_arr[Action.LEFT.value]),
        "prob_right": float(probs_arr[Action.RIGHT.value]),
        "prob_stay": float(probs_arr[Action.STAY.value]),
    }
    for tau in range(env_cfg.n_task_types):
        row[f"prob_pick_{tau}"] = float(probs_arr[make_pick_action(tau).value])
    return row


def build_phase1_policy_prob_row(
    step: int,
    wall_time: float,
    state_id: int | str,
    state: State,
    probs,
    env_cfg: EnvConfig,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "step": int(step),
        "wall_time": float(wall_time),
        "state_id": state_id,
        "actor_id": int(state.actor),
        "state_json": serialize_state(state),
    }
    row.update(_probability_columns(probs, env_cfg))
    return row


def build_phase2_policy_prob_row(
    step: int,
    wall_time: float,
    state_id: int | str,
    state_label: str,
    state: State,
    probs,
    env_cfg: EnvConfig,
) -> dict[str, float | int | str]:
    actor = state.actor
    row: dict[str, float | int | str] = {
        "step": int(step),
        "wall_time": float(wall_time),
        "state_id": state_id,
        "state_label": state_label,
        "actor_id": int(actor),
        "state_json": serialize_state(state),
    }

    present_types = {tau for _, tau in state.tasks_at(state.agent_positions[actor])}
    assigned_types = (
        set(env_cfg.task_assignments[actor])
        if env_cfg.task_assignments is not None
        else {0}
    )
    for tau in range(env_cfg.n_task_types):
        row[f"present_type_{tau}"] = int(tau in present_types)
    for tau in range(env_cfg.n_task_types):
        row[f"assigned_type_{tau}"] = int(tau in assigned_types)

    row.update(_probability_columns(probs, env_cfg))
    return row


__all__ = [
    "build_phase1_policy_prob_row",
    "build_phase2_policy_prob_row",
]
