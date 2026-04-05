"""Actor-critic action-space helpers for orchard."""

from __future__ import annotations

import numpy as np

from orchard.datatypes import EnvConfig, State
from orchard.enums import Action, make_pick_action


_MOVE_ACTIONS_BY_VALUE: tuple[Action, ...] = (
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT,
    Action.STAY,
)

_MOVE_ACTION_SINGLETONS: dict[int, Action] = {
    action.value: action for action in _MOVE_ACTIONS_BY_VALUE
}


def full_action_head_dim(env_cfg: EnvConfig) -> int:
    """Size of the fixed actor head derived from orchard's action encoding."""
    if env_cfg.n_task_types <= 0:
        return Action.STAY.value + 1
    return make_pick_action(env_cfg.n_task_types - 1).value + 1


def action_to_policy_index(action: Action) -> int:
    """Map orchard actions directly onto policy-head indices."""
    return int(action.value)


def policy_index_to_action(idx: int) -> Action:
    """Inverse of action_to_policy_index for the fixed actor head."""
    value = int(idx)
    if value in _MOVE_ACTION_SINGLETONS:
        return _MOVE_ACTION_SINGLETONS[value]
    if value >= Action.PICK.value:
        return make_pick_action(value - Action.PICK.value)
    raise ValueError(f"Unsupported policy index: {idx}")


def build_phase1_legal_mask(state: State, env_cfg: EnvConfig) -> np.ndarray:
    """Mask for move selection.

    Orchard's phase-1 action space always contains all movement actions; the
    environment clamps off-grid moves back into the grid.
    """
    del state

    mask = np.zeros(full_action_head_dim(env_cfg), dtype=bool)
    for action in _MOVE_ACTIONS_BY_VALUE:
        mask[action_to_policy_index(action)] = True
    return mask


def build_phase2_legal_mask(state: State, env_cfg: EnvConfig) -> np.ndarray:
    """Mask for pick/stay selection after landing on a task cell."""
    mask = np.zeros(full_action_head_dim(env_cfg), dtype=bool)
    mask[action_to_policy_index(Action.STAY)] = True

    actor_pos = state.agent_positions[state.actor]
    for _, tau in state.tasks_at(actor_pos):
        if 0 <= tau < env_cfg.n_task_types:
            mask[action_to_policy_index(make_pick_action(tau))] = True
    return mask


__all__ = [
    "full_action_head_dim",
    "action_to_policy_index",
    "policy_index_to_action",
    "build_phase1_legal_mask",
    "build_phase2_legal_mask",
]
