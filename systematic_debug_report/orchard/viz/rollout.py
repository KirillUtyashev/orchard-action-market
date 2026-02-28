"""Rollout → Frame generator: wraps eval.rollout_trajectory with rendering metadata."""

from __future__ import annotations

from typing import Callable

import torch

import orchard.encoding as encoding
from orchard.datatypes import State
from orchard.enums import Action, ACTION_PRIORITY
from orchard.env.base import BaseEnv
from orchard.eval import rollout_trajectory
from orchard.model import ValueNetwork
from orchard.policy import Q_team
from orchard.viz.frame import Decision, Frame


def generate_frames(
    start_state: State,
    policy_fn: Callable[[State], Action],
    env: BaseEnv,
    n_steps: int,
    policy_name: str = "",
    networks: list[ValueNetwork] | None = None,
    include_decisions: bool = False,
    include_values: bool = False,
) -> list[Frame]:
    """Run a rollout and produce a Frame for every transition.

    A pick step yields two Frames (move + pick). A non-pick step yields one.
    n_steps counts agent decisions, not transitions.

    Parameters
    ----------
    start_state : starting env state
    policy_fn : s -> Action
    env : environment instance
    n_steps : how many agent decisions to roll out
    policy_name : label shown in the viewer
    networks : value networks (needed for --decisions and --values)
    include_decisions : compute Q_team for all actions at decision points
    include_values : compute V_i(s) for each agent at decision points
    """
    frames: list[Frame] = []
    total_picks = 0
    decision_count = 0
    state_index = 0
    transition_index = 0

    for transition in rollout_trajectory(start_state, policy_fn, env, n_steps):
        is_pick = (transition.action == Action.PICK)
        picked = any(r != 0.0 for r in transition.rewards)

        if picked:
            total_picks += 1

        # Increment decision count on non-PICK transitions (actual agent choices)
        if not is_pick:
            decision_count += 1

        # --- Optional: decision introspection (only at decision points) ---
        decisions: list[Decision] | None = None
        if include_decisions and networks is not None and not is_pick:
            decisions = []
            for action in ACTION_PRIORITY:
                q = Q_team(transition.s_t, action, networks, env)
                decisions.append(Decision(
                    action=action,
                    q_value=q,
                    is_chosen=(action == transition.action),
                ))

        # --- Optional: per-agent values (only at decision points) ---
        agent_values: dict[int, float] | None = None
        if include_values and networks is not None and not is_pick:
            agent_values = {}
            with torch.no_grad():
                for i, net in enumerate(networks):
                    agent_values[i] = net(encoding.encode(transition.s_t, i)).item()

        frame = Frame(
            step=decision_count,
            transition_index=transition_index,
            state_index=state_index,
            state=transition.s_t,
            state_after=transition.s_t_after,
            height=env.cfg.height,
            width=env.cfg.width,
            actor=transition.s_t.actor,
            action=transition.action,
            rewards=transition.rewards,
            discount=transition.discount,
            picked=picked,
            policy_name=policy_name,
            total_picks=total_picks,
            total_decisions=decision_count,
            apples_on_grid=len(transition.s_t.apple_positions),
            apples_after=len(transition.s_t_after.apple_positions),
            decisions=decisions,
            agent_values=agent_values,
        )
        frames.append(frame)

        state_index += 1
        transition_index += 1

    return frames
