"""Rollout → Frame generator: wraps eval.rollout_trajectory with rendering metadata."""

from __future__ import annotations

from typing import Callable

import torch

import orchard.encoding as encoding
from orchard.datatypes import State
from orchard.enums import Action, PickMode
from orchard.env.base import BaseEnv
from orchard.eval import rollout_trajectory
from orchard.model import ValueNetwork
from orchard.policy import get_all_actions, get_phase2_actions
from orchard.viz.frame import Decision, Frame


def generate_frames(
    start_state: State,
    policy_fn: "Callable[[State], Action]",
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
    """
    frames: list[Frame] = []
    total_picks = 0
    total_correct_picks = 0
    total_wrong_picks = 0
    total_reward = 0.0
    total_team_reward = 0.0
    decision_count = 0
    state_index = 0
    transition_index = 0
    n_agents = env.cfg.n_agents
    agent_pick_counts: dict[int, int] = {i: 0 for i in range(n_agents)}
    is_decentralized = networks is not None and len(networks) > 1
    # Phase-1 actions for Q-value display; phase-2 handled per-transition
    all_actions = get_all_actions(env.cfg)

    for transition in rollout_trajectory(start_state, policy_fn, env, n_steps):
        picked = any(r != 0.0 for r in transition.rewards)

        # Determine pick type and correctness (always, regardless of n_task_types)
        picked_task_type: int | None = None
        picked_correct: bool | None = None

        if picked and env.cfg.task_assignments is not None:
            total_picks += 1
            agent_pick_counts[transition.s_t.actor] += 1

            # Figure out what type was picked by comparing before/after tasks
            before_tasks = set(zip(transition.s_t.task_positions,
                                   transition.s_t.task_types or ()))
            after_tasks = set(zip(transition.s_t_after.task_positions,
                                  transition.s_t_after.task_types or ()))
            removed = before_tasks - after_tasks
            if removed:
                _, tau = next(iter(removed))
                picked_task_type = tau
                g_actor = set(env.cfg.task_assignments[transition.s_t.actor])
                picked_correct = (tau in g_actor)
                if picked_correct:
                    total_correct_picks += 1
                else:
                    total_wrong_picks += 1

        # Accumulate reward
        total_reward += transition.rewards[transition.s_t.actor]
        total_team_reward += sum(transition.rewards)

        # Increment decision count on non-PICK transitions (actual agent choices)
        if transition.action.is_move():
            decision_count += 1

        # --- Optional: decision introspection (only at decision points) ---
        decisions: list[Decision] | None = None
        # Introspect Q-values only at phase-1 decision points (move actions)
        if include_decisions and networks is not None and transition.action.is_move():
            decisions = []
            for action in all_actions:
                s_after = env.apply_action(transition.s_t, action)
                agent_qvals: dict[int, float] = {}
                q_total: float = 0.0
                with torch.no_grad():
                    for i, network in enumerate(networks):
                        v_i = network(encoding.encode(s_after, i)).item()
                        agent_qvals[i] = v_i
                        q_total += v_i
                decisions.append(Decision(
                    action=action,
                    q_value=q_total,
                    is_chosen=(action == transition.action),
                    agent_q_values=agent_qvals if is_decentralized else None,
                ))

        # --- Optional: per-agent values (only at decision points) ---
        agent_values: dict[int, float] | None = None
        if include_values and networks is not None and transition.action.is_move():
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
            tasks_on_grid=len(transition.s_t.task_positions),
            tasks_after=len(transition.s_t_after.task_positions),
            picked_task_type=picked_task_type,
            picked_correct=picked_correct,
            total_correct_picks=total_correct_picks,
            total_wrong_picks=total_wrong_picks,
            total_reward=total_reward,
            total_team_reward=total_team_reward,
            decisions=decisions,
            agent_values=agent_values,
            agent_picks=dict(agent_pick_counts),
        )
        frames.append(frame)

        state_index += 1
        transition_index += 1

    return frames
