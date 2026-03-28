"""Base environment: shared logic for apply_action, resolve_pick, advance_actor, step."""

from __future__ import annotations

from abc import ABC, abstractmethod

from orchard.enums import Action, PickMode
from orchard.datatypes import EnvConfig, Grid, State, Transition, sort_tasks


class BaseEnv(ABC):
    """Abstract base for orchard environments."""

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def init_state(self) -> State:
        ...

    def apply_action(self, state: State, action: Action) -> State:
        """Apply movement action to current actor. Returns movement after-state.

        For pick actions, returns state unchanged (no movement).
        Clamps to grid boundaries. Actor remains the same.
        Does NOT resolve pick, spawn, despawn, or advance actor.
        """
        if action.is_pick():
            return state  # pick actions don't move

        actor = state.actor
        r, c = state.agent_positions[actor]
        dr, dc = action.delta
        nr = max(0, min(self.cfg.height - 1, r + dr))
        nc = max(0, min(self.cfg.width - 1, c + dc))
        new_positions = list(state.agent_positions)
        new_positions[actor] = Grid(nr, nc)
        return State(
            agent_positions=tuple(new_positions),
            task_positions=state.task_positions,
            actor=actor,
            task_types=state.task_types,
        )

    def resolve_pick(
        self, state: State, pick_type: int | None = None
    ) -> tuple[State, tuple[float, ...]]:
        """Resolve task pickup.

        Args:
            state: current state (after movement for forced, or current pos for choice)
            pick_type: for CHOICE mode, which type τ to pick. None for FORCED.

        Two reward paths:
            n_task_types == 1: legacy r_picker / r_other distribution
            n_task_types > 1:  actor gets R_high (τ ∈ G_actor) or R_low (τ ∉ G_actor),
                               others get 0

        Returns (new_state, rewards). If no pick occurs, returns (state, zero_rewards).
        Does NOT spawn, despawn, or advance actor.
        """
        actor = state.actor
        n = self.cfg.n_agents
        zero_rewards = tuple(0.0 for _ in range(n))

        if self.cfg.n_task_types == 1:
            # --- Legacy path ---
            if not (self.cfg.pick_mode == PickMode.FORCED
                    and state.is_agent_on_task(actor)):
                return state, zero_rewards

            pos = state.agent_positions[actor]
            picked_idx = state.task_positions.index(pos)
            new_tasks = (state.task_positions[:picked_idx]
                         + state.task_positions[picked_idx + 1:])

            r_other = ((1.0 - self.cfg.r_picker) / (n - 1)) if n > 1 else 0.0
            rewards = tuple(
                self.cfg.r_picker if i == actor else r_other
                for i in range(n)
            )
            return State(
                agent_positions=state.agent_positions,
                task_positions=new_tasks,
                actor=actor,
                task_types=None,
            ), rewards

        # --- Task specialization path (n_task_types > 1) ---
        if self.cfg.pick_mode == PickMode.FORCED:
            if not state.is_agent_on_task(actor):
                return state, zero_rewards

            pos = state.agent_positions[actor]
            # Exactly 1 task per cell in forced mode
            picked_idx = state.task_positions.index(pos)
            tau = state.task_types[picked_idx]

        else:
            # CHOICE mode: pick_type=None means STAY (agent declined to pick)
            if pick_type is None:
                return state, zero_rewards
            pos = state.agent_positions[actor]
            # Find task of matching type at actor's position
            picked_idx = None
            for i, (tp, tt) in enumerate(
                zip(state.task_positions, state.task_types)
            ):
                if tp == pos and tt == pick_type:
                    picked_idx = i
                    break
            if picked_idx is None:
                # No matching task — wasted action
                return state, zero_rewards
            tau = pick_type

        # Remove the picked task
        new_positions = (state.task_positions[:picked_idx]
                         + state.task_positions[picked_idx + 1:])
        new_types = (state.task_types[:picked_idx]
                     + state.task_types[picked_idx + 1:])
        new_positions, new_types = sort_tasks(new_positions, new_types)

        # Compute reward
        g_actor = set(self.cfg.task_assignments[actor])
        if tau in g_actor:
            actor_reward = self.cfg.r_high
        else:
            actor_reward = self.cfg.r_low

        rewards = tuple(
            actor_reward if i == actor else 0.0
            for i in range(n)
        )

        return State(
            agent_positions=state.agent_positions,
            task_positions=new_positions,
            actor=actor,
            task_types=new_types,
        ), rewards

    @abstractmethod
    def spawn_and_despawn(self, state: State) -> State:
        """Handle task spawning and despawning. Does NOT advance actor."""
        ...

    def advance_actor(self, state: State) -> State:
        """Advance actor to next agent (round-robin)."""
        return State(
            agent_positions=state.agent_positions,
            task_positions=state.task_positions,
            actor=(state.actor + 1) % self.cfg.n_agents,
            task_types=state.task_types,
        )

    def step(self, state: State, action: Action) -> Transition:
        """Full step: apply move action → resolve pick (FORCED only) → spawn/despawn → advance actor.

        In CHOICE mode, pick resolution is handled externally by the train/eval
        two-phase loop. step() only handles move actions.

        Returns Transition with the pick after-state as s_t_after.
        """
        actor = state.actor
        zero_rewards = tuple(0.0 for _ in range(self.cfg.n_agents))

        assert action.is_move(), (
            f"step() only accepts move actions. Got {action}. "
            "Use resolve_pick() directly for phase-2 pick actions."
        )

        s_moved = self.apply_action(state, action)

        if self.cfg.pick_mode == PickMode.FORCED and s_moved.is_agent_on_task(actor):
            s_picked, rewards = self.resolve_pick(s_moved)
        else:
            s_picked, rewards = s_moved, zero_rewards

        s_spawned = self.spawn_and_despawn(s_picked)
        s_next = self.advance_actor(s_spawned)

        return Transition(
            s_t=state,
            action=action,
            s_t_after=s_picked,
            s_t_next=s_next,
            rewards=rewards,
            discount=self.cfg.gamma,
        )

    @property
    def n_agents(self) -> int:
        return self.cfg.n_agents
