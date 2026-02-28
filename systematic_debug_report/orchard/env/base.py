"""Base environment: shared logic for apply_action, resolve_pick, advance_actor, step."""

from __future__ import annotations

from abc import ABC, abstractmethod

from orchard.enums import Action
from orchard.datatypes import EnvConfig, Grid, State, Transition


class BaseEnv(ABC):
    """Abstract base for orchard environments."""

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def init_state(self) -> State:
        ...

    def apply_action(self, state: State, action: Action) -> State:
        """Apply action to current actor. Returns movement after-state.

        Clamps to grid boundaries. Actor remains the same.
        Does NOT resolve pick, spawn, despawn, or advance actor.
        """
        actor = state.actor
        r, c = state.agent_positions[actor]
        dr, dc = action.delta
        nr = max(0, min(self.cfg.height - 1, r + dr))
        nc = max(0, min(self.cfg.width - 1, c + dc))
        new_positions = list(state.agent_positions)
        new_positions[actor] = Grid(nr, nc)
        return State(
            agent_positions=tuple(new_positions),
            apple_positions=state.apple_positions,
            actor=actor,
            apple_ages=state.apple_ages,
            apple_ids=state.apple_ids,
        )

    def resolve_pick(self, state: State) -> tuple[State, tuple[float, ...]]:
        """If actor is on an apple and force_pick is true, remove apple and compute rewards.

        Returns (pick_after_state, rewards).
        If no pick: returns (state unchanged, all-zero rewards).
        Does NOT spawn, despawn, or advance actor.
        """
        actor = state.actor
        zero_rewards = tuple(0.0 for _ in range(self.cfg.n_agents))

        if not (self.cfg.force_pick and state.is_agent_on_apple(actor)):
            return state, zero_rewards

        pos = state.agent_positions[actor]
        picked_idx = state.apple_positions.index(pos)

        new_apples = state.apple_positions[:picked_idx] + state.apple_positions[picked_idx + 1:]
        new_ages: tuple[int, ...] | None = None
        if state.apple_ages is not None:
            new_ages = state.apple_ages[:picked_idx] + state.apple_ages[picked_idx + 1:]
        new_ids: tuple[int, ...] | None = None
        if state.apple_ids is not None:
            new_ids = state.apple_ids[:picked_idx] + state.apple_ids[picked_idx + 1:]

        r_other = (1.0 - self.cfg.r_picker) / (self.cfg.n_agents - 1) if self.cfg.n_agents > 1 else 0.0
        rewards = tuple(
            self.cfg.r_picker if i == actor else r_other
            for i in range(self.cfg.n_agents)
        )
        return State(
            agent_positions=state.agent_positions,
            apple_positions=new_apples,
            actor=actor,
            apple_ages=new_ages,
            apple_ids=new_ids,
        ), rewards

    @abstractmethod
    def spawn_and_despawn(self, state: State) -> State:
        """Handle apple spawning and despawning. Does NOT advance actor."""
        ...

    def advance_actor(self, state: State) -> State:
        """Advance actor to next agent (round-robin)."""
        return State(
            agent_positions=state.agent_positions,
            apple_positions=state.apple_positions,
            actor=(state.actor + 1) % self.cfg.n_agents,
            apple_ages=state.apple_ages,
            apple_ids=state.apple_ids,
        )

    def step(self, state: State, action: Action) -> Transition:
        """Full step: apply action → resolve pick → spawn/despawn → advance actor.

        Returns Transition. The training loop uses apply_action directly for
        the intermediate movement after-state when a pick occurs.
        """
        s_moved = self.apply_action(state, action)
        s_picked, rewards = self.resolve_pick(s_moved)
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
