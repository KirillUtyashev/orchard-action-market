"""Base environment: apply_action, resolve_pick, advance_actor, step."""

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
        """Apply movement action to current actor. Clamps to grid boundaries.

        Does NOT resolve pick, spawn, despawn, or advance actor.
        """
        if action.is_pick():
            return state

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

    def _find_task_to_pick(
        self, state: State, pick_type: int | None,
    ) -> tuple[int, int] | None:
        """Find task index and type for a pick attempt. Returns None if no pick."""
        actor = state.actor
        pos = state.agent_positions[actor]

        if self.cfg.pick_mode == PickMode.FORCED:
            if not state.is_agent_on_task(actor):
                return None
            g_actor = set(self.cfg.task_assignments[actor])
            for i, (tp, tt) in enumerate(zip(state.task_positions, state.task_types)):
                if tp == pos and tt in g_actor:
                    return i, tt
            return None  # only wrong-type tasks at this cell: stay

        # CHOICE mode: pick_type=None means STAY (decline)
        if pick_type is None:
            return None
        for i, (tp, tt) in enumerate(zip(state.task_positions, state.task_types)):
            if tp == pos and tt == pick_type:
                return i, pick_type
        return None

    def _compute_pick_rewards(self, actor: int, tau: int) -> tuple[float, ...]:
        """Per-agent rewards for picking task of type τ.

        Correct (τ ∈ G_actor): picker gets r_picker, groupmates share (1 - r_picker).
        Wrong   (τ ∉ G_actor): picker gets r_low, everyone else 0.
        """
        n = self.cfg.n_agents
        g_actor = set(self.cfg.task_assignments[actor])

        if tau not in g_actor:
            return tuple(self.cfg.r_low if j == actor else 0.0 for j in range(n))

        group = [j for j in range(n) if tau in self.cfg.task_assignments[j]]
        groupmate_r = (1.0 - self.cfg.r_picker) / (len(group) - 1) if len(group) > 1 else 0.0
        group_set = set(group)
        return tuple(
            self.cfg.r_picker if j == actor
            else groupmate_r if j in group_set
            else 0.0
            for j in range(n)
        )

    def resolve_pick(
        self, state: State, pick_type: int | None = None,
    ) -> tuple[State, tuple[float, ...]]:
        """Resolve task pickup. Returns (new_state, per_agent_rewards)."""
        assert self.cfg.task_assignments is not None, "task_assignments must be set"
        assert state.task_types is not None, "task_types must be set"

        zero_rewards = tuple(0.0 for _ in range(self.cfg.n_agents))
        result = self._find_task_to_pick(state, pick_type)
        if result is None:
            return state, zero_rewards
        picked_idx, tau = result

        # Remove the picked task
        new_positions = state.task_positions[:picked_idx] + state.task_positions[picked_idx + 1:]
        new_types = state.task_types[:picked_idx] + state.task_types[picked_idx + 1:]
        new_positions, new_types = sort_tasks(new_positions, new_types)

        new_state = State(
            agent_positions=state.agent_positions,
            task_positions=new_positions,
            actor=state.actor,
            task_types=new_types,
        )
        return new_state, self._compute_pick_rewards(state.actor, tau)

    @abstractmethod
    def spawn_and_despawn(self, state: State) -> State:
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
        """Full step: move → resolve pick (FORCED only) → spawn/despawn → advance.

        In CHOICE mode, pick resolution is handled externally by the train loop.
        """
        actor = state.actor
        zero_rewards = tuple(0.0 for _ in range(self.cfg.n_agents))

        assert action.is_move(), f"step() only accepts move actions, got {action}"

        s_moved = self.apply_action(state, action)

        if self.cfg.pick_mode == PickMode.FORCED and s_moved.is_agent_on_task(actor):
            s_picked, rewards = self.resolve_pick(s_moved)
        else:
            s_picked, rewards = s_moved, zero_rewards

        s_spawned = self.spawn_and_despawn(s_picked)
        s_next = self.advance_actor(s_spawned)

        return Transition(
            s_t=state, action=action, s_t_after=s_picked,
            s_t_next=s_next, rewards=rewards, discount=self.cfg.gamma,
        )

    @property
    def n_agents(self) -> int:
        return self.cfg.n_agents
