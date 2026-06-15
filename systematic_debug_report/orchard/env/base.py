"""Base environment: apply_action, resolve_pick, advance_actor, step."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod

import numpy as np
import torch

from orchard.enums import Action
from orchard.datatypes import EnvConfig, Grid, State, Transition, sort_tasks



def task_center_for_agent(agent: int, n_agents: int, n_task_types: int) -> int:
    """Place agents at representative centers in task-type space.

    The rule floor((i + 1) * T / (N + 1)) preserves the old task_id ==
    agent_id mapping whenever T == N, while giving sensible centers for
    T != N, e.g. T=11,N=1 -> 5 and T=11,N=2 -> 3,7.
    """
    if n_agents <= 0 or n_task_types <= 0:
        raise ValueError("n_agents and n_task_types must be positive")
    return min(int((agent + 1) * n_task_types // (n_agents + 1)), n_task_types - 1)


def circular_task_distance(a: int, b: int, n_task_types: int) -> int:
    d = abs(int(a) - int(b))
    return min(d, int(n_task_types) - d)

class BaseEnv(ABC):
    """Abstract base for orchard environments."""

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        N = cfg.n_agents
        T = cfg.n_task_types
        C = cfg.relatedness_width
        S = cfg.proficiency_width

        # Agents live in task-type space. For T == N this reduces to the old
        # identity mapping (agent i centered on task i). For T != N, agents are
        # placed at evenly spaced representative task centers, e.g. T=11,N=2 ->
        # centers 3 and 7.
        self.task_centers: np.ndarray = np.array(
            [task_center_for_agent(i, N, T) for i in range(N)],
            dtype=np.int64,
        )

        # proficiency[i, kappa] = 1 if task kappa is within proficiency_width of
        # agent i's base task in task-type space.
        self.proficiency: np.ndarray = np.array(
            [
                [
                    1.0 if circular_task_distance(center, kappa, T) <= S else 0.0
                    for kappa in range(T)
                ]
                for center in self.task_centers
            ],
            dtype=np.float32,
        )

        # task_interest[i, kappa] = 1 if agent i is interested in task kappa.
        self.task_interest: np.ndarray = np.array(
            [
                [
                    1.0 if circular_task_distance(center, kappa, T) <= C else 0.0
                    for kappa in range(T)
                ]
                for center in self.task_centers
            ],
            dtype=np.float32,
        )

        # relatedness remains an agent-agent compatibility mask for legacy code;
        # it is derived from distance between agents' base tasks.
        self.relatedness: np.ndarray = np.array(
            [
                [
                    1.0 if circular_task_distance(self.task_centers[i], self.task_centers[j], T) <= C else 0.0
                    for j in range(N)
                ]
                for i in range(N)
            ],
            dtype=np.float32,
        )

        # teammate_mask[i, j] = relatedness[i,j] > 0  (N x N bool)
        self.teammate_mask: np.ndarray = self.relatedness > 0
        # proficiency_positive_types[i] = frozenset of kappa where proficiency[i, kappa] > 0
        self.proficiency_positive_types: tuple[frozenset[int], ...] = tuple(
            frozenset(kappa for kappa in range(T) if self.proficiency[i, kappa] > 0)
            for i in range(N)
        )

        # Torch tensors for use in encoders
        self._proficiency_t: torch.Tensor = torch.from_numpy(self.proficiency)  # (N, T)
        self._rel_t: torch.Tensor = torch.from_numpy(self.relatedness)           # (N, N)

        # category_rewards is set by StochasticEnv after super().__init__
        # Shape: (T, N) — category_rewards[kappa, j] = r'_j^(kappa)
        self.category_rewards: np.ndarray = np.zeros((T, N), dtype=np.float32)

        # Precomputed pick reward table; rebuilt by _precompute_pick_rewards() after
        # category_rewards is populated. Shape: (N, T, N).
        self._pick_rewards: np.ndarray = np.zeros((N, T, N), dtype=np.float32)

    def _precompute_pick_rewards(self) -> None:
        """Build _pick_rewards[actor, tau, j] = proficiency[actor,tau] * r'[tau,j].

        The relatedness/C^(tau) mask is already baked into category_rewards r'
        (see StochasticEnv._generate_category_rewards), so no separate R factor
        is needed here. r'[tau,j] is nonzero only for j in C^(tau).
        """
        self._pick_rewards = (
            self.proficiency[:, :, np.newaxis]
            * self.category_rewards[np.newaxis, :, :]
        ).astype(np.float32)

    def set_eval_mode(
        self,
        eval_mode: bool,
        seed: int | None = None,
        fixed_spawn_zones: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        """Switch between training and eval mode. No-op by default."""

    @abstractmethod
    def init_state(self) -> State:
        ...

    def apply_action(self, state: State, action: Action) -> State:
        """Apply movement action to current actor. Clamps to grid boundaries."""
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
        """Find task index and type for a pick attempt. Returns None if no pick (STAY)."""
        # CHOICE mode: pick_type=None means STAY (decline)
        if pick_type is None:
            return None
        actor = state.actor
        pos = state.agent_positions[actor]
        for i, (tp, tt) in enumerate(zip(state.task_positions, state.task_types or [])):
            if tp == pos and tt == pick_type:
                return i, pick_type
        return None

    def _compute_pick_rewards(
        self, actor: int, tau: int,
    ) -> tuple[float, ...]:
        """Per-agent rewards: r_j = phi[actor, tau] * r'[tau, j] (r' carries the C^(tau) mask)."""
        return tuple(self._pick_rewards[actor, tau].tolist())

    def resolve_pick(
        self, state: State, pick_type: int | None = None,
    ) -> tuple[State, tuple[float, ...]]:
        """Resolve task pickup. Returns (new_state, per_agent_rewards)."""
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
        """Full step: move → spawn/despawn → advance.

        Pick resolution is handled externally by the training loop (CHOICE mode).
        """
        assert action.is_move(), f"step() only accepts move actions, got {action}"

        s_moved = self.apply_action(state, action)
        s_spawned = self.spawn_and_despawn(s_moved)
        s_next = self.advance_actor(s_spawned)

        zero_rewards = tuple(0.0 for _ in range(self.cfg.n_agents))
        return Transition(
            s_t=state, action=action, s_t_after=s_moved,
            s_t_next=s_next, rewards=zero_rewards, discount=self.cfg.gamma,
        )

    @property
    def n_agents(self) -> int:
        return self.cfg.n_agents
