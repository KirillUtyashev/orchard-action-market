"""Frame: all information needed to render one transition."""

from __future__ import annotations

from dataclasses import dataclass

from orchard.datatypes import State
from orchard.enums import Action


@dataclass(frozen=True)
class Decision:
    """One candidate action and its Q-value."""
    action: Action
    q_value: float
    is_chosen: bool

    # Per-agent V_i(after_state(s, a)) breakdown (decentralized only)
    agent_q_values: dict[int, float] | None = None


@dataclass(frozen=True)
class Frame:
    """Everything needed to render one transition.

    Naming convention (Sutton & Barto):
      Transition from s_t with discount γ_{t+1} and reward r_{t+1} to s_{t+1}.
      state_index = t.
    """

    # Indexing
    step: int                           # agent decision index (0, 1, 2, ...)
    transition_index: int               # global transition index (0, 1, 2, 3, ...)
    state_index: int                    # t in s_t (increments per transition)

    # States
    state: State                        # s_t
    state_after: State                  # s_{t+1} as after-state (post-action, pre-env)
    height: int
    width: int

    # Transition info: s_t --(action, r_{t+1}, γ_{t+1})--> s_{t+1}
    actor: int
    action: Action                      # PICK (forced) or PICK_τ (choice mode)
    rewards: tuple[float, ...]          # r_{t+1}
    discount: float                     # γ_{t+1}
    picked: bool                        # True only on PICK transitions

    # Metadata
    policy_name: str

    # Running stats (based on agent decisions, not transitions)
    total_picks: int
    total_decisions: int                # number of agent decisions so far
    tasks_on_grid: int                  # len(state.task_positions) at s_t
    tasks_after: int                    # len(state_after.task_positions) at s_{t+1}

    # Pick tracking (always active, regardless of n_task_types)
    picked_task_type: int | None = None     # type of task picked (None if no pick)
    picked_correct: bool | None = None      # True if τ ∈ G_actor, False if not, None if no pick
    total_correct_picks: int = 0
    total_wrong_picks: int = 0
    total_reward: float = 0.0               # cumulative actor reward across all transitions
    total_team_reward: float = 0.0          # cumulative team reward (sum over agents)

    # Optional: decision introspection (--decisions)
    decisions: list[Decision] | None = None

    # Optional: per-agent values (--values)
    agent_values: dict[int, float] | None = None

    # Per-agent cumulative picks
    agent_picks: dict[int, int] | None = None

    # Backward compat aliases
    @property
    def apples_on_grid(self) -> int:
        return self.tasks_on_grid

    @property
    def apples_after(self) -> int:
        return self.tasks_after

    @property
    def picks_per_step(self) -> float:
        return self.total_picks / self.total_decisions if self.total_decisions > 0 else 0.0

    @property
    def reward_per_step(self) -> float:
        return self.total_reward / self.total_decisions if self.total_decisions > 0 else 0.0

    @property
    def team_reward_per_step(self) -> float:
        return self.total_team_reward / self.total_decisions if self.total_decisions > 0 else 0.0

    def agent_picks_per_step(self, agent: int) -> float:
        """Per-agent picks / total decisions."""
        if self.agent_picks is None or self.total_decisions == 0:
            return 0.0
        return self.agent_picks.get(agent, 0) / self.total_decisions
