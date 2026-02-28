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
    action: Action                      # includes PICK for forced picks
    rewards: tuple[float, ...]          # r_{t+1}
    discount: float                     # γ_{t+1}
    picked: bool                        # True only on PICK transitions

    # Metadata
    policy_name: str

    # Running stats (based on agent decisions, not transitions)
    total_picks: int
    total_decisions: int                # number of agent decisions so far
    apples_on_grid: int                 # len(state.apple_positions) at s_t
    apples_after: int                   # len(state_after.apple_positions) at s_{t+1}

    # Optional: decision introspection (--decisions)
    decisions: list[Decision] | None = None

    # Optional: per-agent values (--values)
    agent_values: dict[int, float] | None = None

    @property
    def picks_per_step(self) -> float:
        return self.total_picks / self.total_decisions if self.total_decisions > 0 else 0.0
