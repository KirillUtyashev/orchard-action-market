"""Trainer base class: unified interface for CPU and GPU training paths."""

from __future__ import annotations

from abc import ABC, abstractmethod

from orchard.datatypes import EnvConfig, EvalConfig, State
from orchard.enums import Action
from orchard.env.base import BaseEnv
from orchard.model import ValueNetwork


class TrainerBase(ABC):
    """Abstract base for all trainers.

    Two concrete implementations:
      - CpuTrainer: sequential forward passes on CPU (any n_networks)
      - GpuTrainer: vmap-batched forward passes on GPU (any n_networks)

    The main loop calls:
      1. select_move(state, t) → move action
      2. select_pick(s_moved, t) → pick/stay action (only if on task)
      3. train_move(s_moved, on_task, t) → TD update: prev_after → move_after
      4. train_pick(s_picked, rewards, t) → TD update: move_after → pick_after
    """

    @abstractmethod
    def select_move(self, state: State, t: int) -> Action:
        """Epsilon-greedy move action selection (phase 1)."""
        ...

    @abstractmethod
    def select_pick(self, state: State, t: int) -> Action:
        """Epsilon-greedy pick/stay action selection (phase 2)."""
        ...

    @abstractmethod
    def train_move(self, s_moved: State, on_task: bool, t: int) -> None:
        """Encode move_after. TD update: prev_after → move_after (r=0, γ=γ).

        If not on_task, stores move_after as prev for next step.
        On first step (no prev), just stores the encoding.
        """
        ...

    @abstractmethod
    def train_pick(self, s_picked: State, rewards: tuple[float, ...], t: int) -> None:
        """Encode pick_after. TD update: move_after → pick_after (r=rewards, γ=1).

        Stores pick_after as prev for next step.
        """
        ...

    @abstractmethod
    def evaluate(self, env: BaseEnv, eval_cfg: EvalConfig) -> dict[str, float | int]:
        """Run eval rollouts and return metrics dict."""
        ...

    @abstractmethod
    def sync_to_cpu(self) -> None:
        """Push GPU params to CPU ValueNetwork objects. No-op for CpuTrainer."""
        ...

    @abstractmethod
    def get_td_loss(self) -> float:
        """Return average TD loss since last call, then reset accumulator."""
        ...

    @property
    @abstractmethod
    def networks(self) -> list[ValueNetwork]:
        """The underlying ValueNetwork list (for checkpointing and detail logging)."""
        ...
