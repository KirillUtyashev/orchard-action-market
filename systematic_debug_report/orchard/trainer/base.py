"""Trainer base class: unified interface for CPU and GPU training paths."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from orchard.datatypes import EnvConfig, EvalConfig, State
from orchard.env.base import BaseEnv
from orchard.model import ValueNetwork
from orchard.actor_critic import PolicyNetwork


class TrainerBase(ABC):
    """Abstract base for all orchard trainers."""

    @abstractmethod
    def step(self, state: State, t: int) -> State:
        """Advance training by one actor turn and return the next state."""
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

    def get_main_metrics(self) -> dict[str, float | int | str]:
        return {}

    def get_detail_metrics(self) -> dict[str, float | int | str]:
        return {}

    def setup_aux_loggers(self, run_dir: Path) -> None:
        del run_dir

    def log_auxiliary(self, step: int, wall_time: float) -> None:
        del step, wall_time

    def close(self) -> None:
        """Release trainer-owned resources such as auxiliary loggers."""

    def flush_pending_updates(self) -> None:
        """Apply any deferred training updates. Default is no-op."""

    @abstractmethod
    def save_checkpoint(self, path: Path, step: int) -> None:
        """Persist trainer state to disk."""
        ...

    @abstractmethod
    def load_checkpoint(self, path: str | Path) -> int | None:
        """Load trainer state from disk and return the saved step if present."""
        ...

    def load_critic_checkpoint(self, path: str | Path) -> int | None:
            """Load critic weights only. Only supported by actor-critic trainers."""
            raise NotImplementedError(f"{type(self).__name__} does not support load_critic_checkpoint.")

    def load_actor_checkpoint(self, path: str | Path) -> int | None:
        """Load actor weights only. Only supported by actor-critic trainers."""
        raise NotImplementedError(f"{type(self).__name__} does not support load_actor_checkpoint.")
    
    @property
    @abstractmethod
    def critic_networks(self) -> list[ValueNetwork]:
        """Underlying critic networks."""
        ...

    @property
    def actor_networks(self) -> list[PolicyNetwork]:
        """Underlying actor networks, if any."""
        return []
