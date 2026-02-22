"""Base encoder classes: BaseEncoder, GridEncoder."""

from __future__ import annotations

from abc import ABC, abstractmethod

from orchard.datatypes import EncoderOutput, EnvConfig, State


class BaseEncoder(ABC):
    """Encoder producing a flat feature vector."""

    def __init__(self, env_cfg: EnvConfig) -> None:
        self.env_cfg = env_cfg

    @abstractmethod
    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        ...

    @abstractmethod
    def input_dim(self) -> int:
        """Scalar feature dimension."""
        ...


class GridEncoder(BaseEncoder):
    """Encoder producing a (C, H, W) grid tensor."""

    @abstractmethod
    def grid_channels(self) -> int:
        ...

    def input_dim(self) -> int:
        """For grid encoders, input_dim returns channel count."""
        return self.grid_channels()
