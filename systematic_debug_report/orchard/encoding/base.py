"""Encoder base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod

from orchard.datatypes import EncoderOutput, EnvConfig, State


class BaseEncoder(ABC):
    """Base encoder. Every encoder produces scalar features."""

    def __init__(self, env_cfg: EnvConfig) -> None:
        self.env_cfg = env_cfg

    @abstractmethod
    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        ...

    @abstractmethod
    def scalar_dim(self) -> int:
        """Scalar feature dimension.

        MLP encoders: full input vector length.
        Grid encoders: extra scalars concatenated after conv flatten.
        """
        ...

    def grid_channels(self) -> int:
        """Grid channel count. 0 means no grid output (MLP encoder)."""
        return 0


class GridEncoder(BaseEncoder):
    """Base for encoders that produce grid + scalar output."""

    @abstractmethod
    def grid_channels(self) -> int:
        ...