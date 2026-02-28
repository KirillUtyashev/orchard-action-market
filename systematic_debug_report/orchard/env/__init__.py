"""Environment package: create_env factory."""

from __future__ import annotations

from orchard.enums import EnvType
from orchard.env.base import BaseEnv
from orchard.env.deterministic import DeterministicEnv
from orchard.env.stochastic import StochasticEnv
from orchard.datatypes import EnvConfig


def create_env(cfg: EnvConfig) -> BaseEnv:
    """Factory: create environment based on config."""
    if cfg.env_type == EnvType.DETERMINISTIC:
        return DeterministicEnv(cfg)
    elif cfg.env_type == EnvType.STOCHASTIC:
        return StochasticEnv(cfg)
    else:
        raise ValueError(f"Unknown env type: {cfg.env_type}")
