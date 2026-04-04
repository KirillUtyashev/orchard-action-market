"""Environment package: create_env factory."""

from __future__ import annotations

from orchard.env.base import BaseEnv
from orchard.env.stochastic import StochasticEnv
from orchard.datatypes import EnvConfig


def create_env(cfg: EnvConfig) -> BaseEnv:
    return StochasticEnv(cfg)
