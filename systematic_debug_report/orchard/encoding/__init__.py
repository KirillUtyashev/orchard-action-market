"""Encoding singleton API: init once, use everywhere."""

from __future__ import annotations

import torch

from orchard.encoding.base import BaseEncoder
from orchard.encoding.grid import GeneralDecEncoder, GeneralCenEncoder, EverythingEncoder
from orchard.enums import EncoderType
from orchard.datatypes import EncoderOutput, State


# Module-level singleton
_encoder: BaseEncoder | None = None


def init_encoder(encoder_type: EncoderType, env, n_networks: int | None = None) -> None:
    """Initialize the global encoder singleton.

    env must be a StochasticEnv (or any BaseEnv subclass) with attributes:
      .cfg, .phi, .relatedness, .category_rewards

    n_networks is required for EVERYTHING_CNN_GRID (pass 1 for centralized,
    N for decentralized). It is ignored for all other encoder types.
    """
    global _encoder
    cfg = env.cfg
    phi = env.phi
    rel = env.relatedness
    cr = env.category_rewards

    if encoder_type == EncoderType.GENERAL_DEC_CNN_GRID:
        _encoder = GeneralDecEncoder(cfg, phi, rel, cr)
    elif encoder_type == EncoderType.GENERAL_CEN_CNN_GRID:
        _encoder = GeneralCenEncoder(cfg, phi, rel, cr)
    elif encoder_type == EncoderType.EVERYTHING_CNN_GRID:
        n = n_networks if n_networks is not None else cfg.n_agents
        _encoder = EverythingEncoder(cfg, n)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def encode(state: State, agent_idx: int) -> EncoderOutput:
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.encode(state, agent_idx)


def encode_batch_for_actions(state: State, agent_idx: int, after_states: list[State]) -> EncoderOutput:
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.encode_batch_for_actions(state, agent_idx, after_states)


def encode_all_agents(state: State) -> tuple[torch.Tensor, torch.Tensor]:
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.encode_all_agents(state)


def encode_all_agents_for_actions(state: State, after_states: list[State]) -> tuple[torch.Tensor, torch.Tensor]:
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.encode_all_agents_for_actions(state, after_states)


def get_scalar_dim() -> int:
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.scalar_dim()


def get_grid_channels() -> int:
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.grid_channels()


def get_grid_height() -> int:
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.grid_height()


def get_grid_width() -> int:
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.grid_width()


def get_encoder() -> BaseEncoder:
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder
