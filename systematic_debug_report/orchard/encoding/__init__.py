"""Encoding singleton API: init once, use everywhere."""

from __future__ import annotations

import torch

from orchard.encoding.base import BaseEncoder
from orchard.encoding.grid import (
    BlindTaskGridEncoder,
    CentralizedTaskGridEncoder,
    FilteredTaskGridEncoder,
)
from orchard.enums import EncoderType
from orchard.datatypes import EncoderOutput, EnvConfig, State


# Module-level singleton
_encoder: BaseEncoder | None = None


_ENCODER_MAP = {
    EncoderType.BLIND_TASK_CNN_GRID: BlindTaskGridEncoder,
    EncoderType.FILTERED_TASK_CNN_GRID: FilteredTaskGridEncoder,
    EncoderType.CENTRALIZED_TASK_CNN_GRID: CentralizedTaskGridEncoder,
}


def init_encoder(encoder_type: EncoderType, env_cfg: EnvConfig) -> None:
    global _encoder
    cls = _ENCODER_MAP.get(encoder_type)
    if cls is None:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    _encoder = cls(env_cfg)


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
