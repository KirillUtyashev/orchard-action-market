"""Encoding singleton API: init once, use everywhere."""

from __future__ import annotations

from orchard.encoding.base import BaseEncoder
from orchard.encoding.grid import BasicGridEncoder, CentralizedGridEncoder, EgoCentricGridEncoder, GridMLPEncoder, NoRedundantAgentGridEncoder, TaskGridEncoder, CentralizedTaskGridEncoder
from orchard.encoding.relative import RelativeEncoder, RelativeKEncoder
from orchard.enums import EncoderType
from orchard.datatypes import EncoderOutput, EnvConfig, State


# Module-level singleton
_encoder: BaseEncoder | None = None


def _create_encoder(encoder_type, env_cfg, k=None, use_vec_encode=True):
    if encoder_type == EncoderType.RELATIVE:
        return RelativeEncoder(env_cfg)
    elif encoder_type == EncoderType.RELATIVE_K:
        assert k is not None, "k_nearest required for RELATIVE_K"
        return RelativeKEncoder(env_cfg, k)
    elif encoder_type == EncoderType.CNN_GRID:
        return BasicGridEncoder(env_cfg)
    elif encoder_type == EncoderType.GRID_MLP:
        return GridMLPEncoder(env_cfg)
    elif encoder_type == EncoderType.CENTRALIZED_CNN_GRID:
        return CentralizedGridEncoder(env_cfg)
    elif encoder_type == EncoderType.EGOCENTRIC_CNN_GRID:
        return EgoCentricGridEncoder(env_cfg)
    elif encoder_type == EncoderType.NO_REDUNDANT_AGENT_GRID:
        return NoRedundantAgentGridEncoder(env_cfg)
    elif encoder_type == EncoderType.TASK_CNN_GRID:
        return TaskGridEncoder(env_cfg, use_vec_encode=use_vec_encode)
    elif encoder_type == EncoderType.CENTRALIZED_TASK_CNN_GRID:
        return CentralizedTaskGridEncoder(env_cfg, use_vec_encode=use_vec_encode)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

def init_encoder(encoder_type, env_cfg, k=None, use_vec_encode=True):
    global _encoder
    _encoder = _create_encoder(encoder_type, env_cfg, k, use_vec_encode=use_vec_encode)


def encode(state: State, agent_idx: int) -> EncoderOutput:
    """Encode state from agent's perspective. Uses the global encoder."""
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.encode(state, agent_idx)

def encode_batch_for_actions(state: State, agent_idx: int, after_states: list[State]) -> EncoderOutput:
    """Batch-encode multiple after-states for one agent. Uses the global encoder."""
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.encode_batch_for_actions(state, agent_idx, after_states)


def get_scalar_dim() -> int:
    """Scalar feature dim (full input for MLP, extra scalars for CNN)."""
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.scalar_dim()


def get_grid_channels() -> int:
    """Grid channel count. 0 if MLP encoder."""
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.grid_channels()

def get_grid_height() -> int:
    """Spatial height of the encoder grid."""
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.grid_height()


def get_grid_width() -> int:
    """Spatial width of the encoder grid."""
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.grid_width()


def get_encoder() -> BaseEncoder:
    """Return the global encoder instance."""
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder
