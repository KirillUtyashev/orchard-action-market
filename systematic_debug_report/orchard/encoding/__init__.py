"""Encoding singleton API: init once, use everywhere."""

from __future__ import annotations

from orchard.encoding.base import BaseEncoder
from orchard.encoding.grid import BasicGridEncoder, CentralizedGridEncoder, GridMLPEncoder
from orchard.encoding.relative import PositionalKEncoder, RelativeEncoder, RelativeKEncoder, StableIdEncoder
from orchard.enums import EncoderType
from orchard.datatypes import EncoderOutput, EnvConfig, State


# Module-level singleton
_encoder: BaseEncoder | None = None


def _create_encoder(encoder_type, env_cfg, k=None):
    if encoder_type == EncoderType.RELATIVE:
        return RelativeEncoder(env_cfg)
    elif encoder_type == EncoderType.RELATIVE_K:
        assert k is not None, "k_nearest required for RELATIVE_K"
        return RelativeKEncoder(env_cfg, k)
    elif encoder_type == EncoderType.CNN_GRID:
        return BasicGridEncoder(env_cfg)
    elif encoder_type == EncoderType.POSITIONAL_K:
            assert k is not None, "k_nearest required for POSITIONAL_K"
            return PositionalKEncoder(env_cfg, k)
    elif encoder_type == EncoderType.STABLE_ID:
        return StableIdEncoder(env_cfg)
    elif encoder_type == EncoderType.GRID_MLP:
        return GridMLPEncoder(env_cfg)
    elif encoder_type == EncoderType.CENTRALIZED_CNN_GRID:
        return CentralizedGridEncoder(env_cfg)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

def init_encoder(encoder_type, env_cfg, k=None):
    global _encoder
    _encoder = _create_encoder(encoder_type, env_cfg, k)


def encode(state: State, agent_idx: int) -> EncoderOutput:
    """Encode state from agent's perspective. Uses the global encoder."""
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.encode(state, agent_idx)


def get_scalar_dim() -> int:
    """Scalar feature dim (full input for MLP, extra scalars for CNN)."""
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.scalar_dim()


def get_grid_channels() -> int:
    """Grid channel count. 0 if MLP encoder."""
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.grid_channels()


def get_encoder() -> BaseEncoder:
    """Return the global encoder instance."""
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder
