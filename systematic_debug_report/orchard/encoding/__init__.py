"""Encoding singleton API: init once, use everywhere."""

from __future__ import annotations

from orchard.encoding.base import BaseEncoder
from orchard.encoding.grid import BasicGridEncoder
from orchard.encoding.relative import RelativeEncoder, RelativeKEncoder
from orchard.enums import EncoderType
from orchard.datatypes import EncoderOutput, EnvConfig, State

# Module-level singleton
_encoder: BaseEncoder | None = None


def _create_encoder(encoder_type, env_cfg, k_nearest=None):
    if encoder_type == EncoderType.RELATIVE:
        return RelativeEncoder(env_cfg)
    elif encoder_type == EncoderType.RELATIVE_K:
        assert k_nearest is not None, "k_nearest required for RELATIVE_K"
        return RelativeKEncoder(env_cfg, k_nearest)
    elif encoder_type == EncoderType.CNN_GRID:
        return BasicGridEncoder(env_cfg)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

def init_encoder(encoder_type, env_cfg, k_nearest=None):
    global _encoder
    _encoder = _create_encoder(encoder_type, env_cfg, k_nearest)


def encode(state: State, agent_idx: int) -> EncoderOutput:
    """Encode state from agent's perspective. Uses the global encoder."""
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.encode(state, agent_idx)


def get_input_dim() -> int:
    """Return scalar input dim (for MLP) or channel count (for CNN)."""
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder.input_dim()


def get_encoder() -> BaseEncoder:
    """Return the global encoder instance."""
    assert _encoder is not None, "Call init_encoder() first"
    return _encoder
