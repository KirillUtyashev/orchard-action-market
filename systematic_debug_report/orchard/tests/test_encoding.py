"""Tests for encoding: dimension checks and known outputs."""

import pytest
import torch

from orchard.enums import EncoderType, EnvType
from orchard.datatypes import EnvConfig, Grid, State
import orchard.encoding as encoding
from orchard.encoding.relative import RelativeEncoder, RelativeSimpleEncoder
from orchard.encoding.grid import BasicGridEncoder


def _make_cfg(**overrides) -> EnvConfig:
    defaults = dict(
        height=2, width=2, n_agents=2, n_apples=1,
        gamma=0.9, r_picker=-1.0, force_pick=True,
        max_apples=1, env_type=EnvType.DETERMINISTIC,
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


class TestRelativeEncoder:
    def test_dim(self):
        cfg = _make_cfg()
        enc = RelativeEncoder(cfg)
        # n_apples*2 + (n_agents-1)*3 + 4 = 1*2 + 1*3 + 4 = 9
        assert enc.input_dim() == 9

    def test_output_shape(self):
        cfg = _make_cfg()
        enc = RelativeEncoder(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0),),
            actor=0,
        )
        out = enc.encode(s, 0)
        assert out.scalar is not None
        assert out.grid is None
        assert out.scalar.shape == (9,)

    def test_known_output(self):
        cfg = _make_cfg()
        enc = RelativeEncoder(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1)),
            apple_positions=(Grid(1, 0),),
            actor=0,
        )
        out = enc.encode(s, 0)
        feats = out.scalar.tolist()
        # Apple at (1,0), agent at (0,0): dr=1/2=0.5, dc=0/2=0.0
        assert pytest.approx(feats[0], abs=1e-6) == 0.5
        assert pytest.approx(feats[1], abs=1e-6) == 0.0


class TestRelativeSimpleEncoder:
    def test_dim(self):
        cfg = _make_cfg()
        enc = RelativeSimpleEncoder(cfg)
        # n_apples*2 + (n_agents-1)*3 + 1 = 1*2 + 1*3 + 1 = 6
        assert enc.input_dim() == 6

    def test_output_shape(self):
        cfg = _make_cfg()
        enc = RelativeSimpleEncoder(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0),),
            actor=0,
        )
        out = enc.encode(s, 0)
        assert out.scalar.shape == (6,)


class TestBasicGridEncoder:
    def test_channels(self):
        cfg = _make_cfg()
        enc = BasicGridEncoder(cfg)
        assert enc.grid_channels() == 4
        assert enc.input_dim() == 4

    def test_output_shape(self):
        cfg = _make_cfg()
        enc = BasicGridEncoder(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0),),
            actor=0,
        )
        out = enc.encode(s, 0)
        assert out.grid is not None
        assert out.scalar is None
        assert out.grid.shape == (4, 2, 2)

    def test_self_channel(self):
        cfg = _make_cfg()
        enc = BasicGridEncoder(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0),),
            actor=0,
        )
        out = enc.encode(s, 0)
        assert out.grid[0, 0, 0] == 1.0
        assert out.grid[0, 0, 1] == 0.0

    def test_apple_channel(self):
        cfg = _make_cfg()
        enc = BasicGridEncoder(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0),),
            actor=0,
        )
        out = enc.encode(s, 0)
        assert out.grid[2, 1, 0] == 1.0
        assert out.grid[2, 0, 0] == 0.0


class TestSingletonAPI:
    def test_init_and_encode(self):
        cfg = _make_cfg()
        encoding.init_encoder(EncoderType.RELATIVE, cfg)
        assert encoding.get_input_dim() == 9

        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0),),
            actor=0,
        )
        out = encoding.encode(s, 0)
        assert out.scalar is not None
        assert out.scalar.shape == (9,)
