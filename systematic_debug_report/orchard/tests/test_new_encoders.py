"""Tests for EgoCentricGridEncoder and NoRedundantAgentGridEncoder."""

import pytest
import torch

from orchard.enums import EncoderType, EnvType, PickMode
from orchard.datatypes import EnvConfig, Grid, State
from orchard.encoding.grid import EgoCentricGridEncoder, NoRedundantAgentGridEncoder


def _make_cfg(**overrides) -> EnvConfig:
    defaults = dict(
        height=6, width=6, n_agents=2, n_tasks=9,
        gamma=0.99, r_picker=-1.0, pick_mode=PickMode.FORCED,
        max_tasks=9, env_type=EnvType.STOCHASTIC,
        stochastic=None,
        task_assignments=((0,), (0,)),
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


def _make_small_cfg(**overrides) -> EnvConfig:
    defaults = dict(
        height=3, width=3, n_agents=2, n_tasks=1,
        gamma=0.99, r_picker=-1.0, pick_mode=PickMode.FORCED,
        max_tasks=1, env_type=EnvType.DETERMINISTIC,
        task_assignments=((0,), (0,)),
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


# ═══════════════════════════════════════════════════════════════════════
# EgoCentricGridEncoder
# ═══════════════════════════════════════════════════════════════════════

class TestEgoCentricGridEncoder:

    def test_grid_dimensions_6x6(self):
        cfg = _make_cfg()
        enc = EgoCentricGridEncoder(cfg)
        assert enc.grid_channels() == 4
        assert enc.grid_height() == 11  # 2*6 - 1
        assert enc.grid_width() == 11
        assert enc.scalar_dim() == 1

    def test_grid_dimensions_3x3(self):
        cfg = _make_small_cfg()
        enc = EgoCentricGridEncoder(cfg)
        assert enc.grid_channels() == 4
        assert enc.grid_height() == 5  # 2*3 - 1
        assert enc.grid_width() == 5
        assert enc.scalar_dim() == 1

    def test_output_shape_6x6(self):
        cfg = _make_cfg()
        enc = EgoCentricGridEncoder(cfg)
        s = State(
            agent_positions=(Grid(2, 3), Grid(4, 1)),
            task_positions=(Grid(0, 0), Grid(5, 5)),
            actor=0,
        )
        out = enc.encode(s, 0)
        assert out.grid is not None
        assert out.grid.shape == (4, 11, 11)
        assert out.scalar is not None
        assert out.scalar.shape == (1,)

    def test_self_always_at_center(self):
        """Self channel should be 1 at center (H-1, W-1) regardless of agent position."""
        cfg = _make_small_cfg()  # 3x3 → 5x5 ego grid, center=(2,2)
        enc = EgoCentricGridEncoder(cfg)

        for r, c in [(0, 0), (1, 1), (2, 2), (0, 2)]:
            s = State(
                agent_positions=(Grid(r, c), Grid(1, 1) if (r, c) != (1, 1) else Grid(0, 0)),
                task_positions=(Grid(2, 2),),
                actor=0,
            )
            out = enc.encode(s, 0)
            # Ch1 (self) should be 1 at center (2,2) and 0 elsewhere in-bounds
            assert out.grid[1, 2, 2] == 1.0, f"Self not at center for agent at ({r},{c})"

    def test_oob_is_negative_one(self):
        """Out-of-bounds cells should be -1 on all channels."""
        cfg = _make_small_cfg()  # 3x3 → 5x5 ego grid
        enc = EgoCentricGridEncoder(cfg)

        # Agent at corner (0,0): top-left of ego grid is OOB
        s = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(Grid(1, 1),),
            actor=0,
        )
        out = enc.encode(s, 0)
        # (0,0) in ego grid → world = (0,0) + (0-2, 0-2) = (-2,-2) → OOB
        for ch in range(4):
            assert out.grid[ch, 0, 0] == -1.0, f"Ch{ch} at (0,0) should be -1 (OOB)"

    def test_inbounds_not_negative_one(self):
        """In-bounds cells should not be -1."""
        cfg = _make_small_cfg()  # 3x3 → 5x5 ego grid
        enc = EgoCentricGridEncoder(cfg)

        # Agent at center (1,1): all world cells are reachable
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0)),
            task_positions=(Grid(2, 2),),
            actor=0,
        )
        out = enc.encode(s, 0)
        # Center agent: ego (2,2) → world (1,1), ego (0,0) → world (-1,-1) OOB
        # ego (1,1) → world (0,0) in bounds
        # Check that the center cell is not -1 on any channel
        for ch in range(4):
            assert out.grid[ch, 2, 2] != -1.0

    def test_apple_placement(self):
        """Apple at world (2,2), agent at (1,1) on 3x3 → apple at ego (3,3)."""
        cfg = _make_small_cfg()
        enc = EgoCentricGridEncoder(cfg)
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0)),
            task_positions=(Grid(2, 2),),
            actor=0,
        )
        out = enc.encode(s, 0)
        # offset: (2-1) = (1,1) from center (2,2) → ego (3,3)
        assert out.grid[0, 3, 3] == 1.0

    def test_other_agent_placement(self):
        """Other agent at world (0,0), self at (1,1) on 3x3 → other at ego (1,1)."""
        cfg = _make_small_cfg()
        enc = EgoCentricGridEncoder(cfg)
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0)),
            task_positions=(Grid(2, 2),),
            actor=0,
        )
        out = enc.encode(s, 0)
        # other at world (0,0): offset = (0-1, 0-1) = (-1,-1) from center (2,2) → ego (1,1)
        assert out.grid[2, 1, 1] == 1.0

    def test_actor_channel(self):
        """Actor channel marks the actor's position in ego coords."""
        cfg = _make_small_cfg()
        enc = EgoCentricGridEncoder(cfg)
        # agent 0 at (1,1), agent 1 at (0,0), actor=0
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0)),
            task_positions=(Grid(2, 2),),
            actor=0,
        )
        out = enc.encode(s, 0)
        # Actor is agent 0 at (1,1) = self → at center (2,2)
        assert out.grid[3, 2, 2] == 1.0

    def test_scalar_is_actor(self):
        cfg = _make_small_cfg()
        enc = EgoCentricGridEncoder(cfg)
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0)),
            task_positions=(Grid(2, 2),),
            actor=0,
        )
        # Agent 0 is actor
        out0 = enc.encode(s, 0)
        assert out0.scalar.item() == 1.0
        # Agent 1 is not actor
        out1 = enc.encode(s, 1)
        assert out1.scalar.item() == 0.0

    def test_batch_shape_is_actor(self):
        """encode_batch_for_actions shape when agent IS the actor."""
        cfg = _make_small_cfg()
        enc = EgoCentricGridEncoder(cfg)
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0)),
            task_positions=(Grid(2, 2),),
            actor=0,
        )
        # 3 fake after-states (actor moved to different positions)
        after_states = [
            State(agent_positions=(Grid(0, 1), Grid(0, 0)), task_positions=(Grid(2, 2),), actor=0),
            State(agent_positions=(Grid(1, 0), Grid(0, 0)), task_positions=(Grid(2, 2),), actor=0),
            State(agent_positions=(Grid(1, 1), Grid(0, 0)), task_positions=(Grid(2, 2),), actor=0),
        ]
        out = enc.encode_batch_for_actions(s, 0, after_states)
        assert out.grid.shape == (3, 4, 5, 5)
        assert out.scalar.shape == (3, 1)

    def test_batch_shape_not_actor(self):
        """encode_batch_for_actions shape when agent is NOT the actor."""
        cfg = _make_small_cfg()
        enc = EgoCentricGridEncoder(cfg)
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0)),
            task_positions=(Grid(2, 2),),
            actor=0,
        )
        after_states = [
            State(agent_positions=(Grid(0, 1), Grid(0, 0)), task_positions=(Grid(2, 2),), actor=0),
            State(agent_positions=(Grid(1, 0), Grid(0, 0)), task_positions=(Grid(2, 2),), actor=0),
        ]
        # Encode from agent 1's perspective (not actor)
        out = enc.encode_batch_for_actions(s, 1, after_states)
        assert out.grid.shape == (2, 4, 5, 5)
        assert out.scalar.shape == (2, 1)
        assert (out.scalar == 0.0).all()


# ═══════════════════════════════════════════════════════════════════════
# NoRedundantAgentGridEncoder
# ═══════════════════════════════════════════════════════════════════════

class TestNoRedundantAgentGridEncoder:

    def test_channels_and_dims(self):
        cfg = _make_small_cfg()
        enc = NoRedundantAgentGridEncoder(cfg)
        assert enc.grid_channels() == 4
        assert enc.scalar_dim() == 1
        assert enc.grid_height() == 3
        assert enc.grid_width() == 3

    def test_output_shape(self):
        cfg = _make_small_cfg()
        enc = NoRedundantAgentGridEncoder(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(Grid(1, 1),),
            actor=0,
        )
        out = enc.encode(s, 0)
        assert out.grid.shape == (4, 3, 3)
        assert out.scalar.shape == (1,)

    def test_self_zeroed_when_actor(self):
        """When agent IS the actor, self channel should be all zeros."""
        cfg = _make_small_cfg()
        enc = NoRedundantAgentGridEncoder(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(Grid(1, 1),),
            actor=0,
        )
        out = enc.encode(s, 0)  # agent 0 is actor
        # Ch1 (self) should be all zeros
        assert (out.grid[1] == 0.0).all()

    def test_self_present_when_not_actor(self):
        """When agent is NOT the actor, self channel should mark agent's position."""
        cfg = _make_small_cfg()
        enc = NoRedundantAgentGridEncoder(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(Grid(1, 1),),
            actor=0,
        )
        out = enc.encode(s, 1)  # agent 1 is not actor
        # Ch1 (self) should be 1 at (2,2)
        assert out.grid[1, 2, 2] == 1.0
        # And 0 elsewhere
        assert out.grid[1, 0, 0] == 0.0
        assert out.grid[1, 1, 1] == 0.0

    def test_others_excludes_actor_when_not_self(self):
        """Others channel excludes both self AND actor."""
        cfg = _make_small_cfg(n_agents=3, n_tasks=1)
        enc = NoRedundantAgentGridEncoder(cfg)
        # 3 agents: 0 at (0,0), 1 at (1,1), 2 at (2,2). Actor=0
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1), Grid(2, 2)),
            task_positions=(Grid(0, 2),),
            actor=0,
        )
        # From agent 1's perspective: self=1, actor=0
        # Ch2 (others) should only have agent 2 (exclude self=1 and actor=0)
        out = enc.encode(s, 1)
        assert out.grid[2, 2, 2] == 1.0  # agent 2
        assert out.grid[2, 0, 0] == 0.0  # actor excluded
        assert out.grid[2, 1, 1] == 0.0  # self excluded

    def test_others_excludes_self_when_actor(self):
        """When agent IS the actor, others = everyone except self."""
        cfg = _make_small_cfg()
        enc = NoRedundantAgentGridEncoder(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(Grid(1, 1),),
            actor=0,
        )
        out = enc.encode(s, 0)  # agent 0 is actor
        # Ch2: others excluding self AND actor (same person) → just agent 1
        assert out.grid[2, 2, 2] == 1.0  # agent 1
        assert out.grid[2, 0, 0] == 0.0  # self/actor excluded

    def test_actor_channel_always_present(self):
        """Actor channel should always mark actor's position."""
        cfg = _make_small_cfg()
        enc = NoRedundantAgentGridEncoder(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(Grid(1, 1),),
            actor=0,
        )
        # From agent 0 (is actor)
        out0 = enc.encode(s, 0)
        assert out0.grid[3, 0, 0] == 1.0
        # From agent 1 (not actor)
        out1 = enc.encode(s, 1)
        assert out1.grid[3, 0, 0] == 1.0

    def test_scalar_is_actor(self):
        cfg = _make_small_cfg()
        enc = NoRedundantAgentGridEncoder(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(Grid(1, 1),),
            actor=0,
        )
        out0 = enc.encode(s, 0)
        assert out0.scalar.item() == 1.0
        out1 = enc.encode(s, 1)
        assert out1.scalar.item() == 0.0

    def test_batch_shape(self):
        cfg = _make_small_cfg()
        enc = NoRedundantAgentGridEncoder(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(Grid(1, 1),),
            actor=0,
        )
        after_states = [
            State(agent_positions=(Grid(0, 1), Grid(2, 2)), task_positions=(Grid(1, 1),), actor=0),
            State(agent_positions=(Grid(1, 0), Grid(2, 2)), task_positions=(Grid(1, 1),), actor=0),
        ]
        out = enc.encode_batch_for_actions(s, 0, after_states)
        assert out.grid.shape == (2, 4, 3, 3)
        assert out.scalar.shape == (2, 1)

    def test_matches_basic_when_no_redundancy(self):
        """With 2 agents, when agent is NOT actor, BasicGridEncoder and
        NoRedundantAgentGridEncoder should differ only on Ch1 and Ch2."""
        from orchard.encoding.grid import BasicGridEncoder
        cfg = _make_small_cfg()
        basic = BasicGridEncoder(cfg)
        noreg = NoRedundantAgentGridEncoder(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(Grid(1, 1),),
            actor=0,
        )
        out_b = basic.encode(s, 1)
        out_n = noreg.encode(s, 1)
        # Ch0 (apples) should be identical
        assert torch.equal(out_b.grid[0], out_n.grid[0])
        # Ch3 (actor) should be identical
        assert torch.equal(out_b.grid[3], out_n.grid[3])

