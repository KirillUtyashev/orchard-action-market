"""Tests for PositionAwareTaskGridEncoder (O(1) decentralized, 5 channels)."""

import pytest
import torch
from orchard.enums import Action, PickMode
from orchard.datatypes import EnvConfig, StochasticConfig, Grid, State
from orchard.env.stochastic import StochasticEnv
from orchard.encoding.grid import PositionAwareTaskGridEncoder

def _make_cfg() -> EnvConfig:
    # 4 agents, 2 types.
    # A0, A1 -> Type 0 (Teammates)
    # A2, A3 -> Type 1 (Teammates)
    return EnvConfig(
        height=5, width=5, n_agents=4, n_tasks=2, gamma=0.99, r_picker=1.0,
        n_task_types=2, pick_mode=PickMode.FORCED, max_tasks_per_type=2,
        task_assignments=((0,), (0,), (1,), (1,)),
        stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=None, despawn_prob=0.0)
    )

def _make_state() -> State:
    return State(
        agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
        task_positions=(Grid(1, 0), Grid(1, 1)),
        actor=0, task_types=(0, 1), pick_phase=False
    )


class TestPositionAwareEncoderChannels:
    def test_dimensions(self):
        cfg = _make_cfg()
        enc = PositionAwareTaskGridEncoder(cfg)
        assert enc.grid_channels() == 5
        assert enc.scalar_dim() == 3

    def test_my_tasks_only(self):
        """Ch0 shows only tasks matching agent's type — no irrelevant tasks channel."""
        enc = PositionAwareTaskGridEncoder(_make_cfg())
        out = enc.encode(_make_state(), agent_idx=0)

        # Agent 0 owns Type 0. Task at (1,0) is Type 0 → visible.
        assert out.grid[0, 1, 0].item() == 1.0
        # Task at (1,1) is Type 1 → invisible (no Ch1 for irrelevant tasks).
        assert out.grid[0, 1, 1].item() == 0.0

    def test_no_irrelevant_tasks_channel(self):
        """Unlike Filtered, there is no irrelevant tasks channel."""
        enc = PositionAwareTaskGridEncoder(_make_cfg())
        out = enc.encode(_make_state(), agent_idx=0)

        # Ch1 is Self position, not irrelevant tasks.
        # Agent 0 at (0,0) → Ch1 should have self there.
        assert out.grid[1, 0, 0].item() == 1.0
        # (1,1) has irrelevant task but Ch1 should NOT show it.
        assert out.grid[1, 1, 1].item() == 0.0

    def test_self_position(self):
        enc = PositionAwareTaskGridEncoder(_make_cfg())
        out = enc.encode(_make_state(), agent_idx=2)

        # Agent 2 at (0,2)
        assert out.grid[1, 0, 2].item() == 1.0
        assert out.grid[1].sum().item() == 1.0

    def test_teammates(self):
        enc = PositionAwareTaskGridEncoder(_make_cfg())
        out = enc.encode(_make_state(), agent_idx=0)

        # Agent 0's teammate is A1 at (0,1)
        assert out.grid[2, 0, 1].item() == 1.0
        # A2, A3 are strangers, not here
        assert out.grid[2, 0, 2].item() == 0.0
        assert out.grid[2, 0, 3].item() == 0.0

    def test_strangers_visible(self):
        """Key difference from Blind: strangers appear in Ch3."""
        enc = PositionAwareTaskGridEncoder(_make_cfg())
        out = enc.encode(_make_state(), agent_idx=0)

        # A2 at (0,2), A3 at (0,3) are strangers to A0
        assert out.grid[3, 0, 2].item() == 1.0
        assert out.grid[3, 0, 3].item() == 1.0
        # A1 is teammate, not stranger
        assert out.grid[3, 0, 1].item() == 0.0

    def test_actor_channel(self):
        enc = PositionAwareTaskGridEncoder(_make_cfg())
        s = _make_state()  # Actor is 0 at (0,0)
        out = enc.encode(s, agent_idx=2)

        assert out.grid[4, 0, 0].item() == 1.0
        assert out.grid[4].sum().item() == 1.0

    def test_scalars(self):
        enc = PositionAwareTaskGridEncoder(_make_cfg())
        s = _make_state()  # Actor is 0

        # From actor's perspective
        out0 = enc.encode(s, agent_idx=0)
        assert out0.scalar[0].item() == 1.0  # is_self_actor
        assert out0.scalar[1].item() == 0.0  # is_teammate_actor (actor is not own teammate)

        # From teammate of actor
        out1 = enc.encode(s, agent_idx=1)
        assert out1.scalar[0].item() == 0.0  # is_self_actor
        assert out1.scalar[1].item() == 1.0  # is_teammate_actor

        # From stranger to actor
        out2 = enc.encode(s, agent_idx=2)
        assert out2.scalar[0].item() == 0.0
        assert out2.scalar[1].item() == 0.0

    def test_pick_phase_scalar(self):
        enc = PositionAwareTaskGridEncoder(_make_cfg())
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(1, 0),),
            actor=0, task_types=(0,), pick_phase=True
        )
        out = enc.encode(s, agent_idx=0)
        assert out.scalar[2].item() == 1.0


class TestPositionAwareVsBlind:
    """Verify the key structural difference: strangers visible here, invisible in Blind."""

    def test_stranger_channel_nonzero(self):
        from orchard.encoding.grid import BlindTaskGridEncoder
        cfg = _make_cfg()
        blind = BlindTaskGridEncoder(cfg)
        posaware = PositionAwareTaskGridEncoder(cfg)
        s = _make_state()

        b_out = blind.encode(s, agent_idx=0)
        p_out = posaware.encode(s, agent_idx=0)

        # Blind has 4 channels, no stranger info anywhere
        assert b_out.grid.shape[0] == 4

        # PositionAware has 5 channels, Ch3 has strangers
        assert p_out.grid.shape[0] == 5
        assert p_out.grid[3].sum().item() > 0  # strangers visible


class TestPositionAwareEncoderBatching:
    def test_encode_batch_matches_loop(self):
        cfg = _make_cfg()
        enc = PositionAwareTaskGridEncoder(cfg)
        env = StochasticEnv(cfg)
        s = _make_state()

        after_states = [env.apply_action(s, Action(a)) for a in range(5)]
        batch = enc.encode_batch_for_actions(s, agent_idx=0, after_states=after_states)

        assert batch.grid.shape == (5, 5, 5, 5)

        for k, s_after in enumerate(after_states):
            single = enc.encode(s_after, agent_idx=0)
            assert torch.allclose(batch.grid[k], single.grid), f"Grid mismatch at action {k}"
            assert torch.allclose(batch.scalar[k], single.scalar), f"Scalar mismatch at action {k}"

    def test_encode_batch_non_actor(self):
        """Batch encode from a non-actor agent's perspective."""
        cfg = _make_cfg()
        enc = PositionAwareTaskGridEncoder(cfg)
        env = StochasticEnv(cfg)
        s = _make_state()  # actor=0

        after_states = [env.apply_action(s, Action(a)) for a in range(5)]
        batch = enc.encode_batch_for_actions(s, agent_idx=2, after_states=after_states)

        for k, s_after in enumerate(after_states):
            single = enc.encode(s_after, agent_idx=2)
            assert torch.allclose(batch.grid[k], single.grid), f"Grid mismatch at action {k} for agent 2"
            assert torch.allclose(batch.scalar[k], single.scalar), f"Scalar mismatch at action {k} for agent 2"

    def test_encode_all_agents(self):
        cfg = _make_cfg()
        enc = PositionAwareTaskGridEncoder(cfg)
        s = _make_state()

        grids, scalars = enc.encode_all_agents(s)
        assert grids.shape == (4, 5, 5, 5)

        for i in range(4):
            single = enc.encode(s, agent_idx=i)
            assert torch.allclose(grids[i], single.grid), f"Grid mismatch for agent {i}"
            assert torch.allclose(scalars[i], single.scalar), f"Scalar mismatch for agent {i}"

    def test_encode_all_agents_for_actions(self):
        cfg = _make_cfg()
        enc = PositionAwareTaskGridEncoder(cfg)
        env = StochasticEnv(cfg)
        s = _make_state()

        after_states = [env.apply_action(s, Action(a)) for a in range(5)]
        grids, scalars = enc.encode_all_agents_for_actions(s, after_states)
        assert grids.shape == (4, 5, 5, 5, 5)  # (N, B, C, H, W)

        for i in range(4):
            for k, s_after in enumerate(after_states):
                single = enc.encode(s_after, agent_idx=i)
                assert torch.allclose(grids[i, k], single.grid), \
                    f"Grid mismatch for agent {i}, action {k}"
                assert torch.allclose(scalars[i, k], single.scalar), \
                    f"Scalar mismatch for agent {i}, action {k}"


class TestPositionAwareStrangerCounting:
    """Test stranger counting with agents on the same cell."""

    def test_strangers_same_cell(self):
        cfg = _make_cfg()
        enc = PositionAwareTaskGridEncoder(cfg)
        # Put two strangers on the same cell
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(2, 2), Grid(2, 2)),
            task_positions=(), actor=0, task_types=(), pick_phase=False
        )
        out = enc.encode(s, agent_idx=0)
        # A2 and A3 both at (2,2), both strangers to A0
        assert out.grid[3, 2, 2].item() == 2.0
