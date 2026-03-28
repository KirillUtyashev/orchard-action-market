"""Tests for new task-specialization encoders (chunk 3)."""

import pytest
import torch

from orchard.enums import Action, EncoderType, EnvType, PickMode, make_pick_action
from orchard.encoding import init_encoder, encode, encode_batch_for_actions, get_scalar_dim, get_grid_channels
from orchard.encoding.grid import TaskGridEncoder, CentralizedTaskGridEncoder
from orchard.datatypes import EnvConfig, Grid, State
from orchard.env.deterministic import DeterministicEnv


def _make_cfg(n_task_types=4, n_agents=4, pick_mode=PickMode.FORCED, **overrides):
    defaults = dict(
        height=5, width=5, n_agents=n_agents, n_tasks=3,
        gamma=0.99, r_picker=1.0,
        n_task_types=n_task_types, r_high=1.0, r_low=0.0,
        task_assignments=tuple((i,) for i in range(n_task_types)) if n_task_types == n_agents else
            tuple(tuple(range(n_task_types)) for _ in range(n_agents)),
        pick_mode=pick_mode,
        max_tasks_per_type=3, max_tasks=12,
        env_type=EnvType.DETERMINISTIC,
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


def _make_state():
    """4 agents, tasks of types 0,1,2 at known positions."""
    return State(
        agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
        task_positions=(Grid(1, 0), Grid(1, 1), Grid(2, 0)),
        actor=0,
        task_types=(0, 1, 2),
    )


# ---------------------------------------------------------------------------
# TaskGridEncoder (decentralized)
# ---------------------------------------------------------------------------
class TestTaskGridEncoder:
    def test_channel_count(self):
        cfg = _make_cfg(n_task_types=4)
        enc = TaskGridEncoder(cfg)
        assert enc.grid_channels() == 7  # T+3 = 4+3
        assert enc.scalar_dim() == 2  # is_actor + phase2_pending

    def test_encode_task_channels(self):
        cfg = _make_cfg(n_task_types=4)
        enc = TaskGridEncoder(cfg)
        s = _make_state()
        out = enc.encode(s, agent_idx=0)

        # Type 0 at (1,0)
        assert out.grid[0, 1, 0].item() == 1.0
        # Type 1 at (1,1)
        assert out.grid[1, 1, 1].item() == 1.0
        # Type 2 at (2,0)
        assert out.grid[2, 2, 0].item() == 1.0
        # Type 3 — no tasks
        assert out.grid[3].sum().item() == 0.0

    def test_encode_self_channel(self):
        cfg = _make_cfg()
        enc = TaskGridEncoder(cfg)
        s = _make_state()
        T = 4

        # Agent 0 at (0,0)
        out0 = enc.encode(s, agent_idx=0)
        assert out0.grid[T, 0, 0].item() == 1.0
        assert out0.grid[T].sum().item() == 1.0

        # Agent 2 at (0,2)
        out2 = enc.encode(s, agent_idx=2)
        assert out2.grid[T, 0, 2].item() == 1.0

    def test_encode_others_channel(self):
        cfg = _make_cfg()
        enc = TaskGridEncoder(cfg)
        s = _make_state()
        T = 4

        out = enc.encode(s, agent_idx=0)
        # Others: agents 1,2,3 at (0,1),(0,2),(0,3)
        assert out.grid[T + 1, 0, 1].item() == 1.0
        assert out.grid[T + 1, 0, 2].item() == 1.0
        assert out.grid[T + 1, 0, 3].item() == 1.0
        # Self position should NOT be in others
        assert out.grid[T + 1, 0, 0].item() == 0.0

    def test_encode_actor_channel(self):
        cfg = _make_cfg()
        enc = TaskGridEncoder(cfg)
        s = _make_state()  # actor=0 at (0,0)
        T = 4

        out = enc.encode(s, agent_idx=1)
        assert out.grid[T + 2, 0, 0].item() == 1.0  # actor at (0,0)

    def test_encode_scalar(self):
        cfg = _make_cfg()
        enc = TaskGridEncoder(cfg)
        s = _make_state()  # actor=0

        out0 = enc.encode(s, agent_idx=0)
        assert out0.scalar[0].item() == 1.0  # is actor

        out1 = enc.encode(s, agent_idx=1)
        assert out1.scalar[0].item() == 0.0  # not actor

    def test_batch_encoding_matches_loop(self):
        """Batch encoding should match individual encode() calls."""
        cfg = _make_cfg()
        enc = TaskGridEncoder(cfg)
        env = DeterministicEnv(cfg)
        s = _make_state()

        # Generate after-states for movement actions
        after_states = [env.apply_action(s, Action(a)) for a in range(5)]

        batch = enc.encode_batch_for_actions(s, agent_idx=0, after_states=after_states)
        for k, s_after in enumerate(after_states):
            single = enc.encode(s_after, agent_idx=0)
            assert torch.allclose(batch.grid[k], single.grid), f"Mismatch at action {k}"

    def test_batch_encoding_with_pick(self):
        """Batch encoding handles pick actions (task removal)."""
        cfg = _make_cfg(pick_mode=PickMode.CHOICE)
        enc = TaskGridEncoder(cfg)
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(1, 0), Grid(1, 1)),
            actor=0,
            task_types=(0, 1),
        )

        # Movement after-states + pick after-states
        after_states = [env.apply_action(s, Action(a)) for a in range(5)]
        # Add pick(0) after-state: removes type 0 at (1,0)
        s_picked, _ = env.resolve_pick(s, pick_type=0)
        after_states.append(s_picked)

        batch = enc.encode_batch_for_actions(s, agent_idx=0, after_states=after_states)

        # Movement after-states should have task at (1,0)
        assert batch.grid[4, 0, 1, 0].item() == 1.0  # STAY: type 0 still there

        # Pick after-state should NOT have task at (1,0)
        assert batch.grid[5, 0, 1, 0].item() == 0.0  # type 0 removed


# ---------------------------------------------------------------------------
# CentralizedTaskGridEncoder
# ---------------------------------------------------------------------------
class TestCentralizedTaskGridEncoder:
    def test_channel_count(self):
        cfg = _make_cfg(n_task_types=4, n_agents=4)
        enc = CentralizedTaskGridEncoder(cfg)
        assert enc.grid_channels() == 9  # T+N+1 = 4+4+1
        assert enc.scalar_dim() == 5     # one-hot of length N + phase2_pending

    def test_encode_task_channels(self):
        cfg = _make_cfg()
        enc = CentralizedTaskGridEncoder(cfg)
        s = _make_state()
        out = enc.encode(s, agent_idx=0)

        assert out.grid[0, 1, 0].item() == 1.0  # type 0
        assert out.grid[1, 1, 1].item() == 1.0  # type 1
        assert out.grid[2, 2, 0].item() == 1.0  # type 2
        assert out.grid[3].sum().item() == 0.0   # type 3 absent

    def test_encode_per_agent_channels(self):
        cfg = _make_cfg()
        enc = CentralizedTaskGridEncoder(cfg)
        s = _make_state()
        T = 4
        out = enc.encode(s, agent_idx=0)

        # Agent 0 at (0,0) → channel T+0
        assert out.grid[T + 0, 0, 0].item() == 1.0
        # Agent 1 at (0,1) → channel T+1
        assert out.grid[T + 1, 0, 1].item() == 1.0
        # Agent 2 at (0,2) → channel T+2
        assert out.grid[T + 2, 0, 2].item() == 1.0
        # Agent 3 at (0,3) → channel T+3
        assert out.grid[T + 3, 0, 3].item() == 1.0

    def test_encode_actor_channel(self):
        cfg = _make_cfg()
        enc = CentralizedTaskGridEncoder(cfg)
        s = _make_state()  # actor=0 at (0,0)
        T, N = 4, 4
        out = enc.encode(s, agent_idx=0)
        assert out.grid[T + N, 0, 0].item() == 1.0

    def test_encode_one_hot_scalar(self):
        cfg = _make_cfg()
        enc = CentralizedTaskGridEncoder(cfg)
        s = _make_state()  # actor=0

        out = enc.encode(s, agent_idx=0)
        assert out.scalar.shape == (5,)
        assert out.scalar[0].item() == 1.0
        assert out.scalar[1].item() == 0.0
        assert out.scalar[2].item() == 0.0
        assert out.scalar[3].item() == 0.0
        assert out.scalar[4].item() == 0.0  # phase2_pending

    def test_one_hot_independent_of_agent_idx(self):
        """One-hot encodes actor, not agent_idx."""
        cfg = _make_cfg()
        enc = CentralizedTaskGridEncoder(cfg)
        s = _make_state()  # actor=0

        # Encoding from agent 2's perspective should still one-hot actor=0
        out = enc.encode(s, agent_idx=2)
        assert out.scalar[0].item() == 1.0

    def test_batch_encoding_matches_loop(self):
        cfg = _make_cfg()
        enc = CentralizedTaskGridEncoder(cfg)
        env = DeterministicEnv(cfg)
        s = _make_state()

        after_states = [env.apply_action(s, Action(a)) for a in range(5)]
        batch = enc.encode_batch_for_actions(s, agent_idx=0, after_states=after_states)

        for k, s_after in enumerate(after_states):
            single = enc.encode(s_after, agent_idx=0)
            assert torch.allclose(batch.grid[k], single.grid), f"Mismatch at action {k}"
            assert torch.allclose(batch.scalar[k], single.scalar), f"Scalar mismatch at action {k}"


# ---------------------------------------------------------------------------
# Channel dimensions for various T and N
# ---------------------------------------------------------------------------
class TestChannelDimensions:
    @pytest.mark.parametrize("T,N", [(2, 2), (4, 4), (8, 4), (4, 7)])
    def test_dec_channels(self, T, N):
        cfg = _make_cfg(n_task_types=T, n_agents=N)
        enc = TaskGridEncoder(cfg)
        assert enc.grid_channels() == T + 3
        assert enc.scalar_dim() == 2  # is_actor + phase2_pending

    @pytest.mark.parametrize("T,N", [(2, 2), (4, 4), (8, 4), (4, 7)])
    def test_cen_channels(self, T, N):
        cfg = _make_cfg(n_task_types=T, n_agents=N)
        enc = CentralizedTaskGridEncoder(cfg)
        assert enc.grid_channels() == T + N + 1
        assert enc.scalar_dim() == N + 1  # one-hot + phase2_pending


# ---------------------------------------------------------------------------
# Global encoder API
# ---------------------------------------------------------------------------
class TestGlobalEncoderAPI:
    def test_init_and_use_task_encoder(self):
        cfg = _make_cfg()
        init_encoder(EncoderType.TASK_CNN_GRID, cfg)
        assert get_grid_channels() == 7  # 4+3
        assert get_scalar_dim() == 2

        out = encode(_make_state(), agent_idx=0)
        assert out.grid.shape == (7, 5, 5)

    def test_init_and_use_centralized_task_encoder(self):
        cfg = _make_cfg()
        init_encoder(EncoderType.CENTRALIZED_TASK_CNN_GRID, cfg)
        assert get_grid_channels() == 9  # 4+4+1
        assert get_scalar_dim() == 5

        out = encode(_make_state(), agent_idx=0)
        assert out.grid.shape == (9, 5, 5)
        assert out.scalar.shape == (5,)

    def test_old_encoder_still_works(self):
        """BasicGridEncoder still works with legacy states."""
        cfg = EnvConfig(
            height=3, width=3, n_agents=2, n_tasks=1,
            gamma=0.9, r_picker=1.0, max_tasks=4,
            env_type=EnvType.DETERMINISTIC,
        )
        init_encoder(EncoderType.CNN_GRID, cfg)
        assert get_grid_channels() == 4
        assert get_scalar_dim() == 1

        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            task_positions=(Grid(1, 0),),
            actor=0,
        )
        out = encode(s, agent_idx=0)
        assert out.grid.shape == (4, 3, 3)
