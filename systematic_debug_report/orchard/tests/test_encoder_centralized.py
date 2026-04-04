"""Tests for CentralizedTaskGridEncoder (Global viewpoint, T+N+1 channels)."""

import pytest
import torch
from orchard.enums import Action, PickMode
from orchard.datatypes import EnvConfig, StochasticConfig, Grid, State
from orchard.env.stochastic import StochasticEnv
from orchard.encoding.grid import CentralizedTaskGridEncoder

def _make_cfg() -> EnvConfig:
    # 4 agents, 2 task types.
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
        actor=1,  # Note: Actor is 1
        task_types=(0, 1), pick_phase=True
    )

class TestCentralizedEncoderChannels:
    def test_dimensions(self):
        cfg = _make_cfg() # T=2, N=4
        enc = CentralizedTaskGridEncoder(cfg)
        # Channels: T + N + 1 = 2 + 4 + 1 = 7
        assert enc.grid_channels() == 7
        # Scalars: N + 1 = 4 + 1 = 5
        assert enc.scalar_dim() == 5

    def test_task_and_agent_channels(self):
        enc = CentralizedTaskGridEncoder(_make_cfg())
        out = enc.encode(_make_state(), agent_idx=0)
        
        # T=2 Channels for tasks
        assert out.grid[0, 1, 0].item() == 1.0 # Type 0
        assert out.grid[1, 1, 1].item() == 1.0 # Type 1
        
        # N=4 Channels for agents (Channels 2,3,4,5)
        assert out.grid[2, 0, 0].item() == 1.0 # Agent 0
        assert out.grid[3, 0, 1].item() == 1.0 # Agent 1
        assert out.grid[4, 0, 2].item() == 1.0 # Agent 2
        assert out.grid[5, 0, 3].item() == 1.0 # Agent 3

    def test_actor_and_scalars(self):
        enc = CentralizedTaskGridEncoder(_make_cfg())
        s = _make_state() # Actor is 1, pick_phase=True
        
        out = enc.encode(s, agent_idx=0)
        
        # Actor channel (Channel 6)
        assert out.grid[6, 0, 1].item() == 1.0
        
        # Scalars: One-hot for actor 1 + pick_phase
        assert out.scalar[0].item() == 0.0
        assert out.scalar[1].item() == 1.0 # Actor 1
        assert out.scalar[2].item() == 0.0
        assert out.scalar[3].item() == 0.0
        assert out.scalar[4].item() == 1.0 # pick_phase

    def test_independent_of_agent_idx(self):
        """Centralized encoder output should be identical regardless of agent_idx requested."""
        enc = CentralizedTaskGridEncoder(_make_cfg())
        s = _make_state()
        
        out0 = enc.encode(s, agent_idx=0)
        out2 = enc.encode(s, agent_idx=2)
        
        assert torch.allclose(out0.grid, out2.grid)
        assert torch.allclose(out0.scalar, out2.scalar)

class TestCentralizedEncoderBatching:
    def test_encode_batch_for_actions(self):
        cfg = _make_cfg()
        enc = CentralizedTaskGridEncoder(cfg)
        env = StochasticEnv(cfg)
        s = _make_state()
        
        after_states = [env.apply_action(s, Action(a)) for a in range(5)]
        batch = enc.encode_batch_for_actions(s, agent_idx=0, after_states=after_states)
        
        assert batch.grid.shape == (5, 7, 5, 5)
        
        for k, s_after in enumerate(after_states):
            single = enc.encode(s_after, agent_idx=0)
            assert torch.allclose(batch.grid[k], single.grid)
            assert torch.allclose(batch.scalar[k], single.scalar)

    def test_encode_all_agents(self):
        """Centralized encoder's all_agents wraps the single representation with an N=1 dim."""
        cfg = _make_cfg()
        enc = CentralizedTaskGridEncoder(cfg)
        s = _make_state()
        
        grids, scalars = enc.encode_all_agents(s)
        
        assert grids.shape == (1, 7, 5, 5)
        assert scalars.shape == (1, 5)
        
        single = enc.encode(s, agent_idx=0)
        assert torch.allclose(grids[0], single.grid)
        assert torch.allclose(scalars[0], single.scalar)