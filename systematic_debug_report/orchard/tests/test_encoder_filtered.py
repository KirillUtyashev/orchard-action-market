"""Tests for FilteredTaskGridEncoder (O(1) decentralized, 6 channels)."""

import pytest
import torch
from orchard.enums import Action, PickMode
from orchard.datatypes import EnvConfig, StochasticConfig, Grid, State
from orchard.env.stochastic import StochasticEnv
from orchard.encoding.grid import FilteredTaskGridEncoder

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


def _make_choice_cfg() -> EnvConfig:
    return EnvConfig(
        height=5, width=5, n_agents=4, n_tasks=3, gamma=0.99, r_picker=1.0,
        n_task_types=2, pick_mode=PickMode.CHOICE, max_tasks_per_type=3,
        task_assignments=((0,), (0,), (1,), (1,)),
        stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=None, despawn_prob=0.0)
    )


def _make_choice_pick_state() -> State:
    return State(
        agent_positions=(Grid(2, 2), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
        task_positions=(Grid(2, 2), Grid(2, 2), Grid(1, 1)),
        actor=0,
        task_types=(0, 1, 1),
        pick_phase=True,
    )

class TestFilteredEncoderChannels:
    def test_dimensions(self):
        cfg = _make_cfg()
        enc = FilteredTaskGridEncoder(cfg)
        assert enc.grid_channels() == 6
        assert enc.scalar_dim() == 3

    def test_task_filtering(self):
        enc = FilteredTaskGridEncoder(_make_cfg())
        out = enc.encode(_make_state(), agent_idx=0)
        
        # Ch0: My Tasks. Agent 0 owns Type 0 at (1,0).
        assert out.grid[0, 1, 0].item() == 1.0
        assert out.grid[0, 1, 1].item() == 0.0
        
        # Ch1: Irrelevant Tasks. Type 1 at (1,1).
        assert out.grid[1, 1, 1].item() == 1.0
        assert out.grid[1, 1, 0].item() == 0.0

    def test_agent_filtering(self):
        enc = FilteredTaskGridEncoder(_make_cfg())
        out = enc.encode(_make_state(), agent_idx=0)
        
        # Ch2: Self. A0 at (0,0)
        assert out.grid[2, 0, 0].item() == 1.0
        
        # Ch3: Teammates. A1 at (0,1)
        assert out.grid[3, 0, 1].item() == 1.0
        assert out.grid[3, 0, 2].item() == 0.0
        
        # Ch4: Strangers. A2 at (0,2), A3 at (0,3)
        assert out.grid[4, 0, 2].item() == 1.0
        assert out.grid[4, 0, 3].item() == 1.0
        assert out.grid[4, 0, 1].item() == 0.0

    def test_actor_and_scalars(self):
        enc = FilteredTaskGridEncoder(_make_cfg())
        s = _make_state() # Actor is 0
        
        # From A2 (stranger to actor 0)
        out2 = enc.encode(s, agent_idx=2)
        assert out2.grid[5, 0, 0].item() == 1.0 # Actor channel
        assert out2.scalar[0].item() == 0.0 # is_self_actor
        assert out2.scalar[1].item() == 0.0 # is_teammate_actor

class TestFilteredEncoderBatching:
    def test_encode_batch_for_actions(self):
        cfg = _make_cfg()
        enc = FilteredTaskGridEncoder(cfg)
        env = StochasticEnv(cfg)
        s = _make_state()
        
        after_states = [env.apply_action(s, Action(a)) for a in range(5)]
        batch = enc.encode_batch_for_actions(s, agent_idx=0, after_states=after_states)
        
        assert batch.grid.shape == (5, 6, 5, 5)
        
        for k, s_after in enumerate(after_states):
            single = enc.encode(s_after, agent_idx=0)
            assert torch.allclose(batch.grid[k], single.grid)
            assert torch.allclose(batch.scalar[k], single.scalar)

    def test_encode_all_agents(self):
        cfg = _make_cfg()
        enc = FilteredTaskGridEncoder(cfg)
        s = _make_state()
        
        grids, scalars = enc.encode_all_agents(s)
        assert grids.shape == (4, 6, 5, 5)
        
        for i in range(4):
            single = enc.encode(s, agent_idx=i)
            assert torch.allclose(grids[i], single.grid)
            assert torch.allclose(scalars[i], single.scalar)

    def test_encode_all_agents_for_actions_move_phase_matches_single_encode(self):
        cfg = _make_cfg()
        enc = FilteredTaskGridEncoder(cfg)
        env = StochasticEnv(cfg)
        s = _make_state()

        after_states = [env.apply_action(s, Action(a)) for a in range(5)]
        grids, scalars = enc.encode_all_agents_for_actions(s, after_states)

        assert grids.shape == (4, 5, 6, 5, 5)
        assert scalars.shape == (4, 5, 3)

        for i in range(4):
            for k, s_after in enumerate(after_states):
                single = enc.encode(s_after, agent_idx=i)
                assert torch.equal(grids[i, k], single.grid), f"Grid mismatch for agent {i}, action {k}"
                assert torch.equal(scalars[i, k], single.scalar), f"Scalar mismatch for agent {i}, action {k}"

    def test_encode_all_agents_for_actions_pick_phase_matches_single_encode(self):
        cfg = _make_choice_cfg()
        enc = FilteredTaskGridEncoder(cfg)
        env = StochasticEnv(cfg)
        s = _make_choice_pick_state()

        after_pick_type0, _ = env.resolve_pick(s, pick_type=0)
        after_pick_type1, _ = env.resolve_pick(s, pick_type=1)
        after_states = [after_pick_type0, after_pick_type1, s]
        grids, scalars = enc.encode_all_agents_for_actions(s, after_states)

        assert grids.shape == (4, 3, 6, 5, 5)
        assert scalars.shape == (4, 3, 3)

        for i in range(4):
            for k, s_after in enumerate(after_states):
                single = enc.encode(s_after, agent_idx=i)
                assert torch.equal(grids[i, k], single.grid), f"Grid mismatch for agent {i}, action {k}"
                assert torch.equal(scalars[i, k], single.scalar), f"Scalar mismatch for agent {i}, action {k}"

    def test_encode_all_agents_for_actions_overlapping_tasks_stay_binary(self):
        cfg = _make_choice_cfg()
        enc = FilteredTaskGridEncoder(cfg)
        env = StochasticEnv(cfg)
        s = _make_choice_pick_state()

        after_pick_type0, _ = env.resolve_pick(s, pick_type=0)
        after_pick_type1, _ = env.resolve_pick(s, pick_type=1)
        grids, _ = enc.encode_all_agents_for_actions(s, [s, after_pick_type0, after_pick_type1])

        assert grids[:, :, 0].amax().item() == 1.0
        assert grids[:, :, 1].amax().item() == 1.0

        for i, s_after in enumerate([s, after_pick_type0, after_pick_type1]):
            single_agent0 = enc.encode(s_after, agent_idx=0)
            assert torch.equal(grids[0, i, 0], single_agent0.grid[0])
            assert torch.equal(grids[0, i, 1], single_agent0.grid[1])

    def test_encode_all_agents_for_actions_empty_batch(self):
        cfg = _make_cfg()
        enc = FilteredTaskGridEncoder(cfg)
        s = _make_state()

        grids, scalars = enc.encode_all_agents_for_actions(s, [])
        assert grids.shape == (4, 0, 6, 5, 5)
        assert scalars.shape == (4, 0, 3)
