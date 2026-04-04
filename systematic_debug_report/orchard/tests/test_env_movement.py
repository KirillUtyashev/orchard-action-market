"""Tests for environment initialization, movement, and actor advancement."""

import pytest
from orchard.enums import Action, PickMode, DespawnMode
from orchard.datatypes import EnvConfig, StochasticConfig, Grid, State
from orchard.env.stochastic import StochasticEnv
from orchard.seed import set_all_seeds

def _make_move_cfg() -> EnvConfig:
    return EnvConfig(
        height=3, width=3, n_agents=2, n_tasks=1, gamma=0.99, r_picker=1.0,
        n_task_types=1, pick_mode=PickMode.FORCED, max_tasks_per_type=1,
        task_assignments=((0,), (0,)),
        stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=DespawnMode.NONE, despawn_prob=0.0)
    )

class TestInitialization:
    def test_init_state_no_overlap(self):
        set_all_seeds(42)
        cfg = _make_move_cfg()
        env = StochasticEnv(cfg)
        s = env.init_state()
        
        # Agents and tasks shouldn't overlap on init
        for t_pos in s.task_positions:
            assert t_pos not in s.agent_positions

        assert s.actor == 0
        assert s.n_agents == cfg.n_agents
        assert len(s.task_positions) == cfg.n_tasks
        assert s.task_types is not None

class TestApplyAction:
    def test_movement_directions(self):
        cfg = _make_move_cfg()
        env = StochasticEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0)),
            task_positions=(), actor=0, task_types=()
        )
        
        # UP: row - 1
        s_up = env.apply_action(s, Action.UP)
        assert s_up.agent_positions[0] == Grid(0, 1)

        # DOWN: row + 1
        s_down = env.apply_action(s, Action.DOWN)
        assert s_down.agent_positions[0] == Grid(2, 1)

        # LEFT: col - 1
        s_left = env.apply_action(s, Action.LEFT)
        assert s_left.agent_positions[0] == Grid(1, 0)

        # RIGHT: col + 1
        s_right = env.apply_action(s, Action.RIGHT)
        assert s_right.agent_positions[0] == Grid(1, 2)

        # STAY: no change
        s_stay = env.apply_action(s, Action.STAY)
        assert s_stay.agent_positions[0] == Grid(1, 1)

    def test_boundary_clamping(self):
        cfg = _make_move_cfg()  # 3x3 grid
        env = StochasticEnv(cfg)
        
        # Top-left corner (0,0)
        s_tl = State(agent_positions=(Grid(0, 0),), task_positions=(), actor=0, task_types=())
        assert env.apply_action(s_tl, Action.UP).agent_positions[0] == Grid(0, 0)
        assert env.apply_action(s_tl, Action.LEFT).agent_positions[0] == Grid(0, 0)

        # Bottom-right corner (2,2)
        s_br = State(agent_positions=(Grid(2, 2),), task_positions=(), actor=0, task_types=())
        assert env.apply_action(s_br, Action.DOWN).agent_positions[0] == Grid(2, 2)
        assert env.apply_action(s_br, Action.RIGHT).agent_positions[0] == Grid(2, 2)

    def test_pick_action_causes_no_movement(self):
        cfg = _make_move_cfg()
        env = StochasticEnv(cfg)
        s = State(agent_positions=(Grid(1, 1),), task_positions=(), actor=0, task_types=())
        s_after = env.apply_action(s, Action.PICK)
        assert s_after is s  # Should return exact same state object

class TestAdvanceActor:
    def test_round_robin(self):
        cfg = _make_move_cfg() # 2 agents
        env = StochasticEnv(cfg)
        s = State(agent_positions=(Grid(0,0), Grid(1,1)), task_positions=(), actor=0, task_types=())
        
        s1 = env.advance_actor(s)
        assert s1.actor == 1
        
        s2 = env.advance_actor(s1)
        assert s2.actor == 0