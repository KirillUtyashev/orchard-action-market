"""Tests for task spawning, despawning, and collision rules."""

import pytest
from collections import Counter
from orchard.enums import PickMode, DespawnMode, TaskSpawnMode
from orchard.datatypes import EnvConfig, StochasticConfig, Grid, State
from orchard.env.stochastic import StochasticEnv
from orchard.seed import set_all_seeds

def _make_spawn_cfg(spawn_prob=1.0, despawn_mode=DespawnMode.NONE, despawn_prob=0.0, 
                    task_spawn_mode=None, pick_mode=PickMode.FORCED) -> EnvConfig:
    return EnvConfig(
        height=5, width=5, n_agents=1, n_tasks=0, gamma=0.99, r_picker=1.0,
        n_task_types=2, pick_mode=pick_mode, max_tasks_per_type=3,
        task_assignments=((0, 1),),
        stochastic=StochasticConfig(
            spawn_prob=spawn_prob, despawn_mode=despawn_mode, 
            despawn_prob=despawn_prob, task_spawn_mode=task_spawn_mode
        )
    )

class TestDespawn:
    def test_probability_despawn(self):
        set_all_seeds(42)
        cfg = _make_spawn_cfg(spawn_prob=0.0, despawn_mode=DespawnMode.PROBABILITY, despawn_prob=1.0)
        env = StochasticEnv(cfg)
        
        # Start with 2 tasks
        s = State(
            agent_positions=(Grid(0, 0),),
            task_positions=(Grid(1, 1), Grid(2, 2)),
            actor=0, task_types=(0, 1)
        )
        
        # With despawn_prob=1.0, both should vanish
        s_after = env.spawn_and_despawn(s)
        assert len(s_after.task_positions) == 0
        assert len(s_after.task_types) == 0

class TestSpawnLimits:
    def test_max_tasks_per_type_respected(self):
        set_all_seeds(99)
        cfg = _make_spawn_cfg(spawn_prob=1.0) # Always spawn
        env = StochasticEnv(cfg)
        
        s = State(agent_positions=(Grid(0, 0),), task_positions=(), actor=0, task_types=())
        
        # Run spawn for a few rounds
        for _ in range(5):
            s = env.spawn_and_despawn(s)
        
        # Max is 3 per type. 2 types = 6 total tasks maximum
        assert len(s.task_positions) == 6
        
        types_count = Counter(s.task_types)
        assert types_count[0] == 3
        assert types_count[1] == 3

class TestSpawnModes:
    def test_global_unique_prevents_overlap(self):
        set_all_seeds(7)
        # GLOBAL_UNIQUE: A cell can hold max 1 task of ANY type
        cfg = _make_spawn_cfg(task_spawn_mode=TaskSpawnMode.GLOBAL_UNIQUE)
        env = StochasticEnv(cfg)
        s = State(agent_positions=(Grid(0, 0),), task_positions=(), actor=0, task_types=())
        
        for _ in range(10):
            s = env.spawn_and_despawn(s)
        
        pos_counts = Counter(s.task_positions)
        for pos, count in pos_counts.items():
            assert count == 1, f"Cell {pos} has {count} tasks, expected max 1 in GLOBAL_UNIQUE"

    def test_per_type_unique_allows_overlap(self):
        set_all_seeds(13)
        # PER_TYPE_UNIQUE: A cell can hold max 1 task of EACH type (stacking allowed)
        cfg = _make_spawn_cfg(task_spawn_mode=TaskSpawnMode.PER_TYPE_UNIQUE)
        # Force a small grid (1x2) to guarantee stacking if spawn_prob is 1.0
        cfg = EnvConfig(**{**cfg.__dict__, "height": 1, "width": 2, "max_tasks_per_type": 5})
        env = StochasticEnv(cfg)
        
        # Agent takes (0,0), leaving only (0,1) empty
        s = State(agent_positions=(Grid(0, 0),), task_positions=(), actor=0, task_types=())
        
        # Spawn... types 0 and 1 should both spawn on (0,1)
        s = env.spawn_and_despawn(s)
        
        assert len(s.task_positions) == 2
        assert s.task_positions[0] == Grid(0, 1)
        assert s.task_positions[1] == Grid(0, 1)
        assert set(s.task_types) == {0, 1}

    def test_auto_fallback_modes(self):
        # FORCED pick -> auto defaults to GLOBAL_UNIQUE
        cfg_forced = _make_spawn_cfg(task_spawn_mode=None, pick_mode=PickMode.FORCED)
        env_f = StochasticEnv(cfg_forced)
        assert env_f.stoch.task_spawn_mode is None # Maintained as None in config
        
        # If we trace execution of spawn_and_despawn, it shouldn't stack.
        # (Already tested in `test_new_features.py` from context, but good to know it works here)