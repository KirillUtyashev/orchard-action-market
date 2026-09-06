"""Tests for task spawning, despawning, and spawn-on-agent-cell rules."""

import pytest
from collections import Counter
from orchard.enums import DespawnMode
from orchard.datatypes import EnvConfig, StochasticConfig, Grid, State
from orchard.env.stochastic import StochasticEnv
from orchard.seed import set_all_seeds

def _make_spawn_cfg(spawn_prob=1.0, despawn_mode=DespawnMode.NONE, despawn_prob=0.0,
                    spawn_on_agent_cells=False, spawn_at_round_end=False) -> EnvConfig:
    return EnvConfig(
        height=5, width=5, n_agents=1, n_tasks=0, gamma=0.99,
        n_task_types=2, max_tasks_per_type=3,
        stochastic=StochasticConfig(
            spawn_prob=spawn_prob, despawn_mode=despawn_mode,
            despawn_prob=despawn_prob,
            spawn_on_agent_cells=spawn_on_agent_cells,
            spawn_at_round_end=spawn_at_round_end,
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
        cfg = _make_spawn_cfg(spawn_prob=1.0)  # Always spawn
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

class TestSpawnOnAgentCells:
    def test_agent_cell_blocked_by_default(self):
        """Default (spawn_on_agent_cells=False): tasks never spawn on agent cells."""
        set_all_seeds(0)
        cfg = _make_spawn_cfg(spawn_prob=1.0, spawn_on_agent_cells=False)
        env = StochasticEnv(cfg)
        agent_pos = Grid(2, 2)
        s = State(agent_positions=(agent_pos,), task_positions=(), actor=0, task_types=())
        for _ in range(20):
            s = env.spawn_and_despawn(s)
        assert agent_pos not in s.task_positions, \
            "Agent cell must never receive a task when spawn_on_agent_cells=False"

    def test_agent_cell_reachable_when_enabled(self):
        """spawn_on_agent_cells=True: agent cell is eligible; fills under spawn_prob=1.0."""
        set_all_seeds(0)
        # 1x2 grid: agent at (0,0), max_tasks=2 per type so both cells can hold tasks
        cfg = EnvConfig(
            height=1, width=2, n_agents=1, n_tasks=0, gamma=0.99,
            n_task_types=1, max_tasks_per_type=2,
            stochastic=StochasticConfig(
                spawn_prob=1.0, despawn_mode=DespawnMode.NONE, despawn_prob=0.0,
                spawn_on_agent_cells=True,
            ),
        )
        env = StochasticEnv(cfg)
        s = State(agent_positions=(Grid(0, 0),), task_positions=(), actor=0, task_types=())
        s = env.spawn_and_despawn(s)
        # Both cells (0,0) and (0,1) should have type-0 tasks since spawn_prob=1.0
        assert Grid(0, 0) in s.task_positions, \
            "Agent cell must be eligible for spawning when spawn_on_agent_cells=True"
        assert len(s.task_positions) == 2

    def test_spawn_distribution_is_uniform(self):
        """rng.shuffle ensures no row-major bias: bottom rows get tasks too."""
        set_all_seeds(42)
        # 10x1 grid (10 rows, 1 col), 1 agent at (0,0), max_tasks=1 per type.
        # Without shuffle, the single task would always land at row 0 (first empty cell).
        # With shuffle it should be distributed across all rows over many cycles.
        cfg = EnvConfig(
            height=10, width=1, n_agents=1, n_tasks=0, gamma=0.99,
            n_task_types=1, max_tasks_per_type=1,
            stochastic=StochasticConfig(
                spawn_prob=1.0, despawn_mode=DespawnMode.PROBABILITY, despawn_prob=1.0,
                spawn_on_agent_cells=False,
            ),
        )
        env = StochasticEnv(cfg)
        s = State(agent_positions=(Grid(0, 0),), task_positions=(), actor=0, task_types=())
        row_hits = Counter()
        n_cycles = 500
        for _ in range(n_cycles):
            s = env.spawn_and_despawn(s)  # despawn_prob=1.0 clears, then respawn 1 task
            for pos in s.task_positions:
                row_hits[pos.row] += 1
        # Every row except row 0 (agent) should receive at least some tasks.
        # With uniform distribution across 9 eligible rows, each should appear ~500/9 ≈ 55 times.
        for row in range(1, 10):
            assert row_hits[row] > 10, \
                f"Row {row} only got {row_hits[row]} tasks in {n_cycles} cycles — possible bias"


class TestSpawnAtRoundEnd:
    def test_noop_for_non_last_actor(self):
        """spawn_at_round_end=True: no state change when actor != n_agents-1."""
        set_all_seeds(0)
        # 2 agents, despawn_prob=1.0 so any fire clears all tasks
        cfg = EnvConfig(
            height=5, width=5, n_agents=2, n_tasks=0, gamma=0.99,
            n_task_types=1, max_tasks_per_type=3,
            stochastic=StochasticConfig(
                spawn_prob=0.0, despawn_mode=DespawnMode.PROBABILITY, despawn_prob=1.0,
                spawn_at_round_end=True,
            ),
        )
        env = StochasticEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1)),
            task_positions=(Grid(2, 2), Grid(3, 3)),
            actor=0,  # NOT the last actor (n_agents-1 = 1)
            task_types=(0, 0),
        )
        s_after = env.spawn_and_despawn(s)
        # Tasks must be unchanged since actor=0 != n_agents-1=1
        assert s_after.task_positions == s.task_positions
        assert s_after.task_types == s.task_types

    def test_fires_for_last_actor(self):
        """spawn_at_round_end=True: spawn/despawn fires normally when actor == n_agents-1."""
        set_all_seeds(0)
        cfg = EnvConfig(
            height=5, width=5, n_agents=2, n_tasks=0, gamma=0.99,
            n_task_types=1, max_tasks_per_type=3,
            stochastic=StochasticConfig(
                spawn_prob=0.0, despawn_mode=DespawnMode.PROBABILITY, despawn_prob=1.0,
                spawn_at_round_end=True,
            ),
        )
        env = StochasticEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1)),
            task_positions=(Grid(2, 2), Grid(3, 3)),
            actor=1,  # last actor (n_agents-1 = 1)
            task_types=(0, 0),
        )
        s_after = env.spawn_and_despawn(s)
        # despawn_prob=1.0 should clear all tasks
        assert len(s_after.task_positions) == 0

    def test_default_false_fires_every_step(self):
        """Default (spawn_at_round_end=False): despawn fires regardless of actor."""
        set_all_seeds(0)
        cfg = EnvConfig(
            height=5, width=5, n_agents=2, n_tasks=0, gamma=0.99,
            n_task_types=1, max_tasks_per_type=3,
            stochastic=StochasticConfig(
                spawn_prob=0.0, despawn_mode=DespawnMode.PROBABILITY, despawn_prob=1.0,
                spawn_at_round_end=False,
            ),
        )
        env = StochasticEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1)),
            task_positions=(Grid(2, 2), Grid(3, 3)),
            actor=0,  # not the last actor, but should still fire
            task_types=(0, 0),
        )
        s_after = env.spawn_and_despawn(s)
        assert len(s_after.task_positions) == 0
