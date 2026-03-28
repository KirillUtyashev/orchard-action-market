"""Tests for environment: init, step, pick, spawn, boundaries (chunk 2)."""

import pytest
from orchard.enums import Action, EnvType, PickMode, DespawnMode, make_pick_action
from orchard.env import create_env
from orchard.env.deterministic import DeterministicEnv
from orchard.env.stochastic import StochasticEnv
from orchard.datatypes import EnvConfig, Grid, State, StochasticConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_det_cfg(**overrides) -> EnvConfig:
    defaults = dict(
        height=2, width=2, n_agents=2, n_tasks=1,
        gamma=0.9, r_picker=-1.0, pick_mode=PickMode.FORCED,
        max_tasks=1, env_type=EnvType.DETERMINISTIC,
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


def _make_stoch_cfg(**overrides) -> EnvConfig:
    defaults = dict(
        height=9, width=9, n_agents=4, n_tasks=4,
        gamma=0.99, r_picker=1.0, pick_mode=PickMode.FORCED,
        max_tasks=12, env_type=EnvType.STOCHASTIC,
        stochastic=StochasticConfig(
            spawn_prob=0.04, despawn_mode=DespawnMode.PROBABILITY,
            despawn_prob=0.05,
        ),
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


def _make_task_spec_cfg(**overrides) -> EnvConfig:
    """Multi-type config: 4 agents, 4 types, rho=0.25."""
    defaults = dict(
        height=9, width=9, n_agents=4, n_tasks=3,
        gamma=0.99, r_picker=1.0,
        n_task_types=4, r_high=1.0, r_low=0.0,
        task_assignments=((0,), (1,), (2,), (3,)),
        pick_mode=PickMode.FORCED,
        max_tasks_per_type=3, max_tasks=12,
        env_type=EnvType.DETERMINISTIC,
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


# ---------------------------------------------------------------------------
# Legacy (n_task_types=1) tests — same as old tests but with new API
# ---------------------------------------------------------------------------
class TestDeterministicInitLegacy:
    def test_init_state_positions(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = env.init_state()
        assert s.agent_positions == (Grid(0, 0), Grid(0, 1))
        assert s.task_positions == (Grid(1, 0),)
        assert s.actor == 0
        assert s.task_types is None

    def test_init_state_3x3(self):
        cfg = _make_det_cfg(height=3, width=3, n_agents=2, n_tasks=2)
        env = DeterministicEnv(cfg)
        s = env.init_state()
        assert s.agent_positions == (Grid(0, 0), Grid(0, 1))
        assert s.task_positions == (Grid(0, 2), Grid(1, 0))


class TestApplyActionLegacy:
    def test_move_down(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            task_positions=(Grid(1, 0),),
            actor=0,
        )
        s_after = env.apply_action(s, Action.DOWN)
        assert s_after.agent_positions[0] == Grid(1, 0)
        assert s_after.actor == 0
        assert s_after.task_positions == s.task_positions

    def test_boundary_clamp(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            task_positions=(Grid(1, 0),),
            actor=0,
        )
        s_after = env.apply_action(s, Action.UP)
        assert s_after.agent_positions[0] == Grid(0, 0)

    def test_stay(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            task_positions=(Grid(1, 0),),
            actor=0,
        )
        s_after = env.apply_action(s, Action.STAY)
        assert s_after.agent_positions[0] == Grid(0, 0)

    def test_pick_action_no_movement(self):
        """Pick actions should return state unchanged (no movement)."""
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            task_positions=(Grid(1, 0),),
            actor=0,
        )
        s_after = env.apply_action(s, Action.PICK)
        assert s_after is s


class TestResolvePickLegacy:
    def test_no_pick_empty_cell(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            task_positions=(Grid(1, 0),),
            actor=0,
        )
        s_picked, rewards = env.resolve_pick(s)
        assert s_picked is s
        assert rewards == (0.0, 0.0)

    def test_pick_on_task(self):
        cfg = _make_det_cfg(r_picker=-1.0)
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 1)),
            task_positions=(Grid(1, 0),),
            actor=0,
        )
        s_picked, rewards = env.resolve_pick(s)
        assert s_picked.task_positions == ()
        assert rewards[0] == -1.0
        assert rewards[1] == 2.0


class TestSpawnLegacy:
    def test_spawn_after_pick(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 1)),
            task_positions=(),
            actor=0,
        )
        s_spawned = env.spawn_and_despawn(s)
        assert len(s_spawned.task_positions) == 1
        assert s_spawned.task_positions[0] == Grid(0, 0)

    def test_no_spawn_when_full(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            task_positions=(Grid(1, 0),),
            actor=0,
        )
        s_spawned = env.spawn_and_despawn(s)
        assert s_spawned.task_positions == s.task_positions


class TestAdvanceActor:
    def test_round_robin(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            task_positions=(Grid(1, 0),),
            actor=0,
        )
        s2 = env.advance_actor(s)
        assert s2.actor == 1
        s3 = env.advance_actor(s2)
        assert s3.actor == 0


class TestFullStepLegacy:
    def test_step_no_pick(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            task_positions=(Grid(1, 0),),
            actor=0,
        )
        tr = env.step(s, Action.RIGHT)
        assert tr.s_t is s
        assert tr.rewards == (0.0, 0.0)
        assert tr.s_t_next.actor == 1

    def test_step_with_pick(self):
        cfg = _make_det_cfg(r_picker=-1.0)
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            task_positions=(Grid(1, 0),),
            actor=0,
        )
        tr = env.step(s, Action.DOWN)
        assert tr.rewards[0] == -1.0
        assert tr.rewards[1] == 2.0
        assert len(tr.s_t_next.task_positions) == 1
        assert tr.s_t_next.actor == 1


class TestCreateEnv:
    def test_factory_deterministic(self):
        cfg = _make_det_cfg()
        env = create_env(cfg)
        assert isinstance(env, DeterministicEnv)


# ---------------------------------------------------------------------------
# Task specialization (n_task_types > 1)
# ---------------------------------------------------------------------------
class TestInitMultiType:
    def test_deterministic_init_spawns_per_type(self):
        cfg = _make_task_spec_cfg()
        env = DeterministicEnv(cfg)
        s = env.init_state()
        assert s.task_types is not None
        # Should have tasks of each type
        type_counts = {}
        for t in s.task_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        for tau in range(4):
            assert tau in type_counts, f"Type {tau} missing from init"

    def test_stochastic_init_spawns_per_type(self):
        cfg = _make_task_spec_cfg(
            env_type=EnvType.STOCHASTIC,
            stochastic=StochasticConfig(
                spawn_prob=0.04, despawn_mode=DespawnMode.NONE,
                despawn_prob=0.0,
            ),
        )
        env = StochasticEnv(cfg)
        s = env.init_state()
        assert s.task_types is not None
        type_counts = {}
        for t in s.task_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        for tau in range(4):
            assert tau in type_counts


class TestResolvePickMultiType:
    def _make_state(self):
        """Agent 0 at (1,0), task type 0 at (1,0), task type 1 at (2,0)."""
        return State(
            agent_positions=(Grid(1, 0), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0), Grid(2, 0)),
            actor=0,
            task_types=(0, 1),
        )

    def test_forced_pick_correct_type(self):
        """Agent 0 picks type 0 (its assigned type) → R_high=1.0."""
        cfg = _make_task_spec_cfg(r_high=1.0, r_low=0.0)
        env = DeterministicEnv(cfg)
        s = self._make_state()
        s_picked, rewards = env.resolve_pick(s)
        assert rewards[0] == 1.0  # actor gets R_high
        assert rewards[1] == 0.0  # others get 0
        assert rewards[2] == 0.0
        assert rewards[3] == 0.0
        # Task removed
        assert Grid(1, 0) not in s_picked.task_positions

    def test_forced_pick_wrong_type(self):
        """Agent 1 picks type 0 (not its assigned type 1) → R_low."""
        cfg = _make_task_spec_cfg(r_high=1.0, r_low=-1.0)
        env = DeterministicEnv(cfg)
        # Agent 1 at (1,0), task type 0 at (1,0)
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0), Grid(2, 0)),
            actor=1,
            task_types=(0, 1),
        )
        s_picked, rewards = env.resolve_pick(s)
        assert rewards[1] == -1.0  # actor gets R_low
        assert rewards[0] == 0.0   # others get 0

    def test_forced_pick_no_task(self):
        """Actor not on any task → no pick."""
        cfg = _make_task_spec_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(3, 3), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(0,),
        )
        s_picked, rewards = env.resolve_pick(s)
        assert s_picked is s
        assert all(r == 0.0 for r in rewards)

    def test_forced_pick_others_always_zero(self):
        """Non-actor agents always get 0 reward."""
        cfg = _make_task_spec_cfg(r_high=1.0, r_low=-1.0)
        env = DeterministicEnv(cfg)
        s = self._make_state()
        _, rewards = env.resolve_pick(s)
        for i in range(1, 4):
            assert rewards[i] == 0.0


class TestChoicePick:
    def _make_cfg(self, **overrides):
        defaults = dict(
            height=5, width=5, n_agents=4, n_tasks=3,
            gamma=0.99, r_picker=1.0,
            n_task_types=4, r_high=1.0, r_low=0.0,
            task_assignments=((0,), (1,), (2,), (3,)),
            pick_mode=PickMode.CHOICE,
            max_tasks_per_type=3, max_tasks=12,
            env_type=EnvType.DETERMINISTIC,
        )
        defaults.update(overrides)
        return EnvConfig(**defaults)

    def test_choice_pick_correct_type(self):
        cfg = self._make_cfg()
        env = DeterministicEnv(cfg)
        # Agent 0 at (1,1), type 0 task at (1,1)
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 1),),
            actor=0,
            task_types=(0,),
        )
        s_picked, rewards = env.resolve_pick(s, pick_type=0)
        assert rewards[0] == 1.0
        assert s_picked.task_positions == ()

    def test_choice_pick_wrong_type_at_cell(self):
        """Pick type that doesn't exist at actor's cell → wasted action."""
        cfg = self._make_cfg()
        env = DeterministicEnv(cfg)
        # Agent 0 at (1,1), type 0 task at (1,1), but agent tries to pick type 1
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 1),),
            actor=0,
            task_types=(0,),
        )
        s_picked, rewards = env.resolve_pick(s, pick_type=1)
        assert s_picked is s  # no change
        assert all(r == 0.0 for r in rewards)

    def test_choice_pick_multiple_types_at_cell(self):
        """Multiple types at same cell — only pick the specified type."""
        cfg = self._make_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 1), Grid(1, 1), Grid(2, 2)),
            actor=0,
            task_types=(0, 2, 1),
        )
        s_picked, rewards = env.resolve_pick(s, pick_type=2)
        assert rewards[0] == 0.0  # type 2 ∉ G_0={0}, so R_low=0.0
        # Only type 2 at (1,1) removed, type 0 at (1,1) remains
        assert len(s_picked.task_positions) == 2
        remaining_types = set(s_picked.task_types)
        assert 0 in remaining_types  # type 0 still there
        assert 2 not in s_picked.task_types or \
            sum(1 for t in s_picked.task_types if t == 2) == 0  # type 2 gone

    def test_choice_movement_no_autopick(self):
        """Movement in choice mode never auto-picks."""
        cfg = self._make_cfg()
        env = DeterministicEnv(cfg)
        # Agent 0 moves onto a task cell — should NOT auto-pick
        s = State(
            agent_positions=(Grid(0, 0), Grid(2, 2), Grid(3, 3), Grid(4, 4)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(0,),
        )
        tr = env.step(s, Action.DOWN)
        # Agent moved to (1,0) where task is, but no pick in choice mode
        assert tr.rewards == (0.0, 0.0, 0.0, 0.0)
        # Task should still be there (after spawn/despawn, but no despawn in det)
        assert Grid(1, 0) in tr.s_t_after.task_positions

    def test_step_with_pick_action(self):
        """Phase-2 pick via resolve_pick in choice mode."""
        cfg = self._make_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 1),),
            actor=0,
            task_types=(0,),
        )
        # Phase 2: resolve pick directly (step() is move-only now)
        s_picked, rewards = env.resolve_pick(s, pick_type=0)
        assert rewards[0] == 1.0  # correct pick
        assert Grid(1, 1) not in s_picked.task_positions


# ---------------------------------------------------------------------------
# Spawn with multiple types
# ---------------------------------------------------------------------------
class TestSpawnMultiType:
    def test_despawn_maintains_parallel_arrays(self):
        """After despawn, task_positions and task_types stay parallel."""
        cfg = _make_task_spec_cfg(
            env_type=EnvType.STOCHASTIC,
            stochastic=StochasticConfig(
                spawn_prob=0.0,  # no spawning
                despawn_mode=DespawnMode.PROBABILITY,
                despawn_prob=1.0,  # despawn everything
            ),
        )
        env = StochasticEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(1, 0), Grid(1, 1), Grid(2, 0)),
            actor=0,
            task_types=(0, 1, 2),
        )
        s2 = env.spawn_and_despawn(s)
        assert len(s2.task_positions) == 0
        assert len(s2.task_types) == 0

    def test_spawn_respects_max_per_type(self):
        """Spawning should not exceed max_tasks_per_type for any type."""
        cfg = _make_task_spec_cfg(
            max_tasks_per_type=2,
            env_type=EnvType.STOCHASTIC,
            stochastic=StochasticConfig(
                spawn_prob=1.0,  # always spawn
                despawn_mode=DespawnMode.NONE,
                despawn_prob=0.0,
            ),
        )
        env = StochasticEnv(cfg)
        # Start with 2 of type 0 (at cap), 0 of type 1-3
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(0, 3)),
            task_positions=(Grid(1, 0), Grid(1, 1)),
            actor=0,
            task_types=(0, 0),
        )
        s2 = env.spawn_and_despawn(s)
        # Type 0 should not have grown past 2
        type_0_count = sum(1 for t in s2.task_types if t == 0)
        assert type_0_count <= 2


class TestCentralizedRewardEquivalence:
    """Centralized reward = sum of decentralized = actor's reward."""
    def test_cen_equals_dec_sum(self):
        cfg = _make_task_spec_cfg(r_high=1.0, r_low=-1.0)
        env = DeterministicEnv(cfg)
        # Agent 0 picks correct type
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(0,),
        )
        _, rewards = env.resolve_pick(s)
        # Centralized = sum of dec = actor's reward (others are 0)
        assert sum(rewards) == rewards[0]
        assert rewards[0] == 1.0

    def test_cen_equals_dec_sum_wrong_pick(self):
        cfg = _make_task_spec_cfg(r_high=1.0, r_low=-1.0)
        env = DeterministicEnv(cfg)
        # Agent 0 picks wrong type (type 1, agent 0 owns type 0)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(1,),
        )
        _, rewards = env.resolve_pick(s)
        assert sum(rewards) == rewards[0]
        assert rewards[0] == -1.0
