"""Tests for environment: init, step, pick, spawn, boundaries."""

import pytest
from orchard.enums import Action, EnvType
from orchard.env import create_env
from orchard.env.deterministic import DeterministicEnv
from orchard.datatypes import EnvConfig, Grid, State


def _make_det_cfg(**overrides) -> EnvConfig:
    defaults = dict(
        height=2, width=2, n_agents=2, n_apples=1,
        gamma=0.9, r_picker=-1.0, force_pick=True,
        max_apples=1, env_type=EnvType.DETERMINISTIC,
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


class TestDeterministicInit:
    def test_init_state_positions(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = env.init_state()
        # 2x2: agents at (0,0),(0,1), apple at (1,0)
        assert s.agent_positions == (Grid(0, 0), Grid(0, 1))
        assert s.apple_positions == (Grid(1, 0),)
        assert s.actor == 0

    def test_init_state_3x3(self):
        cfg = _make_det_cfg(height=3, width=3, n_agents=2, n_apples=2)
        env = DeterministicEnv(cfg)
        s = env.init_state()
        assert s.agent_positions == (Grid(0, 0), Grid(0, 1))
        assert s.apple_positions == (Grid(0, 2), Grid(1, 0))


class TestApplyAction:
    def test_move_down(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0),),
            actor=0,
        )
        s_after = env.apply_action(s, Action.DOWN)
        assert s_after.agent_positions[0] == Grid(1, 0)
        assert s_after.actor == 0  # actor unchanged
        assert s_after.apple_positions == s.apple_positions  # apples unchanged

    def test_boundary_clamp_up(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0),),
            actor=0,
        )
        s_after = env.apply_action(s, Action.UP)
        assert s_after.agent_positions[0] == Grid(0, 0)  # clamped

    def test_boundary_clamp_left(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0),),
            actor=0,
        )
        s_after = env.apply_action(s, Action.LEFT)
        assert s_after.agent_positions[0] == Grid(0, 0)

    def test_stay(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0),),
            actor=0,
        )
        s_after = env.apply_action(s, Action.STAY)
        assert s_after.agent_positions[0] == Grid(0, 0)


class TestResolvePick:
    def test_no_pick_empty_cell(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0),),
            actor=0,
        )
        s_picked, rewards = env.resolve_pick(s)
        assert s_picked is s  # unchanged
        assert rewards == (0.0, 0.0)

    def test_pick_on_apple(self):
        cfg = _make_det_cfg(r_picker=-1.0)
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0),),
            actor=0,
        )
        s_picked, rewards = env.resolve_pick(s)
        assert s_picked.apple_positions == ()  # apple removed
        assert rewards[0] == -1.0  # picker reward
        assert rewards[1] == 2.0   # other agent reward

    def test_no_pick_force_pick_false(self):
        cfg = _make_det_cfg(force_pick=False)
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0),),
            actor=0,
        )
        s_picked, rewards = env.resolve_pick(s)
        assert s_picked is s
        assert rewards == (0.0, 0.0)


class TestSpawnAndDespawn:
    def test_spawn_after_pick(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        # State with no apples (just picked)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 1)),
            apple_positions=(),
            actor=0,
        )
        s_spawned = env.spawn_and_despawn(s)
        assert len(s_spawned.apple_positions) == 1
        # First empty cell not occupied by agents: (0,0)
        assert s_spawned.apple_positions[0] == Grid(0, 0)

    def test_no_spawn_when_full(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0),),
            actor=0,
        )
        s_spawned = env.spawn_and_despawn(s)
        assert s_spawned.apple_positions == s.apple_positions


class TestAdvanceActor:
    def test_round_robin(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0),),
            actor=0,
        )
        s2 = env.advance_actor(s)
        assert s2.actor == 1
        s3 = env.advance_actor(s2)
        assert s3.actor == 0


class TestFullStep:
    def test_step_no_pick(self):
        cfg = _make_det_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0),),
            actor=0,
        )
        tr = env.step(s, Action.RIGHT)
        assert tr.s_t is s
        assert tr.rewards == (0.0, 0.0)
        assert tr.s_t_next.actor == 1  # advanced

    def test_step_with_pick(self):
        cfg = _make_det_cfg(r_picker=-1.0)
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1)),
            apple_positions=(Grid(1, 0),),
            actor=0,
        )
        tr = env.step(s, Action.DOWN)
        # Agent moved to (1,0) which has apple → pick
        assert tr.rewards[0] == -1.0
        assert tr.rewards[1] == 2.0
        # After spawn, should have 1 apple again
        assert len(tr.s_t_next.apple_positions) == 1
        assert tr.s_t_next.actor == 1


class TestCreateEnv:
    def test_factory_deterministic(self):
        cfg = _make_det_cfg()
        env = create_env(cfg)
        assert isinstance(env, DeterministicEnv)
