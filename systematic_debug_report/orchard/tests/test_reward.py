"""Tests for reward computation (chunk 2).

Focused on verifying the reward function:
- Legacy (n_task_types=1): r_picker / r_other distribution
- Multi-type: actor gets R_high or R_low, others get 0
- Centralized reward = sum of decentralized = actor's reward
"""

import pytest
from orchard.enums import EnvType, PickMode
from orchard.env.deterministic import DeterministicEnv
from orchard.datatypes import EnvConfig, Grid, State


def _make_cfg(n_task_types=1, r_high=1.0, r_low=0.0, r_picker=1.0,
              task_assignments=None, **overrides):
    defaults = dict(
        height=5, width=5, n_agents=4, n_tasks=3,
        gamma=0.99, r_picker=r_picker,
        n_task_types=n_task_types, r_high=r_high, r_low=r_low,
        task_assignments=task_assignments,
        pick_mode=PickMode.FORCED,
        max_tasks_per_type=3, max_tasks=12,
        env_type=EnvType.DETERMINISTIC,
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


class TestLegacyReward:
    """n_task_types == 1: old r_picker / r_other distribution."""

    def test_picker_gets_r_picker(self):
        cfg = _make_cfg(r_picker=1.0)
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0),),
            actor=0,
        )
        _, rewards = env.resolve_pick(s)
        assert rewards[0] == 1.0

    def test_others_get_complement(self):
        """r_other = (1 - r_picker) / (N-1)."""
        cfg = _make_cfg(r_picker=-1.0)
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0),),
            actor=0,
        )
        _, rewards = env.resolve_pick(s)
        assert rewards[0] == -1.0
        # r_other = (1 - (-1)) / 3 = 2/3
        expected_other = (1.0 - (-1.0)) / 3
        for i in range(1, 4):
            assert abs(rewards[i] - expected_other) < 1e-10

    def test_no_pick_zero_reward(self):
        cfg = _make_cfg()
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(3, 3), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0),),
            actor=0,
        )
        _, rewards = env.resolve_pick(s)
        assert all(r == 0.0 for r in rewards)


class TestMultiTypeReward:
    """n_task_types > 1: actor gets R_high or R_low, others get 0."""

    def test_correct_pick_r_high(self):
        """Agent 0 picks type 0 (its type) → R_high."""
        cfg = _make_cfg(
            n_task_types=4, r_high=1.0, r_low=0.0,
            task_assignments=((0,), (1,), (2,), (3,)),
        )
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(0,),
        )
        _, rewards = env.resolve_pick(s)
        assert rewards[0] == 1.0

    def test_wrong_pick_r_low_zero(self):
        """Agent 0 picks type 1 (not its type) → R_low=0."""
        cfg = _make_cfg(
            n_task_types=4, r_high=1.0, r_low=0.0,
            task_assignments=((0,), (1,), (2,), (3,)),
        )
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(1,),
        )
        _, rewards = env.resolve_pick(s)
        assert rewards[0] == 0.0

    def test_wrong_pick_r_low_negative(self):
        """Agent 0 picks type 1 → R_low=-1."""
        cfg = _make_cfg(
            n_task_types=4, r_high=1.0, r_low=-1.0,
            task_assignments=((0,), (1,), (2,), (3,)),
        )
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(1,),
        )
        _, rewards = env.resolve_pick(s)
        assert rewards[0] == -1.0

    def test_others_always_zero(self):
        """Non-actor agents always get 0 regardless of pick correctness."""
        cfg = _make_cfg(
            n_task_types=4, r_high=1.0, r_low=-1.0,
            task_assignments=((0,), (1,), (2,), (3,)),
        )
        env = DeterministicEnv(cfg)

        # Correct pick
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(0,),
        )
        _, rewards = env.resolve_pick(s)
        for i in range(1, 4):
            assert rewards[i] == 0.0

        # Wrong pick
        s2 = State(
            agent_positions=(Grid(1, 0), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(2,),
        )
        _, rewards2 = env.resolve_pick(s2)
        for i in range(1, 4):
            assert rewards2[i] == 0.0

    def test_no_pick_zero_for_all(self):
        cfg = _make_cfg(
            n_task_types=4, r_high=1.0, r_low=-1.0,
            task_assignments=((0,), (1,), (2,), (3,)),
        )
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(3, 3), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(0,),
        )
        _, rewards = env.resolve_pick(s)
        assert all(r == 0.0 for r in rewards)

    def test_different_actors_different_rewards(self):
        """Same task type gives different rewards depending on actor."""
        cfg = _make_cfg(
            n_task_types=4, r_high=1.0, r_low=-1.0,
            task_assignments=((0,), (1,), (2,), (3,)),
        )
        env = DeterministicEnv(cfg)

        # Agent 0 picks type 0 → correct
        s0 = State(
            agent_positions=(Grid(1, 0), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(0,),
        )
        _, r0 = env.resolve_pick(s0)
        assert r0[0] == 1.0

        # Agent 1 picks type 0 → wrong
        s1 = State(
            agent_positions=(Grid(0, 0), Grid(1, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0),),
            actor=1,
            task_types=(0,),
        )
        _, r1 = env.resolve_pick(s1)
        assert r1[1] == -1.0


class TestCentralizedEquivalence:
    """cen reward = sum of dec rewards = actor's reward."""

    def test_sum_equals_actor(self):
        cfg = _make_cfg(
            n_task_types=4, r_high=1.0, r_low=-1.0,
            task_assignments=((0,), (1,), (2,), (3,)),
        )
        env = DeterministicEnv(cfg)

        for actor in range(4):
            for task_type in range(4):
                positions = [Grid(3, 3)] * 4
                positions[actor] = Grid(1, 0)
                s = State(
                    agent_positions=tuple(positions),
                    task_positions=(Grid(1, 0),),
                    actor=actor,
                    task_types=(task_type,),
                )
                _, rewards = env.resolve_pick(s)
                # sum of all agent rewards = actor's reward
                assert abs(sum(rewards) - rewards[actor]) < 1e-10
