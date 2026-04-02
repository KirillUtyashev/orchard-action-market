"""Tests for unified reward computation.

Unified formula:
  Correct pick (τ ∈ G_actor):
      picker gets r_picker
      each groupmate j ∈ G_τ \\ {actor} gets (1 - r_picker) / (n_τ - 1)
      everyone else gets 0
  Wrong pick (τ ∉ G_actor):
      picker gets r_low
      everyone else gets 0
  Centralized sum: +1 for correct, r_low for wrong.

Tests verify:
- n_task_types=1 (single group of all N agents): recovers old r_picker / r_other
- n_task_types>1, r_picker=1: picker-only reward (groupmates get 0)
- n_task_types>1, r_picker=-1: local sharing (groupmates get +2/(n_τ-1))
- Centralized equivalence: sum(dec rewards) = cen reward
"""

import pytest
from orchard.enums import EnvType, PickMode
from orchard.env.deterministic import DeterministicEnv
from orchard.datatypes import EnvConfig, Grid, State


def _make_cfg(n_task_types=1, r_low=0.0, r_picker=1.0,
              task_assignments=None, n_agents=4, **overrides):
    # Auto-generate task_assignments if not provided
    if task_assignments is None:
        if n_task_types == 1:
            task_assignments = tuple((0,) for _ in range(n_agents))
        else:
            # Default: each agent owns one type (requires n_agents >= n_task_types)
            task_assignments = tuple((i,) for i in range(n_agents))

    defaults = dict(
        height=5, width=5, n_agents=n_agents, n_tasks=3,
        gamma=0.99, r_picker=r_picker,
        n_task_types=n_task_types, r_low=r_low,
        task_assignments=task_assignments,
        pick_mode=PickMode.FORCED,
        max_tasks_per_type=3, max_tasks=12,
        env_type=EnvType.DETERMINISTIC,
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


class TestSingleTaskType:
    """n_task_types == 1: all agents in one group. Recovers old r_picker / r_other."""

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
        """r_other = (1 - r_picker) / (N-1) via group sharing."""
        cfg = _make_cfg(r_picker=-1.0)
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0),),
            actor=0,
        )
        _, rewards = env.resolve_pick(s)
        assert rewards[0] == -1.0
        # n_τ = 4 (all agents in one group), groupmate_r = (1 - (-1)) / 3 = 2/3
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

    def test_cen_sum_equals_one(self):
        """With 1 task type, any pick is correct → cen sum = +1."""
        cfg = _make_cfg(r_picker=-1.0)
        env = DeterministicEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0),),
            actor=0,
        )
        _, rewards = env.resolve_pick(s)
        assert abs(sum(rewards) - 1.0) < 1e-10


class TestMultiTypePickerOnly:
    """n_task_types > 1, r_picker=1: only picker rewarded (groupmates get 0)."""

    def test_correct_pick_picker_only(self):
        """Agent 0 picks type 0 (its type) → r_picker=1, others 0."""
        cfg = _make_cfg(
            n_task_types=4, r_picker=1.0, r_low=0.0,
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
        for i in range(1, 4):
            assert rewards[i] == 0.0

    def test_wrong_pick_r_low_zero(self):
        """Agent 0 picks type 1 (not its type) → r_low=0."""
        cfg = _make_cfg(
            n_task_types=4, r_picker=1.0, r_low=0.0,
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
        """Agent 0 picks type 1 → r_low=-1."""
        cfg = _make_cfg(
            n_task_types=4, r_picker=1.0, r_low=-1.0,
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


class TestMultiTypeLocalSharing:
    """n_task_types > 1, r_picker=-1: local group sharing."""

    def test_pair_sharing(self):
        """2 agents per task. Picker=-1, groupmate=+2."""
        cfg = _make_cfg(
            n_task_types=2, r_picker=-1.0, r_low=-1.0, n_agents=4,
            task_assignments=((0,), (0,), (1,), (1,)),
        )
        env = DeterministicEnv(cfg)
        # Agent 0 picks type 0. Agent 1 is groupmate.
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(0,),
        )
        _, rewards = env.resolve_pick(s)
        assert rewards[0] == -1.0
        # n_τ = 2, groupmate_r = (1 - (-1)) / (2-1) = 2.0
        assert abs(rewards[1] - 2.0) < 1e-10
        # Agents 2, 3 are on type 1 → not groupmates → 0
        assert rewards[2] == 0.0
        assert rewards[3] == 0.0
        # Cen sum = -1 + 2 = +1
        assert abs(sum(rewards) - 1.0) < 1e-10

    def test_triple_sharing(self):
        """3 agents per task. Picker=-1, each groupmate=+1."""
        cfg = _make_cfg(
            n_task_types=2, r_picker=-1.0, r_low=-1.0, n_agents=6,
            task_assignments=((0,), (0,), (0,), (1,), (1,), (1,)),
        )
        env = DeterministicEnv(cfg)
        # Agent 0 picks type 0. Agents 1, 2 are groupmates.
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 0), Grid(0, 1), Grid(0, 2),
                             Grid(2, 0), Grid(2, 1)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(0,),
        )
        _, rewards = env.resolve_pick(s)
        assert rewards[0] == -1.0
        # n_τ = 3, groupmate_r = (1 - (-1)) / (3-1) = 1.0
        assert abs(rewards[1] - 1.0) < 1e-10
        assert abs(rewards[2] - 1.0) < 1e-10
        # Non-groupmates get 0
        for i in range(3, 6):
            assert rewards[i] == 0.0
        # Cen sum = -1 + 1 + 1 = +1
        assert abs(sum(rewards) - 1.0) < 1e-10

    def test_wrong_pick_no_sharing(self):
        """Wrong pick: only picker penalized, no groupmate reward."""
        cfg = _make_cfg(
            n_task_types=2, r_picker=-1.0, r_low=-1.0, n_agents=4,
            task_assignments=((0,), (0,), (1,), (1,)),
        )
        env = DeterministicEnv(cfg)
        # Agent 0 (assigned type 0) picks type 1 → wrong
        s = State(
            agent_positions=(Grid(1, 0), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 0),),
            actor=0,
            task_types=(1,),
        )
        _, rewards = env.resolve_pick(s)
        assert rewards[0] == -1.0
        for i in range(1, 4):
            assert rewards[i] == 0.0

    def test_no_pick_zero_for_all(self):
        cfg = _make_cfg(
            n_task_types=4, r_picker=-1.0, r_low=-1.0,
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


class TestCentralizedEquivalence:
    """Centralized sum = +1 for correct, r_low for wrong, regardless of r_picker."""

    def test_sum_correct_pick_always_one(self):
        """Sum of dec rewards on correct pick = +1, for any r_picker."""
        for r_picker in [1.0, 0.0, -1.0, -5.0]:
            cfg = _make_cfg(
                n_task_types=2, r_picker=r_picker, r_low=-1.0, n_agents=4,
                task_assignments=((0,), (0,), (1,), (1,)),
            )
            env = DeterministicEnv(cfg)
            # Agent 0 picks type 0 → correct
            s = State(
                agent_positions=(Grid(1, 0), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
                task_positions=(Grid(1, 0),),
                actor=0,
                task_types=(0,),
            )
            _, rewards = env.resolve_pick(s)
            assert abs(sum(rewards) - 1.0) < 1e-10, (
                f"r_picker={r_picker}: sum={sum(rewards)}, expected 1.0"
            )

    def test_sum_wrong_pick_equals_r_low(self):
        """Sum of dec rewards on wrong pick = r_low."""
        for r_low in [0.0, -1.0, -0.5]:
            cfg = _make_cfg(
                n_task_types=4, r_picker=-1.0, r_low=r_low,
                task_assignments=((0,), (1,), (2,), (3,)),
            )
            env = DeterministicEnv(cfg)
            # Agent 0 picks type 1 → wrong
            s = State(
                agent_positions=(Grid(1, 0), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
                task_positions=(Grid(1, 0),),
                actor=0,
                task_types=(1,),
            )
            _, rewards = env.resolve_pick(s)
            assert abs(sum(rewards) - r_low) < 1e-10

    def test_different_actors_different_rewards(self):
        """Same task type gives different rewards depending on actor."""
        cfg = _make_cfg(
            n_task_types=4, r_picker=1.0, r_low=-1.0,
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
