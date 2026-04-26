"""Tests for picking mechanics, FORCED/CHOICE modes, and reward distribution."""

import pytest
from orchard.enums import Action, PickMode, DespawnMode, make_pick_action
from orchard.datatypes import EnvConfig, StochasticConfig, Grid, State
from orchard.env.stochastic import StochasticEnv

def _make_pick_cfg(n_agents=4, n_task_types=2, pick_mode=PickMode.FORCED, r_picker=1.0, r_low=-1.0):
    # Group assignments: Agent 0,1 -> Type 0 | Agent 2,3 -> Type 1
    assignments = ((0,), (0,), (1,), (1,)) if n_task_types == 2 else tuple((0,) for _ in range(n_agents))
    
    return EnvConfig(
        height=3, width=3, n_agents=n_agents, n_tasks=2, gamma=0.99,
        r_picker=r_picker, r_low=r_low, n_task_types=n_task_types,
        pick_mode=pick_mode, task_assignments=assignments, max_tasks_per_type=2,
        stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=DespawnMode.NONE, despawn_prob=0.0)
    )

class TestRewardDistribution:
    def test_correct_pick_rewards_groupmates(self):
        # r_picker = -1.0, so the 1.0 total reward minus -1.0 picker reward leaves 2.0 to distribute.
        # 1 groupmate -> gets +2.0. Strangers get 0.
        cfg = _make_pick_cfg(n_task_types=2, r_picker=-1.0)
        env = StochasticEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 1),), actor=0, task_types=(0,)
        )
        
        _, rewards = env.resolve_pick(s, pick_type=0)
        
        assert rewards[0] == -1.0  # Picker (Type 0)
        assert rewards[1] == 2.0   # Groupmate (Type 0)
        assert rewards[2] == 0.0   # Stranger (Type 1)
        assert rewards[3] == 0.0   # Stranger (Type 1)
        assert sum(rewards) == 1.0 # Total team reward is +1.0 for a correct pick

    def test_wrong_pick_penalizes_picker_only(self):
        # In CHOICE mode: agent explicitly picks a wrong-type task → r_low penalty
        cfg = _make_pick_cfg(n_task_types=2, r_low=-0.5, pick_mode=PickMode.CHOICE)
        env = StochasticEnv(cfg)

        # Agent 0 (assigned Type 0) explicitly picks a Type 1 task
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 1),), actor=0, task_types=(1,)
        )

        _, rewards = env.resolve_pick(s, pick_type=1)

        assert rewards[0] == -0.5
        assert rewards[1] == 0.0
        assert rewards[2] == 0.0
        assert rewards[3] == 0.0
        assert sum(rewards) == -0.5

    def test_forced_wrong_type_stays(self):
        # In FORCED mode: agent on a wrong-type task stays (no pick, 0 reward)
        cfg = _make_pick_cfg(n_task_types=2, r_low=-0.5)
        env = StochasticEnv(cfg)

        # Agent 0 (assigned Type 0) is on a Type 1 task
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 1),), actor=0, task_types=(1,)
        )

        s_after, rewards = env.resolve_pick(s)

        assert len(s_after.task_positions) == 1  # task not picked
        assert sum(rewards) == 0.0

class TestPickModeForced:
    def test_auto_picks_on_resolve(self):
        cfg = _make_pick_cfg(pick_mode=PickMode.FORCED)
        env = StochasticEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 1),), actor=0, task_types=(0,)
        )
        
        # pick_type=None is ignored in FORCED, it auto-detects what is under the agent
        s_after, rewards = env.resolve_pick(s, pick_type=None)
        
        assert len(s_after.task_positions) == 0
        assert rewards[0] == 1.0

    def test_step_function_auto_picks(self):
        cfg = _make_pick_cfg(pick_mode=PickMode.FORCED)
        env = StochasticEnv(cfg)
        # Agent moves DOWN onto the task
        s = State(
            agent_positions=(Grid(0, 1), Grid(0, 0), Grid(2, 2), Grid(2, 0)),
            task_positions=(Grid(1, 1),), actor=0, task_types=(0,)
        )
        
        tr = env.step(s, Action.DOWN)
        
        assert tr.rewards[0] == 1.0
        assert len(tr.s_t_next.task_positions) == 0

class TestPickModeChoice:
    def test_requires_explicit_pick_type(self):
        cfg = _make_pick_cfg(pick_mode=PickMode.CHOICE)
        env = StochasticEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 1),), actor=0, task_types=(0,)
        )
        
        # Passing None (or implicitly STAYing) means no pick happens
        s_after, rewards = env.resolve_pick(s, pick_type=None)
        assert len(s_after.task_positions) == 1
        assert sum(rewards) == 0.0
        
        # Passing correct pick_type succeeds
        s_picked, rewards_picked = env.resolve_pick(s, pick_type=0)
        assert len(s_picked.task_positions) == 0
        assert rewards_picked[0] == 1.0

    def test_wrong_type_request_ignored(self):
        cfg = _make_pick_cfg(pick_mode=PickMode.CHOICE)
        env = StochasticEnv(cfg)
        # Task is Type 0, agent asks for Type 1
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0), Grid(0, 1), Grid(0, 2)),
            task_positions=(Grid(1, 1),), actor=0, task_types=(0,)
        )
        
        s_after, rewards = env.resolve_pick(s, pick_type=1)
        assert len(s_after.task_positions) == 1  # Pick failed, task remains
        assert sum(rewards) == 0.0

    def test_step_function_does_not_auto_pick(self):
        cfg = _make_pick_cfg(pick_mode=PickMode.CHOICE)
        env = StochasticEnv(cfg)
        # Agent moves DOWN onto the task
        s = State(
            agent_positions=(Grid(0, 1), Grid(0, 0), Grid(2, 2), Grid(2, 0)),
            task_positions=(Grid(1, 1),), actor=0, task_types=(0,)
        )
        
        tr = env.step(s, Action.DOWN)
        
        # In CHOICE mode, movement does not trigger a pick
        assert tr.rewards[0] == 0.0
        assert len(tr.s_t_next.task_positions) == 1