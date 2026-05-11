"""Tests for picking mechanics and reward distribution in the φ/R framework."""

import pytest
import numpy as np
from orchard.enums import Action, DespawnMode, make_pick_action
from orchard.datatypes import EnvConfig, StochasticConfig, Grid, State
from orchard.env.stochastic import StochasticEnv
from orchard.seed import set_all_seeds


def _make_pick_cfg(n_agents=2, n_task_types=2, clustering=0, specialization=0) -> EnvConfig:
    return EnvConfig(
        height=3, width=3, n_agents=n_agents, n_tasks=2, gamma=0.99,
        n_task_types=n_task_types, clustering=clustering, specialization=specialization,
        max_tasks_per_type=2,
        stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=DespawnMode.NONE, despawn_prob=0.0)
    )


class TestComputePickRewards:
    def test_default_params_self_reward_only(self):
        """With clustering=0, specialization=0, sigma_a=0, sigma_b=0:
        phi[i, kappa] = 1 iff i == kappa (specialization=0 → S=0 → |i-κ|<=0)
        R[i, j] = 1 iff i == j (clustering=0 → C=0 → |i-j|<=0)
        r'[kappa, j] = 1/N for all j (sigma_a=sigma_b=0)
        norm[actor] = N / group_size = 2/1 = 2
        So r_j = phi[actor, tau] * R[actor, j] * r'[tau,j] * norm
        With actor=0, tau=0: phi[0,0]=1, R[0,0]=1, R[0,1]=0, norm=2 → rewards=(1, 0)
        """
        set_all_seeds(0)
        cfg = _make_pick_cfg(n_agents=2, n_task_types=2, clustering=0, specialization=0)
        env = StochasticEnv(cfg)

        rewards = env._compute_pick_rewards(actor=0, tau=0)
        # phi[0,0]=1, R[0,0]=1, R[0,1]=0, r'[0,j]=1/2, norm=2/1=2
        # r_0 = 1*1*(1/2)*2 = 1.0; r_1 = 0
        assert len(rewards) == 2
        assert pytest.approx(rewards[0], abs=1e-6) == 1.0  # picker gets phi=1 (norm compensates)
        assert pytest.approx(rewards[1], abs=1e-6) == 0.0  # non-teammate gets 0

    def test_wrong_type_zero_reward(self):
        """phi[0, 1] = 0 when specialization=0 → all rewards are 0."""
        set_all_seeds(0)
        cfg = _make_pick_cfg(n_agents=2, n_task_types=2, clustering=0, specialization=0)
        env = StochasticEnv(cfg)

        rewards = env._compute_pick_rewards(actor=0, tau=1)
        assert all(r == 0.0 for r in rewards)

    def test_clustering_spreads_reward_to_groupmates(self):
        """With clustering=1, agents 0 and 1 are teammates (|0-1|<=1).
        So reward spreads to both from a pick by actor=0.
        """
        set_all_seeds(0)
        cfg = _make_pick_cfg(n_agents=2, n_task_types=2, clustering=1, specialization=1)
        env = StochasticEnv(cfg)

        rewards = env._compute_pick_rewards(actor=0, tau=0)
        # With clustering=1: R[0,0]=1, R[0,1]=1
        # With specialization=1: phi[0,0]=1, phi[0,1]=1
        # r'[0, j] = 1/N (sigma_b=sigma_a=0)
        # r_j = phi[0,0] * R[0,j] * r'[0,j] = 1 * 1 * 0.5 = 0.5 for both j
        assert len(rewards) == 2
        assert all(r > 0.0 for r in rewards)

    def test_resolve_pick_removes_task(self):
        """resolve_pick removes the picked task from the state."""
        set_all_seeds(0)
        cfg = _make_pick_cfg()
        env = StochasticEnv(cfg)

        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0)),
            task_positions=(Grid(1, 1), Grid(2, 2)),
            actor=0, task_types=(0, 1)
        )
        s_after, rewards = env.resolve_pick(s, pick_type=0)
        assert len(s_after.task_positions) == 1
        assert Grid(1, 1) not in s_after.task_positions
        assert Grid(2, 2) in s_after.task_positions
        assert len(rewards) == 2

    def test_resolve_pick_none_is_noop(self):
        """resolve_pick with pick_type=None leaves state unchanged (STAY)."""
        set_all_seeds(0)
        cfg = _make_pick_cfg()
        env = StochasticEnv(cfg)

        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0)),
            task_positions=(Grid(1, 1),),
            actor=0, task_types=(0,)
        )
        s_after, rewards = env.resolve_pick(s, pick_type=None)
        assert s_after is s
        assert all(r == 0.0 for r in rewards)

    def test_resolve_pick_wrong_type_is_noop(self):
        """resolve_pick with a type not matching any task at actor's cell → noop."""
        set_all_seeds(0)
        cfg = _make_pick_cfg()
        env = StochasticEnv(cfg)

        # Actor at (1,1), task is type 0, but we try to pick type 1
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0)),
            task_positions=(Grid(1, 1),),
            actor=0, task_types=(0,)
        )
        s_after, rewards = env.resolve_pick(s, pick_type=1)
        assert len(s_after.task_positions) == 1  # task remains
        assert all(r == 0.0 for r in rewards)

    def test_step_does_not_auto_pick(self):
        """step() only applies movement — pick must be resolved explicitly."""
        set_all_seeds(0)
        cfg = _make_pick_cfg()
        env = StochasticEnv(cfg)

        # Agent moves RIGHT onto task at (0,1)
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1)),
            task_positions=(Grid(0, 1),),
            actor=0, task_types=(0,)
        )
        tr = env.step(s, Action.RIGHT)
        # No auto-pick: task should still be there, reward=0
        assert tr.rewards == (0.0, 0.0)
        assert len(tr.s_t_next.task_positions) == 1
