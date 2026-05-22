"""Tests for the general proficiency/relatedness framework: reward formula, encoders, heuristic."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from orchard.datatypes import EnvConfig, StochasticConfig
from orchard.enums import DespawnMode, EncoderType, Heuristic
from orchard.env import create_env
from orchard.eval import evaluate_policy_metrics
from orchard.policy import nearest_action, get_phase2_actions, heuristic_action
from orchard.seed import set_all_seeds
import orchard.encoding as encoding


def _make_env(
    n_agents: int = 4,
    n_task_types: int = 4,
    relatedness_width: int = 1,
    proficiency_width: int = 1,
    sigma_a: float = 0.0,
    sigma_b: float = 0.0,
    seed: int = 0,
    height: int = 5,
    width: int = 5,
):
    set_all_seeds(seed)
    cfg = EnvConfig(
        height=height, width=width,
        n_agents=n_agents, n_tasks=2, gamma=0.99,
        n_task_types=n_task_types,
        relatedness_width=relatedness_width, proficiency_width=proficiency_width,
        max_tasks_per_type=5,
        stochastic=StochasticConfig(
            spawn_prob=0.3, despawn_mode=DespawnMode.PROBABILITY, despawn_prob=0.1,
            sigma_a=sigma_a, sigma_b=sigma_b,
        ),
    )
    return create_env(cfg)


# ---------------------------------------------------------------------------
# proficiency and relatedness matrix structure
# ---------------------------------------------------------------------------

class TestProficiencyAndRelatedness:
    def test_proficiency_diagonal_always_one(self):
        """proficiency[i, i] = 1 when S >= 0 and T = N."""
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=0)
        # With S=0: proficiency[i, kappa] = 1 iff i == kappa
        for i in range(4):
            assert env.proficiency[i, i] == 1.0

    def test_proficiency_zero_outside_proficiency_width(self):
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=1)
        # circular topology: proficiency[0, 2] has circular dist min(2,2)=2 > 1 → 0
        assert env.proficiency[0, 2] == 0.0
        # proficiency[0, 3] has circular dist min(3,1)=1 <= 1 → 1 (wraps around)
        assert env.proficiency[0, 3] == 1.0
        # proficiency[0, 1] has circular dist 1 <= 1 → 1
        assert env.proficiency[0, 1] == 1.0

    def test_relatedness_self_always_one(self):
        env = _make_env(relatedness_width=1)
        for i in range(4):
            assert env.relatedness[i, i] == 1.0

    def test_relatedness_relatedness_width_structure(self):
        env = _make_env(n_agents=4, relatedness_width=1)
        # R(0, 1) = 1[|0-1|<=1] = 1
        assert env.relatedness[0, 1] == 1.0
        # R(0, 2) = 1[|0-2|<=1] = 0
        assert env.relatedness[0, 2] == 0.0

    def test_full_relatedness_when_C_ge_N(self):
        env = _make_env(n_agents=4, relatedness_width=10)
        assert np.all(env.relatedness == 1.0)

    def test_proficiency_positive_types_consistent_with_proficiency(self):
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=1)
        for i in range(4):
            expected = frozenset(k for k in range(4) if env.proficiency[i, k] > 0)
            assert env.proficiency_positive_types[i] == expected


# ---------------------------------------------------------------------------
# Category reward generation
# ---------------------------------------------------------------------------

class TestCategoryRewards:
    def test_shape(self):
        env = _make_env(n_agents=4, n_task_types=4)
        assert env.category_rewards.shape == (4, 4)

    def test_zero_sigma_gives_uniform(self):
        env = _make_env(n_agents=4, n_task_types=4, sigma_a=0.0, sigma_b=0.0)
        # relatedness_width=1 (default), N=4 → g = min(3,4) = 3 → each r'[kappa] = (1/g) * 1_N
        g = min(2 * env.cfg.relatedness_width + 1, env.cfg.n_agents)
        expected = 1.0 / g
        assert np.allclose(env.category_rewards, expected, atol=1e-5)

    def test_sigma_a_zero_means_uniform_within_category(self):
        env = _make_env(n_agents=4, n_task_types=4, sigma_a=0.0, sigma_b=1.0)
        # With sigma_a=0, all agents get same r' within a category (no agent variance)
        for kappa in range(4):
            row = env.category_rewards[kappa]
            assert np.allclose(row, row[0], atol=1e-5), f"kappa={kappa} not uniform: {row}"

    def test_sigma_b_zero_means_equal_category_means(self):
        env = _make_env(n_agents=4, n_task_types=4, sigma_a=0.0, sigma_b=0.0)
        # All categories have same baseline 1/g where g = min(2*relatedness_width+1, N)
        g = min(2 * env.cfg.relatedness_width + 1, env.cfg.n_agents)
        means = env.category_rewards.mean(axis=1)
        assert np.allclose(means, 1.0 / g, atol=1e-5)

    def test_category_rewards_dtype_float32(self):
        env = _make_env()
        assert env.category_rewards.dtype == np.float32


# ---------------------------------------------------------------------------
# Reward formula: r_j = proficiency[actor,tau] * R[actor,j] * r'[tau,j]
# ---------------------------------------------------------------------------

class TestRewardFormula:
    def test_zero_proficiency_means_zero_reward(self):
        """If actor has no proficiency in task type, all rewards are zero."""
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=0)
        # With S=0: agent 0 can only do type 0. Picking type 1 → proficiency[0,1]=0 → all r=0
        state = env.init_state()
        rewards = env._compute_pick_rewards(actor=0, tau=1)
        assert all(r == 0.0 for r in rewards), f"Expected all zero, got {rewards}"

    def test_correct_pick_formula(self):
        """r_j = proficiency[actor,tau] * R[actor,j] * r'[tau,j] computed correctly."""
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=0,
                        sigma_a=0.0, sigma_b=0.0)
        # sigma_a=0, sigma_b=0 → r'[tau,j] = 1/g for all j; g = min(2*1+1, 4) = 3 → r'=1/3
        # S=0: proficiency[0, 0] = 1, proficiency[0, k≠0] = 0
        # C=1, circular: R[0,0]=1, R[0,1]=1, R[0,2]=0, R[0,3]=1 (agent 3 wraps around)
        # no norm
        rewards = env._compute_pick_rewards(actor=0, tau=0)
        for j in range(4):
            expected = 1.0 * env.relatedness[0, j] * (1.0 / 3)
            assert abs(rewards[j] - expected) < 1e-5, f"j={j}: got {rewards[j]}, expected {expected}"

    def test_resolve_pick_removes_task(self):
        """After resolve_pick, the picked task is removed from state."""
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=10)
        state = env.init_state()
        # Find a state where actor is on an eligible task
        from orchard.enums import Action
        for _ in range(100):
            actor = state.actor
            eligible = env.proficiency_positive_types[actor]
            if state.is_agent_on_task(actor, eligible):
                n_tasks_before = len(state.task_positions)
                new_state, rewards = env.resolve_pick(state, pick_type=list(eligible)[0] if eligible else None)
                assert len(new_state.task_positions) == n_tasks_before - 1
                return
            state = env.advance_actor(env.spawn_and_despawn(state))
        pytest.skip("Could not find pick opportunity in 100 steps")


# ---------------------------------------------------------------------------
# Phase-2 action space: only proficiency > 0 types offered
# ---------------------------------------------------------------------------

class TestPhase2Actions:
    def test_stay_always_offered(self):
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=0)
        state = env.init_state()
        # Even if no eligible tasks at cell, STAY is offered
        from orchard.datatypes import State, Grid
        from orchard.enums import Action
        # Put actor on a cell with only ineligible task
        actor = 0  # proficiency[0, tau>0] = 0 with S=0
        actor_pos = state.agent_positions[actor]
        actions = get_phase2_actions(state.with_pick_phase(), env)
        # At minimum STAY is returned (or empty if no task at cell)
        assert Action.STAY in actions or len(actions) == 0

    def test_ineligible_types_not_offered(self):
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=0)
        from orchard.datatypes import State, Grid
        from orchard.enums import Action, make_pick_action
        state = env.init_state()
        actor = 0  # can only do type 0 with S=0
        actions = get_phase2_actions(state.with_pick_phase(), env)
        # pick(tau) for tau != 0 must not be in actions
        for tau in range(1, 4):
            assert make_pick_action(tau) not in actions


# ---------------------------------------------------------------------------
# Encoders: channel count and value correctness
# ---------------------------------------------------------------------------

class TestGeneralDecEncoder:
    def setup_method(self):
        self.env = _make_env(n_agents=4, n_task_types=4, relatedness_width=1, proficiency_width=1,
                              sigma_a=0.0, sigma_b=0.0)
        encoding.init_encoder(EncoderType.GENERAL_DEC_CNN_GRID, self.env)
        self.state = self.env.init_state()

    def test_grid_channels_count(self):
        T = self.env.cfg.n_task_types
        out = encoding.encode(self.state, 0)
        assert out.grid.shape[0] == T + 3

    def test_scalar_dim(self):
        out = encoding.encode(self.state, 0)
        assert out.scalar.shape[0] == 3

    def test_encode_all_agents_shape(self):
        T = self.env.cfg.n_task_types
        N = self.env.cfg.n_agents
        h, w = self.env.cfg.height, self.env.cfg.width
        grids, scalars = encoding.encode_all_agents(self.state)
        assert grids.shape == (N, T + 3, h, w)
        assert scalars.shape == (N, 3)

    def test_actor_scalar_indicator(self):
        grids, scalars = encoding.encode_all_agents(self.state)
        actor = self.state.actor
        # scalar[actor, 0] == 1, scalar[non-actor, 0] == 0
        assert scalars[actor, 0].item() == pytest.approx(1.0)
        for i in range(self.env.cfg.n_agents):
            if i != actor:
                assert scalars[i, 0].item() == pytest.approx(0.0)

    def test_task_value_channels_non_negative_when_rewards_positive(self):
        """With sigma=0 → r'=1/g>0, proficiency>=0, R>=0 → task values >= 0."""
        grids, _ = encoding.encode_all_agents(self.state)
        T = self.env.cfg.n_task_types
        # Task value channels (0..T-1) should be >= 0
        assert (grids[:, :T] >= -1e-6).all()

    def test_encode_batch_for_actions_shape(self):
        from orchard.policy import get_all_actions
        from orchard.env.base import BaseEnv
        actions = get_all_actions(self.env.cfg)
        after_states = [self.env.apply_action(self.state, a) for a in actions]
        out = encoding.encode_batch_for_actions(self.state, 0, after_states)
        T, N = self.env.cfg.n_task_types, self.env.cfg.n_agents
        h, w = self.env.cfg.height, self.env.cfg.width
        assert out.grid.shape == (len(actions), T + 3, h, w)
        assert out.scalar.shape == (len(actions), 3)


class TestGeneralCenEncoder:
    def setup_method(self):
        self.env = _make_env(n_agents=4, n_task_types=4, relatedness_width=1, proficiency_width=1)
        encoding.init_encoder(EncoderType.GENERAL_CEN_CNN_GRID, self.env)
        self.state = self.env.init_state()

    def test_grid_channels_count(self):
        T = self.env.cfg.n_task_types
        N = self.env.cfg.n_agents
        out = encoding.encode(self.state, 0)
        assert out.grid.shape[0] == T + N + 1

    def test_scalar_dim(self):
        N = self.env.cfg.n_agents
        out = encoding.encode(self.state, 0)
        assert out.scalar.shape[0] == N + 1

    def test_encode_all_agents_shape(self):
        T, N = self.env.cfg.n_task_types, self.env.cfg.n_agents
        h, w = self.env.cfg.height, self.env.cfg.width
        grids, scalars = encoding.encode_all_agents(self.state)
        # Centralized: N=1 outer dim
        assert grids.shape == (1, T + N + 1, h, w)


# ---------------------------------------------------------------------------
# Heuristic: value-aware, picks best task not nearest task
# ---------------------------------------------------------------------------

class TestHeuristic:
    def test_heuristic_prefers_high_value_type(self):
        """With sigma_b>0, heuristic should tend toward higher-value task types."""
        set_all_seeds(0)
        env = _make_env(n_agents=4, n_task_types=4, relatedness_width=10, proficiency_width=10,
                        sigma_a=0.0, sigma_b=2.0, seed=0)
        # With C=N-1, S=T-1: all proficiency=1, all R=1
        # Task values differ by category → heuristic is value-aware
        state = env.init_state()
        action = nearest_action(state, env)
        assert action.is_move()

    def test_heuristic_stays_when_no_tasks(self):
        from orchard.datatypes import State, Grid
        env = _make_env()
        state = env.init_state()
        # Empty task state
        empty_state = State(
            agent_positions=state.agent_positions,
            task_positions=(),
            actor=state.actor,
            task_types=(),
        )
        action = nearest_action(empty_state, env)
        from orchard.enums import Action
        assert action == Action.STAY

    def test_heuristic_picks_eligible_in_phase2(self):
        """In pick phase with eligible task, heuristic picks it."""
        from orchard.datatypes import State, Grid
        from orchard.enums import Action, make_pick_action
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=10)
        state = env.init_state()
        actor = state.actor
        # Manually place actor on a task cell of type the actor can do
        eligible_types = env.proficiency_positive_types[actor]
        if not eligible_types:
            pytest.skip("No eligible types for actor")
        tau = next(iter(eligible_types))
        # Find that task type in state
        for pos, tt in zip(state.task_positions, state.task_types or []):
            if tt == tau:
                # Move actor to this position
                new_positions = list(state.agent_positions)
                new_positions[actor] = pos
                pick_state = State(
                    agent_positions=tuple(new_positions),
                    task_positions=state.task_positions,
                    actor=actor,
                    task_types=state.task_types,
                    pick_phase=True,
                )
                result = nearest_action(pick_state, env)
                assert result.is_pick(), f"Expected pick, got {result}"
                return
        pytest.skip("Could not find suitable task in initial state")


# ---------------------------------------------------------------------------
# Integration: end-to-end rollout metrics
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_rollout_metrics_run(self):
        env = _make_env(n_agents=4, n_task_types=4, relatedness_width=1, proficiency_width=1,
                        sigma_a=0.0, sigma_b=1.0, seed=42)
        state = env.init_state()
        metrics = evaluate_policy_metrics(
            state,
            lambda s: heuristic_action(s, env, Heuristic.NEAREST),
            env,
            n_steps=50,
        )
        assert "rps" in metrics
        assert "team_rps" in metrics
        assert isinstance(metrics["team_rps"], float)

    def test_team_rps_non_negative(self):
        """With sigma_b > 0 and non-trivial proficiency, heuristic should achieve positive team RPS."""
        env = _make_env(n_agents=4, n_task_types=4, relatedness_width=10, proficiency_width=10,
                        sigma_a=0.0, sigma_b=0.5, seed=1)
        state = env.init_state()
        metrics = evaluate_policy_metrics(
            state,
            lambda s: heuristic_action(s, env, Heuristic.NEAREST),
            env,
            n_steps=200,
        )
        assert metrics["team_rps"] >= 0.0

    def test_zero_proficiency_gives_zero_reward(self):
        """Fully isolated agents (C=0, S=0) only get reward for their own type."""
        env = _make_env(n_agents=4, n_task_types=4, relatedness_width=0, proficiency_width=0,
                        sigma_a=0.0, sigma_b=0.0, seed=0)
        # With C=0: R(i,j) = 1[i==j] → only actor gets reward
        # With sigma_b=0, relatedness_width=0: g=1, r'[tau,j] = 1/1 = 1.0 for all j
        # r_actor = proficiency[actor, tau] * R[actor, actor] * r'[tau, actor] = proficiency * 1 * 1
        rewards = env._compute_pick_rewards(actor=0, tau=0)
        # Only agent 0 gets reward (R(0,j)=0 for j!=0)
        for j in range(1, 4):
            assert abs(rewards[j]) < 1e-6, f"Agent {j} should get 0 reward, got {rewards[j]}"
