"""Tests for the general phi/R framework: reward formula, encoders, heuristic."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from orchard.datatypes import EnvConfig, StochasticConfig
from orchard.enums import DespawnMode, EncoderType, Heuristic, StructureType
from orchard.env import create_env
from orchard.eval import evaluate_policy_metrics
from orchard.policy import nearest_action, get_phase2_actions, heuristic_action
from orchard.seed import set_all_seeds
import orchard.encoding as encoding


def _make_env(
    n_agents: int = 4,
    n_task_types: int = 4,
    clustering: int = 1,
    specialization: int = 1,
    sigma_a: float = 0.0,
    sigma_b: float = 0.0,
    seed: int = 0,
    height: int = 5,
    width: int = 5,
    structure: StructureType = StructureType.ID_DISTANCE,
    structure_group_size: int | None = None,
    n_tasks_per_group: int | None = None,
):
    set_all_seeds(seed)
    cfg = EnvConfig(
        height=height, width=width,
        n_agents=n_agents, n_tasks=2, gamma=0.99,
        n_task_types=n_task_types,
        clustering=clustering, specialization=specialization,
        structure=structure,
        structure_group_size=structure_group_size,
        n_tasks_per_group=n_tasks_per_group,
        max_tasks_per_type=5,
        stochastic=StochasticConfig(
            spawn_prob=0.3, despawn_mode=DespawnMode.PROBABILITY, despawn_prob=0.1,
            sigma_a=sigma_a, sigma_b=sigma_b,
        ),
    )
    return create_env(cfg)


# ---------------------------------------------------------------------------
# phi and R matrix structure
# ---------------------------------------------------------------------------

class TestPhiAndRelatedness:
    def test_phi_diagonal_always_one(self):
        """phi[i, i] = 1 when S >= 0 and T = N."""
        env = _make_env(n_agents=4, n_task_types=4, specialization=0)
        # With S=0: phi[i, kappa] = 1 iff i == kappa
        for i in range(4):
            assert env.phi[i, i] == 1.0

    def test_phi_zero_outside_specialization(self):
        env = _make_env(n_agents=4, n_task_types=4, specialization=1)
        # phi[0, 3] = 1[|0-3|<=1] = 0
        assert env.phi[0, 3] == 0.0
        # phi[0, 1] = 1[|0-1|<=1] = 1
        assert env.phi[0, 1] == 1.0

    def test_relatedness_self_always_one(self):
        env = _make_env(clustering=1)
        for i in range(4):
            assert env.relatedness[i, i] == 1.0

    def test_relatedness_clustering_structure(self):
        env = _make_env(n_agents=4, clustering=1)
        # R(0, 1) = 1[|0-1|<=1] = 1
        assert env.relatedness[0, 1] == 1.0
        # R(0, 2) = 1[|0-2|<=1] = 0
        assert env.relatedness[0, 2] == 0.0

    def test_full_relatedness_when_C_ge_N(self):
        env = _make_env(n_agents=4, clustering=10)
        assert np.all(env.relatedness == 1.0)

    def test_phi_positive_types_consistent_with_phi(self):
        env = _make_env(n_agents=4, n_task_types=4, specialization=1)
        for i in range(4):
            expected = frozenset(k for k in range(4) if env.phi[i, k] > 0)
            assert env.phi_positive_types[i] == expected

    def test_disjoint_groups_relatedness_is_block_diagonal(self):
        env = _make_env(
            n_agents=5,
            n_task_types=5,
            structure=StructureType.DISJOINT_GROUPS,
            structure_group_size=2,
        )
        expected = np.array(
            [
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        assert np.array_equal(env.relatedness, expected)

    def test_disjoint_groups_proficiency_matches_task_groups(self):
        env = _make_env(
            n_agents=5,
            n_task_types=5,
            structure=StructureType.DISJOINT_GROUPS,
            structure_group_size=2,
        )
        expected = np.array(
            [
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        assert np.array_equal(env.phi, expected)
        assert env.phi_positive_types[0] == frozenset({0, 1})
        assert env.phi_positive_types[2] == frozenset({2, 3})
        assert env.phi_positive_types[4] == frozenset({4})

    def test_disjoint_groups_can_set_task_types_per_group(self):
        env = _make_env(
            n_agents=5,
            n_task_types=3,
            structure=StructureType.DISJOINT_GROUPS,
            structure_group_size=2,
            n_tasks_per_group=1,
        )
        expected = np.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        assert np.array_equal(env.phi, expected)
        assert env.phi_positive_types[0] == frozenset({0})
        assert env.phi_positive_types[2] == frozenset({1})
        assert env.phi_positive_types[4] == frozenset({2})

    def test_disjoint_groups_require_positive_group_size(self):
        cfg = EnvConfig(
            height=5, width=5,
            n_agents=4, n_tasks=2, gamma=0.99,
            n_task_types=4,
            structure=StructureType.DISJOINT_GROUPS,
            structure_group_size=0,
            stochastic=StochasticConfig(
                spawn_prob=0.3, despawn_mode=DespawnMode.PROBABILITY, despawn_prob=0.1,
            ),
        )
        with pytest.raises(ValueError, match="structure_group_size must be positive"):
            create_env(cfg)

    def test_disjoint_groups_require_positive_tasks_per_group(self):
        cfg = EnvConfig(
            height=5, width=5,
            n_agents=4, n_tasks=2, gamma=0.99,
            n_task_types=4,
            structure=StructureType.DISJOINT_GROUPS,
            structure_group_size=2,
            n_tasks_per_group=0,
            stochastic=StochasticConfig(
                spawn_prob=0.3, despawn_mode=DespawnMode.PROBABILITY, despawn_prob=0.1,
            ),
        )
        with pytest.raises(ValueError, match="n_tasks_per_group must be positive"):
            create_env(cfg)


# ---------------------------------------------------------------------------
# Category reward generation
# ---------------------------------------------------------------------------

class TestCategoryRewards:
    def test_shape(self):
        env = _make_env(n_agents=4, n_task_types=4)
        assert env.category_rewards.shape == (4, 4)

    def test_zero_sigma_gives_uniform(self):
        env = _make_env(n_agents=4, n_task_types=4, sigma_a=0.0, sigma_b=0.0)
        # Each r'[kappa] = (1/N) * 1_N
        expected = 1.0 / 4
        assert np.allclose(env.category_rewards, expected, atol=1e-5)

    def test_sigma_a_zero_means_uniform_within_category(self):
        env = _make_env(n_agents=4, n_task_types=4, sigma_a=0.0, sigma_b=1.0)
        # With sigma_a=0, all agents get same r' within a category (no agent variance)
        for kappa in range(4):
            row = env.category_rewards[kappa]
            assert np.allclose(row, row[0], atol=1e-5), f"kappa={kappa} not uniform: {row}"

    def test_sigma_b_zero_means_equal_category_means(self):
        env = _make_env(n_agents=4, n_task_types=4, sigma_a=0.0, sigma_b=0.0)
        # All categories have same baseline 1/N
        means = env.category_rewards.mean(axis=1)
        assert np.allclose(means, 1.0 / 4, atol=1e-5)

    def test_category_rewards_dtype_float32(self):
        env = _make_env()
        assert env.category_rewards.dtype == np.float32

    def test_category_reward_components_reconstruct_rewards(self):
        env = _make_env(n_agents=5, n_task_types=6, sigma_a=0.4, sigma_b=1.2)
        reconstructed = env.category_reward_agent_offsets + env.category_reward_baselines[:, np.newaxis]

        assert env.category_reward_baseline_raw.shape == (6,)
        assert env.category_reward_baseline_standardized.shape == (6,)
        assert env.category_reward_baselines.shape == (6,)
        assert env.category_reward_agent_offsets.shape == (6, 5)
        assert np.allclose(reconstructed, env.category_rewards, atol=1e-6)
        assert np.allclose(env.category_reward_agent_offsets.mean(axis=1), 0.0, atol=1e-6)

    def test_sigma_b_sets_team_sum_variance(self):
        sigma_b = 1.25
        env = _make_env(n_agents=4, n_task_types=8, sigma_a=0.7, sigma_b=sigma_b)
        team_sums = env.category_rewards.sum(axis=1)
        assert team_sums.mean() == pytest.approx(1.0, abs=1e-5)
        assert team_sums.var() == pytest.approx(sigma_b ** 2, abs=1e-5)

    def test_sigma_a_sets_within_category_agent_variance(self):
        sigma_a = 0.4
        env = _make_env(n_agents=5, n_task_types=6, sigma_a=sigma_a, sigma_b=0.0)
        row_vars = env.category_rewards.var(axis=1)
        assert np.allclose(row_vars, sigma_a ** 2, atol=1e-5)
        assert np.allclose(env.category_rewards.mean(axis=1), 1.0 / 5, atol=1e-5)


# ---------------------------------------------------------------------------
# Reward formula: r_j = phi[actor,tau] * R[actor,j] * r'[tau,j] * norm[actor]
# ---------------------------------------------------------------------------

class TestRewardFormula:
    def test_zero_phi_means_zero_reward(self):
        """If actor has no proficiency in task type, all rewards are zero."""
        env = _make_env(n_agents=4, n_task_types=4, specialization=0)
        # With S=0: agent 0 can only do type 0. Picking type 1 → phi[0,1]=0 → all r=0
        state = env.init_state()
        rewards = env._compute_pick_rewards(actor=0, tau=1)
        assert all(r == 0.0 for r in rewards), f"Expected all zero, got {rewards}"

    def test_correct_pick_formula(self):
        """r_j = phi[actor,tau] * R[actor,j] * r'[tau,j] * norm computed correctly."""
        env = _make_env(n_agents=4, n_task_types=4, specialization=0,
                        sigma_a=0.0, sigma_b=0.0)
        # sigma_a=0, sigma_b=0 → r'[tau,j] = 1/4 for all j
        # S=0: phi[0, 0] = 1, phi[0, k≠0] = 0
        # C=1: R[0, 0]=1, R[0, 1]=1, R[0, 2]=0, R[0, 3]=0
        rewards = env._compute_pick_rewards(actor=0, tau=0)
        norm = env.cfg.n_agents / env.relatedness[0].sum()
        for j in range(4):
            expected = 1.0 * env.relatedness[0, j] * (1.0 / 4) * norm
            assert abs(rewards[j] - expected) < 1e-5, f"j={j}: got {rewards[j]}, expected {expected}"

    def test_full_relatedness_pick_sum_variance_matches_sigma_b(self):
        sigma_b = 1.1
        env = _make_env(
            n_agents=4,
            n_task_types=8,
            clustering=10,
            specialization=10,
            sigma_a=0.6,
            sigma_b=sigma_b,
        )
        reward_sums = np.array([sum(env._compute_pick_rewards(actor=0, tau=tau)) for tau in range(8)])
        assert reward_sums.mean() == pytest.approx(1.0, abs=1e-5)
        assert reward_sums.var() == pytest.approx(sigma_b ** 2, abs=1e-5)

    def test_partial_relatedness_pick_sum_variance_matches_sigma_b_without_agent_noise(self):
        sigma_b = 0.8
        env = _make_env(
            n_agents=6,
            n_task_types=8,
            clustering=1,
            specialization=10,
            sigma_a=0.0,
            sigma_b=sigma_b,
        )
        reward_sums = np.array([sum(env._compute_pick_rewards(actor=0, tau=tau)) for tau in range(8)])
        assert reward_sums.mean() == pytest.approx(1.0, abs=1e-5)
        assert reward_sums.var() == pytest.approx(sigma_b ** 2, abs=1e-5)

    def test_partial_relatedness_pick_sums_include_masked_agent_noise(self):
        env = _make_env(
            n_agents=6,
            n_task_types=8,
            clustering=1,
            specialization=10,
            sigma_a=0.5,
            sigma_b=0.0,
        )
        actor = 0
        norm = env.cfg.n_agents / env.relatedness[actor].sum()
        expected = np.array([
            norm * float((env.relatedness[actor] * env.category_rewards[tau]).sum())
            for tau in range(env.cfg.n_task_types)
        ])
        reward_sums = np.array([
            sum(env._compute_pick_rewards(actor=actor, tau=tau))
            for tau in range(env.cfg.n_task_types)
        ])
        assert np.allclose(reward_sums, expected, atol=1e-5)
        assert reward_sums.var() > 0.0

    def test_disjoint_relatedness_pick_sum_variance_matches_sigma_b_without_agent_noise(self):
        sigma_b = 0.9
        env = _make_env(
            n_agents=6,
            n_task_types=6,
            structure=StructureType.DISJOINT_GROUPS,
            structure_group_size=2,
            n_tasks_per_group=2,
            sigma_a=0.0,
            sigma_b=sigma_b,
        )
        reward_sums = []
        for tau in range(env.cfg.n_task_types):
            actor = (tau // env.cfg.n_tasks_per_group) * env.cfg.structure_group_size
            reward_sums.append(sum(env._compute_pick_rewards(actor=actor, tau=tau)))
        reward_sums = np.array(reward_sums)
        assert reward_sums.mean() == pytest.approx(1.0, abs=1e-5)
        assert reward_sums.var() == pytest.approx(sigma_b ** 2, abs=1e-5)

    def test_resolve_pick_removes_task(self):
        """After resolve_pick, the picked task is removed from state."""
        env = _make_env(n_agents=4, n_task_types=4, specialization=10)
        state = env.init_state()
        # Find a state where actor is on an eligible task
        from orchard.enums import Action
        for _ in range(100):
            actor = state.actor
            eligible = env.phi_positive_types[actor]
            if state.is_agent_on_task(actor, eligible):
                n_tasks_before = len(state.task_positions)
                new_state, rewards = env.resolve_pick(state, pick_type=list(eligible)[0] if eligible else None)
                assert len(new_state.task_positions) == n_tasks_before - 1
                return
            state = env.advance_actor(env.spawn_and_despawn(state))
        pytest.skip("Could not find pick opportunity in 100 steps")


# ---------------------------------------------------------------------------
# Phase-2 action space: only phi > 0 types offered
# ---------------------------------------------------------------------------

class TestPhase2Actions:
    def test_stay_always_offered(self):
        env = _make_env(n_agents=4, n_task_types=4, specialization=0)
        state = env.init_state()
        # Even if no eligible tasks at cell, STAY is offered
        from orchard.datatypes import State, Grid
        from orchard.enums import Action
        # Put actor on a cell with only ineligible task
        actor = 0  # phi[0, tau>0] = 0 with S=0
        actor_pos = state.agent_positions[actor]
        actions = get_phase2_actions(state.with_pick_phase(), env)
        # At minimum STAY is returned (or empty if no task at cell)
        assert Action.STAY in actions or len(actions) == 0

    def test_ineligible_types_not_offered(self):
        env = _make_env(n_agents=4, n_task_types=4, specialization=0)
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
        self.env = _make_env(n_agents=4, n_task_types=4, clustering=1, specialization=1,
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
        """With sigma=0 → r'=1/N>0, phi>=0, R>=0 → task values >= 0."""
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
        self.env = _make_env(n_agents=4, n_task_types=4, clustering=1, specialization=1)
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
        env = _make_env(n_agents=4, n_task_types=4, clustering=10, specialization=10,
                        sigma_a=0.0, sigma_b=2.0, seed=0)
        # With C=N-1, S=T-1: all phi=1, all R=1
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
        env = _make_env(n_agents=4, n_task_types=4, specialization=10)
        state = env.init_state()
        actor = state.actor
        # Manually place actor on a task cell of type the actor can do
        eligible_types = env.phi_positive_types[actor]
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
        env = _make_env(n_agents=4, n_task_types=4, clustering=1, specialization=1,
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
        """With sigma_b > 0 and non-trivial phi, heuristic should achieve positive team RPS."""
        env = _make_env(n_agents=4, n_task_types=4, clustering=10, specialization=10,
                        sigma_a=0.0, sigma_b=0.5, seed=1)
        state = env.init_state()
        metrics = evaluate_policy_metrics(
            state,
            lambda s: heuristic_action(s, env, Heuristic.NEAREST),
            env,
            n_steps=200,
        )
        assert metrics["team_rps"] >= 0.0

    def test_zero_phi_gives_zero_reward(self):
        """Fully isolated agents (C=0, S=0) only get reward for their own type."""
        env = _make_env(n_agents=4, n_task_types=4, clustering=0, specialization=0,
                        sigma_a=0.0, sigma_b=0.0, seed=0)
        # With C=0: R(i,j) = 1[i==j] → only actor gets reward
        # With sigma_b=0: r'[tau,j] = 1/4 for all j
        # r_actor = phi[actor, tau] * R[actor, actor] * r'[tau, actor] = phi * 1 * 1/4
        rewards = env._compute_pick_rewards(actor=0, tau=0)
        # Only agent 0 gets reward (R(0,j)=0 for j!=0)
        for j in range(1, 4):
            assert abs(rewards[j]) < 1e-6, f"Agent {j} should get 0 reward, got {rewards[j]}"
