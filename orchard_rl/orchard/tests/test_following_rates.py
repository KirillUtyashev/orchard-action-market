"""Tests for following-rate helpers."""

import numpy as np

from orchard.following_rates import (
    FollowingRateAgentState,
    build_following_rate_snapshot_row,
    compute_following_rate_stats,
    initial_following_rate_to_influencer,
    initial_following_rate_vector,
)
from orchard.influencer import ExternalInfluencer
from orchard.logging_ import build_following_rate_csv_fieldnames


class TestFollowingRateHelpers:
    def test_initial_following_rate_vector_without_influencer(self):
        rates = initial_following_rate_vector(3, agent_id=1, budget=2.0, influencer_enabled=False)

        assert np.allclose(rates, np.array([1.0, 0.0, 1.0]))

    def test_initial_following_rate_vector_with_influencer(self):
        rates = initial_following_rate_vector(3, agent_id=1, budget=3.0, influencer_enabled=True)

        assert np.allclose(rates, np.array([1.0, 0.0, 1.0]))

    def test_initial_following_rate_to_influencer(self):
        assert initial_following_rate_to_influencer(3, 3.0, influencer_enabled=True) == 1.0
        assert initial_following_rate_to_influencer(3, 3.0, influencer_enabled=False) == 0.0

    def test_initial_following_rate_vector_with_dual_budgets(self):
        rates = initial_following_rate_vector(
            4,
            agent_id=0,
            budget=0.0,
            teammate_mask=[False, True, False, False],
            teammate_budget=2.0,
            non_teammate_budget=6.0,
        )

        assert np.allclose(rates, np.array([0.0, 2.0, 3.0, 3.0]))

    def test_initial_following_rate_vector_with_dual_budgets_leaves_empty_class_unused(self):
        rates = initial_following_rate_vector(
            3,
            agent_id=0,
            budget=0.0,
            teammate_mask=[False, False, False],
            teammate_budget=2.0,
            non_teammate_budget=4.0,
        )

        assert np.allclose(rates, np.array([0.0, 2.0, 2.0]))


class TestFollowingRateAgentState:
    def test_initialization_sanitizes_and_zeros_self_edge(self):
        state = FollowingRateAgentState(
            agent_id=1,
            agent_alphas=np.array([1.0, 5.0, float("nan")]),
            budget=2.0,
            following_rates=np.array([1.0, 2.0, float("inf")]),
        )

        assert np.allclose(state.agent_alphas, np.array([1.0, 0.0, 0.0]))
        assert state.following_rates[1] == 0.0
        assert np.isclose(state.agent_observing_probabilities[0], 1.0 - np.exp(-1.0))
        assert state.agent_observing_probabilities[1] == 0.0

    def test_update_alpha(self):
        state = FollowingRateAgentState(
            agent_id=0,
            agent_alphas=np.array([0.0, 1.0, 2.0]),
            budget=1.0,
            following_rates=np.array([0.0, 0.5, 0.5]),
        )

        state.update_alpha(acting_agent_id=2, q_estimate=6.0, rho=0.25)

        assert np.isclose(state.agent_alphas[2], 3.0)
        assert state.agent_alphas[0] == 0.0

    def test_update_following_rates_without_influencer(self):
        state = FollowingRateAgentState(
            agent_id=0,
            agent_alphas=np.array([0.0, 2.0, 1.0]),
            budget=1.5,
            following_rates=np.array([0.0, 0.75, 0.75]),
        )

        updated = state.update_following_rates()

        assert updated.shape == (3,)
        assert updated[0] == 0.0
        assert np.isclose(updated.sum(), 1.5)
        assert state.following_rate_to_influencer == 0.0

    def test_update_following_rates_with_influencer(self):
        state = FollowingRateAgentState(
            agent_id=1,
            agent_alphas=np.array([3.0, 0.0, 1.0]),
            budget=2.0,
            following_rates=np.array([1.0, 0.0, 1.0]),
            following_rate_to_influencer=0.0,
        )

        updated = state.update_following_rates(influencer_value=4.0)

        assert updated.shape == (3,)
        assert updated[1] == 0.0
        assert np.isclose(updated.sum() + state.following_rate_to_influencer, 2.0)
        assert state.influencer_value == 4.0

    def test_effective_observing_probability_includes_influencer_weight(self):
        follower = FollowingRateAgentState(
            agent_id=1,
            agent_alphas=np.array([3.0, 0.0, 1.0]),
            budget=2.0,
            following_rates=np.array([1.0, 0.0, 1.0]),
            following_rate_to_influencer=0.5,
        )
        influencer = ExternalInfluencer(
            budget=1.0,
            num_agents=3,
            init_outgoing_rates=[0.2, 0.4, 0.6],
        )

        effective = follower.get_effective_observing_probability(0, influencer)

        expected = follower.agent_observing_probabilities[0] + (
            follower.influencer_observing_probability * influencer.outgoing_weights[0]
        )
        assert np.isclose(effective, expected)

    def test_build_following_rate_snapshot_row_matches_fieldnames(self):
        state = FollowingRateAgentState(
            agent_id=1,
            agent_alphas=np.array([3.0, 0.0, 1.0]),
            budget=2.0,
            following_rates=np.array([1.0, 0.0, 0.5]),
            following_rate_to_influencer=0.25,
            influencer_value=1.5,
        )

        row = build_following_rate_snapshot_row(step=7, wall_time=1.25, agent_state=state)

        assert list(row.keys()) == build_following_rate_csv_fieldnames(3)
        assert row["step"] == 7
        assert row["alpha_to_0"] == 3.0
        assert row["lambda_to_2"] == 0.5
        assert np.isclose(row["weight_to_0"], 1.0 - np.exp(-1.0))
        assert np.isclose(row["weight_to_influencer"], 1.0 - np.exp(-0.25))
        assert row["influencer_value"] == 1.5

    def test_compute_following_rate_stats_matches_manual_expectations(self):
        state_a = FollowingRateAgentState(
            agent_id=0,
            agent_alphas=np.array([0.0, 2.0]),
            budget=2.0,
            following_rates=np.array([0.0, 1.0]),
            following_rate_to_influencer=0.3,
        )
        state_b = FollowingRateAgentState(
            agent_id=1,
            agent_alphas=np.array([4.0, 0.0]),
            budget=2.0,
            following_rates=np.array([0.5, 0.0]),
            following_rate_to_influencer=0.7,
        )
        influencer = ExternalInfluencer(
            budget=1.0,
            num_agents=2,
            init_outgoing_rates=[0.2, 0.8],
            init_beta=[1.0, 3.0],
        )

        stats = compute_following_rate_stats([state_a, state_b], influencer)

        weights = np.array([1.0 - np.exp(-1.0), 1.0 - np.exp(-0.5)])
        follower_to_influencer = np.array([1.0 - np.exp(-0.3), 1.0 - np.exp(-0.7)])
        influencer_weights = 1.0 - np.exp(-np.array([0.2, 0.8]))
        effective_weights = np.array([
            weights[0] + follower_to_influencer[0] * influencer_weights[1],
            weights[1] + follower_to_influencer[1] * influencer_weights[0],
        ])

        assert np.isclose(stats["alpha_mean"], 3.0)
        assert np.isclose(stats["alpha_positive_frac"], 1.0)
        assert np.isclose(stats["following_weight_mean"], weights.mean())
        assert np.isclose(stats["active_follow_edges_mean"], 1.0)
        assert np.isclose(stats["beta_mean"], 2.0)
        assert np.isclose(stats["influencer_weight_mean"], influencer_weights.mean())
        assert np.isclose(
            stats["follower_to_influencer_weight_mean"],
            follower_to_influencer.mean(),
        )
        assert np.isclose(stats["effective_follow_weight_mean"], effective_weights.mean())
