"""Tests for influencer helpers."""

from dataclasses import dataclass

import numpy as np
import pytest

from orchard.influencer import (
    ExternalInfluencer,
    build_influencer_snapshot_row,
    initial_influencer_outgoing_rates,
)
from orchard.logging_ import build_influencer_csv_fieldnames


@dataclass
class _FollowerStub:
    agent_alphas: np.ndarray
    influencer_observing_probability: float


class TestInfluencerHelpers:
    def test_initial_influencer_outgoing_rates(self):
        assert np.allclose(
            initial_influencer_outgoing_rates(3, 3.0, influencer_enabled=True),
            np.array([1.0, 1.0, 1.0]),
        )
        assert np.allclose(
            initial_influencer_outgoing_rates(3, 3.0, influencer_enabled=False),
            np.array([0.0, 0.0, 0.0]),
        )


class TestExternalInfluencer:
    def test_set_outgoing_rates_computes_weights(self):
        influencer = ExternalInfluencer(budget=1.0, num_agents=3)

        influencer.set_outgoing_rates([0.0, 1.0, 2.0])

        assert np.allclose(influencer.outgoing_rates, np.array([0.0, 1.0, 2.0]))
        assert np.allclose(influencer.outgoing_weights, 1.0 - np.exp(-np.array([0.0, 1.0, 2.0])))

    def test_set_beta_validates_shape_and_sanitizes_values(self):
        influencer = ExternalInfluencer(budget=1.0, num_agents=3)

        with pytest.raises(ValueError, match="Expected influencer beta shape"):
            influencer.set_beta([1.0, 2.0])

        influencer.set_beta([1.0, float("nan"), float("inf")])

        assert np.allclose(influencer.beta, np.array([1.0, 0.0, 0.0]))

    def test_recompute_beta_matches_weighted_sum(self):
        influencer = ExternalInfluencer(budget=1.0, num_agents=3)
        followers = [
            _FollowerStub(np.array([1.0, 2.0, 3.0]), 0.25),
            _FollowerStub(np.array([4.0, 5.0, 6.0]), 0.5),
        ]

        beta = influencer.recompute_beta(followers)

        expected = 0.25 * np.array([1.0, 2.0, 3.0]) + 0.5 * np.array([4.0, 5.0, 6.0])
        assert np.allclose(beta, expected)
        assert np.allclose(influencer.beta, expected)

    def test_update_outgoing_rates_preserves_budget_and_nonnegativity(self):
        influencer = ExternalInfluencer(budget=1.5, num_agents=3, init_beta=[1.0, 2.0, 3.0])

        rates = influencer.update_outgoing_rates()

        assert rates.shape == (3,)
        assert np.all(rates >= 0.0)
        assert np.isclose(rates.sum(), 1.5)

    def test_build_influencer_snapshot_row_matches_fieldnames(self):
        influencer = ExternalInfluencer(
            budget=1.5,
            num_agents=3,
            init_outgoing_rates=[0.1, 0.2, 0.3],
            init_beta=[1.0, 2.0, 3.0],
        )

        row = build_influencer_snapshot_row(step=9, wall_time=2.75, influencer=influencer)

        assert list(row.keys()) == build_influencer_csv_fieldnames(3)
        assert row["step"] == 9
        assert row["beta_to_actor_2"] == 3.0
        assert row["lambda_to_actor_1"] == 0.2
        assert np.isclose(row["weight_to_actor_0"], 1.0 - np.exp(-0.1))
