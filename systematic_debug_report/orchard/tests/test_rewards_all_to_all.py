"""Tests for deterministic all-to-all Fourier reward generation."""

import numpy as np
import pytest

from orchard.env.stochastic import StochasticEnv, generate_rewards_all_to_all


@pytest.mark.parametrize(
    ("sigma_a", "sigma_b"),
    [(2.0, 6.0), (0.0, 6.0), (2.0, 0.0), (0.0, 0.0)],
)
def test_all_to_all_reward_invariants(sigma_a, sigma_b):
    N = 11
    rewards = generate_rewards_all_to_all(N, sigma_a, sigma_b)

    assert rewards.shape == (N, N)
    assert np.allclose(rewards.std(axis=0, ddof=0), sigma_a)
    assert np.isclose(rewards.sum(axis=0).std(ddof=0), sigma_b)
    assert np.allclose(rewards.mean(axis=1), 1.0 / N)
    assert np.allclose(
        rewards.std(axis=1, ddof=0),
        np.sqrt(sigma_a**2 + sigma_b**2 / N**2),
    )


def test_main_example_agent_standard_deviation():
    rewards = generate_rewards_all_to_all(11, 2.0, 6.0)
    assert np.allclose(rewards.std(axis=1, ddof=0), np.sqrt(2**2 + 6**2 / 11**2))


@pytest.mark.parametrize("sigma", [1.0, 2.0, 4.0, 8.0, 16.0])
def test_equal_sigma_sweep_for_plot_matrices(sigma):
    rewards = generate_rewards_all_to_all(11, sigma, sigma)

    assert np.allclose(rewards.std(axis=0, ddof=0), sigma)
    assert np.isclose(rewards.sum(axis=0).std(ddof=0), sigma)
    assert np.allclose(rewards.mean(axis=1), 1.0 / 11)
    assert np.allclose(
        rewards.std(axis=1, ddof=0),
        np.sqrt(sigma**2 + sigma**2 / 11**2),
    )


@pytest.mark.parametrize("sigma_b", [1.0, 2.0, 4.0, 8.0, 16.0])
def test_fixed_sigma_a_sweep_for_plot_matrices(sigma_b):
    rewards = generate_rewards_all_to_all(11, 4.0, sigma_b)

    assert np.allclose(
        rewards.std(axis=1, ddof=0),
        np.sqrt(4**2 + sigma_b**2 / 11**2),
    )


@pytest.mark.parametrize("num_agents", [0, 4])
def test_all_to_all_requires_at_least_five_agents(num_agents):
    with pytest.raises(ValueError, match="num_agents >= 5"):
        generate_rewards_all_to_all(num_agents, 1.0, 1.0)


@pytest.mark.parametrize(("sigma_a", "sigma_b"), [(-1.0, 0.0), (0.0, -1.0)])
def test_all_to_all_rejects_negative_sigmas(sigma_a, sigma_b):
    with pytest.raises(ValueError, match="non-negative"):
        generate_rewards_all_to_all(11, sigma_a, sigma_b)


def test_environment_uses_task_by_agent_orientation():
    rewards = StochasticEnv._generate_category_rewards(
        seed=123,
        n_task_types=11,
        N=11,
        sigma_a=2.0,
        sigma_b=6.0,
        relatedness_width=5,
        reward_generation="circulant_all_to_all",
    )

    assert rewards.shape == (11, 11)
    assert np.allclose(rewards.std(axis=1, ddof=0), 2.0)
    assert np.isclose(rewards.sum(axis=1).std(ddof=0), 6.0)

    other_seed_rewards = StochasticEnv._generate_category_rewards(
        seed=999,
        n_task_types=11,
        N=11,
        sigma_a=2.0,
        sigma_b=6.0,
        relatedness_width=5,
        reward_generation="circulant_all_to_all",
    )
    assert np.array_equal(rewards, other_seed_rewards)


def test_partial_interest_keeps_independent_generation():
    rewards = StochasticEnv._generate_category_rewards(
        seed=123,
        n_task_types=11,
        N=11,
        sigma_a=2.0,
        sigma_b=6.0,
        relatedness_width=2,
        reward_generation="circulant_all_to_all",
    )

    assert np.all(np.count_nonzero(rewards, axis=1) <= 5)
