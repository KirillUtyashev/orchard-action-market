"""Tests for following-rate budget allocators."""

import numpy as np
import pytest

from orchard.following_rates.rate_allocators import (
    CommunicationBudgetAllocator,
    FollowingRateUpdater,
    ScipyCommunicationBudgetAllocator,
    is_scipy_rate_solver_available,
)


class TestCommunicationBudgetAllocator:
    def test_closed_form_respects_budget_nonnegativity_and_exclusions(self):
        allocator = CommunicationBudgetAllocator(budget=1.5)

        rates = allocator.solve([3.0, 2.0, 1.0], excluded_indices=[1])

        assert rates.shape == (3,)
        assert np.isclose(rates.sum(), 1.5)
        assert rates[1] == 0.0
        assert np.all(rates >= 0.0)

    def test_fallback_uses_previous_rates_when_coefficients_non_positive(self):
        allocator = CommunicationBudgetAllocator(budget=3.0)

        rates = allocator.solve([0.0, -1.0, 0.0], prev_rates=[1.0, 2.0, 0.0])

        assert np.allclose(rates, np.array([1.0, 2.0, 0.0]))

    def test_fallback_becomes_uniform_when_previous_rates_are_unusable(self):
        allocator = CommunicationBudgetAllocator(budget=2.0)

        rates = allocator.solve([0.0, 0.0, 0.0], prev_rates=[1.0, 2.0], excluded_indices=[1])

        assert np.allclose(rates, np.array([1.0, 0.0, 1.0]))


class TestFollowingRateUpdater:
    def test_solve_always_zeros_self_edge(self):
        updater = FollowingRateUpdater(budget=1.0, solver_name="closed_form")

        rates = updater.solve([1.0, 2.0, 3.0], self_id=1)

        assert rates.shape == (3,)
        assert rates[1] == 0.0
        assert np.isclose(rates.sum(), 1.0)


class TestScipyAllocator:
    def test_scipy_allocator_handles_optional_dependency(self):
        if not is_scipy_rate_solver_available():
            with pytest.raises(ModuleNotFoundError):
                ScipyCommunicationBudgetAllocator(1.0)
            return

        allocator = ScipyCommunicationBudgetAllocator(1.0)
        rates = allocator.solve([1.0, 2.0, 3.0], excluded_indices=[0])

        assert rates.shape == (3,)
        assert rates[0] == 0.0
        assert np.all(rates >= 0.0)
        assert np.isclose(rates.sum(), 1.0)
