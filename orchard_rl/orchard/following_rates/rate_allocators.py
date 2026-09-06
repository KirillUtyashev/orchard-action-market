"""Utilities for allocating finite communication budgets across targets."""

from __future__ import annotations

import numpy as np


def _normalize_excluded_indices(excluded_indices, num_targets: int) -> np.ndarray:
    excluded = np.zeros(num_targets, dtype=bool)
    if excluded_indices is not None:
        for idx in np.asarray(excluded_indices, dtype=int).reshape(-1):
            if 0 <= int(idx) < num_targets:
                excluded[int(idx)] = True
    return excluded


class CommunicationBudgetAllocator:
    def __init__(self, budget: float):
        self.budget = max(0.0, float(budget))

    @staticmethod
    def _sanitize_prev_rates(prev_rates, num_targets: int) -> np.ndarray | None:
        if prev_rates is None:
            return None
        prev = np.asarray(prev_rates, dtype=float).reshape(-1)
        if prev.shape != (num_targets,):
            return None
        prev = np.where(np.isfinite(prev), prev, 0.0)
        return np.maximum(prev, 0.0)

    @staticmethod
    def _fallback_rates(
        num_targets: int,
        budget: float,
        prev_rates=None,
        excluded_indices=None,
    ) -> np.ndarray:
        rates = np.zeros(num_targets, dtype=float)
        if num_targets <= 0 or budget <= 0.0:
            return rates

        excluded = _normalize_excluded_indices(excluded_indices, num_targets)
        allowed = ~excluded
        allowed_count = int(np.count_nonzero(allowed))
        if allowed_count <= 0:
            return rates

        prev = CommunicationBudgetAllocator._sanitize_prev_rates(prev_rates, num_targets)
        if prev is not None:
            rates = prev.copy()
            rates[excluded] = 0.0
            total = float(rates.sum())
            if total > 0.0:
                rates *= budget / total
                rates[excluded] = 0.0
                return rates

        rates[allowed] = budget / float(allowed_count)
        return rates

    def solve(self, coefficients, prev_rates=None, excluded_indices=None) -> np.ndarray:
        coeffs = np.asarray(coefficients, dtype=float).reshape(-1)
        num_targets = int(coeffs.shape[0])
        rates = np.zeros(num_targets, dtype=float)
        if num_targets <= 0 or self.budget <= 0.0:
            return rates

        excluded = _normalize_excluded_indices(excluded_indices, num_targets)
        valid_targets = np.flatnonzero((~excluded) & (coeffs > 0.0))
        if valid_targets.size == 0:
            return self._fallback_rates(
                num_targets,
                self.budget,
                prev_rates=prev_rates,
                excluded_indices=excluded_indices,
            )

        target_coeffs = np.asarray(coeffs[valid_targets], dtype=float)
        order = np.argsort(target_coeffs)[::-1]
        sorted_targets = valid_targets[order]
        sorted_coeffs = target_coeffs[order]
        prefix_log_coeffs = np.cumsum(np.log(sorted_coeffs))

        tol = 1e-12
        active_count = int(sorted_coeffs.size)
        for k in range(1, int(sorted_coeffs.size) + 1):
            log_mu = (float(prefix_log_coeffs[k - 1]) - self.budget) / float(k)
            mu = float(np.exp(log_mu))
            smallest_active = float(sorted_coeffs[k - 1])
            next_inactive = float(sorted_coeffs[k]) if k < int(sorted_coeffs.size) else None
            if smallest_active > mu + tol and (next_inactive is None or next_inactive <= mu + tol):
                active_count = k
                break

        active_targets = sorted_targets[:active_count]
        active_coeffs = sorted_coeffs[:active_count]
        log_mu = (float(np.log(active_coeffs).sum()) - self.budget) / float(active_count)
        active_rates = np.maximum(0.0, np.log(active_coeffs) - log_mu)
        rates[active_targets] = active_rates
        rates[excluded] = 0.0

        total = float(rates.sum())
        if active_targets.size > 0 and total > 0.0:
            rates[active_targets[-1]] = max(0.0, float(rates[active_targets[-1]]) + (self.budget - total))
        return rates


class ScipyCommunicationBudgetAllocator:
    def __init__(self, budget: float):
        self.budget = max(0.0, float(budget))
        self._fallback_allocator = CommunicationBudgetAllocator(self.budget)
        try:
            from scipy.optimize import minimize
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The scipy following-rate solver requires scipy to be installed."
            ) from exc
        self._minimize = minimize

    def solve(self, coefficients, prev_rates=None, excluded_indices=None) -> np.ndarray:
        coeffs = np.asarray(coefficients, dtype=float).reshape(-1)
        num_targets = int(coeffs.shape[0])
        rates = np.zeros(num_targets, dtype=float)
        if num_targets <= 0 or self.budget <= 0.0:
            return rates

        excluded = _normalize_excluded_indices(excluded_indices, num_targets)
        valid_targets = np.flatnonzero((~excluded) & (coeffs > 0.0))
        if valid_targets.size == 0:
            return self._fallback_allocator.solve(
                coeffs,
                prev_rates=prev_rates,
                excluded_indices=excluded_indices,
            )

        if valid_targets.size == 1:
            rates[valid_targets[0]] = self.budget
            return rates

        valid_coeffs = np.asarray(coeffs[valid_targets], dtype=float)
        prev = CommunicationBudgetAllocator._sanitize_prev_rates(prev_rates, num_targets)
        if prev is not None:
            x0 = np.maximum(prev[valid_targets], 0.0)
            total = float(x0.sum())
            if total > 0.0:
                x0 *= self.budget / total
            else:
                x0.fill(self.budget / float(valid_targets.size))
        else:
            x0 = np.full(valid_targets.size, self.budget / float(valid_targets.size), dtype=float)

        bounds = [(0.0, None)] * int(valid_targets.size)

        def objective(x):
            return -float(np.sum((1.0 - np.exp(-x)) * valid_coeffs))

        constraints = [
            {
                "type": "eq",
                "fun": lambda x: float(np.sum(x) - self.budget),
            }
        ]

        result = self._minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if not result.success:
            return self._fallback_allocator.solve(
                coeffs,
                prev_rates=prev_rates,
                excluded_indices=excluded_indices,
            )

        solved = np.maximum(np.asarray(result.x, dtype=float), 0.0)
        total = float(solved.sum())
        if total > 0.0:
            solved *= self.budget / total
        rates[valid_targets] = solved
        rates[excluded] = 0.0
        total = float(rates.sum())
        if valid_targets.size > 0 and total > 0.0:
            rates[valid_targets[-1]] = max(0.0, float(rates[valid_targets[-1]]) + (self.budget - total))
        return rates


def get_supported_rate_solver_names() -> tuple[str, ...]:
    return ("closed_form", "scipy")


def is_scipy_rate_solver_available() -> bool:
    try:
        import scipy  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def make_budget_allocator(solver_name: str, budget: float):
    solver = str(solver_name).strip().lower() or "closed_form"
    if solver == "closed_form":
        return CommunicationBudgetAllocator(budget)
    if solver == "scipy":
        return ScipyCommunicationBudgetAllocator(budget)
    raise ValueError(
        f"Unsupported following-rate solver '{solver_name}'. Expected one of {get_supported_rate_solver_names()}."
    )


class FollowingRateUpdater:
    def __init__(self, budget: float, solver_name: str = "closed_form"):
        self._allocator = make_budget_allocator(solver_name, budget)

    def solve(self, alpha_row, self_id: int, prev_rates=None) -> np.ndarray:
        return self._allocator.solve(alpha_row, prev_rates=prev_rates, excluded_indices=[int(self_id)])


__all__ = [
    "CommunicationBudgetAllocator",
    "ScipyCommunicationBudgetAllocator",
    "FollowingRateUpdater",
    "get_supported_rate_solver_names",
    "is_scipy_rate_solver_available",
    "make_budget_allocator",
]
