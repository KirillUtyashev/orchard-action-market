"""Influencer-side state and helpers for communication experiments."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import numpy as np

from orchard.following_rates.rate_allocators import make_budget_allocator


class FollowerLike(Protocol):
    agent_alphas: object
    influencer_observing_probability: float


def initial_influencer_outgoing_rates(
    num_agents: int,
    budget: float,
    *,
    influencer_enabled: bool = False,
) -> np.ndarray:
    rates = np.zeros(int(num_agents), dtype=float)
    if not influencer_enabled or num_agents <= 0 or budget <= 0.0:
        return rates
    rates.fill(float(budget) / float(num_agents))
    return rates


class ExternalInfluencer:
    def __init__(
        self,
        budget: float,
        num_agents: int,
        init_outgoing_rates=None,
        init_beta=None,
        rate_solver_name: str = "closed_form",
    ):
        self.budget = float(budget)
        self.num_agents = int(num_agents)
        self.rate_solver_name = str(rate_solver_name)
        self._rate_allocator = make_budget_allocator(self.rate_solver_name, self.budget)
        self.outgoing_rates = np.zeros(self.num_agents, dtype=float)
        self.outgoing_weights = np.zeros(self.num_agents, dtype=float)
        self.beta = np.zeros(self.num_agents, dtype=float)
        self.set_outgoing_rates(
            init_outgoing_rates if init_outgoing_rates is not None else self.outgoing_rates
        )
        self.set_beta(init_beta if init_beta is not None else self.beta)

    def set_outgoing_rates(self, outgoing_rates) -> None:
        rates = np.asarray(outgoing_rates, dtype=float).reshape(-1)
        if rates.shape != (self.num_agents,):
            raise ValueError(
                f"Expected outgoing influencer rates shape {(self.num_agents,)}, got {rates.shape}."
            )
        rates = np.where(np.isfinite(rates), rates, 0.0)
        rates = np.maximum(rates, 0.0)
        self.outgoing_rates = rates
        self.outgoing_weights = 1.0 - np.exp(-rates)

    def set_beta(self, beta) -> None:
        beta_arr = np.asarray(beta, dtype=float).reshape(-1)
        if beta_arr.shape != (self.num_agents,):
            raise ValueError(f"Expected influencer beta shape {(self.num_agents,)}, got {beta_arr.shape}.")
        beta_arr = np.where(np.isfinite(beta_arr), beta_arr, 0.0)
        self.beta = beta_arr

    def recompute_beta(self, follower_agents: Iterable[FollowerLike]) -> np.ndarray:
        beta = np.zeros(self.num_agents, dtype=float)
        for agent in follower_agents:
            beta += float(agent.influencer_observing_probability) * np.asarray(agent.agent_alphas, dtype=float)
        self.beta = beta
        return self.beta.copy()

    def update_outgoing_rates(self) -> np.ndarray:
        updated = self._rate_allocator.solve(self.beta, prev_rates=self.outgoing_rates)
        self.set_outgoing_rates(updated)
        return self.outgoing_rates.copy()


__all__ = [
    "FollowerLike",
    "ExternalInfluencer",
    "initial_influencer_outgoing_rates",
]
