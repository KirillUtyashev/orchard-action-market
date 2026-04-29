"""Follower-side state and helpers for following-rate experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from orchard.following_rates.rate_allocators import make_budget_allocator


class InfluencerLike(Protocol):
    outgoing_weights: object


def initial_following_rate_vector(
    num_agents: int,
    agent_id: int,
    budget: float,
    *,
    influencer_enabled: bool = False,
    teammate_mask=None,
    teammate_budget: float | None = None,
    non_teammate_budget: float | None = None,
) -> np.ndarray:
    rates = np.zeros(int(num_agents), dtype=float)
    if num_agents <= 1:
        return rates

    if teammate_budget is not None or non_teammate_budget is not None:
        teammate_budget = max(0.0, float(teammate_budget or 0.0))
        non_teammate_budget = max(0.0, float(non_teammate_budget or 0.0))
        teammate_flags = np.zeros(int(num_agents), dtype=bool)
        if teammate_mask is not None:
            teammate_flags = np.asarray(teammate_mask, dtype=bool).reshape(-1)
            if teammate_flags.shape != (int(num_agents),):
                raise ValueError(
                    f"Expected teammate mask shape {(int(num_agents),)}, got {teammate_flags.shape}."
                )
        teammate_flags[int(agent_id)] = False
        non_teammate_flags = ~teammate_flags
        non_teammate_flags[int(agent_id)] = False

        teammate_count = int(np.count_nonzero(teammate_flags))
        non_teammate_count = int(np.count_nonzero(non_teammate_flags))
        if teammate_count > 0 and teammate_budget > 0.0:
            rates[teammate_flags] = teammate_budget / float(teammate_count)
        if non_teammate_count > 0 and non_teammate_budget > 0.0:
            rates[non_teammate_flags] = non_teammate_budget / float(non_teammate_count)
        return rates

    if budget <= 0.0:
        return rates

    if influencer_enabled:
        share = float(budget) / float(num_agents)
        for idx in range(int(num_agents)):
            if idx != int(agent_id):
                rates[idx] = share
        return rates

    rates.fill(float(budget) / float(num_agents - 1))
    rates[int(agent_id)] = 0.0
    return rates


def initial_following_rate_to_influencer(
    num_agents: int,
    budget: float,
    *,
    influencer_enabled: bool = False,
) -> float:
    if not influencer_enabled or num_agents <= 0 or budget <= 0.0:
        return 0.0
    return float(budget) / float(num_agents)


@dataclass
class FollowingRateAgentState:
    agent_id: int
    agent_alphas: np.ndarray
    budget: float
    following_rates: np.ndarray
    following_rate_to_influencer: float = 0.0
    rate_solver_name: str = "closed_form"
    influencer_value: float = 0.0

    def __post_init__(self) -> None:
        self.agent_id = int(self.agent_id)
        self.agent_alphas = np.asarray(self.agent_alphas, dtype=float).reshape(-1)
        self.agent_alphas = np.where(np.isfinite(self.agent_alphas), self.agent_alphas, 0.0)
        if self.agent_alphas.size > 0:
            self.agent_alphas[self.agent_id] = 0.0
        self.budget = float(self.budget)
        self.rate_solver_name = str(self.rate_solver_name)
        self._rate_allocator = make_budget_allocator(self.rate_solver_name, self.budget)
        self.agent_observing_probabilities = np.zeros_like(self.agent_alphas, dtype=float)
        self.influencer_observing_probability = 0.0
        initial_rates = np.asarray(self.following_rates, dtype=float).reshape(-1)
        self.following_rates = np.zeros_like(self.agent_alphas, dtype=float)
        self.set_following_rates(initial_rates)
        self.set_influencer_rate(self.following_rate_to_influencer)
        self.set_influencer_value(self.influencer_value)

    def set_following_rates(self, following_rates) -> None:
        rates = np.asarray(following_rates, dtype=float).reshape(-1)
        if rates.shape != self.agent_alphas.shape:
            raise ValueError(
                f"Expected following rates shape {self.agent_alphas.shape}, got {rates.shape}."
            )
        rates = np.where(np.isfinite(rates), rates, 0.0)
        rates = np.maximum(rates, 0.0)
        rates[self.agent_id] = 0.0
        self.following_rates = rates
        self.agent_observing_probabilities = 1.0 - np.exp(-rates)
        self.agent_observing_probabilities[self.agent_id] = 0.0

    def set_influencer_rate(self, following_rate_to_influencer: float) -> None:
        rate = float(following_rate_to_influencer)
        if not np.isfinite(rate):
            rate = 0.0
        rate = max(0.0, rate)
        self.following_rate_to_influencer = rate
        self.influencer_observing_probability = 1.0 - float(np.exp(-rate))

    def set_influencer_value(self, influencer_value: float) -> None:
        value = float(influencer_value)
        if not np.isfinite(value):
            value = 0.0
        self.influencer_value = value

    def update_alpha(self, acting_agent_id: int, q_estimate: float, rho: float) -> None:
        target_id = int(acting_agent_id)
        if target_id == self.agent_id:
            self.agent_alphas[target_id] = 0.0
            return
        self.agent_alphas[target_id] = (
            (1.0 - float(rho)) * float(self.agent_alphas[target_id]) +
            float(rho) * float(q_estimate)
        )
        self.agent_alphas[self.agent_id] = 0.0

    def get_effective_observing_probability(
        self,
        target_id: int,
        influencer: InfluencerLike | None = None,
    ) -> float:
        target_id = int(target_id)
        weight = float(self.agent_observing_probabilities[target_id])
        if influencer is not None:
            weight += (
                float(self.influencer_observing_probability) *
                float(np.asarray(influencer.outgoing_weights, dtype=float)[target_id])
            )
        return weight

    def update_following_rates(self, influencer_value: float | None = None) -> np.ndarray:
        if influencer_value is None:
            updated = self._rate_allocator.solve(
                self.agent_alphas,
                prev_rates=self.following_rates,
                excluded_indices=[self.agent_id],
            )
            self.set_following_rates(updated)
            self.set_influencer_rate(0.0)
            self.set_influencer_value(0.0)
            return self.following_rates.copy()

        self.set_influencer_value(influencer_value)
        coeffs = np.concatenate(
            [np.asarray(self.agent_alphas, dtype=float), np.array([self.influencer_value], dtype=float)],
            axis=0,
        )
        prev_rates = np.concatenate(
            [
                np.asarray(self.following_rates, dtype=float),
                np.array([self.following_rate_to_influencer], dtype=float),
            ],
            axis=0,
        )
        updated = self._rate_allocator.solve(
            coeffs,
            prev_rates=prev_rates,
            excluded_indices=[self.agent_id],
        )
        self.set_following_rates(updated[:-1])
        self.set_influencer_rate(float(updated[-1]))
        return self.following_rates.copy()


__all__ = [
    "InfluencerLike",
    "FollowingRateAgentState",
    "initial_following_rate_vector",
    "initial_following_rate_to_influencer",
]
