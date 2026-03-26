import numpy as np

from debug.code.agents.rate_allocators import FollowingRateUpdater, make_budget_allocator
from debug.code.agents.simple_agent import SimpleAgent

class ACAgent(SimpleAgent):
    def __init__(self, policy, id_, value_network, policy_network, init_alphas):
        super().__init__(policy, id_, value_network)
        self.policy_network = policy_network
        self.agent_alphas = np.asarray(init_alphas, dtype=float)


class ACAgentRates(ACAgent):
    def __init__(
        self,
        policy,
        id_,
        value_network,
        policy_network,
        init_alphas,
        budget,
        init_following_rates,
        init_influencer_rate: float = 0.0,
        rate_solver_name: str = "closed_form",
    ):
        super().__init__(policy, id_, value_network, policy_network, init_alphas)
        self.budget = float(budget)
        self.rate_solver_name = str(rate_solver_name)
        self._rate_allocator = make_budget_allocator(self.rate_solver_name, self.budget)
        self.following_rates = np.zeros_like(self.agent_alphas, dtype=float)
        self.agent_observing_probabilities = np.zeros_like(self.agent_alphas, dtype=float)
        self.following_rate_to_influencer = 0.0
        self.influencer_observing_probability = 0.0
        self.influencer_value = 0.0
        self.set_following_rates(init_following_rates)
        self.set_influencer_rate(init_influencer_rate)

    def set_following_rates(self, following_rates) -> None:
        rates = np.asarray(following_rates, dtype=float).reshape(-1)
        if rates.shape != self.agent_alphas.shape:
            raise ValueError(
                f"Expected following rates shape {self.agent_alphas.shape}, got {rates.shape}."
            )
        rates = np.where(np.isfinite(rates), rates, 0.0)
        rates = np.maximum(rates, 0.0)
        rates[self.id] = 0.0
        self.following_rates = rates
        self.agent_observing_probabilities = 1 - np.exp(-rates)
        self.agent_observing_probabilities[self.id] = 0.0

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
        if target_id == self.id:
            self.agent_alphas[target_id] = 0.0
            return
        self.agent_alphas[target_id] = (1.0 - float(rho)) * float(self.agent_alphas[target_id]) + float(rho) * float(
            q_estimate
        )
        self.agent_alphas[self.id] = 0.0

    def get_effective_observing_probability(self, target_id: int, influencer=None) -> float:
        target_id = int(target_id)
        weight = float(self.agent_observing_probabilities[target_id])
        if influencer is not None:
            weight += float(self.influencer_observing_probability) * float(influencer.outgoing_weights[target_id])
        return weight

    def update_following_rates(self, influencer_value: float | None = None):
        if influencer_value is None:
            updated = self._rate_allocator.solve(
                self.agent_alphas,
                prev_rates=self.following_rates,
                excluded_indices=[self.id],
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
            [np.asarray(self.following_rates, dtype=float), np.array([self.following_rate_to_influencer], dtype=float)],
            axis=0,
        )
        updated = self._rate_allocator.solve(
            coeffs,
            prev_rates=prev_rates,
            excluded_indices=[self.id],
        )
        self.set_following_rates(updated[:-1])
        self.set_influencer_rate(float(updated[-1]))
        return self.following_rates.copy()


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
        self.set_outgoing_rates(init_outgoing_rates if init_outgoing_rates is not None else self.outgoing_rates)
        self.set_beta(init_beta if init_beta is not None else self.beta)

    def set_outgoing_rates(self, outgoing_rates) -> None:
        rates = np.asarray(outgoing_rates, dtype=float).reshape(-1)
        if rates.shape != (self.num_agents,):
            raise ValueError(f"Expected outgoing influencer rates shape {(self.num_agents,)}, got {rates.shape}.")
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

    def recompute_beta(self, follower_agents: list[ACAgentRates]) -> np.ndarray:
        beta = np.zeros(self.num_agents, dtype=float)
        for agent in follower_agents:
            beta += float(agent.influencer_observing_probability) * np.asarray(agent.agent_alphas, dtype=float)
        self.beta = beta
        return self.beta.copy()

    def update_outgoing_rates(self) -> np.ndarray:
        updated = self._rate_allocator.solve(self.beta, prev_rates=self.outgoing_rates)
        self.set_outgoing_rates(updated)
        return self.outgoing_rates.copy()
