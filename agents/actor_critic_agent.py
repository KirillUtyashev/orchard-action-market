import numpy as np
import random

from helpers import get_discounted_value
from agents.communicating_agent import CommAgent


class ACAgent(CommAgent):
    def __init__(self, policy, id_):
        super().__init__(policy, id_)
        self.policy_network = None

    def add_experience_actor_network(self, state, new_state, reward, action, old_positions, new_positions):
        self.policy_network.add_experience(state, new_state, reward, action, old_positions, new_positions, self.id)


class ACAgentBeta(ACAgent):
    def __init__(self, policy, num_agents):
        super().__init__(policy)
        self.alphas = np.zeros(num_agents)
        self.beta = 0
        self.sum_betas = []
        self.beta_batch = []

    def update_beta(self):
        mean = np.mean(self.beta_batch)
        if not np.isnan(mean):
            self.beta = get_discounted_value(self.beta, mean)
        self.beta_batch = []


class ACAgentRates(ACAgentBeta):
    def __init__(self, policy, num_agents, id_, budget=4):
        super().__init__(policy, num_agents)
        self.id_ = id_
        self.agent_rates = np.zeros(num_agents)
        self.baseline_beta = []
        self.budget = budget
        for i in range(self.agent_rates.size):
            self.agent_rates[i] = (1 / (num_agents - 1)) * self.budget
        self.agent_rates[id_] = 0
        # self.acting_rate = (1 / num_agents) * self.budget


    # def get_value_function_bin(self, a, b, pos=None):
    #     assert self.avg_alpha is not None
    #     if pos is None:
    #         pos = self.position
    #     v = self.policy_value.get_sum_value(a, b, pos)[0]
    #     val = (v - self.avg_alpha) / self.avg_alpha
    #     bound = 0.885
    #     val = ((val + 1) / 2) / bound
    #     if val > 1:
    #         val = 1
    #     elif val < 0:
    #         val = 0
    #     return np.array([val])
