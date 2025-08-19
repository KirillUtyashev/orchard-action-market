import numpy as np

from agents.rate_updater import RateUpdater
from agents.agent import Agent
from config import get_config
from helpers import convert_input, convert_position

"""
The "Communicating Agent" - The decentralized agent that has its own value functions. Retrieves Q-values from other agents in the list.
"""


class CommAgent(Agent):
    def __init__(self, policy, id_, num_agents=None):
        super().__init__(policy, id_)
        # self.budget = float(1)
        # self.neigh_ids = [j for j in range(num_agents) if j != id_]
        # self.agent_alphas = np.zeros(num_agents)
        # self.updater = RateUpdater(neigh_ids=self.neigh_ids, budget=self.budget)
        # self.num_agents = num_agents
        #
        # # neighbors: global IDs excluding self
        #
        # self.n_neigh = len(self.neigh_ids)
        #
        # self.agent_rates = np.full(num_agents, self.budget / (num_agents - 1))
        # self.agent_rates[self.id] = 0
        # self.agent_observing_probabilities = 1 - np.exp(-self.agent_rates)
        # self.agent_observing_probabilities[self.id] = 0
        #
        # self.agent_alphas = np.zeros(num_agents)
        # self.beta_temp_batch = []
        # self.beta = 0.0
        #
        # assert self.agent_rates[self.id] == 0.0
        # assert self.agent_observing_probabilities[self.id] == 0.0

    def get_value_function(self, state):
        return self.policy_value.get_value_function(state)

    def add_experience(self, old_state, new_state, reward):
        self.policy_value.add_experience(old_state, new_state, reward)
