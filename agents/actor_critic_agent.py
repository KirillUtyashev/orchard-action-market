import numpy as np

from helpers import get_discounted_value
from agents.communicating_agent import CommAgent
from agents.rate_updater import RateUpdater


class ACAgent(CommAgent):
    def __init__(self, policy, id_):
        super().__init__(policy, id_)
        self.policy_network = None


    def add_experience_actor_network(self, state, new_state, reward, action, old_positions, new_positions):
        self.policy_network.add_experience(state, new_state, reward, action, old_positions, new_positions, self.id)


class ACAgentBeta(ACAgent):
    def __init__(self, policy, id_):
        super().__init__(policy,  id_)
        self.beta_temp_batch = []
        self.beta = 0.0

    def update_beta(self):
        if len(self.beta_temp_batch) != 0:
            avg_beta_for_this_sec = np.mean(self.beta_temp_batch)
            new_beta = get_discounted_value(self.beta, avg_beta_for_this_sec)
            self.beta = new_beta
            self.beta_temp_batch = []


class ACAgentRates(ACAgentBeta):
    def __init__(self, policy, num_agents, id_, budget=4):
        super().__init__(policy, id_)
        self.num_agents = num_agents
        self.budget = float(budget)

        # neighbors: global IDs excluding self
        self.neigh_ids = [j for j in range(num_agents) if j != id_]
        self.n_neigh = len(self.neigh_ids)

        # pass NEIGHBOR IDS, not n_neigh
        self.updater = RateUpdater(neigh_ids=self.neigh_ids, budget=self.budget)

        self.agent_rates = np.full(num_agents, self.budget / (num_agents - 1))
        self.agent_rates[self.id] = 0
        self.agent_observing_probabilities = 1 - np.exp(-self.agent_rates)
        self.agent_observing_probabilities[self.id] = 0

        self.agent_alphas = np.zeros(num_agents)

        assert self.agent_rates[self.id] == 0.0
        assert self.agent_observing_probabilities[self.id] == 0.0

    def learn_rates(self):
        # agent_id_global is a global ID in [0..N-1]
        new_rates = self.updater.update_many_by_global(self.neigh_ids, self.agent_alphas)
        for k, j_global in enumerate(self.neigh_ids):
            self.agent_rates[j_global] = new_rates[k]
            self.agent_observing_probabilities[j_global] = 1 - np.exp(-new_rates[k])
        self.agent_rates[self.id] = 0.0
        self.agent_observing_probabilities[self.id] = 0.0


