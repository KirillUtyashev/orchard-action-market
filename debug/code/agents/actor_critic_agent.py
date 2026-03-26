import numpy as np

from debug.code.agents.simple_agent import SimpleAgent


class ACAgent(SimpleAgent):
    def __init__(self, policy, id_, value_network, policy_network, init_alphas):
        super().__init__(policy, id_, value_network)
        self.policy_network = policy_network
        self.agent_alphas = np.asarray(init_alphas, dtype=float)


class ACAgentRates(ACAgent):
    def __init__(self, policy, id_, value_network, policy_network, init_alphas, budget, init_following_rates):
        super().__init__(policy, id_, value_network, policy_network, init_alphas)
        self.budget = float(budget)
        self.following_rates = np.asarray(init_following_rates, dtype=float)
        self.agent_observing_probabilities = 1 - np.exp(-self.following_rates)

    def update_following_rates(self):
        # TODO: implement rate updater here via closed form solution + some other solver
        pass
