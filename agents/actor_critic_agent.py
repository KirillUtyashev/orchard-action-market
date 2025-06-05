import numpy as np
import random
from agents.communicating_agent import CommAgent


class ACAgent(CommAgent):
    def __init__(self, policy):
        super().__init__(policy)
        self.policy_network = None

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

    def get_learned_action(self, state, tau=1):
        a = state["agents"]
        b = state["apples"]

        actions = [0, 1, 4]
        output = self.policy_network.get_function_output(a, b, self.position, tau)
        try:
            action = random.choices(actions, weights=output)[0]
        except Exception as e:
            print(1)
        return action

    def get_action(self, state, agents_list=None):
        if self.policy == "learned_policy":
            return self.get_learned_action(state)
        else:
            return super().get_action(state, agents_list)


class ACAgentBeta(ACAgent):
    def __init__(self, policy, num_agents):
        super().__init__(policy)
        self.alphas = np.zeros(num_agents)
        self.beta = 0
        self.sum_betas = []


class ACAgentRates(ACAgentBeta):
    def __init__(self, policy, num_agents, id_, budget=1):
        super().__init__(policy, num_agents)
        self.id_ = id_
        self.agent_rates = np.zeros(num_agents)
        self.baseline_beta = []
        self.budget = budget
        for i in range(self.agent_rates.size):
            self.agent_rates[i] = (1 / num_agents) * self.budget
        self.agent_rates[id_] = 1
        self.acting_rate = (1 / num_agents) * self.budget
