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
        action = random.choices(actions, weights=output)[0]
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
        self.baseline_beta = 0
