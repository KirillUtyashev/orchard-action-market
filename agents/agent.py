import numpy as np
from abc import abstractmethod
from policies.random_policy import random_policy
from config import get_config


action_vectors = [
    np.array([-1, 0]),
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([0, -1]),
    np.array([0, 0])
]


def calculate_ir(a, b, pos, action):
    new_pos = np.clip(pos + action_vectors[action], [0, 0], a.shape-np.array([1, 1]))
    agents = a.copy()
    apples = b.copy()
    agents[new_pos[0], new_pos[1]] += 1
    agents[pos[0], pos[1]] -= 1
    if apples[new_pos[0], new_pos[1]] > 0:
        apples[new_pos[0], new_pos[1]] -= 1
        return 1, agents, apples, new_pos
    else:
        return 0, agents, apples, new_pos


class Agent:
    def __init__(self, policy=random_policy):
        self.position = np.array([0, 0])
        self.policy = policy
        self.policy_value = None

    @abstractmethod
    def get_value_function(self, agents, apples, position=None):
        raise NotImplementedError

    def get_best_action(self, state, agents_list=None):
        a = state["agents"]
        b = state["apples"]
        action = 0
        best_val = 0

        for act in [0, 1, 4]:
            val, new_a, new_b, new_pos = calculate_ir(a, b, self.position, act)
            val += get_config()["discount"] * self.get_value_for_agent(new_a, new_b, agents_list, new_pos)
            if val > best_val:
                action = act
                best_val = val
        return action

    def get_action(self, state, agents_list=None):
        if self.policy == "value_function":
            return self.get_best_action(state, agents_list)
        else:
            return self.policy(state, self.position)

    @abstractmethod
    def evaluate_interface(self, agents, apples, agents_list=None, positions=None):
        raise NotImplementedError

    @abstractmethod
    def get_value_for_agent(self, agents, apples, agents_list=None, hypothetical_pos=None):
        """
        Use this function when evaluating actions
        """
        raise NotImplementedError
