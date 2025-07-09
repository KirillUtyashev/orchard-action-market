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
    new_pos = np.clip(pos + action, [0, 0], a.shape-np.array([1, 1]))
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
    def get_value_function(self, state):
        raise NotImplementedError

    @abstractmethod
    def add_experience(self, old_state, new_state, reward, action=None):
        raise NotImplementedError

    @abstractmethod
    def get_value_for_agent(self, agents, apples, agents_list=None, hypothetical_pos=None):
        """
        Use this function when evaluating actions
        """
        raise NotImplementedError
