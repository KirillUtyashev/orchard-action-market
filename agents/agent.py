import numpy as np
from abc import abstractmethod
from policies.random_policy import random_policy
from config import get_config


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
    def __init__(self, policy, id_):
        self.position = np.array([0, 0])
        self.policy = policy
        self.policy_value = None
        self.id = id_
        self.collected_apples = 0

    @abstractmethod
    def get_value_function(self, state):
        raise NotImplementedError
