import numpy as np
from policies.random_policy import random_policy

action_vectors = [
            np.array([-1, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([0, -1]),
            np.array([0, 0])
        ]


class Agent:
    def __init__(self, policy=random_policy):
        self.position = np.array([0, 0])
        self.policy = policy

    def get_action(self, state):
        return self.policy(state, self.position)



