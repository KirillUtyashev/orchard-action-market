import numpy as np
from agents.agent import Agent
from policies.random_policy import random_policy


class SimpleAgent(Agent):
    def __init__(self, policy=random_policy):
        super().__init__(policy)

    def evaluate_interface(self, agents, apples, agents_list=None, positions=None):
        return self.get_value_function(agents, apples)

    def get_value_function(self, agents, apples, position=None):
        return self.policy_value.get_value_function(np.concatenate((agents, apples), axis=0).T)

    def get_value_for_agent(self, agents, apples, agents_list=None, hypothetical_pos=None):
        return self.evaluate_interface(agents, apples)



