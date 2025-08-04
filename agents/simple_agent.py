from agents.agent import Agent
from policies.random_policy import random_policy


class SimpleAgent(Agent):
    def add_experience(self, old_state, new_state, reward):
        self.policy_value.add_experience(old_state, new_state, reward)

    def __init__(self, policy, id_):
        super().__init__(policy, id_)

    def get_value_function(self, state):
        return self.policy_value.get_value_function(state[:self.policy_value.get_input_dim()])
