from agents.agent import Agent
from policies.random_policy import random_policy


class SimpleAgent(Agent):
    def add_experience(self, old_state, new_state, reward, action=None):
        self.policy_value.add_experience(old_state, new_state, reward)

    def __init__(self, policy=random_policy):
        super().__init__(policy)

    def evaluate_interface(self, agents, apples, agents_list=None, positions=None):
        return self.get_value_function(agents)

    def get_value_function(self, state):
        return self.policy_value.get_value_function(state[:len(state) - 2])

    def get_value_for_agent(self, agents, apples, agents_list=None, hypothetical_pos=None):
        return self.evaluate_interface(agents, apples)
