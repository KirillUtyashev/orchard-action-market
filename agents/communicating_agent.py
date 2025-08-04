from agents.agent import Agent
from config import get_config
from helpers import convert_input, convert_position

"""
The "Communicating Agent" - The decentralized agent that has its own value functions. Retrieves Q-values from other agents in the list.
"""


class CommAgent(Agent):
    def __init__(self, policy, id_):
        super().__init__(policy, id_)

    def get_value_function(self, state):
        return self.policy_value.get_value_function(state)

    def add_experience(self, old_state, new_state, reward):
        self.policy_value.add_experience(old_state, new_state, reward)
