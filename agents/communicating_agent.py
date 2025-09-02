from agents.value_agent import ValueAgent
import numpy as np

"""
The "Communicating Agent" - The decentralized agent that has its own value functions. Retrieves Q-values from other agents in the list.
"""


class CommAgent(ValueAgent):
    def __init__(self, agent_info):
        super().__init__(agent_info)
        self.agent_alphas = np.zeros(agent_info.num_agents)

    def get_value_function(self, state):
        return self.policy_value.get_value_function(state)

    def add_experience(self, old_state, new_state, reward):
        self.policy_value.add_experience(old_state, new_state, reward)
