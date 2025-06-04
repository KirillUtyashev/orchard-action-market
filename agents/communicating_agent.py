import numpy as np

from agents.agent import Agent
from helpers import convert_position

"""
The "Communicating Agent" - The decentralized agent that has its own value functions. Retrieves Q-values from other agents in the list.
"""


class CommAgent(Agent):
    def __init__(self, policy):
        super().__init__(policy)

    def get_value_function(self, agents, apples, position=None):
        f = self.policy_value.get_value_function(np.concatenate([agents, apples, convert_position(self.position if position is None else position)], axis=0))
        return f

    def evaluate_interface(self, agents, apples, agents_list=None, positions=None):
        sum_ = 0
        for num, agent in enumerate(agents_list):
            sum_ += agent.get_value_function(agents, apples, np.array(positions[num]))
        return sum_

    def get_value_for_agent(self, agents, apples, agents_list=None, hypothetical_pos=None):
        sum_ = 0
        for agent in agents_list:
            if agent is self:
                sum_ += agent.get_value_function(agents, apples, hypothetical_pos)
            else:
                sum_ += agent.get_value_function(agents, apples, agent.position)
        return sum_
