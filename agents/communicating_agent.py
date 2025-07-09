from agents.agent import Agent
from config import get_config
from helpers import convert_input, convert_position

"""
The "Communicating Agent" - The decentralized agent that has its own value functions. Retrieves Q-values from other agents in the list.
"""


class CommAgent(Agent):
    def __init__(self, policy):
        super().__init__(policy)

    def get_value_function(self, state):
        return self.policy_value.get_value_function(state)

    def get_value_for_agent(self, agents, apples, agents_list=None, hypothetical_pos=None):
        sum_ = 0
        for agent in agents_list:
            if agent is self:
                if get_config()["alt_input"]:
                    alt_state = convert_input({"agents": agents, "apples": apples}, hypothetical_pos)
                    sum_ += agent.get_q_value(alt_state["agents"])
                else:
                    sum_ += agent.get_q_value(agents)
            else:
                if get_config()["alt_input"]:
                    alt_state = convert_input({"agents": agents, "apples": apples}, agent.position)
                    sum_ += agent.get_q_value(alt_state["agents"])
                else:
                    sum_ += agent.get_q_value(agents)
        return sum_

    def add_experience(self, old_state, new_state, reward, action=None):
        self.policy_value.add_experience(old_state, new_state, reward)
