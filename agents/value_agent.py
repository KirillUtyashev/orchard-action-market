from abc import abstractmethod

from agents.agent import Agent, AgentInfo


class ValueAgent(Agent):
    def __init__(self, agent_info: AgentInfo):
        super().__init__(agent_info)
        self.policy_value = None

    @abstractmethod
    def get_value_function(self, state):
        pass
