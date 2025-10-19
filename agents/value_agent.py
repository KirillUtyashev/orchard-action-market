from abc import abstractmethod
from typing_extensions import override

from agents.agent import Agent, AgentInfo


class ValueAgent(Agent):
    def __init__(self, agent_info: AgentInfo):
        super().__init__(agent_info)
        self.policy_value = None

    @override
    def get_primary_network(self):
        # The primary network for a value agent is its value function.
        return self.policy_value

    @abstractmethod
    def get_value_function(self, state):
        raise NotImplementedError
