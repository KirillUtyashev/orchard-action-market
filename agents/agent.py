from dataclasses import dataclass
from typing import Optional

import numpy as np
from abc import abstractmethod

from models.actor_network import ActorNetwork
from models.reward_network import RewardNetwork
from models.value_function import VNetwork


@dataclass
class AgentInfo:
    policy: callable
    agent_id: int = 0
    num_agents: int = None
    budget: Optional[int] = None


class Agent:
    def __init__(self, agent_info: AgentInfo):
        self.position = np.array([0, 0])
        self.policy = agent_info.policy
        self.id = agent_info.agent_id
        self.collected_apples = 0

    def get_primary_network(self):
        """
        Returns the main network this agent uses for training.
        Subclasses should override this. Returns None if the agent has no network.
        """
        # Base agent has no network.
        return None
