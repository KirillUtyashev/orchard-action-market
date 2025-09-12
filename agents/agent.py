from dataclasses import dataclass
from typing import Optional

import numpy as np
from abc import abstractmethod


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
