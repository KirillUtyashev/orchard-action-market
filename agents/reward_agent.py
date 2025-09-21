from typing import Optional

from agents.agent import Agent, AgentInfo
from models.reward_network import RewardNetwork
from models.reward_cnn import RewardCNN
from typing import Union


class RewardAgent(Agent):
    reward_network: Optional[Union[RewardNetwork, RewardCNN]]

    def __init__(self, agent_info: AgentInfo):
        super().__init__(agent_info)
        self.correct_predictions = 0
        self.reward_network = None
        self.correct_predictions_by_reward = {"-1.0": 0, "0.0": 0, "1.0": 0, "other": 0}

        self.total_predictions_by_reward = {"-1.0": 0, "0.0": 0, "1.0": 0, "other": 0}

        self.total_predictions = 0
        self.prediction_accuracy_history = []
        self.prediction_accuracy_by_reward = {
            "-1.0": [],
            "0.0": [],
            "1.0": [],
            "other": [],
        }
