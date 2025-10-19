from typing import Optional

from agents.agent import Agent, AgentInfo
from models.reward_network import RewardNetwork
from models.cnn import CNN
from typing import Union

from enum import Enum


class RewardType(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class RewardKeys(Enum):
    NEG_ONE = "-1.0"
    ZERO = "reward = 0.0"
    ONE = "reward = 1.0"
    OTHER = "other"
    GREATER_THAN_ZERO = "reward > 0"
    LESS_THAN_ZERO = "reward < 0"
    CORRECT = "correct"
    TOTAL = "total"
    PREDICTION_ERRORS = "prediction_error (|label - pred|)"


reward_plot_keys_discrete = [
    RewardKeys.NEG_ONE,
    RewardKeys.ZERO,
    RewardKeys.ONE,
    RewardKeys.OTHER,
]
reward_plot_keys_continuous = [
    RewardKeys.ZERO,
    RewardKeys.GREATER_THAN_ZERO,
    RewardKeys.LESS_THAN_ZERO,
]


class RewardAgent(Agent):
    reward_network: Optional[Union[RewardNetwork, CNN]]

    def __init__(self, agent_info: AgentInfo):
        super().__init__(agent_info)
        self.correct_predictions = 0
        self.reward_network = None
        self.correct_predictions_by_reward = {
            "-1.0": 0,
            "0.0": 0,
            "1.0": 0,
            "other": 0,
        }

        self.total_predictions_by_reward = {
            "-1.0": 0,
            "0.0": 0,
            "1.0": 0,
            "other": 0,
        }

        self.total_predictions = 0
        self.prediction_accuracy_history = []
        self.prediction_accuracy_by_reward = {
            "-1.0": [],
            "0.0": [],
            "1.0": [],
            "other": [],
        }

        #### NEW CODE THAT USES ENUMS (SAFER) ###
        self.prediction_metrics = {
            RewardKeys.CORRECT: 0,
            RewardKeys.TOTAL: 0,
            RewardKeys.PREDICTION_ERRORS: [],
            RewardType.DISCRETE: {
                RewardKeys.NEG_ONE: {RewardKeys.CORRECT: 0, RewardKeys.TOTAL: 0},
                RewardKeys.ZERO: {RewardKeys.CORRECT: 0, RewardKeys.TOTAL: 0},
                RewardKeys.ONE: {RewardKeys.CORRECT: 0, RewardKeys.TOTAL: 0},
                RewardKeys.OTHER: {RewardKeys.CORRECT: 0, RewardKeys.TOTAL: 0},
            },
            RewardType.CONTINUOUS: {
                RewardKeys.ZERO: {RewardKeys.CORRECT: 0, RewardKeys.TOTAL: 0},
                RewardKeys.GREATER_THAN_ZERO: {
                    RewardKeys.CORRECT: 0,
                    RewardKeys.TOTAL: 0,
                },
                RewardKeys.LESS_THAN_ZERO: {
                    RewardKeys.CORRECT: 0,
                    RewardKeys.TOTAL: 0,
                },
            },
        }
