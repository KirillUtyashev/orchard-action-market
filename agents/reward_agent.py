from agents.agent import Agent, AgentInfo


class RewardAgent(Agent):
    def __init__(self, agent_info: AgentInfo):
        super().__init__(agent_info)
        self.reward_network = None
        self.correct_predictions = 0

        self.correct_predictions_by_reward = {
            "-1.0": 0,
            "0.0": 0,
            "1.0": 0,
            "other": 0
        }

        self.total_predictions_by_reward = {
            "-1.0": 0,
            "0.0": 0,
            "1.0": 0,
            "other": 0
        }

        self.total_predictions = 0
        self.prediction_accuracy_history = []
        self.prediction_accuracy_by_reward = {
            "-1.0": [],
            "0.0": [],
            "1.0": [],
            "other": []
        }
