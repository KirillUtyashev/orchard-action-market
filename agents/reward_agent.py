from agents.agent import Agent, AgentInfo


class RewardAgent(Agent):
    def __init__(self, agent_info: AgentInfo):
        super().__init__(agent_info)
        self.reward_network = None
        self.correct_predictions = 0
        self.total_predictions = 0
        self.prediction_accuracy_history = []
