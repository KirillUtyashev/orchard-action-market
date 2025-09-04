from abc import ABC
from algorithm import Algorithm
from configs.config import ExperimentConfig


class ActorCritic(Algorithm, ABC):
    def __init__(self, config: ExperimentConfig, name: str):
        super().__init__(config, name)
        self.p_network_list = []
        self.v_network_list = []
        self.prob_sample_action_0 = []
        self.prob_sample_action_1 = []
        self.prob_sample_action_2 = []

    def update_lr(self, i):
        pass

    def log_progress(self, sample_state, sample_state5, sample_state6):
        super().log_progress(sample_state, sample_state5, sample_state6)

        observation = self.actor_view_controller.process_state(sample_state, sample_state["poses"][0])
        res = self.agents_list[0].policy_network.get_function_output(observation)

        self.prob_sample_action_0.append(res[0])
        self.prob_sample_action_1.append(res[1])
        self.prob_sample_action_2.append(res[2])

        print(res[0])
        print(res[1])
