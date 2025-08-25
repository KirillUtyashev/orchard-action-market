from abc import ABC
import torch
from algorithm import Algorithm
from configs.config import ExperimentConfig
from models.actor_network import ActorNetwork
from models.value_function import VNetwork


class ActorCritic(Algorithm, ABC):
    def __init__(self, config: ExperimentConfig, name: str):
        super().__init__(config, name)
        self.p_network_list = []
        self.v_network_list = []
        self.prob_sample_action_0 = []
        self.prob_sample_action_1 = []
        self.prob_sample_action_2 = []

    def _format_env_step_return(self, state, new_state, reward, agent_id, positions, action, old_pos):
        return state, new_state, reward, agent_id, positions, action

    def update_critic(self):
        losses = []
        for agent in self.agents_list:
            losses.append(agent.policy_value.train())
        return losses[-1]

    def train_batch(self):
        pass

    def update_lr(self, i):
        pass

    def log_progress(self, sample_state, sample_state5, sample_state6):
        super().log_progress(sample_state, sample_state5, sample_state6)

        observation = self.view_controller.process_state(sample_state, sample_state["poses"][0])
        res = self.agents_list[0].policy_network.get_function_output(observation)

        self.prob_sample_action_0.append(res[0])
        self.prob_sample_action_1.append(res[1])
        self.prob_sample_action_2.append(res[2])

        print(res[0])
        print(res[1])

    def agent_get_action(self, agent_id: int) -> int:
        with torch.no_grad():
            action = self.agent_controller.get_best_action(self.env.get_state(),
                                                           agent_id,
                                                           self.env.available_actions)
        return action

    def init_networks(self):
        if self.train_config.alt_input:
            if self.env_config.width != 1:
                input_dim = self.train_config.vision ** 2 + 1
            else:
                input_dim = self.train_config.vision + 1
        else:
            input_dim = self.env_config.length * self.env_config.width + 1
        return (ActorNetwork(input_dim, 5 if self.env_config.width > 1 else 3, self.train_config.actor_alpha, self.train_config.discount, self.train_config.hidden_dimensions_actor, self.train_config.num_layers_actor),
                VNetwork(input_dim, 1, self.train_config.alpha, self.train_config.discount, self.train_config.hidden_dimensions, self.train_config.num_layers))
