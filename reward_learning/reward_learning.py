from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch

from agents.agent import Agent
from agents.reward_agent import RewardAgent
from configs.config import ExperimentConfig
from helpers.controllers import AgentControllerRandom, ViewController
from helpers.helpers import create_env, step_reward_learning_centralized, \
    step_reward_learning_decentralized
from main import eval_performance
from models.actor_network import ActorNetwork
from algorithm import Algorithm, EvalResult
import matplotlib.pyplot as plt

from models.reward_network import RewardNetwork
from plots import plot_smoothed


class RewardLearning(Algorithm, ABC):
    """Base class for reward function learning."""

    def __init__(self, config: ExperimentConfig, name):
        """Initialize the value function algorithm."""
        super().__init__(config, name)

    def update_lr(self, step: int) -> None:
        """Update a learning rate based on training progress."""
        pass

    def _init_actor_networks(self, actor_network_cls=ActorNetwork):
        return []

    def init_agents_for_eval(self) -> Tuple[List[RewardAgent], AgentControllerRandom]:
        a_list = []
        info = self.agent_info
        for ii in range(len(self.agents_list)):
            info.agent_id = ii
            trained_agent = RewardAgent(info)
            trained_agent.reward_network = self.network_for_eval[ii]
            a_list.append(trained_agent)
        return a_list, AgentControllerRandom(a_list, self.critic_view_controller)

    def build_experiment(self, view_controller_cls=ViewController,
                         agent_controller_cls=AgentControllerRandom,
                         agent_type=RewardAgent, value_network_cls=None,
                         actor_network_cls=None, **kwargs):
        super().build_experiment(view_controller_cls, agent_controller_cls, agent_type, value_network_cls, **kwargs)
        self.network_for_eval = [agent.reward_network for agent in self.agents_list]

    def log_progress(self, sample_state, sample_state5, sample_state6):
        print(self.env.dummy_counter)

    @abstractmethod
    def process_accuracy(self, agents_list):
        pass


class RewardLearningCentralized(RewardLearning):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config, f"RewardLearningCentralized-<{config.train_config.num_agents}>_agents-_length-<{config.env_config.length}>_width-<{config.env_config.width}>_s_target-<{config.env_config.s_target}>-alpha-<{config.train_config.alpha}>-apple_mean_lifetime-<{config.env_config.apple_mean_lifetime}>-<{config.train_config.hidden_dimensions}>-<{config.train_config.num_layers}>-vision-<{config.train_config.critic_vision}>-<{config.train_config.new_input}>-batch_size-<{config.train_config.batch_size}>-env-<{config.env_config.env_cls}>-new_dynamic-<{config.train_config.new_dynamic}>")

    def _init_reward_networks(self, reward_network_cls=RewardNetwork):
        if self.train_config.critic_vision != 0:
            if self.env_config.width != 1:
                reward_input_dim = self.train_config.critic_vision ** 2
            else:
                reward_input_dim = self.train_config.critic_vision
        else:
            reward_input_dim = 2 * self.env_config.length * self.env_config.width
        network = reward_network_cls(reward_input_dim, 1, self.train_config.alpha, self.train_config.discount, self.train_config.hidden_dimensions, self.train_config.num_layers)
        return [network for _ in range(self.train_config.num_agents)]

    def collect_observation(self, step: int) -> None:
        try:
            for tick in range(self.train_config.num_agents):
                env_step_result = self.env_step(tick)
                reward_sum = np.sum(env_step_result.reward_vector)
                if not self.train_config.new_dynamic:
                    state = self.critic_view_controller.process_state(env_step_result.old_state, None, None)
                else:
                    state = self.critic_view_controller.process_state(env_step_result.new_state, None, None)
                self.agents_list[env_step_result.acting_agent_id].reward_network.add_experience(
                    state, reward_sum)
                if self.train_config.new_dynamic and np.sum(env_step_result.reward_vector) > 0:
                    self.env.remove_apple(self.agents_list[env_step_result.acting_agent_id].position)
        except Exception as e:
            self.logger.error(f"Error collecting observations: {e}")
            raise

    def process_accuracy(self, agents_list):
        self.agents_list[0].prediction_accuracy_history.append(agents_list[0].correct_predictions / agents_list[0].total_predictions)

        for key in self.agents_list[0].prediction_accuracy_by_reward.keys():
            if agents_list[0].total_predictions_by_reward[key] != 0:
                self.agents_list[0].prediction_accuracy_by_reward[key].append(agents_list[0].correct_predictions_by_reward[key] / agents_list[0].total_predictions_by_reward[key])
            else:
                self.agents_list[0].prediction_accuracy_by_reward[key].append(0)

        fig = plot_smoothed(
            [self.agents_list[0].prediction_accuracy_history],
            labels=[f"Centralized"],
            title=f"Immediate Reward Prediction Accuracy",
            xlabel="Training step",
            ylabel="Reward Prediction Accuracy",
            num_points=50  # this controls smoothing granularity
        )
        fig.savefig(self.graphs_out_path / f"reward_loss.png")
        plt.close(fig)

        fig = plot_smoothed(
            [self.agents_list[0].prediction_accuracy_by_reward["1.0"]],
            labels=[f"Centralized"],
            title=f"Immediate Reward Prediction Accuracy",
            xlabel="Training step",
            ylabel="Reward Prediction Accuracy",
            num_points=50  # this controls smoothing granularity
        )
        fig.savefig(self.graphs_out_path / f"reward_loss_1.png")
        plt.close(fig)

        fig = plot_smoothed(
            [self.agents_list[0].prediction_accuracy_by_reward["0.0"]],
            labels=[f"Centralized"],
            title=f"Immediate Reward Prediction Accuracy",
            xlabel="Training step",
            ylabel="Reward Prediction Accuracy",
            num_points=50  # this controls smoothing granularity
        )
        fig.savefig(self.graphs_out_path / f"reward_loss_0.png")
        plt.close(fig)

    def run_inference(self):
        agents_list, agent_controller = self.init_agents_for_eval()

        env = create_env(self.env_config, self.train_config.num_agents, None, None, agents_list, self.env_cls)

        with torch.no_grad():
            results = eval_performance(
                num_agents=self.train_config.num_agents,
                agent_controller=agent_controller,
                env=env,
                name=self.name,
                agents_list=agents_list,
                timesteps=10000,
                epsilon=self.train_config.epsilon,
                env_step=step_reward_learning_centralized
            )
        self.process_accuracy(agents_list)
        print(env.dummy_counter)

        return EvalResult(*results)

    def _evaluate_final(self) -> Tuple[np.floating, ...]:
        res = super()._evaluate_final()
        self.logger.info(f"Final accuracy: {self.agents_list[0].prediction_accuracy_history[-1]}")

        # Average accuracy over rewards
        avg_acc_by_reward = {
            "-1.0": 0,
            "0.0": 0,
            "1.0": 0,
            "other": 0
        }
        for key in self.agents_list[0].prediction_accuracy_by_reward.keys():
            avg_acc_by_reward[key] = self.agents_list[0].prediction_accuracy_by_reward[key][-1]
        self.logger.info(f"Average accuracy by reward: {avg_acc_by_reward}")

        return res


class RewardLearningDecentralized(RewardLearning):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config, f"RewardLearningDecentralized-<{config.train_config.num_agents}>_agents-_length-<{config.env_config.length}>_width-<{config.env_config.width}>_s_target-<{config.env_config.s_target}>-alpha-<{config.train_config.alpha}>-apple_mean_lifetime-<{config.env_config.apple_mean_lifetime}>-<{config.train_config.hidden_dimensions}>-<{config.train_config.num_layers}>-vision-<{config.train_config.critic_vision}>-<{config.train_config.new_input}>-batch_size-<{config.train_config.batch_size}>-env-<{config.env_config.env_cls}>-new_dynamic-<{config.train_config.new_dynamic}>")
        self.same_cell_no_reward = [0]
        self.total_number_of_states_with_reward = 0

    def collect_observation(self, step: int) -> None:
        """Collect observations for reward learning."""
        try:
            for tick in range(self.train_config.num_agents):
                env_step_result = self.env_step(tick)
                for each_agent in range(len(self.agents_list)):
                    reward = env_step_result.reward_vector[each_agent]
                    if not self.train_config.new_dynamic:
                        state = self.critic_view_controller.process_state(env_step_result.old_state, env_step_result.old_positions[each_agent], each_agent + 1)
                    else:
                        state = self.critic_view_controller.process_state(env_step_result.new_state, self.agents_list[each_agent].position, each_agent + 1)
                    self.agents_list[each_agent].reward_network.add_experience(
                        state, reward)

                    if self.env.apples[self.agents_list[each_agent].position[0]][self.agents_list[each_agent].position[1]] == 1 and reward != 1.0 and np.sum(env_step_result.reward_vector) > 0:
                        self.same_cell_no_reward.append(self.same_cell_no_reward[-1] + 1)

                if np.sum(env_step_result.reward_vector) > 0:
                    self.total_number_of_states_with_reward += 1

                if len(self.same_cell_no_reward) == (step + 1) * (tick + 1):
                    self.same_cell_no_reward.append(self.same_cell_no_reward[-1])

                if self.train_config.new_dynamic:
                    self.env.remove_apple(self.agents_list[env_step_result.acting_agent_id].position)
        except Exception as e:
            self.logger.error(f"Error collecting observations: {e}")
            raise

    def process_accuracy(self, agents_list):
        for num, agent in enumerate(agents_list):
            self.agents_list[num].prediction_accuracy_history.append(agent.correct_predictions / agent.total_predictions)
            for key in agent.prediction_accuracy_by_reward.keys():
                if agent.total_predictions_by_reward[key] != 0:
                    self.agents_list[num].prediction_accuracy_by_reward[key].append(agent.correct_predictions_by_reward[key] / agent.total_predictions_by_reward[key])
                else:
                    self.agents_list[num].prediction_accuracy_by_reward[key].append(0)

        fig = plot_smoothed(
            [self.same_cell_no_reward],
            labels=[f"Same Cell, No Reward"],
            title=f"Number of states",
            xlabel="Training step",
            ylabel="Reward Prediction Accuracy",
            num_points=10  # this controls smoothing granularity
        )
        fig.savefig(self.graphs_out_path / f"same_cell_no_reward.png")
        plt.close(fig)

        for agent in self.agents_list:
            fig = plot_smoothed(
                [agent.prediction_accuracy_history],
                labels=[f"Agent {agent.id}"],
                title=f"Agent {agent.id} – Immediate Reward Prediction Accuracy",
                xlabel="Training step",
                ylabel="Reward Prediction Accuracy",
                num_points=50  # this controls smoothing granularity
            )
            fig.savefig(self.graphs_out_path / f"{agent.id}_reward_loss.png")
            plt.close(fig)
            fig = plot_smoothed(
                [agent.prediction_accuracy_by_reward["1.0"]],
                labels=[f"Agent {agent.id}"],
                title=f"Agent {agent.id} – Immediate Reward Prediction Accuracy",
                xlabel="Training step",
                ylabel="Reward Prediction Accuracy",
                num_points=50  # this controls smoothing granularity
            )
            fig.savefig(self.graphs_out_path / f"{agent.id}_reward_loss_reward_1.png")
            plt.close(fig)

            fig = plot_smoothed(
                [agent.prediction_accuracy_by_reward["0.0"]],
                labels=[f"Agent {agent.id}"],
                title=f"Agent {agent.id} – Immediate Reward Prediction Accuracy",
                xlabel="Training step",
                ylabel="Reward Prediction Accuracy",
                num_points=50  # this controls smoothing granularity
            )
            fig.savefig(self.graphs_out_path / f"{agent.id}_reward_loss_reward_0.png")
            plt.close(fig)

            fig = plot_smoothed(
                [agent.prediction_accuracy_by_reward["other"]],
                labels=[f"Agent {agent.id}"],
                title=f"Agent {agent.id} – Immediate Reward Prediction Accuracy",
                xlabel="Training step",
                ylabel="Reward Prediction Accuracy",
                num_points=50  # this controls smoothing granularity
            )
            fig.savefig(self.graphs_out_path / f"{agent.id}_reward_loss_reward_other.png")
            plt.close(fig)

            fig = plot_smoothed(
                [agent.prediction_accuracy_by_reward["-1.0"]],
                labels=[f"Agent {agent.id}"],
                title=f"Agent {agent.id} – Immediate Reward Prediction Accuracy",
                xlabel="Training step",
                ylabel="Reward Prediction Accuracy",
                num_points=50  # this controls smoothing granularity
            )
            fig.savefig(self.graphs_out_path / f"{agent.id}_reward_loss_reward_minus_one.png")
            plt.close(fig)

    def run_inference(self):
        agents_list, agent_controller = self.init_agents_for_eval()

        env = create_env(self.env_config, self.train_config.num_agents, None, None, agents_list, self.env_cls)

        with torch.no_grad():
            results = eval_performance(
                num_agents=self.train_config.num_agents,
                agent_controller=agent_controller,
                env=env,
                name=self.name,
                agents_list=agents_list,
                timesteps=10000,
                epsilon=self.train_config.epsilon,
                env_step=step_reward_learning_decentralized
            )
        self.process_accuracy(agents_list)

        return EvalResult(*results)

    def _evaluate_final(self) -> Tuple[np.floating, ...]:
        res = super()._evaluate_final()
        sum_ = 0
        for agent in self.agents_list:
            sum_ += agent.prediction_accuracy_history[-1]
        av = sum_ / len(self.agents_list)
        self.logger.info(f"Final accuracy: {av}")

        # Average accuracy over rewards
        avg_acc_by_reward = {
            "-1.0": 0,
            "0.0": 0,
            "1.0": 0,
            "other": 0
        }
        for agent in self.agents_list:
            for key in agent.prediction_accuracy_by_reward.keys():
                avg_acc_by_reward[key] += agent.prediction_accuracy_by_reward[key][-1]
        for key in avg_acc_by_reward.keys():
            avg_acc_by_reward[key] /= len(self.agents_list)
        self.logger.info(f"Average accuracy by reward: {avg_acc_by_reward}")

        return res
