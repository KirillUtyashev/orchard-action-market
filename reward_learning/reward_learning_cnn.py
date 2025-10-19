from datetime import date
import random
from typing import Sequence
from matplotlib import pyplot as plt
import numpy as np
import torch
from typing_extensions import override


from agents.agent import Agent
from agents.agent import Agent
from agents.reward_agent import RewardAgent
from algorithm import EnvStep, EvalResult
from config import FINAL_DIR
from configs.config import ExperimentConfig
from helpers.controllers import AgentController, AgentControllerUsingPolicy
from helpers.helpers import (
    create_env,
)


from main import eval_performance
from models.cnn import CNN
from orchard.environment import (
    Orchard,
    OrchardBasic,
    OrchardBasicNewDynamic,
    OrchardEuclideanNegativeRewards,
    OrchardEuclideanRewards,
)
from plots import plot_smoothed
from reward_learning.reward_learning import RewardLearningDecentralized
from agents.reward_agent import (
    RewardType,
    RewardKeys,
    reward_plot_keys_discrete,
    reward_plot_keys_continuous,
)


class RewardLearningCNNDecentralized(RewardLearningDecentralized):
    """
    An algorithm for decentralized reward learning that uses a Convolutional
    Neural Network (CNN) to process spatial state information.
    """

    def __init__(self, config: ExperimentConfig):
        # Call the parent __init__ to set up the name and data structures
        super().__init__(
            config,
            name=f"CNNRewardLearning-<{config.train_config.num_agents}>_agents-_length-<{config.env_config.length}>_width-<{config.env_config.width}>_s_target-<{config.env_config.s_target}>-alpha-<{config.train_config.alpha}>-apple_mean_lifetime-<{config.env_config.apple_mean_lifetime}>-<{config.train_config.hidden_dimensions}>-<{config.train_config.num_layers}>-vision-<{config.train_config.critic_vision}>-<{config.train_config.new_input}>-batch_size-<{config.train_config.batch_size}>-env-<{config.env_config.env_cls}>-new_dynamic-<{config.train_config.new_dynamic}>",
        )
        # self.accuracy_over_time_discrete = {
        #     key: [] for key in reward_plot_keys_discrete
        # }
        # self.accuracy_over_time_continuous = {
        #     key: [] for key in reward_plot_keys_continuous
        # }
        # # Note total refers to over all agents
        # self.total_true_rewards_discrete: dict[RewardKeys, int] = {
        #     key: 0 for key in reward_plot_keys_discrete
        # }
        # self.total_true_rewards_continuous: dict[RewardKeys, int] = {
        #     key: 0 for key in reward_plot_keys_continuous
        # }
        # self.total_predicted_rewards_discrete: dict[RewardKeys, int] = {
        #     key: 0 for key in reward_plot_keys_discrete
        # }
        # self.total_predicted_rewards_continuous: dict[RewardKeys, int] = {
        #     key: 0 for key in reward_plot_keys_continuous
        # }
        # self.loss_over_time = []
        # self.prediction_error_over_time = []

    # --- THIS IS THE ONLY METHOD WE NEED TO OVERRIDE ---
    def build_experiment(  # FIXME we should refactor this because this is improper override.
        self,
        agent_controller_cls=AgentControllerUsingPolicy,
        agent_type=RewardAgent,
        **kwargs,
    ):
        # We are overriding the build_experiment from the parent Algorithm class
        # to inject our specific CNN components.

        # 1. This is needed because envstep uses self.agent_controller
        self.agent_controller = AgentControllerUsingPolicy(
            self._agents_list, critic_view_controller=None
        )

        # --- 2. Create the CNNs for the agents ---
        # We manually create the list of networks here.
        networks: list[CNN] = []
        for _ in range(self.train_config.num_agents):
            net = CNN(
                input_channels=3,
                height=self.env_config.width,
                width=self.env_config.length,
                alpha=self.train_config.alpha,
                mlp_hidden_features=self.train_config.hidden_dimensions,
            )
            networks.append(net)

        # --- 3. Initialize the agents and give them their CNNs ---
        self._init_agents_for_training(
            agent_type,
            value_networks=[],  # No value networks
            actor_networks=[],  # No actor networks
            reward_networks=networks,  # Give them the CNNs
        )

        # --- 4. Create the environment ---
        # This logic is the same as the parent.
        if not self.train_config.test:
            self.env: Orchard = create_env(
                self.env_config,
                self.train_config.num_agents,
                *self.restore_all() if self.train_config.skip else (None, None),
                self._agents_list,
                self.env_cls,
                debug=self.debug,
            )

    @override
    def init_agents_for_eval(
        self,
    ) -> tuple[list[RewardAgent], AgentController]:  # FIXME we should
        # refactor the signature to be more generic but this works for now.
        """
        Creates a new set of agents for evaluation and assigns them the
        currently trained networks.

        Precondition: self.agents_list is populated with trained agents.
        """
        eval_agents = []
        info = self.agent_info

        # self.agents_list is the property that returns the typed list from self._agents_list
        # It contains the agents with the networks that have been training.
        for i, trained_agent in enumerate(self.agents_list):
            info.agent_id = i

            # Create a fresh agent for the evaluation environment
            eval_agent = RewardAgent(info)

            # Get the trained network from the corresponding agent in the main list
            eval_agent.reward_network = trained_agent.reward_network

            eval_agents.append(eval_agent)

        eval_controller = AgentControllerUsingPolicy(
            eval_agents, critic_view_controller=None
        )

        return eval_agents, eval_controller

    def calculate_average_prediction_error(self, eval_agents_list):
        total_prediction_error = sum(
            sum(ag.prediction_metrics[RewardKeys.PREDICTION_ERRORS])
            for ag in eval_agents_list
        )
        total_counts = sum(
            len(ag.prediction_metrics[RewardKeys.PREDICTION_ERRORS])
            for ag in eval_agents_list
        )
        if total_counts > 0:
            avg_prediction_error = total_prediction_error / total_counts
        else:
            avg_prediction_error = 0.0
        return avg_prediction_error

    def generate_plots(self):  # BOOKMARK
        """Generates plots for training loss, final prediction accuracy."""
        output_graph_dir = FINAL_DIR / f"{self.name}_{date.today()}"
        output_graph_dir.mkdir(parents=True, exist_ok=True)
        # Create an x-axis representing the evaluation checkpoints
        num_evaluations = len(self.prediction_error_over_time)
        if num_evaluations == 0:
            print("No evaluation data to plot.")
            return

        eval_steps = [x for x in range(0, num_evaluations)]

        # --- PLOT 1: TRAINING LOSS CURVE (one per agent) ---
        for i, agent in enumerate(self.agents_list):
            network = agent.reward_network
            plt.figure(figsize=(10, 6))
            assert isinstance(network, CNN)
            # plot raw data
            plt.plot(
                range(len(network.loss_history)),
                network.loss_history,
                label="Raw Loss",
                alpha=0.3,
            )
            plt.title(f"Training Loss Curve for Agent {i}")
            plt.xlabel("eval step")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.savefig(output_graph_dir / f"Training_Loss_Agent_{i}.png")
            plt.close()
        # --- PLOT 2: ACCURACY BY REWARD TYPE OVER TIME ---
        plt.figure(figsize=(10, 6))

        # Determine which set of keys to use based on the environment
        DISCRETE_ENVS = [OrchardBasic, Orchard, OrchardBasicNewDynamic]
        is_discrete = type(self.env) in DISCRETE_ENVS

        if is_discrete:
            accuracy_history = self.accuracy_over_time_discrete
            keys_to_plot = reward_plot_keys_discrete
        else:
            accuracy_history = self.accuracy_over_time_continuous
            keys_to_plot = reward_plot_keys_continuous

        for key in keys_to_plot:
            if key in accuracy_history and accuracy_history[key]:
                plt.plot(
                    eval_steps,
                    accuracy_history[key],
                    label=f"Accuracy ({key.value})",
                    marker="o",
                    linestyle="--",
                )

        plt.title("Prediction Accuracy Over Training")
        plt.xlabel("Training Timestep")
        plt.ylabel("Accuracy")
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.grid(True)
        plt.savefig(output_graph_dir / "Accuracy_over_Time.png")
        plt.close()

        # --- PLOT 3: AVERAGE PREDICTION ERROR OVER TIME ---
        if self.prediction_error_over_time:
            plt.figure(figsize=(10, 6))
            plt.plot(
                eval_steps,
                self.prediction_error_over_time,
                label="Avg. Prediction Error",
                marker="o",
            )
            plt.axhline(
                0, color="black", linestyle="--", linewidth=1, label="Ideal (No Bias)"
            )
            plt.title("Average Prediction Error Over Training")
            plt.xlabel("Training Timestep")
            plt.ylabel("Avg. Error (Prediction - True Reward)")
            plt.legend()
            plt.grid(True)
            plt.savefig(output_graph_dir / "Average_Error_over_Time.png")
            plt.close()

        # --- PLOT 4: HISTOGRAM OF TOTAL TRUE REWARDS AND TOTAL PREDICTED REWARDS ---
        if is_discrete:
            true_rewards = self.total_true_rewards_discrete
            predicted_rewards = self.total_predicted_rewards_discrete
        else:
            true_rewards = self.total_true_rewards_continuous
            predicted_rewards = self.total_predicted_rewards_continuous
        true_reward_names = [key.value for key in true_rewards.keys()]
        true_reward_values = [true_rewards[key] for key in true_rewards.keys()]
        predicted_names = [key.value for key in predicted_rewards.keys()]
        predicted_values = [predicted_rewards[key] for key in predicted_rewards.keys()]

        # Create the figure and axes
        fig, ax = plt.subplots()

        # Plot true rewards as one set of bars
        ax.bar(
            true_reward_names, true_reward_values, color="blue", label="True Rewards"
        )

        # Plot predicted rewards as another set of bars
        # We use the same x-axis labels for comparison
        ax.bar(
            predicted_names, predicted_values, color="orange", label="Predicted Rewards"
        )

        # Add a legend
        ax.legend()

        # Set an appropriate title and labels
        ax.set_title("Total True vs Predicted Rewards")
        ax.set_xlabel("Reward Categories")
        ax.set_ylabel("Total Rewards")

        plt.savefig(output_graph_dir / "Total_Rewards_Bar_Chart.png")
        plt.close()
