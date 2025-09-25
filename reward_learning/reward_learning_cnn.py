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
from config import GRAPHS_DIR
from configs.config import ExperimentConfig
from helpers.controllers import AgentController, AgentControllerUsingPolicy
from helpers.helpers import (
    create_env,
)


from main import eval_performance
from models.reward_cnn import RewardCNN
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
        self.accuracy_over_time_discrete = {
            key: [] for key in reward_plot_keys_discrete
        }
        self.accuracy_over_time_continuous = {
            key: [] for key in reward_plot_keys_continuous
        }
        # Note total refers to over all agents
        self.total_true_rewards_discrete: dict[RewardKeys, int] = {
            key: 0 for key in reward_plot_keys_discrete
        }
        self.total_true_rewards_continuous: dict[RewardKeys, int] = {
            key: 0 for key in reward_plot_keys_continuous
        }
        self.total_predicted_rewards_discrete: dict[RewardKeys, int] = {
            key: 0 for key in reward_plot_keys_discrete
        }
        self.total_predicted_rewards_continuous: dict[RewardKeys, int] = {
            key: 0 for key in reward_plot_keys_continuous
        }
        self.loss_over_time = []
        self.prediction_error_over_time = []

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
        networks: list[RewardCNN] = []
        for _ in range(self.train_config.num_agents):
            net = RewardCNN(
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

    # @override
    # def collect_observation(self, step: int) -> None:
    #     """
    #     Collects observations and adds them to the CNN's replay buffer.
    #     This version passes the raw state directly to the network, which
    #     handles its own state processing.
    #     """
    #     try:
    #         for tick in range(self.train_config.num_agents):
    #             env_step_result: EnvStep = self.env_step(tick)

    #             # Use the property to get the correctly typed list
    #             for agent in self.agents_list:
    #                 # FIXME |_ This should also be agents with CNN reward network
    #                 reward = env_step_result.reward_vector[agent.id]

    #                 rewardcnn = agent.reward_network
    #                 assert isinstance(rewardcnn, RewardCNN), "Agent doesn't have a CNN!"

    #                 rewardcnn.add_experience_from_raw(
    #                     env_step_result.new_state, agent.position, reward
    #                 )

    #             if env_step_result.picked:
    #                 self.env.remove_apple(
    #                     self.agents_list[env_step_result.acting_agent_id].position
    #                 )
    #     except Exception as e:
    #         self.logger.error(f"Error collecting observations: {e}")
    #         raise

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

    # def _run_evaluation_simulation(
    #     self, eval_agents: list[RewardAgent], env: Orchard
    # ) -> None:
    #     """
    #     Runs a full evaluation simulation. This method MUTATES the eval_agents
    #     by filling in their prediction_metrics
    #     """
    #     print("--- Starting new, clean evaluation simulation ---")
    #     agent_controller = AgentControllerUsingPolicy(
    #         eval_agents, critic_view_controller=None
    #     )
    #     tol = 1e-1  # Tolerance for a correct prediction

    #     for _ in range(self.train_config.eval_timesteps):
    #         for tick in range(self.train_config.num_agents):
    #             # 1. ONE (random) AGENT TAKES ONE ACTION IN THE ENVIRONMENT
    #             # This is the event we are going to evaluate.
    #             acting_agent_idx = random.randint(0, env.n - 1)
    #             action = agent_controller.agent_get_action(env, acting_agent_idx, None)
    #             result = env.process_action(
    #                 acting_agent_idx,
    #                 eval_agents[acting_agent_idx].position.copy(),
    #                 action,
    #             )

    #             # 2. GET THE RESULTS OF THAT ACTION
    #             # This is the state AFTER the action, where agent and apple might overlap.
    #             post_action_state = env.get_state()
    #             # This is the ground-truth reward vector from that action.
    #             labels = result.reward_vector

    #             # 3. GET PREDICTIONS BASED ON THE POST-ACTION STATE
    #             # Now we ask each agent: "Looking at this new state, what reward do you think you just got?"
    #             reward_predictions = []
    #             for agent in eval_agents:
    #                 network = agent.reward_network
    #                 assert isinstance(network, RewardCNN)

    #                 # Use the network's internal helper to process the post-action state
    #                 processed_state = network._raw_state_to_nn_input(
    #                     post_action_state, agent.position
    #                 )
    #                 # Get the prediction for that processed state
    #                 prediction = network.get_model_reward_prediction(processed_state)
    #                 reward_predictions.append(prediction.item())

    #             tol = 1e-1
    #             for i, reward_agent in enumerate(eval_agents):
    #                 label = labels[i]
    #                 # populate discrete total rewards
    #                 if label == -1.0:
    #                     self.total_true_rewards_discrete[RewardKeys.NEG_ONE] += 1
    #                 elif label == 0.0:
    #                     self.total_true_rewards_discrete[RewardKeys.ZERO] += 1
    #                 elif label == 1.0:
    #                     self.total_true_rewards_discrete[RewardKeys.ONE] += 1
    #                 else:
    #                     self.total_true_rewards_discrete[RewardKeys.OTHER] += 1
    #                 # populate continuous total rewards
    #                 if label < 0.0:
    #                     self.total_true_rewards_continuous[
    #                         RewardKeys.LESS_THAN_ZERO
    #                     ] += 1
    #                 elif label > 0.0:
    #                     self.total_true_rewards_continuous[
    #                         RewardKeys.GREATER_THAN_ZERO
    #                     ] += 1
    #                 else:
    #                     self.total_true_rewards_continuous[RewardKeys.ZERO] += 1

    #                 predicted_reward = reward_predictions[i]
    #                 prediction_error = abs(label - predicted_reward)
    #                 is_correct = 1 if prediction_error <= tol else 0

    #                 reward_agent.prediction_metrics[
    #                     RewardKeys.PREDICTION_ERRORS
    #                 ].append(prediction_error)

    #                 if predicted_reward == -1.0:
    #                     discrete_key = RewardKeys.NEG_ONE
    #                 elif predicted_reward == 0.0:
    #                     discrete_key = RewardKeys.ZERO
    #                 elif predicted_reward == 1.0:
    #                     discrete_key = RewardKeys.ONE
    #                 else:
    #                     discrete_key = RewardKeys.OTHER

    #                 if predicted_reward < 0:
    #                     continuous_key = RewardKeys.LESS_THAN_ZERO
    #                 elif predicted_reward > 0:
    #                     continuous_key = RewardKeys.GREATER_THAN_ZERO
    #                 else:
    #                     continuous_key = RewardKeys.ZERO
    #                 reward_agent.prediction_metrics[RewardKeys.CORRECT] += is_correct
    #                 reward_agent.prediction_metrics[RewardKeys.TOTAL] += 1
    #                 # Update continuous metrics
    #                 reward_agent.prediction_metrics[RewardType.CONTINUOUS][
    #                     continuous_key
    #                 ][RewardKeys.CORRECT] += is_correct
    #                 reward_agent.prediction_metrics[RewardType.CONTINUOUS][
    #                     continuous_key
    #                 ][RewardKeys.TOTAL] += 1
    #                 # Update discrete metrics
    #                 reward_agent.prediction_metrics[RewardType.DISCRETE][discrete_key][
    #                     RewardKeys.CORRECT
    #                 ] += is_correct
    #                 reward_agent.prediction_metrics[RewardType.DISCRETE][discrete_key][
    #                     RewardKeys.TOTAL
    #                 ] += 1

    #             if result.picked:
    #                 env.remove_apple(eval_agents[acting_agent_idx].position.copy())

    #         env.apples_despawned += env.despawn_algorithm(env, env.despawn_rate)
    #         env.total_apples += env.spawn_algorithm(env, env.spawn_rate)

    # @override
    # def run_inference(self) -> EvalResult:
    #     """Custom inference method that uses our new, clean evaluation loop."""
    #     try:
    #         eval_agents_list, _ = self.init_agents_for_eval()

    #         # 2. Create a fresh environment.
    #         env: OrchardBasic = create_env(
    #             self.env_config,
    #             self.train_config.num_agents,
    #             None,
    #             None,
    #             eval_agents_list,  # Use the eval agents
    #             self.env_cls,
    #         )

    #         # 3. Run the simulation. This will populate the accuracy counters on the eval_agents.
    #         with torch.no_grad():
    #             self._run_evaluation_simulation(eval_agents_list, env)

    #         # 5. Create the final result object for the main log file.
    #         #    We get the metrics from the environment that was just run.
    #         total_picked = env.total_picked
    #         num_agents = env.n
    #         total_apples = env.total_apples

    #         # fill self.accuracy_over_time_discrete and self.accuracy_over_time_continuous, and loss_over_time and prediction_error_over_time
    #         # by averaging over all eval_agents
    #         for key in reward_plot_keys_discrete:
    #             if key == RewardKeys.PREDICTION_ERRORS:
    #                 continue
    #             total_correct = sum(
    #                 ag.prediction_metrics[RewardType.DISCRETE][key][RewardKeys.CORRECT]
    #                 for ag in eval_agents_list
    #             )
    #             total_preds = sum(
    #                 ag.prediction_metrics[RewardType.DISCRETE][key][RewardKeys.TOTAL]
    #                 for ag in eval_agents_list
    #             )
    #             accuracy = total_correct / total_preds if total_preds > 0 else 0.0
    #             self.accuracy_over_time_discrete[key].append(accuracy)

    #         for key in reward_plot_keys_continuous:
    #             if key == RewardKeys.PREDICTION_ERRORS:
    #                 continue
    #             total_correct = sum(
    #                 ag.prediction_metrics[RewardType.CONTINUOUS][key][
    #                     RewardKeys.CORRECT
    #                 ]
    #                 for ag in eval_agents_list
    #             )
    #             total_preds = sum(
    #                 ag.prediction_metrics[RewardType.CONTINUOUS][key][RewardKeys.TOTAL]
    #                 for ag in eval_agents_list
    #             )
    #             accuracy = total_correct / total_preds if total_preds > 0 else 0.0
    #             self.accuracy_over_time_continuous[key].append(accuracy)

    #         # PREDICTION ERRORS
    #         avg_prediction_error = self.calculate_average_prediction_error(
    #             eval_agents_list
    #         )
    #         self.prediction_error_over_time.append(avg_prediction_error)

    #         return EvalResult(
    #             total_apples=total_apples,
    #             total_picked=total_picked,
    #             picked_per_agent=total_picked / num_agents if num_agents > 0 else 0,
    #             per_agent=(
    #                 (total_picked / num_agents) / (total_apples / num_agents)
    #                 if total_apples > 0
    #                 else 0
    #             ),
    #             # These other metrics are not calculated in our simple loop, so we can default them.
    #             average_distance=0.0,
    #             apple_per_sec=0.0,
    #             nearest_actions=0,
    #             idle_actions=0,
    #         )
    #     except Exception as e:
    #         self.logger.error(f"Error during inference: {e}")
    #         raise

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
        output_graph_dir = GRAPHS_DIR / f"{self.name}_{date.today()}"
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
            assert isinstance(network, RewardCNN)
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
