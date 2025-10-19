from matplotlib import pyplot as plt
from typing_extensions import override
from agents.reward_agent import RewardAgent
from agents.simple_agent import SimpleAgent
from algorithm import Algorithm
from helpers.controllers import (
    AgentControllerCentralized,
    AgentControllerUsingPolicy,
    ViewController,
)
from helpers.helpers import create_env
from models.cnn import CNN
from models.value_cnn import ValueCNNCentralized
from models.value_function import VNetwork
from orchard.environment import Orchard
from plots import add_to_plots, plot_smoothed
from policies.cnn_controllers import AgentControllerCentralizedCNN
from value_function_learning.train_value_function import CentralizedValueFunction


# inherits algorithm
class CentralizedValueCNNAlgorithm(CentralizedValueFunction):
    """Algorithm class for centralized Critique learning using CNN model."""

    @override
    def log_progress(self, sample_state, sample_state5, sample_state6):
        """
        Custom implementation of log_progress for the CNN-based algorithm.
        It tracks the value of three fixed states over time to monitor convergence,
        but uses the CNN's own methods for state processing.
        """
        # Get a reference to the single, shared network
        network: ValueCNNCentralized = self.network_for_eval[0]

        # 1. Process the three sample states using the CNN's method
        processed_state1 = network.raw_state_to_nn_input(sample_state)
        processed_state2 = network.raw_state_to_nn_input(sample_state5)
        processed_state3 = network.raw_state_to_nn_input(sample_state6)

        # 2. Get the value for each processed state directly from the network
        v_value1 = network.get_value_function(processed_state1)
        v_value2 = network.get_value_function(processed_state2)
        v_value3 = network.get_value_function(processed_state3)

        # 3. Log network weights for plotting (the original code did this)
        add_to_plots(network.state_dict(), self.weights_plot)

        print(f"Sample State Values: {v_value1:.4f}, {v_value2:.4f}, {v_value3:.4f}")

        # 4. Append the values to the lists for plotting
        self.loss_plot.append(v_value1)
        self.loss_plot5.append(v_value2)
        self.loss_plot6.append(v_value3)

    @override
    def generate_plots(self):
        """
        Generates and saves plots specific to this algorithm's training progress.
        This provides a clean alternative to the generic `graph_plots` function.
        """
        # --- 1. Plot the Critic's Training Loss ---
        if self.critic_loss:
            fig = plot_smoothed(
                [self.critic_loss],
                labels=["Critic Training MSE Loss"],
                title="Critic Training Loss (Smoothed)",
                xlabel="Training Step",
                ylabel="MSE Loss",
            )
            # Use a log scale for loss plots, it's almost always better
            ax = fig.gca()
            ax.set_yscale("log")
            ax.grid(True)
            # self.graphs_out_path is defined in the base Algorithm class
            fig.savefig(self.graphs_out_path / "Critic_Training_Loss.png")
            plt.close(fig)

        # --- 2. Plot the Sample State Convergence ---
        if self.loss_plot:  # Check if log_progress was actually run
            plt.figure(figsize=(10, 6))
            plt.plot(self.loss_plot, label="Sample State 1 Value")
            plt.plot(self.loss_plot5, label="Sample State 2 Value")
            plt.plot(self.loss_plot6, label="Sample State 3 Value")
            plt.title("Value of Fixed Sample States During Training")
            plt.xlabel("Logging Step")
            plt.ylabel("Predicted Value")
            plt.legend()
            plt.grid(True)
            plt.savefig(self.graphs_out_path / "Sample_State_Values.png")
            plt.close()

    @override
    def build_experiment(self, **kwargs):
        """
        custom build experiment for cnn centralized value function
        """
        # 1. Initialize our CNN critic network.
        critic_networks = self._init_critic_networks()

        # 2. This creates self._agents_list where each agent is a SimpleAgent with policy_value set to ValueCNNCentralized
        self._init_agents_for_training(SimpleAgent, critic_networks, [], [])

        # 3. Initialize OUR agent controller. ignore test flag.
        self.agent_controller = AgentControllerCentralizedCNN(
            self._agents_list, test=self.train_config.test
        )

        # 4. Create the environment.
        if not self.train_config.test:
            self.env = create_env(
                self.env_config,
                self.train_config.num_agents,
                None,  # agent_pos
                None,  # apples
                self._agents_list,
                self.env_cls,
                debug=self.debug,
            )

        # 5. Set up the network for evaluation, consistent with the parent class.
        self.network_for_eval: list[ValueCNNCentralized] = [
            self._agents_list[0].policy_value  # type: ignore
        ]  # policy_value is the CNN value network type ValueCNNCentralized.

    @override
    def _init_critic_networks(self, value_network_cls=None):
        """Returns an array of ValueCNN of length num_agents. However since this is centrealized, we only care about index 0 of the array.

        Args:
            value_network_cls: Only included for compatibility. Do not use.

        Returns:
            Array of ValueCNNCentralized of length num_agents.
        """
        network = ValueCNNCentralized(
            self.env_config.width,
            self.env_config.length,
            self.train_config.alpha,
            self.train_config.discount,
            mlp_hidden_features=self.train_config.hidden_dimensions,
            num_mlp_hidden_layers=self.train_config.num_layers,
        )
        return [network for _ in range(self.train_config.num_agents)]

    @override
    def step_and_collect_observation(self, step: int) -> None:
        """
        Randomly selects an agent to act, and adds the resulting experience (state, new_state, reward)
        to the centralized CNN's training buffer.
        """
        try:
            # We only need one network instance since it's centralized.
            valueCNN: ValueCNNCentralized = self._agents_list[0].policy_value
            for tick in range(self.train_config.num_agents):
                env_step_result = self.single_agent_env_step(tick)

                # In the centralized case, the reward for the system is the sum of all rewards.
                reward = sum(env_step_result.reward_vector)

                # No ViewController is needed.
                processed_state = valueCNN.raw_state_to_nn_input(
                    env_step_result.old_state
                )
                processed_new_state = valueCNN.raw_state_to_nn_input(
                    env_step_result.new_state
                )

                # Add the processed experience to the network's buffer.
                valueCNN.add_experience(processed_state, processed_new_state, reward)

        except Exception as e:
            self.logger.error(f"Error collecting observations for CNN: {e}")
            raise

    @override
    def init_agents_for_eval(self):
        """
        Overrides the parent method to set up agents and the CONTROLLER
        correctly for the evaluation phase.
        """
        # Create a fresh list of agents for the evaluation environment.
        a_list = []
        info = self.agent_info
        for i in range(self.train_config.num_agents):
            info.agent_id = i
            eval_agent = SimpleAgent(info)

            # Assign the single trained network to every evaluation agent.
            # self.network_for_eval was set up in build_experiment.
            eval_agent.policy_value = self.network_for_eval[0]
            a_list.append(eval_agent)

        # Instantiate OUR CNN-specific controller for the evaluation agents.
        # This controller knows how to work with the CNN and does not need a ViewController.
        controller = AgentControllerCentralizedCNN(a_list, test=self.train_config.test)

        return a_list, controller
