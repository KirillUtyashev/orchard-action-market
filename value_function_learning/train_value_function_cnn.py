from abc import abstractmethod
from random import randint
from matplotlib import pyplot as plt
from typing_extensions import override
from agents.communicating_agent import CommAgent
from agents.reward_agent import RewardAgent
from agents.simple_agent import SimpleAgent
from algorithm import Algorithm, EnvStep
from helpers.controllers import (
    AgentControllerCentralized,
    AgentControllerUsingPolicy,
    ViewController,
)
from helpers.helpers import create_env
from models.cnn import CNN
from models.value_cnn import ValueCNNCentralized, ValueCNNDecentralized
from models.value_function import VNetwork
from orchard.environment import Orchard
from plots import add_to_plots, plot_hybrid_smoothed
from policies.cnn_controllers import (
    AgentControllerCentralizedCNN,
    AgentControllerDecentralizedCNN,
)


class ValueFunctionCNNAlgorithm(Algorithm):
    """Base class for CNN-based value function learning algorithms."""

    def update_lr(self, step: int) -> None:
        pass

    @abstractmethod
    def step_and_collect_observation(self, step: int) -> None:
        """Step and collects observations from the environment step result and adds them to the appropriate networks."""
        pass

    @override
    def log_progress(self, sample_state, sample_state5, sample_state6):
        network = self.network_for_eval[0]

        # Centralized and Decentralized networks have different state processing needs
        def get_val(state):
            if isinstance(network, ValueCNNCentralized):
                processed = network.raw_state_to_nn_input(state)
            else:  # Decentralized
                processed = network.raw_state_to_nn_input(
                    state, agent_pos=state["poses"][0]
                )
            return network.get_value_function(processed)

        v1, v2, v3 = (
            get_val(sample_state),
            get_val(sample_state5),
            get_val(sample_state6),
        )
        add_to_plots(network.state_dict(), self.weights_plot)
        print(f"Sample State Values: {v1:.4f}, {v2:.4f}, {v3:.4f}")
        self.loss_plot.append(v1)
        self.loss_plot5.append(v2)
        self.loss_plot6.append(v3)

    @override
    def generate_plots(self):
        """
        Generates and saves plots specific to this algorithm's training progress.
        This provides a clean alternative to the generic `graph_plots` function.
        """
        # --- 1. Plot the Critic's Training Loss ---
        if self.critic_loss:
            fig = plot_hybrid_smoothed(
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

        # --- 3. Plot the Apple Count History ---
        if self.apple_count_history:
            fig = plot_hybrid_smoothed(
                [self.apple_count_history],
                labels=["Apple Count"],
                title="Number of Apples in Orchard During Training",
                xlabel="Training Step",
                ylabel="Total Apples",
            )
            ax = fig.gca()
            ax.grid(True)
            # save the figure
            fig.savefig(self.graphs_out_path / "Apple_Count_During_Training.png")
            plt.close(fig)


class CentralizedValueCNNAlgorithm(ValueFunctionCNNAlgorithm):
    """Algorithm class for centralized Critique learning using CNN model."""

    def __init__(self, config):
        name = f"CentralizedValueCNN-{config.train_config.num_agents}-agents"
        super().__init__(config, name)

    @override
    def _init_critic_networks(self, value_network_cls=None):
        """Returns an array of ValueCNN of length num_agents. However since this is centrealized, we only care about index 0 of the array.

        Args:
            value_network_cls: Only included for compatibility. Do not use.

        Returns:
            Array of ValueCNNCentralized of length num_agents.
        """
        network = ValueCNNCentralized(
            self.env_config.length,
            self.env_config.width,
            self.train_config.alpha,
            self.train_config.discount,
            self.train_config.hidden_dimensions,
            self.train_config.num_layers,
        )
        return [network for _ in range(self.train_config.num_agents)]

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

    # TODO base class signature should not be so strict
    @override
    def build_experiment(self, **kwargs):
        """
        custom build experiment for cnn centralized value function

        **kwargs not needed
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
    def step_and_collect_observation(self, step: int) -> None:
        """
        Takes the result of a single, clean environment step and adds the
        corresponding experience to the training buffer.
        """
        assert isinstance(self._agents_list[0].policy_value, ValueCNNCentralized)
        valueCNN: ValueCNNCentralized = self._agents_list[0].policy_value
        for tick in range(self.train_config.num_agents):
            env_step_result = self.single_agent_env_step(tick)
            reward = sum(env_step_result.reward_vector)
            processed_state = valueCNN.raw_state_to_nn_input(env_step_result.old_state)
            processed_new_state = valueCNN.raw_state_to_nn_input(
                env_step_result.new_state
            )
            valueCNN.add_experience(processed_state, processed_new_state, reward)


class DecentralizedValueFunctionCNNAlgorithm(ValueFunctionCNNAlgorithm):
    def __init__(self, config):
        name = f"DecentralizedValueCNN-{config.train_config.num_agents}-agents"
        super().__init__(config, name)

    @override
    def step_and_collect_observation(self, step: int) -> None:
        """Implements the multi-tick loop for decentralized experience collection."""
        for tick in range(self.train_config.num_agents):
            env_step_result = self.single_agent_env_step(tick)
            for i, agent in enumerate(self._agents_list):
                assert isinstance(agent.policy_value, ValueCNNDecentralized)
                network: ValueCNNDecentralized = agent.policy_value
                reward = env_step_result.reward_vector[i]
                processed_state = network.raw_state_to_nn_input(
                    env_step_result.old_state,
                    agent_pos=env_step_result.old_positions[i],
                )
                processed_new_state = network.raw_state_to_nn_input(
                    env_step_result.new_state, agent_pos=agent.position
                )
                network.add_experience(processed_state, processed_new_state, reward)

    def _init_critic_networks(self, value_network_cls=None):
        networks = []
        for _ in range(self.train_config.num_agents):
            net = ValueCNNDecentralized(
                self.env_config.length,
                self.env_config.width,
                self.train_config.alpha,
                self.train_config.discount,
                self.train_config.hidden_dimensions,
                self.train_config.num_layers,
            )
            networks.append(net)
        return networks

    @override
    def build_experiment(self, **kwargs):
        critic_networks = self._init_critic_networks()
        self._init_agents_for_training(CommAgent, critic_networks, [], [])
        self.agent_controller = AgentControllerDecentralizedCNN(
            self._agents_list, test=self.train_config.test
        )
        self.env = create_env(
            self.env_config,
            self.train_config.num_agents,
            None,
            None,
            self._agents_list,
            self.env_cls,
            debug=self.debug,
        )
        self.network_for_eval = [agent.policy_value for agent in self._agents_list]

    @override
    def init_agents_for_eval(self):
        a_list = []
        info = self.agent_info
        for i, trained_agent_base in enumerate(self._agents_list):
            info.agent_id = i
            eval_agent = CommAgent(info)
            eval_agent.policy_value = self.network_for_eval[i]
            a_list.append(eval_agent)
        controller = AgentControllerDecentralizedCNN(
            a_list, test=self.train_config.test
        )
        return a_list, controller
