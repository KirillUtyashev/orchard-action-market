from abc import ABC
from typing import List, Tuple
from matplotlib import pyplot as plt
from typing_extensions import override
import numpy as np
from agents.agent import Agent
from configs.config import ExperimentConfig
from helpers.controllers import (
    AgentController,
    AgentControllerCentralized,
    AgentControllerDecentralized,
    AgentControllerDecentralizedPersonal,
    AgentControllerValue,
    ViewController,
)
from helpers.helpers import get_discounted_value
from models.actor_network import ActorNetwork
from models.value_cnn import ValueCNNCentralized
from models.value_function import VNetwork
from agents.simple_agent import SimpleAgent
from agents.communicating_agent import CommAgent
from algorithm import Algorithm
from orchard.environment import OrchardBasic
from plots import graph_plots, plot_hybrid_smoothed
from value_function_learning.train_value_function_cnn import ValueFunctionCNNAlgorithm


class ValueFunction(Algorithm, ABC):
    """Base class for value function implementations."""

    def __init__(self, config: ExperimentConfig, name):
        """Initialize the value function algorithm."""
        super().__init__(config, name)

    def update_lr(self, step: int) -> None:
        """Update a learning rate based on training progress."""
        pass

    def _init_actor_networks(self, actor_network_cls=ActorNetwork):
        return []

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


class CentralizedValueFunction(ValueFunction):
    """Centralized implementation of a value function."""

    def __init__(self, config: ExperimentConfig):
        """Initialize the value function algorithm."""
        super().__init__(
            config,
            f"CentralizedValue",
        )

    def init_agents_for_eval(self) -> Tuple[List[SimpleAgent], AgentController]:
        a_list = []
        info = self.agent_info
        for ii in range(1, len(self._agents_list) + 1):
            info.agent_id = ii
            trained_agent = SimpleAgent(info)
            trained_agent.policy_value = self.network_for_eval[0]
            a_list.append(trained_agent)
        return a_list, AgentControllerCentralized(a_list, self.critic_view_controller)

    def step_and_collect_observation(self, step: int) -> None:
        """Collect observations."""
        try:
            for tick in range(self.train_config.num_agents):
                env_step_result = self.single_agent_env_step(tick)

                processed_state = self.critic_view_controller.state_to_nn_input(
                    env_step_result.old_state, None, None
                )
                processed_new_state = self.critic_view_controller.state_to_nn_input(
                    env_step_result.new_state, None, None
                )

                # Add rewards here
                rewards_sum = np.sum(env_step_result.reward_vector)
                self._agents_list[env_step_result.acting_agent_id].add_experience(
                    processed_state[: self.network_for_eval[0].get_input_dim()],
                    processed_new_state[: self.network_for_eval[0].get_input_dim()],
                    rewards_sum,
                )

                if self.train_config.new_dynamic:
                    self.env.remove_apple(
                        self._agents_list[env_step_result.acting_agent_id].position
                    )
        except Exception as e:
            self.logger.error(f"Error collecting observations: {e}")
            raise

    def _init_critic_networks(self, value_network_cls=VNetwork) -> list:
        # Get critic network vision
        if self.train_config.critic_vision != 0:
            if self.env_config.width != 1:
                critic_input_dim = self.train_config.critic_vision**2
            else:
                critic_input_dim = self.train_config.critic_vision
        else:
            critic_input_dim = 2 * self.env_config.length * self.env_config.width
        network = value_network_cls(
            critic_input_dim,
            1,
            self.train_config.alpha,
            self.train_config.discount,
            self.train_config.hidden_dimensions,
            self.train_config.num_layers,
        )
        return [network for _ in range(self.train_config.num_agents)]

    def build_experiment(
        self,
        view_controller_cls=ViewController,
        agent_controller_cls: AgentControllerValue = AgentControllerCentralized,  # type: ignore
        agent_type=SimpleAgent,
        value_network_cls=VNetwork,
        **kwargs,
    ):
        super().build_experiment(
            view_controller_cls,
            agent_controller_cls,
            agent_type,
            value_network_cls,
            **kwargs,
        )
        self.network_for_eval = [self._agents_list[0].policy_value]


class DecentralizedValueFunction(ValueFunction):
    """Decentralized implementation of a value function."""

    def __init__(self, config: ExperimentConfig, name=None):
        """Initialize the value function algorithm."""
        if name is None:
            super().__init__(
                config,
                f"DecentralizedValue",
            )
        else:
            super().__init__(config, name)
        self.network_list = []

    def build_experiment(
        self,
        view_controller_cls=ViewController,
        agent_controller_cls=AgentControllerDecentralized,
        agent_type=CommAgent,
        value_network_cls=VNetwork,
        actor_network_cls=ActorNetwork,
        **kwargs,
    ):
        super().build_experiment(
            view_controller_cls,
            agent_controller_cls,
            agent_type,
            value_network_cls,
            **kwargs,
        )
        for agent in self._agents_list:
            self.network_list.append(agent.policy_value)
        self.network_for_eval = self.network_list

    def init_agents_for_eval(
        self,
    ) -> Tuple[List[CommAgent], AgentControllerDecentralized]:
        a_list = []
        info = self.agent_info
        for ii in range(len(self._agents_list)):
            info.agent_id = ii
            trained_agent = CommAgent(info)
            trained_agent.policy_value = self.network_list[ii]
            a_list.append(trained_agent)
        return a_list, AgentControllerDecentralized(a_list, self.critic_view_controller)

    def step_and_collect_observation(self, step: int) -> None:
        """Collect observations for decentralized training."""
        try:
            for tick in range(self.train_config.num_agents):
                env_step_result = self.single_agent_env_step(tick)
                for each_agent in range(len(self._agents_list)):
                    reward = env_step_result.reward_vector[each_agent]
                    processed_state = self.critic_view_controller.state_to_nn_input(
                        env_step_result.old_state,
                        env_step_result.old_positions[each_agent],
                        each_agent + 1,
                    )
                    processed_new_state = self.critic_view_controller.state_to_nn_input(
                        env_step_result.new_state,
                        self._agents_list[each_agent].position,
                        each_agent + 1,
                    )

                    self._agents_list[each_agent].add_experience(
                        processed_state, processed_new_state, reward
                    )

                if self.train_config.new_dynamic:
                    self.env.remove_apple(
                        self._agents_list[env_step_result.acting_agent_id].position
                    )
        except Exception as e:
            self.logger.error(f"Error collecting observations: {e}")
            raise


class DecentralizedValueFunctionPersonal(DecentralizedValueFunction):
    def __init__(self, config: ExperimentConfig):
        super().__init__(
            config,
            f"DecentralizedPersonal-<{config.train_config.num_agents}>_agents-_length-<{config.env_config.length}>_width-<{config.env_config.width}>_s_target-<{config.env_config.s_target}>-alpha-<{config.train_config.alpha}>-apple_mean_lifetime-<{config.env_config.apple_mean_lifetime}>-<{config.train_config.hidden_dimensions}>-<{config.train_config.num_layers}>-vision-<{config.train_config.critic_vision}>-batch_size-<{config.train_config.batch_size}>-env-<{config.env_config.env_cls}>",
        )

    def init_agents_for_eval(
        self,
    ) -> Tuple[List[SimpleAgent], AgentControllerDecentralizedPersonal]:
        a_list = []
        info = self.agent_info
        for ii in range(len(self._agents_list)):
            info.agent_id = ii
            trained_agent = SimpleAgent(info)
            trained_agent.policy_value = self.network_list[ii]
            a_list.append(trained_agent)
        return a_list, AgentControllerDecentralizedPersonal(
            a_list, self.critic_view_controller
        )

    def build_experiment(
        self,
        view_controller_cls=ViewController,
        agent_controller_cls=AgentControllerDecentralizedPersonal,
        agent_type=SimpleAgent,
        value_network_cls=VNetwork,
        actor_network_cls=ActorNetwork,
        **kwargs,
    ):
        super().build_experiment(
            view_controller_cls,
            agent_controller_cls,
            agent_type,
            value_network_cls,
            **kwargs,
        )
