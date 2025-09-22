from datetime import date
from matplotlib import pyplot as plt
import torch
from typing_extensions import override


from agents.reward_agent import RewardAgent
from algorithm import EnvStep, EvalResult
from config import GRAPHS_DIR
from configs.config import ExperimentConfig
from helpers.controllers import AgentController, AgentControllerUsingPolicy
from helpers.helpers import (
    create_env,
    step_and_evaluate_reward_prediction_accuracy_cnn_decentralized,
)


from main import eval_performance
from models.reward_cnn import RewardCNN
from orchard.environment import Orchard
from plots import plot_smoothed
from reward_learning.reward_learning import RewardLearningDecentralized


class RewardLearningCNNDecentralized(RewardLearningDecentralized):
    """
    An algorithm for decentralized reward learning that uses a Convolutional
    Neural Network (CNN) to process spatial state information.
    """

    def __init__(self, config: ExperimentConfig):
        # Call the parent __init__ to set up the name and data structures
        super().__init__(config)

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

    @override
    def collect_observation(self, step: int) -> None:
        """
        Collects observations and adds them to the CNN's replay buffer.
        This version passes the raw state directly to the network, which
        handles its own state processing.
        """
        try:
            for tick in range(self.train_config.num_agents):
                env_step_result: EnvStep = self.env_step(tick)

                # Use the property to get the correctly typed list
                for agent in self.agents_list:
                    # FIXME |_ This should also be agents with CNN reward network
                    reward = env_step_result.reward_vector[agent.id]

                    rewardcnn = agent.reward_network
                    assert isinstance(rewardcnn, RewardCNN), "Agent doesn't have a CNN!"

                    rewardcnn.add_experience_from_raw(
                        env_step_result.new_state, agent.position, reward
                    )

                if env_step_result.picked:
                    self.env.remove_apple(
                        self.agents_list[env_step_result.acting_agent_id].position
                    )
        except Exception as e:
            self.logger.error(f"Error collecting observations: {e}")
            raise

    @override
    def init_agents_for_eval(
        self,
    ) -> tuple[list[RewardAgent], AgentController]:  # FIXME we should
        # refactor the signature to be more generic but this works for now.
        """
        Creates a new set of agents for evaluation and assigns them the
        currently trained networks.
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

    @override
    def run_inference(self) -> EvalResult:
        """Custom inference to use the custom

        Returns:
            _description_
        """
        agents_list, agent_controller = self.init_agents_for_eval()

        env = create_env(
            self.env_config,
            self.train_config.num_agents,
            None,
            None,
            agents_list,
            self.env_cls,
        )

        with torch.no_grad():
            try:
                results = eval_performance(
                    num_agents=self.train_config.num_agents,
                    agent_controller=agent_controller,
                    env=env,
                    name=self.name,
                    agents_list=agents_list,
                    timesteps=self.train_config.eval_timesteps,
                    epsilon=self.train_config.epsilon,
                    env_step=step_and_evaluate_reward_prediction_accuracy_cnn_decentralized,
                )
            except Exception as e:
                print(f"Error during inference: {e}")
                raise
        self.process_accuracy(agents_list)

        return EvalResult(*results)

    @override
    def _generate_plots(self):  # BOOKMARK
        """Generates plots for the CNN reward learning experiment."""
        # This logic is simple because it's specific to this algorithm.
        for i, agent in enumerate(self.agents_list):
            dir_to_save = GRAPHS_DIR / date.today().isoformat() / self.name
            dir_to_save.mkdir(parents=True, exist_ok=True)
            network = agent.reward_network
            if isinstance(network, RewardCNN) and network.loss_history:
                plt.figure()
                plt.plot(network.loss_history, label=f"Agent {i} Raw Loss")
                plt.yscale("log")
                plt.title("Raw Training Loss")
                plt.grid(True)
                plt.savefig(dir_to_save / f"Agent_{i}_Raw_Training_Loss.png")
                plt.close()
