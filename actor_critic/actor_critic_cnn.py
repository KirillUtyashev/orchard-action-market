# In a new file: actor_critic/actor_critic_cnn.py

from matplotlib import pyplot as plt
import torch
from algorithm import Algorithm
from agents.actor_critic_agent import ACAgent  # Re-use the agent type
from models.value_cnn import ValueCNNDecentralized
from models.actor_cnn import ActorCNN
from plots import plot_hybrid_smoothed
from policies.cnn_controllers import AgentControllerActorCriticCNN
from helpers.helpers import create_env


class ActorCriticCNNAlgorithm(Algorithm):
    def __init__(self, config):
        name = f"ActorCriticCNN"
        self.actor_loss = []
        self.avg_advantage_history = []
        super().__init__(config, name)

    def build_experiment(self, **kwargs):
        # 1. Initialize networks for each agent (Actor and Critic)
        num_actions = 5 if self.env_config.width > 1 else 3
        critic_networks = []
        actor_networks = []
        for _ in range(self.train_config.num_agents):
            critic_networks.append(
                ValueCNNDecentralized(
                    self.env_config.length,
                    self.env_config.width,
                    self.train_config.alpha,
                    self.train_config.discount,
                    self.train_config.hidden_dimensions,
                    self.train_config.num_layers,
                )
            )
            actor_networks.append(
                ActorCNN(
                    self.env_config.length,
                    self.env_config.width,
                    num_actions,
                    self.train_config.actor_alpha,
                    self.train_config.hidden_dimensions_actor,
                    self.train_config.num_layers_actor,
                )
            )

        # 2. Initialize agents and assign them their networks
        self._init_agents_for_training(ACAgent, critic_networks, actor_networks, [])

        # 3. Initialize the CNN-specific controller
        self.agent_controller = AgentControllerActorCriticCNN(self._agents_list)

        # 4. Create the environment
        self.env = create_env(
            self.env_config,
            self.train_config.num_agents,
            None,
            None,
            self._agents_list,
            self.env_cls,
            debug=self.debug,
        )
        # For evaluation
        self.network_for_eval = [agent.policy_network for agent in self._agents_list]

    def step_and_collect_observation(self, step: int):
        for tick in range(self.train_config.num_agents):
            env_step_result = self.single_agent_env_step(tick)

            if self.train_config.new_dynamic and env_step_result.picked:
                pos = self._agents_list[env_step_result.acting_agent_id].position
                self.env.remove_apple(pos)

            acting_agent_id = env_step_result.acting_agent_id
            acting_agent = self._agents_list[acting_agent_id]
            reward = env_step_result.reward_vector[
                acting_agent_id
            ]  # Simple case: agent's own reward
            assert isinstance(acting_agent, ACAgent)
            critic_net = acting_agent.policy_value
            actor_net = acting_agent.policy_network

            # Process states for the acting agent
            assert isinstance(critic_net, ValueCNNDecentralized)
            processed_old_state = critic_net.raw_state_to_nn_input(
                env_step_result.old_state,
                agent_pos=env_step_result.old_positions[acting_agent_id],
            )
            processed_new_state = critic_net.raw_state_to_nn_input(
                env_step_result.new_state, agent_pos=acting_agent.position
            )

            # Calculate Advantage
            with torch.no_grad():
                v_old = critic_net.get_value_function(processed_old_state)
                v_new = critic_net.get_value_function(processed_new_state)

            advantage = reward + self.train_config.discount * v_new - v_old

            # --- Add experiences to buffers ---
            # All agents' critics learn from all transitions (decentralized value learning)
            for i, agent in enumerate(self._agents_list):
                r_i = env_step_result.reward_vector[i]

                p_old = agent.policy_value.raw_state_to_nn_input(
                    env_step_result.old_state,
                    agent_pos=env_step_result.old_positions[i],
                )
                p_new = agent.policy_value.raw_state_to_nn_input(
                    env_step_result.new_state, agent_pos=agent.position
                )
                agent.policy_value.add_experience(p_old, p_new, r_i)

            # Only the acting agent's actor learns from this action
            actor_processed_state = actor_net.raw_state_to_nn_input(
                env_step_result.old_state,
                agent_pos=env_step_result.old_positions[acting_agent_id],
            )
            actor_net.add_experience(
                actor_processed_state, env_step_result.action, advantage
            )

    def init_agents_for_eval(self):
        """Prepares a fresh set of agents for the evaluation phase."""
        a_list = []
        info = self.agent_info
        for i, trained_agent in enumerate(self._agents_list):
            info.agent_id = i
            eval_agent = ACAgent(info)

            # Assign the trained networks to the evaluation agent
            eval_agent.policy_network = trained_agent.policy_network  # Actor
            eval_agent.policy_value = trained_agent.policy_value  # Critic
            a_list.append(eval_agent)

        # Use the dedicated CNN controller for evaluation
        controller = AgentControllerActorCriticCNN(a_list)
        return a_list, controller

    def training_step(self, step: int):
        """Modified training step to handle actor loss."""
        if self.debug:
            return
        self.step_and_collect_observation(step)
        self.env.spawn_despawn()
        self.apple_count_history.append(self.env.apples.sum())

        for agent in self._agents_list:
            # Train critic if ready
            if (
                agent.policy_value
                and len(agent.policy_value.batch_states) >= self.train_config.batch_size
            ):
                loss = agent.policy_value.train_batch()
                if loss is not None:
                    self.critic_loss.append(loss)
            assert isinstance(agent, ACAgent)
            assert isinstance(agent.policy_network, ActorCNN)
            # Train actor if ready
            if (
                agent.policy_network
                and len(agent.policy_network.batch_states)
                >= self.train_config.batch_size
            ):
                actorCNN: ActorCNN = agent.policy_network
                loss = actorCNN.train_batch()
                if loss is not None:
                    self.actor_loss.append(loss)

    def log_progress(self, sample_state, sample_state5, sample_state6):
        """Logs value predictions and actor probabilities for sample states."""
        # --- Critic Logging (Value of states) ---
        critic_net = self._agents_list[0].policy_value

        def get_val(state):
            processed = critic_net.raw_state_to_nn_input(
                state, agent_pos=state["poses"][0]
            )
            return critic_net.get_value_function(processed)

        v1, v2, v3 = (
            get_val(sample_state),
            get_val(sample_state5),
            get_val(sample_state6),
        )
        print(f"Sample State Values (Critic): {v1:.4f}, {v2:.4f}, {v3:.4f}")
        self.loss_plot.append(v1)
        self.loss_plot5.append(v2)
        self.loss_plot6.append(v3)

        # --- Actor Logging (Action probabilities) ---
        actor_net = self._agents_list[0].policy_network
        processed_actor_state = actor_net.raw_state_to_nn_input(
            sample_state, agent_pos=sample_state["poses"][0]
        )
        action_probs = actor_net.get_action_probabilities(processed_actor_state)
        probs_str = ", ".join([f"{p:.3f}" for p in action_probs])
        print(f"Sample State Action Probs (Actor): [{probs_str}]")

    def generate_plots(self):
        """Generates and saves plots for actor and critic training progress."""
        super().generate_plots()
        # Plot Actor Loss
        if self.actor_loss:
            fig = plot_hybrid_smoothed(
                [self.actor_loss],
                labels=["Actor Policy Gradient Loss"],
                title="Actor Training Loss (Smoothed)",
            )
            fig.savefig(self.graphs_out_path / "Actor_Training_Loss.png")
            plt.close(fig)

    #     if self.critic_loss:
    #         fig = plot_hybrid_smoothed(
    #             [self.critic_loss],
    #             labels=["Critic Training MSE Loss"],
    #             title="Critic Training Loss (Smoothed)",
    #         )
    #         ax = fig.gca()
    #         ax.set_yscale("log")
    #         fig.savefig(self.graphs_out_path / "Critic_Training_Loss.png")
    #         plt.close(fig

    #     # Plot Sample State Values (from critic)
    #     if self.loss_plot:
    #         plt.figure(figsize=(10, 6))
    #         plt.plot(self.loss_plot, label="Sample State 1 Value")
    #         plt.plot(self.loss_plot5, label="Sample State 2 Value")
    #         plt.plot(self.loss_plot6, label="Sample State 3 Value")
    #         plt.title("Value of Fixed Sample States During Training (Critic)")
    #         plt.legend()
    #         plt.grid(True)
    #         plt.savefig(self.graphs_out_path / "Sample_State_Values.png")
    #         plt.close()

    #     # --- 3. Plot the Apple Count History ---
    #     if self.apple_count_history:
    #         fig = plot_hybrid_smoothed(
    #             [self.apple_count_history],
    #             labels=["Apple Count"],
    #             title="Number of Apples in Orchard During Training",
    #             xlabel="Training Step",
    #             ylabel="Total Apples",
    #         )
    #         ax = fig.gca()
    #         ax.grid(True)
    #         # save the figure
    #         fig.savefig(self.graphs_out_path / "Apple_Count_During_Training.png")
    #         plt.close(fig)
