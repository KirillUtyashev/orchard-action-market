from matplotlib import pyplot as plt

from agents.actor_critic_agent import ACAgent
from actor_critic_kirill import ActorCritic
from configs.config import ExperimentConfig
from models.actor_dc_1d import ActorNetwork
from models.value_function import VNetwork
from value_function_learning.controllers import AgentControllerActorCritic, \
    ViewController
import numpy as np


class ActorCriticPerfect(ActorCritic):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config, f"""ActorCritic-<{config.train_config.num_agents}>_agents-_length-<{config.env_config.length}>_width-<{config.env_config.width}>_s_target-<{config.env_config.s_target}>-alpha-<{config.train_config.alpha}>-apple_mean_lifetime-<{config.env_config.apple_mean_lifetime}>-<{config.train_config.hidden_dimensions}>-<{config.train_config.num_layers}>-vision-<{config.train_config.vision}>-batch_size-<{config.train_config.batch_size}>-actor_alpha-<{config.train_config.actor_alpha}>-actor_hidden-<{config.train_config.hidden_dimensions_actor}>-actor_layers-<{config.train_config.num_layers_actor}>""")
        self.alpha_ema = {i: np.zeros((0, self.train_config.num_agents), dtype=float) for i in range(self.train_config.num_agents)}

    def init_agents_for_eval(self):
        a_list = []
        for ii in range(len(self.agents_list)):
            trained_agent = ACAgent("learned_policy", ii)
            trained_agent.policy_network = self.p_network_list[ii]
            a_list.append(trained_agent)
        return a_list

    # def save_networks(self, path):
    #     for nummer, netwk in enumerate(self.p_network_list):
    #         torch.save(netwk.function.state_dict(),
    #                    path + "/" + self.name + "_actor_network_AC_" + str(
    #                        nummer) + ".pt")
    #     for nummer, netwk in enumerate(self.v_network_list):
    #         torch.save(netwk.function.state_dict(),
    #                    path + "/" + self.name + "_critic_network_AC_" + str(
    #                        nummer) + ".pt")

    def update_critic(self):
        super().update_critic()
        for num, agent in enumerate(self.agents_list):
            # Update betas
            self._record_rates(num, agent.agent_alphas)

    def _record_rates(self, agent_i: int, alphas):
        """
        Append a snapshot of agent_i's follow rates to all agents.
        rates_global_vec: iterable length n_agents, in GLOBAL index order.
        """
        # for j in range(self.train_config.num_agents):
        #     new_list = np.append(self.foll_rate_hist[agent_i][j], float(agent_rates[j]))
        #     self.foll_rate_hist[agent_i][j] = new_list
        # make sure we have a 1 x n_agents row
        row_2 = np.asarray(alphas, dtype=float).reshape(1, -1)
        # append the new row to the (K, n_agents) history for this agent
        self.alpha_ema[agent_i] = np.vstack([self.alpha_ema[agent_i], row_2])

    def collect_observation(self, step):
        try:
            for tick in range(self.train_config.num_agents):
                s, new_s, r, agent, positions, action = self.env_step(tick)
                if action is not None:
                    for each_agent in range(len(self.agents_list)):
                        curr_pos = self.agents_list[each_agent].position
                        reward = r if each_agent == agent else 0
                        processed_state = self.view_controller.process_state(s, positions[each_agent])
                        processed_new_state = self.view_controller.process_state(new_s, curr_pos)
                        self.agents_list[each_agent].add_experience(
                            processed_state, processed_new_state, reward)
                        if each_agent == agent:
                            new_positions = []
                            for j in range(len(self.agents_list)):
                                new_positions.append(self.agents_list[j].position)
                            advantage = reward + self.train_config.discount * self.agent_controller.collective_value_from_state(new_s, new_positions, each_agent) - self.agent_controller.collective_value_from_state(s, positions, each_agent)
                            self.agents_list[each_agent].policy_network.add_experience(processed_state, processed_new_state, r, action, advantage)

        except Exception as e:
            self.logger.error(f"Error collecting observations: {e}")
            raise

    def log_progress(self, sample_state, sample_state5, sample_state6):
        super().log_progress(sample_state, sample_state5, sample_state6)
        for agent_id in range(self.train_config.num_agents):
            plt.figure(figsize=(10, 4))
            arr = self.alpha_ema[agent_id]
            for other_agent in range(self.train_config.num_agents):
                series = arr[:, other_agent]  # <-- column j, not row j
                if series.size > 0:
                    plt.plot(series, label=f"Q-value from agent {other_agent}")
                plt.plot(self.agents_list[agent_id].agent_alphas[other_agent])
            plt.legend()
            plt.title(f"Observed Q-values for Agent {agent_id}")
            plt.xlabel("Training Step")
            plt.ylabel("Q-value")
            plt.show()

    def run(self):
        try:
            self.view_controller = ViewController(self.train_config.vision)
            self.agent_controller = AgentControllerActorCritic(self.agents_list, self.view_controller)
            for nummer in range(self.train_config.num_agents):
                agent = ACAgent("learned_policy", nummer)
                agent.policy_network, agent.policy_value = self.init_networks()
                self.agents_list.append(agent)
                self.v_network_list.append(agent.policy_value)
                self.p_network_list.append(agent.policy_network)
            self.network_for_eval = self.p_network_list
            return self.train() if not self.train_config.skip else self.train(*self.restore_all())
        except Exception as e:
            self.logger.error(f"Failed to run decentralized training: {e}")
            raise
