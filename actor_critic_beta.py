from actor_critic_kirill import ActorCritic
from agents.actor_critic_agent import ACAgent, ACAgentBeta
from configs.config import ExperimentConfig
from value_function_learning.controllers import AgentControllerActorCritic, \
    ViewController
import numpy as np


class ActorCriticBeta(ActorCritic):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config, f"""ActorCriticBeta-<{config.train_config.num_agents}>_agents-_length-<{config.env_config.length}>_width-<{config.env_config.width}>_s_target-<{config.env_config.s_target}>-alpha-<{config.train_config.alpha}>-apple_mean_lifetime-<{config.env_config.apple_mean_lifetime}>-<{config.train_config.hidden_dimensions}>-<{config.train_config.num_layers}>-vision-<{config.train_config.vision}>-batch_size-<{config.train_config.batch_size}>-actor_alpha-<{config.train_config.actor_alpha}>-actor_hidden-<{config.train_config.hidden_dimensions_actor}>-actor_layers-<{config.train_config.num_layers_actor}>-beta-<{config.train_config.beta_rate}>""")
        self.alpha_ema = {i: np.zeros((0, self.train_config.num_agents), dtype=float) for i in range(self.train_config.num_agents)}

    def init_agents_for_eval(self):
        a_list = []
        for ii in range(len(self.agents_list)):
            trained_agent = ACAgentBeta("learned_policy", self.train_config.beta_rate, ii)
            trained_agent.policy_network = self.p_network_list[ii]
            a_list.append(trained_agent)
        return a_list

    def update_critic(self):
        super().update_critic()
        for num, agent in enumerate(self.agents_list):
            # Update betas
            agent.update_beta()

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
                            total_feedback = reward + self.train_config.discount * self.agent_controller.collective_value_from_state(new_s, new_positions, agent)
                            self.agents_list[each_agent].beta_temp_batch.append(total_feedback)
                            advantage_for_actor = total_feedback - self.agents_list[each_agent].beta
                            self.agents_list[each_agent].policy_network.add_experience(processed_state, processed_new_state, r, action, advantage_for_actor)
        except Exception as e:
            self.logger.error(f"Error collecting observations: {e}")
            raise

    def run(self):
        try:
            self.view_controller = ViewController(self.train_config.vision)
            self.agent_controller = AgentControllerActorCritic(self.agents_list, self.view_controller)
            for nummer in range(self.train_config.num_agents):
                agent = ACAgentBeta("learned_policy", self.train_config.beta_rate, nummer)
                agent.policy_network, agent.policy_value = self.init_networks()
                self.agents_list.append(agent)
                self.v_network_list.append(agent.policy_value)
                self.p_network_list.append(agent.policy_network)
            self.network_for_eval = self.p_network_list
            return self.train() if not self.train_config.skip else self.train(*self.restore_all())
        except Exception as e:
            self.logger.error(f"Failed to run decentralized training: {e}")
            raise
