from agents.actor_critic_agent import ACAgent
from actor_critic.actor_critic import ActorCritic
from configs.config import ExperimentConfig
from plots import add_to_plots
from helpers.controllers import AgentControllerActorCritic, \
    ViewController


class ActorCriticPerfect(ActorCritic):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config, f"""ActorCritic-<{config.train_config.num_agents}>_agents-_length-<{config.env_config.length}>_width-<{config.env_config.width}>_s_target-<{config.env_config.s_target}>-alpha-<{config.train_config.alpha}>-apple_mean_lifetime-<{config.env_config.apple_mean_lifetime}>-<{config.train_config.hidden_dimensions}>-<{config.train_config.num_layers}>-vision-<{config.train_config.vision}>-batch_size-<{config.train_config.batch_size}>-actor_alpha-<{config.train_config.actor_alpha}>-actor_hidden-<{config.train_config.hidden_dimensions_actor}>-actor_layers-<{config.train_config.num_layers_actor}>""")

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


class ActorCriticPerfectNoAdvantage(ActorCritic):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config, f"""ActorCriticNoAdvantage-<{config.train_config.num_agents}>_agents-_length-<{config.env_config.length}>_width-<{config.env_config.width}>_s_target-<{config.env_config.s_target}>-alpha-<{config.train_config.alpha}>-apple_mean_lifetime-<{config.env_config.apple_mean_lifetime}>-<{config.train_config.hidden_dimensions}>-<{config.train_config.num_layers}>-vision-<{config.train_config.vision}>-batch_size-<{config.train_config.batch_size}>-actor_alpha-<{config.train_config.actor_alpha}>-actor_hidden-<{config.train_config.hidden_dimensions_actor}>-actor_layers-<{config.train_config.num_layers_actor}>""")

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
                            advantage = reward + self.train_config.discount * self.agent_controller.collective_value_from_state(new_s, new_positions, each_agent)
                            self.agents_list[each_agent].policy_network.add_experience(processed_state, processed_new_state, r, action, advantage)

        except Exception as e:
            self.logger.error(f"Error collecting observations: {e}")
            raise

    def restore_all(self):
        name = f"""ActorCritic-<{self.train_config.num_agents}>_agents-_length-<{self.env_config.length}>_width-<{self.env_config.width}>_s_target-<{self.env_config.s_target}>-alpha-<{self.train_config.alpha}>-apple_mean_lifetime-<{self.env_config.apple_mean_lifetime}>-<{self.train_config.hidden_dimensions}>-<{self.train_config.num_layers}>-vision-<{self.train_config.vision}>-batch_size-<{self.train_config.batch_size}>-actor_alpha-<{self.train_config.actor_alpha}>-actor_hidden-<{self.train_config.hidden_dimensions_actor}>-actor_layers-<{self.train_config.num_layers_actor}>"""
        self.load_networks(name)
        agent_pos, apples = self._load_env_state()
        return agent_pos, apples

    def log_progress(self, sample_state, sample_state5, sample_state6):
        super().log_progress(sample_state, sample_state5, sample_state6)
        add_to_plots(self.v_network_list[0].function.state_dict(), self.v_weights)

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
