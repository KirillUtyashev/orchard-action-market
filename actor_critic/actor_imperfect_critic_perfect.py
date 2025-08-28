from agents.actor_critic_agent import ACAgent
from actor_critic.actor_critic import ActorCritic
from configs.config import ExperimentConfig
from helpers.controllers import AgentControllerActorCritic, \
    AgentControllerActorCriticIndividual, ViewController


class ActorImperfectCriticPerfect(ActorCritic):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config, f"""ACImPerf-<{config.train_config.num_agents}>_agents-_length-<{config.env_config.length}>_width-<{config.env_config.width}>_s_target-<{config.env_config.s_target}>-alpha-<{config.train_config.alpha}>-apple_mean_lifetime-<{config.env_config.apple_mean_lifetime}>-<{config.train_config.hidden_dimensions}>-<{config.train_config.num_layers}>-critic_vision-<{config.train_config.critic_vision}>-actor_vision-<{config.train_config.actor_vision}>-batch_size-<{config.train_config.batch_size}>-actor_alpha-<{config.train_config.actor_alpha}>-actor_hidden-<{config.train_config.hidden_dimensions_actor}>-actor_layers-<{config.train_config.num_layers_actor}>-budget-<{config.train_config.budget}>""")
        self.critic_view_controller = ViewController(self.train_config.critic_vision)
        self.actor_view_controller = ViewController(self.train_config.actor_vision)
        if self.train_config.budget == 0:
            self.agent_controller = AgentControllerActorCriticIndividual(self.agents_list, self.critic_view_controller, self.actor_view_controller)
        else:
            self.agent_controller = AgentControllerActorCritic(self.agents_list, self.critic_view_controller, self.actor_view_controller)

    def init_agents_for_eval(self):
        a_list = []
        for ii in range(len(self.agents_list)):
            trained_agent = ACAgent("learned_policy", ii)
            trained_agent.policy_network = self.p_network_list[ii]
            trained_agent.policy_value = self.v_network_list[ii]
            a_list.append(trained_agent)
        if self.train_config.budget == 0:
            agent_controller = AgentControllerActorCriticIndividual(a_list, self.critic_view_controller, self.actor_view_controller)
        else:
            agent_controller = AgentControllerActorCritic(a_list, self.critic_view_controller, self.actor_view_controller)
        return a_list, agent_controller

    def collect_observation(self, step):
        try:
            for tick in range(self.train_config.num_agents):
                s, new_s, r, agent, positions, action = self.env_step(tick)
                if action is not None:
                    for each_agent in range(len(self.agents_list)):
                        curr_pos = self.agents_list[each_agent].position
                        reward = r if each_agent == agent else 0
                        self.agents_list[each_agent].add_experience(
                            self.critic_view_controller.process_state(s, positions[each_agent]), self.critic_view_controller.process_state(new_s, curr_pos), reward)
                        if each_agent == agent:
                            new_positions = []
                            for j in range(len(self.agents_list)):
                                new_positions.append(self.agents_list[j].position)
                            advantage = reward + self.train_config.discount * self.agent_controller.collective_value_from_state(new_s, new_positions, each_agent) - self.agent_controller.collective_value_from_state(s, positions, each_agent)
                            self.agents_list[each_agent].policy_network.add_experience(self.actor_view_controller.process_state(s, positions[each_agent]), self.actor_view_controller.process_state(new_s, curr_pos), r, action, advantage)

        except Exception as e:
            self.logger.error(f"Error collecting observations: {e}")
            raise

    def run(self):
        try:
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
