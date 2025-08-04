from abc import ABC
import torch
from matplotlib import pyplot as plt
from agents.actor_critic_agent import ACAgent, ACAgentBeta, ACAgentRates
from algorithm import Algorithm
from config import get_config
from configs.config import ExperimentConfig
from models.actor_dc_1d import ActorNetwork, ActorNetworkCounterfactual, \
    ActorNetworkWithBeta, ActorNetworkWithRates
from models.value_function import VNetwork
import numpy as np
from helpers import get_discounted_value
from alloc.allocation import find_allocs, rate_allocate
from value_function_learning.controllers import AgentControllerActorCritic, \
    ViewController


class ActorCritic(Algorithm, ABC):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config, f"ActorCritic-<{config.train_config.num_agents}>_agents-_length-<{config.env_config.length}>_width-<{config.env_config.width}>_s_target-<{config.env_config.s_target}>-alpha-<{config.train_config.alpha}>-apple_mean_lifetime-<{config.env_config.apple_mean_lifetime}>-<{config.train_config.hidden_dimensions}>-<{config.train_config.num_layers}>-vision-<{config.train_config.vision}>")
        self.p_network_list = []
        self.v_network_list = []

        self.prob_sample_action_0 = []
        self.prob_sample_action_1 = []
        self.prob_sample_action_2 = []

    def _format_env_step_return(self, state, new_state, reward, agent_id, positions, action, old_pos):
        return state, new_state, reward, agent_id, positions, action

    def init_agents_for_eval(self):
        a_list = []
        for ii in range(len(self.agents_list)):
            trained_agent = ACAgent("learned_policy", ii)
            trained_agent.policy_network = self.p_network_list[ii]
            a_list.append(trained_agent)
        return a_list

    def save_networks(self, path):
        for nummer, netwk in enumerate(self.p_network_list):
            torch.save(netwk.function.state_dict(),
                       path + "/" + self.name + "_actor_network_AC_" + str(
                           nummer) + ".pt")
        for nummer, netwk in enumerate(self.v_network_list):
            torch.save(netwk.function.state_dict(),
                       path + "/" + self.name + "_critic_network_AC_" + str(
                           nummer) + ".pt")

    def update_actor(self):
        for agent in self.agents_list:
            agent.policy_network.train()

        # res_2 = []
        # res_3 = []
        # for agent in self.agents_list:
        #     res = agent.policy_network.train()
        #     if res is not None:
        #         loss, adv_value = res
        #         res_2.append(loss)
        #         res_3.append(adv_value)
        # if len(res_2) > 0 and len(res_3) > 0:
        #     self.actor_loss_history.append(res_2[0])
        #     self.adv_loss_history.append(res_3[0])

    def update_critic(self):
        losses = []
        for agent in self.agents_list:
            losses.append(agent.policy_value.train())
        return losses[-1]

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
                            advantage = reward + self.train_config.discount * self.agent_controller.collective_value_from_state(new_s, new_positions) - self.agent_controller.collective_value_from_state(s, positions)
                            self.agents_list[each_agent].add_experience(s, new_s, r, action, advantage)

        except Exception as e:
            self.logger.error(f"Error collecting observations: {e}")
            raise

    def update_lr(self, i):
        pass

    def log_progress(self, sample_state, sample_state5, sample_state6):
        super(ActorCritic, self).log_progress(sample_state, sample_state5, sample_state6)

        observation = self.view_controller.process_state(sample_state, sample_state["poses"][0])
        res = self.agents_list[0].policy_network.get_function_output(observation)

        self.prob_sample_action_0.append(res[0].item())
        self.prob_sample_action_1.append(res[1].item())
        self.prob_sample_action_2.append(res[2].item())

        print(res[0].item())
        print(res[1].item())

    def train_batch(self):
        self.update_actor()
        self.update_critic()

    def run(self):
        try:
            self.view_controller = ViewController(self.train_config.vision)
            self.agent_controller = AgentControllerActorCritic(self.agents_list, self.view_controller)
            for nummer in range(self.train_config.num_agents):
                agent = ACAgent("learned_policy", nummer)
                if self.train_config.alt_input:
                    if self.env_config.width != 1:
                        input_dim = self.train_config.vision ** 2 + 1
                    else:
                        input_dim = self.train_config.vision + 1
                else:
                    input_dim = self.env_config.length * self.env_config.width + 1
                agent.policy_network = ActorNetwork(input_dim, self.train_config.alpha, self.train_config.discount, self.train_config.hidden_dimensions, self.train_config.num_layers)
                agent.policy_value = VNetwork(input_dim, self.train_config.alpha, self.train_config.discount, self.train_config.hidden_dimensions, self.train_config.num_layers)
                self.agents_list.append(agent)
                self.v_network_list.append(agent.policy_value)
                self.p_network_list.append(agent.policy_network)
            self.network_for_eval = self.p_network_list
            self.train()
        except Exception as e:
            self.logger.error(f"Failed to run decentralized training: {e}")
            raise


class ActorCriticBeta(ActorCritic):
    def __init__(self, batch_size, alpha):
        super().__init__(batch_size, alpha)
        self.name = "AC-BETA"
        self.alpha_1_1 = []
        self.alpha_1_2 = []
        self.alpha_2_1 = []
        self.alpha_2_2 = []
        self.beta_1 = []
        self.beta_2 = []

    def update_actor(self, t_ratio=None):
        super().update_actor()
        for agent in self.agents_list:
            agent.update_beta()
        self.alpha_1_1.append(self.agents_list[0].alphas[0])
        self.alpha_1_2.append(self.agents_list[1].alphas[0])
        self.alpha_2_1.append(self.agents_list[0].alphas[1])
        self.alpha_2_2.append(self.agents_list[1].alphas[1])
        self.beta_1.append(self.agents_list[0].beta)
        self.beta_2.append(self.agents_list[1].beta)

    def plot(self):
        super().plot()
        plt.figure()
        plt.plot(self.alpha_1_1)
        plt.plot(self.alpha_1_2)
        plt.xlabel("Training Step")
        plt.ylabel("Alphas for agent 1")
        plt.title("Alphas for agent 1")
        plt.show()
        plt.figure()
        plt.plot(self.alpha_2_1)
        plt.plot(self.alpha_2_2)
        plt.xlabel("Training Step")
        plt.ylabel("Alphas for agent 2")
        plt.title("Alphas for agent 2")
        plt.show()
        plt.figure()
        plt.plot(self.beta_1)
        plt.xlabel("Training Step")
        plt.ylabel("Beta for agent 1")
        plt.title("Beta for agent 1")
        plt.show()
        plt.plot(self.beta_2)
        plt.xlabel("Training Step")
        plt.ylabel("Beta for agent 2")
        plt.title("Beta for agent 2")
        plt.show()

    def collect_observation(self, step, timesteps):
        new_s, agent, reward, action = super().collect_observation(step, timesteps)
        self.find_ab(new_s, agent, reward, action)

    def find_ab(self, new_s, agent, reward, action):
        """
        Compute and update alphas and betas in place for each agent after collecting an observation
        """
        beta_sum = 0
        for num, each_agent in enumerate(self.agents_list):
            if action is not None:
                value = self.discount * each_agent.get_q_value(
                    new_s["agents"].copy())[0]
                if num == agent:
                    value += reward
            else:
                value = 0
            for agent_num in range(len(self.agents_list)):
                if agent_num != agent:
                    each_agent.alphas[agent_num] = get_discounted_value(each_agent.alphas[agent_num], 0)
                else:
                    each_agent.alphas[agent] = get_discounted_value(each_agent.alphas[agent], value)
            beta_sum += value
        if action is not None:
            self.agents_list[agent].beta_batch.append(beta_sum.item())

    def run(self, timesteps):
        for i in range(get_config()["num_agents"]):
            agent = ACAgentBeta("learned_policy", get_config()["num_agents"])
            agent.policy_network = ActorNetworkWithBeta(get_config()["orchard_length"] + 1, self.agents_list, self.alpha, get_config()["discount"])
            agent.policy_value = VNetwork(get_config()["orchard_length"] + 1, 0.0002, get_config()["discount"])
            self.agents_list.append(agent)
            self.v_network_list.append(agent.policy_value)
            self.p_network_list.append(agent.policy_network)
        self.network_for_eval = self.p_network_list
        self.train(timesteps)
        self.plot()


class ActorCriticCounterfactual(ActorCritic):
    def __init__(self, batch_size, alpha):
        super(ActorCriticCounterfactual, self).__init__(batch_size, alpha)
        self.name = "AC-COUNTERFACTUAL"

    def run(self, timesteps):
        for i in range(get_config()["num_agents"]):
            agent = ACAgent("learned_policy")
            agent.policy_network = ActorNetworkCounterfactual(get_config()["orchard_length"] + 1, self.agents_list, self.alpha, get_config()["discount"])
            agent.policy_value = VNetwork(get_config()["orchard_length"] + 1, 0.0005, get_config()["discount"])
            self.agents_list.append(agent)
            self.v_network_list.append(agent.policy_value)
            self.p_network_list.append(agent.policy_network)
        self.network_for_eval = self.p_network_list
        self.train(timesteps)


class ActorCriticRate(ActorCriticBeta):
    def __init__(self, batch_size, alpha):
        super(ActorCriticRate, self).__init__(batch_size, alpha)
        self.name = "AC-RATE"

    def update_actor(self, t_ratio=None):
        super().update_actor()
        if t_ratio % 0.0001:
            for id_, agent in enumerate(self.agents_list):
                agent.alphas[id_] = 0
                agent.agent_rates = rate_allocate(agent.alphas, np.array([]))[0:len(self.agents_list)]
                agent.agent_rates[id_] = 0

    def find_ab(self, new_s, agent, reward, action):
        beta_sum = 0
        for num, each_agent in enumerate(self.agents_list):
            value = self.discount * each_agent.get_q_value(
                new_s["agents"].copy())[0] * (1 - np.exp(-each_agent.agent_rates[agent]))
            if num == agent:
                value += reward
            for agent_num in range(len(self.agents_list)):
                if agent_num != agent:
                    each_agent.alphas[agent_num] = get_discounted_value(each_agent.alphas[agent_num], 0)
                else:
                    each_agent.alphas[agent] = get_discounted_value(each_agent.alphas[agent], value.item())
            beta_sum += value
        self.agents_list[agent].beta_batch.append(beta_sum.item())

    def run(self, timesteps):
        for i in range(get_config()["num_agents"]):
            agent = ACAgentRates("learned_policy", get_config()["num_agents"], i)
            agent.policy_network = ActorNetworkWithRates(get_config()["orchard_length"] + 1, self.agents_list, self.alpha, get_config()["discount"])
            agent.policy_value = VNetwork(get_config()["orchard_length"] + 1, 0.0005, get_config()["discount"])
            self.agents_list.append(agent)
            self.v_network_list.append(agent.policy_value)
            self.p_network_list.append(agent.policy_network)
        self.network_for_eval = self.p_network_list
        self.train(timesteps)
        self.plot()


if __name__ == "__main__":
    ac = ActorCriticBeta(1, 0.001)
    ac.run(1000000)
