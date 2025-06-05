import time

import torch
from matplotlib import pyplot as plt
from agents.actor_critic_agent import ACAgent, ACAgentBeta, ACAgentRates
from algorithm import Algorithm
from config import get_config
from models.actor_dc_1d import ActorNetwork, ActorNetworkCounterfactual, \
    ActorNetworkWithBeta, ActorNetworkWithRates
from models.value_function import VNetwork
import numpy as np
import random
from helpers import env_step
from alloc.allocation import find_allocs
from value_function_learning.train_value_function import DecentralizedValueFunction


def get_discounted_value(old, new, discount_factor=0.05):
    return old * (1 - discount_factor) + new * discount_factor


class ActorCritic(Algorithm):
    def __init__(self, batch_size, alpha):
        super().__init__(batch_size, alpha, "AC-VALUE")
        self.p_network_list = []
        self.v_network_list = []
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.adv_loss_history = []

    def update_actor(self):
        res_2 = []
        res_3 = []
        for agent in self.agents_list:
            res = agent.policy_network.train()
            if res is not None:
                loss, adv_value = res
                res_2.append(loss)
                res_3.append(adv_value)
        if len(res_2) > 0 and len(res_3) > 0:
            self.actor_loss_history.append(res_2[0])
            self.adv_loss_history.append(res_3[0])

    def update_critic(self):
        res = []
        for agent in self.agents_list:
            res.append(agent.policy_value.train())
        self.critic_loss_history.append(res[0])

    def plot(self):
        plt.figure()
        plt.plot(self.actor_loss_history)
        plt.xlabel("Training Step")
        plt.ylabel("Actor Loss")
        plt.title("Training Loss over Time")
        plt.show()
        plt.figure()
        plt.plot(self.critic_loss_history)
        plt.xlabel("Training Step")
        plt.ylabel("Critic Loss")
        plt.title("Training Loss over Time")
        plt.show()
        plt.figure()
        plt.plot(self.adv_loss_history)
        plt.xlabel("Training Step")
        plt.ylabel("Advantage Loss")
        plt.title("Training Loss over Time")
        plt.show()

    def collect_observation(self, step, timesteps):
        s, new_s, r, agent, positions, action = env_step(self.agents_list, self.env, step, timesteps, "AC")
        if action is not None:
            for each_agent in range(len(self.agents_list)):
                if each_agent == agent:
                    self.agents_list[each_agent].policy_value.add_experience(s, positions[each_agent], new_s, self.agents_list[each_agent].position, r)
                    new_positions = []
                    for j in range(len(self.agents_list)):
                        new_positions.append(self.agents_list[j].position)
                    self.agents_list[each_agent].policy_network.add_experience(s, positions[each_agent], new_s, self.agents_list[each_agent].position, r, action, positions, new_positions, agent)
                else:
                    self.agents_list[each_agent].policy_value.add_experience(s, self.agents_list[each_agent].position, new_s, self.agents_list[each_agent].position, 0)
        return new_s, agent, r, action

    def update_lr(self, step, timesteps):
        if step == (0.33 * timesteps):
            for network in self.v_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = g['lr'] * 0.5
        if step == (0.625 * timesteps):
            for network in self.v_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = g['lr'] * 0.5

    def log_progress(self, sample_state, sample_state5, sample_state6):
        super(ActorCritic, self).log_progress(sample_state, sample_state5, sample_state6)
        prob_value1 = self.agents_list[0].policy_network.get_function_output(sample_state["agents"], sample_state["apples"], pos=sample_state["poses"][0])
        prob_value2 = self.agents_list[1].policy_network.get_function_output(sample_state["agents"], sample_state["apples"], pos=sample_state["poses"][0])
        print(prob_value1)
        print(prob_value2)

    def train_batch(self):
        self.update_actor()
        self.update_critic()

    def evaluate_checkpoint(self, step, timesteps, maxi):
        super().evaluate_checkpoint(step, timesteps, maxi)
        self.plot()

    def run(self, timesteps):
        for i in range(get_config()["num_agents"]):
            agent = ACAgent("learned_policy")
            agent.policy_network = ActorNetwork(get_config()["orchard_length"] + 1, self.agents_list, self.alpha, get_config()["discount"])
            agent.policy_value = VNetwork(get_config()["orchard_length"] + 1, 0.0005, get_config()["discount"])
            self.agents_list.append(agent)
            self.v_network_list.append(agent.policy_value)
            self.p_network_list.append(agent.policy_network)
        self.network_for_eval = self.p_network_list
        self.train(timesteps)
        plt.figure()
        plt.plot(self.actor_loss_history)
        plt.xlabel("Training Step")
        plt.ylabel("Actor Loss")
        plt.title("Training Loss over Time")
        plt.show()
        plt.figure()
        plt.plot(self.critic_loss_history)
        plt.xlabel("Training Step")
        plt.ylabel("Critic Loss")
        plt.title("Training Loss over Time")
        plt.show()


class ActorCriticBeta(ActorCritic):
    def __init__(self, batch_size, alpha):
        super().__init__(batch_size, alpha)
        self.name = "AC-BETA"
        self.beta_batch = [[] for _ in range(get_config()["num_agents"])]

        self.alpha_1_1 = []
        self.alpha_1_2 = []
        self.alpha_2_1 = []
        self.alpha_2_2 = []
        self.beta_1 = []
        self.beta_2 = []

    def update_actor(self):
        super().update_actor()
        for i in range(len(self.beta_batch)):
            mean = np.mean(self.beta_batch[i])
            if not np.isnan(mean):
                self.agents_list[i].beta = get_discounted_value(self.agents_list[i].beta, mean)
            self.beta_batch[i] = []
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
                value = self.discount * each_agent.get_value_function(new_s["agents"].copy(), new_s["apples"].copy(), each_agent.position)[0]
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
            self.beta_batch[agent].append(beta_sum.item())

    def run(self, timesteps):
        for i in range(get_config()["num_agents"]):
            agent = ACAgentBeta("learned_policy", get_config()["num_agents"])
            agent.policy_network = ActorNetworkWithBeta(get_config()["orchard_length"] + 1, self.agents_list, self.alpha, get_config()["discount"])
            agent.policy_value = VNetwork(get_config()["orchard_length"] + 1, 0.0005, get_config()["discount"])
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

    def update_actor(self):
        super().update_actor()
        for id_, agent in enumerate(self.agents_list):
            agent.agent_rates, agent.acting_rate = find_allocs(agent.alphas, id_, agent.beta)

    def find_ab(self, new_s, agent, reward, action):
        beta_sum = 0
        for num, each_agent in enumerate(self.agents_list):
            if action is not None:
                value = self.discount * each_agent.get_value_function(new_s["agents"].copy(), new_s["apples"].copy(), each_agent.position)[0] * each_agent.agent_rates[agent]
                if num == agent:
                    value += reward
            else:
                value = np.array(0)
            for agent_num in range(len(self.agents_list)):
                if agent_num != agent:
                    each_agent.alphas[agent_num] = get_discounted_value(each_agent.alphas[agent_num], 0)
                else:
                    each_agent.alphas[agent] = get_discounted_value(each_agent.alphas[agent], value.item())
            beta_sum += value
        if action is not None:
            self.beta_batch[agent].append(beta_sum.item())

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
    random.seed(42)
    np.random.seed(42)
    # for _ in range(5):
    ac = ActorCriticRate(8, 0.00005)
    ac.run(50000)
