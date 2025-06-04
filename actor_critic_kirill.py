from matplotlib import pyplot as plt
from agents.actor_critic_agent import ACAgent, ACAgentBeta
from algorithm import Algorithm
from config import get_config
from models.actor_dc_1d import ActorNetwork, ActorNetworkCounterfactual, \
    ActorNetworkWithBeta
from models.value_function import VNetwork
import numpy as np
import random
from helpers import env_step


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
        plt.ylabel("Critic Loss")
        plt.title("Training Loss over Time")
        plt.show()


    def collect_observation(self, step, timesteps):
        s, new_s, r, agent, positions, action = env_step(self.agents_list, self.env, step, timesteps, "AC")
        for each_agent in range(len(self.agents_list)):
            if each_agent == agent:
                self.agents_list[each_agent].policy_value.add_experience(s, positions[each_agent], new_s, self.agents_list[each_agent].position, r)
                new_positions = []
                for j in range(len(self.agents_list)):
                    new_positions.append(self.agents_list[j].position)
                self.agents_list[each_agent].policy_network.add_experience(s, positions[each_agent], new_s, self.agents_list[each_agent].position, r, action, positions, new_positions, agent)
            else:
                self.agents_list[each_agent].policy_value.add_experience(s, self.agents_list[each_agent].position, new_s, self.agents_list[each_agent].position, 0)
        return new_s, agent, r

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
        prob_value = self.agents_list[0].policy_network.get_function_output(sample_state["agents"], sample_state["apples"], pos=sample_state["poses"][0])
        print(prob_value)

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

    def collect_observation(self, step, timesteps):
        new_s, agent, reward = super().collect_observation(step, timesteps)
        self.find_ab(new_s, agent, reward)

    def find_ab(self, new_s, agent, reward):
        """
        Compute and update alphas and betas in place for each agent after collecting an observation
        """
        beta_sum = 0
        for num, each_agent in enumerate(self.agents_list):
            value = each_agent.get_value_function(new_s["agents"].copy(), new_s["apples"].copy(), each_agent.position)[0]
            if num == agent:
                value += reward
            for agent_num in range(len(self.agents_list)):
                if agent_num != agent:
                    each_agent.alphas[agent_num] = get_discounted_value(each_agent.alphas[agent_num], 0)
                else:
                    each_agent.alphas[agent] = get_discounted_value(each_agent.alphas[agent], value)
            beta_sum += value
        beta = beta_sum
        self.agents_list[agent].beta = get_discounted_value(self.agents_list[agent].beta, beta.item())

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

    def update_following_rates(self):
        pass

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
        plt.ylabel("Critic Loss")
        plt.title("Training Loss over Time")
        plt.show()


if __name__ == "__main__":
    # random.seed(42)
    # np.random.seed(42)
    # for _ in range(5):
    for _ in range(10):
        ac = ActorCriticBeta(8, 0.00005)
        ac.run(50000)
