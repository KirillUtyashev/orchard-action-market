from agents.communicating_agent import CommAgent
from agents.simple_agent import SimpleAgent
from models.value_function import VNetwork
from algorithm import Algorithm
from helpers import env_step
import torch
from config import get_config


torch.set_default_dtype(torch.float64)


class CentralizedValueFunction(Algorithm):
    def __init__(self, batch_size, alpha):
        super().__init__(batch_size, alpha, "C-RANDOM")
        self.network = None

    def update_actor(self):
        return

    def update_critic(self):
        self.agents_list[0].policy_value.train()

    def collect_observation(self, step, timesteps):
        s, new_s, r = env_step(self.agents_list, self.env, step, timesteps, "C")
        self.agents_list[0].policy_value.add_experience(s, None, new_s, None, r)

    def update_lr(self, step, timesteps):
        if step == (0.33 * timesteps):
            for g in self.network.optimizer.param_groups:
                g['lr'] = 0.0008
        if step == (0.5 * timesteps):
            for g in self.network.optimizer.param_groups:
                g['lr'] = 0.0002
        if step == (0.625 * timesteps):
            for g in self.network.optimizer.param_groups:
                g['lr'] = 0.00005

    def run(self, timesteps):
        network = VNetwork(get_config()["orchard_length"], self.alpha, get_config()["discount"])
        for _ in range(get_config()["num_agents"]):
            agent = SimpleAgent(policy="value_function")
            agent.policy_value = network
            self.agents_list.append(agent)
        self.network_for_eval = [network]
        self.network = network
        self.train(timesteps)


class DecentralizedValueFunction(Algorithm):
    def __init__(self, batch_size, alpha):
        super().__init__(batch_size, alpha, "DC-RANDOM")
        self.network_list = []

    def update_actor(self):
        return

    def update_critic(self):
        for agent in self.agents_list:
            agent.policy_value.train()

    def collect_observation(self, step, timesteps):
        s, new_s, r, old_pos, agent = env_step(self.agents_list, self.env, step, timesteps, "DC")
        for each_agent in range(len(self.agents_list)):
            if each_agent == agent:
                self.agents_list[each_agent].policy_value.add_experience(s, old_pos,
                                                                    new_s, self.agents_list[each_agent].position, r)
            else:
                self.agents_list[each_agent].policy_value.add_experience(s, self.agents_list[each_agent].position,
                                                                    new_s, self.agents_list[each_agent].position, 0)

    def update_lr(self, step, timesteps):
        if step == (0.33 * timesteps):
            for network in self.network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.00025
        if step == (0.625 * timesteps):
            for network in self.network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.0001

    def run(self, timesteps):
        for _ in range(get_config()["num_agents"]):
            agent = CommAgent(policy="value_function")
            network = VNetwork(get_config()["orchard_length"] + 1, self.alpha, get_config()["discount"])
            agent.policy_value = network
            self.network_list.append(network)
            self.agents_list.append(agent)
        self.network_for_eval = self.network_list
        self.train(timesteps)


if __name__ == "__main__":
    test = DecentralizedValueFunction(8, 0.0005)
    test.run(25000)
