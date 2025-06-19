from agents.communicating_agent import CommAgent
from agents.simple_agent import SimpleAgent
from main import run_environment_1d
from models.value_function import VNetwork
from algorithm import Algorithm
from helpers import convert_input, env_step
import torch
import matplotlib.pyplot as plt
from config import get_config
from orchard.algorithms import apple_despawn, apple_spawn, single_apple_despawn, \
    single_apple_spawn
from policies.random_policy import random_policy_1d, random_policy
from controllers import AgentController, ViewController, AgentControllerDecentralized, AgentControllerCentralized

torch.set_default_dtype(torch.float64)


class CentralizedValueFunction(Algorithm):
    def _format_env_step_return(self, state, new_state, reward, agent_id,
                                positions, action, old_pos):
        return state, new_state, reward

    def init_agents_for_eval(self):
        a_list = []
        for ii in range(len(self.agents_list)):
            trained_agent = SimpleAgent(policy="value_function")
            trained_agent.policy_value = self.network_for_eval[0]
            a_list.append(trained_agent)
        return a_list

    def save_networks(self, path):
        torch.save(self.network_for_eval[0].function.state_dict(),
                   path + "/" + self.name + "_cen_" + ".pt")

    def agent_get_action(self, agent_id):
        if self.agents_list[agent_id].policy == "value_function":
            action = self.agent_controller.get_best_action(self.env.get_state(), agent_id, self.env.available_actions)
        else:
            action = self.agents_list[agent_id].policy(self.env.available_actions)
        return action

    def __init__(self, batch_size, alpha, num_agents, orchard_length, orchard_width, alt_input=False, vision=None):
        super().__init__(batch_size, alpha, "C-RANDOM", num_agents, orchard_length, orchard_width, alt_input, vision)
        self.network = None
        self.view_controller = None

    def update_actor(self, t_ratio=None):
        return

    def update_critic(self):
        self.agents_list[0].policy_value.train()

    def collect_observation(self, step, timesteps, alt_vision=False):
        s, new_s, r = env_step(self.agents_list, self.env, step, timesteps, "C")
        self.agents_list[0].add_experience(self.view_controller.process_state(s, None), self.view_controller.process_state(new_s, None), r)

    def update_lr(self, step, timesteps):
        # if step == (0.33 * timesteps):
        #     for g in self.network.optimizer.param_groups:
        #             g['lr'] = 0.0001
        # if step == (0.625 * timesteps):
        #     for g in self.network.optimizer.param_groups:
        #             g['lr'] = 0.00003
        if step == (0.33 * timesteps):
            for g in self.network.optimizer.param_groups:
                    g['lr'] = 0.0005
        if step == (0.625 * timesteps):
            for g in self.network.optimizer.param_groups:
                    g['lr'] = 0.00015

    def run(self, timesteps, spawn_algo, despawn_algo):
        self.view_controller = ViewController(self.vision)
        self.agent_controller = AgentControllerCentralized(self.agents_list, self.view_controller)
        network = VNetwork(self.orchard_width * self.orchard_length, self.alpha, get_config()["discount"])
        for _ in range(self.num_agents):
            agent = SimpleAgent(policy=random_policy)
            agent.policy_value = network
            self.agents_list.append(agent)
        self.network_for_eval = [network]
        self.network = network
        return self.train(timesteps, spawn_algo, despawn_algo)


class DecentralizedValueFunction(Algorithm):

    def __init__(self, batch_size, alpha, num_agents, orchard_length, orchard_width, alt_input=False, vision=None):
        super().__init__(batch_size, alpha, "DC-RANDOM", num_agents, orchard_length, orchard_width, alt_input, vision)
        self.network_list = []

    def agent_get_action(self, agent_id):
        if self.agents_list[agent_id].policy == "value_function":
            action = self.agent_controller.get_best_action(self.env.get_state(), agent_id, self.env.available_actions)
        else:
            action = self.agents_list[agent_id].policy(self.env.available_actions)
        return action

    def _format_env_step_return(self, state, new_state, reward, agent_id,
                                positions, action, old_pos):
        return state, new_state, reward, old_pos, agent_id

    def update_actor(self, t_ratio=None):
        return

    def init_agents_for_eval(self):
        a_list = []
        for ii in range(len(self.agents_list)):
            trained_agent = CommAgent(policy="value_function")
            trained_agent.policy_value = self.network_list[ii]
            a_list.append(trained_agent)
        return a_list

    def save_networks(self, path):
        for nummer, netwk in enumerate(self.network_list):
            torch.save(netwk.function.state_dict(),
                       path + "/" + self.name + "_decen_" + str(
                           nummer) + ".pt")

    def update_critic(self):
        for agent in self.agents_list:
            agent.policy_value.train()

    def collect_observation(self, step, timesteps, alt_vision=False):
        s, new_s, r, old_pos, agent = self.env_step()
        for each_agent in range(len(self.agents_list)):
            if each_agent == agent:
                self.agents_list[each_agent].add_experience(self.view_controller.process_state(s, old_pos), self.view_controller.process_state(new_s, self.agents_list[each_agent].position), r)
            else:
                self.agents_list[each_agent].add_experience(self.view_controller.process_state(s, self.agents_list[each_agent].position), self.view_controller.process_state(new_s, self.agents_list[each_agent].position), 0)

    def update_lr(self, step, timesteps):
        if step == (0.33 * timesteps):
            for network in self.network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.0005
        if step == (0.625 * timesteps):
            for network in self.network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.00015

    def run(self, timesteps, spawn_algo, despawn_algo, num_agents=None, orchard_length=None):
        self.view_controller = ViewController(self.vision)
        self.agent_controller = AgentControllerDecentralized(self.agents_list, self.view_controller)
        for _ in range(self.num_agents):
            agent = CommAgent(policy=random_policy)
            network = VNetwork(self.orchard_length * self.orchard_width + 1, self.alpha, get_config()["discount"])
            agent.policy_value = network
            self.network_list.append(network)
            self.agents_list.append(agent)
        self.network_for_eval = self.network_list
        return self.train(timesteps, spawn_algo, despawn_algo, self.alt_input)


if __name__ == "__main__":
    import random
    import numpy as np

    widths = [1, 2, 3, 4, 5]
    orchard_length = 20
    orchard_width = 3
    random.seed(42)
    np.random.seed(42)
    c = CentralizedValueFunction(256, 0.0025, int(orchard_length * orchard_width * 0.1), orchard_length, orchard_width)
    c.run(20000, apple_spawn, apple_despawn)
    # for width in widths:
    #     print(f"In Decentralized with width {width}")
    #     random.seed(42)
    #     np.random.seed(42)
    #     c = DecentralizedValueFunction(256, 0.0025, 4, 20, width, alt_input=True, vision=9)
    #     c.run(20000)

    # for width in widths:
    #     print(f"In Centralized variable agents with width {width}")
    #     orchard_length = 20
    #     random.seed(42)
    #     np.random.seed(42)
    #     c = CentralizedValueFunction(256, 0.0025, int(orchard_length * width // 5), orchard_length, width)
    #     c.run(20000, apple_spawn, apple_despawn)
    #
    # print("In Centralized")
    # for width in widths:
    #     random.seed(42)
    #     np.random.seed(42)
    #     c = CentralizedValueFunction(256, 0.0025, 4, 20, width)
    #     c.run(20000)

    # random.seed(42)
    # np.random.seed(42)
    # c = DecentralizedValueFunction(256, 0.0005, 2, 10, alt_input=True, vision=3)
    # c.run(10000)
    # for _ in range(2):
    #     for combination in combinations:
    #         a_list = []
    #         directory = f"/Users/utya.kirill/Desktop/orchard-action-market/policyitchk/DC-RANDOM-ALT-INPUT-{combination[0]}-{combination[1]}"
    #         for i in range(combination[0]):
    #             agent = CommAgent(policy="value_function")
    #             agent.policy_value = VNetwork(10, 0.0005, get_config()["discount"])
    #             agent.policy_value.function.load_state_dict(torch.load(f"{directory}/DC-RANDOM-ALT-INPUT-{combination[0]}-{combination[1]}_decen_{0}_it_99.pt"))
    #             a_list.append(agent)
    #         reward, ratio = run_environment_1d(combination[0], "value_function", combination[1], None, None, f"DC-{combination[0]}_{combination[1]}",
    #                            agents_list=a_list,
    #                            spawn_algo=single_apple_spawn,
    #                            despawn_algo=single_apple_despawn,
    #                            timesteps=20000)
        # test = DecentralizedValueFunction(256, 0.0005, combination[0], combination[1], alt_input=True,
        #                                   vision=2)
        # decen_alt_view.append(ratio)
        # print(decen_alt_view)

    # for width in widths:
    #     a_list = []
    #     for _ in range((20 * width) // 10):
    #         a_list.append(SimpleAgent(policy=random_policy))
    #     reward, ratio = run_environment_1d((20 * width) // 10, 20, width, None, None, f"Random{20}_{4}-{width}",
    #                        agents_list=a_list,
    #                        spawn_algo=apple_spawn,
    #                        despawn_algo=apple_despawn,
    #                        timesteps=20000)
    #     print(ratio)
