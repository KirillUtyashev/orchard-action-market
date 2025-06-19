from abc import abstractmethod

import torch
import os
from config import CHECKPOINT_DIR
from main import run_environment_1d
from orchard.algorithms import single_apple_spawn, single_apple_despawn, apple_spawn, apple_despawn
from plots import graph_plots
from eval_network import eval_network
from orchard.environment import *
from helpers import generate_sample_states
from config import get_config


class Algorithm:
    def __init__(self, batch_size, alpha, name, num_agents, orchard_length, orchard_width, alt_input=False, vision=None):
        self.discount = get_config()["discount"]
        self.agents_list = []
        self.env = None  # Orchard environment
        self.alpha = alpha  # Learning rate
        self.num_agents = num_agents
        self.orchard_length = orchard_length
        self.orchard_width = orchard_width
        self.alt_input = alt_input
        self.vision = vision
        name = f"{name}_{self.num_agents}_{self.orchard_length}"  # Name of the experiment
        if self.alt_input:
            name = name + f"-ALT-INPUT-VISION-{vision}-{self.orchard_width}"
        self.name = name

        self.batch_size = batch_size  # Batch size for learning

        # Plots for evaluating value of sample states
        self.loss_plot = []
        self.loss_plot5 = []
        self.loss_plot6 = []

        self.max_ratio = 0

        # Network(s) used for eval_network at the middle and end of training
        self.network_for_eval = []
        self.view_controller = None
        self.agent_controller = None

    def create_env(self, spawn_apples=None, despawn_apples=None):
        if not spawn_apples or not despawn_apples:
            self.env = Orchard(self.orchard_length, self.orchard_width, self.num_agents, get_config()["S"], get_config()["phi"], self.agents_list, spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn)
        else:
            self.env = Orchard(self.orchard_length, self.orchard_width, self.num_agents, get_config()["S"], get_config()["phi"], self.agents_list, spawn_algo=spawn_apples, despawn_algo=despawn_apples)
        self.env.initialize(self.agents_list)

    @abstractmethod
    def update_actor(self, r_ratio=None):
        raise NotImplementedError

    @abstractmethod
    def update_critic(self):
        raise NotImplementedError

    @abstractmethod
    def collect_observation(self, step, timesteps, alt_vision=False):
        raise NotImplementedError

    def train_batch(self, t_ratio=None):
        self.update_critic()
        self.update_actor(t_ratio)

    def log_progress(self, sample_state, sample_state5, sample_state6):
        # print("Spawned", self.env.total_apples)
        # print("Despawned", self.env.apples_despawned)
        print("Number of non-zero apples:", np.count_nonzero(self.env.apples))

        agent_obs = []
        for i in range(len(self.agents_list)):
            agent_obs.append(self.view_controller.process_state(sample_state, sample_state["poses"][i]))
        v_value = self.agent_controller.get_collective_value(agent_obs)
        agent_obs = []
        for i in range(len(self.agents_list)):
            agent_obs.append(self.view_controller.process_state(sample_state5, sample_state5["poses"][i]))
        v_value5 = self.agent_controller.get_collective_value(agent_obs)
        agent_obs = []
        for i in range(len(self.agents_list)):
            agent_obs.append(self.view_controller.process_state(sample_state6, sample_state6["poses"][i]))
        v_value6 = self.agent_controller.get_collective_value(agent_obs)

        print("P", v_value)
        self.loss_plot.append(v_value.item())
        self.loss_plot5.append(v_value5.item())
        self.loss_plot6.append(v_value6.item())

    @abstractmethod
    def update_lr(self, step, timesteps):
        raise NotImplementedError

    def evaluate_checkpoint(self, step, timesteps, maxi, spawn_algo, despawn_algo):
        print("=====Eval at", step, "steps======")
        maxi, ratio = self.eval_network(maxi, spawn_algo, despawn_algo)
        if ratio > self.max_ratio:
            self.max_ratio = ratio
        print("=====Completed Evaluation=====")

    def eval_network(self, maxi, spawn_algo, despawn_algo):
        agents_list = self.init_agents_for_eval()
        with torch.no_grad():
            val, ratio = run_environment_1d(len(self.agents_list), self.orchard_length, self.orchard_width, None, None, self.name,
                                            agents_list=agents_list,
                                            spawn_algo=spawn_algo,
                                            despawn_algo=despawn_algo,
                                            timesteps=20000, vision=self.vision)
        print("saving best")
        path = os.path.join(CHECKPOINT_DIR, self.name)
        if not os.path.isdir(path):
            os.makedirs(path)
        self.save_networks(path)
        return maxi, ratio

    def env_step(self):
        agent_id = random.randint(0, len(self.agents_list) - 1)
        state = self.env.get_state()  # this is assumed to be a dict with "agents" and "apples"
        old_pos = self.agents_list[agent_id].position
        positions = []
        for i in range(len(self.agents_list)):
            positions.append(self.agents_list[i].position)
        action = self.agent_get_action(agent_id)
        reward, new_pos = self.env.main_step(self.agents_list[agent_id].position.copy(), action)
        self.agents_list[agent_id].position = new_pos.copy()
        return self._format_env_step_return(state, self.env.get_state(), reward, agent_id, positions, action, old_pos)

    @abstractmethod
    def agent_get_action(self, agent_id):
        raise NotImplementedError

    @abstractmethod
    def _format_env_step_return(self, state, new_state, reward, agent_id, positions, action, old_pos):
        raise NotImplementedError

    @abstractmethod
    def init_agents_for_eval(self):
        raise NotImplementedError

    @abstractmethod
    def save_networks(self, path):
        raise NotImplementedError

    def train(self, timesteps, spawn_algo, despawn_algo, alt_vision=False):
        self.create_env(spawn_apples=spawn_algo, despawn_apples=despawn_algo)
        sample_state, sample_state5, sample_state6 = generate_sample_states(self.env.length, self.env.width, len(self.agents_list))
        maxi = 0
        total_reward = 0
        for step in range(timesteps):
            for _ in range(self.batch_size):
                self.collect_observation(step, timesteps, alt_vision)
            self.train_batch(step / timesteps)

            if step % (0.02 * timesteps) == 0:
                self.log_progress(sample_state, sample_state5, sample_state6)
            self.update_lr(step, timesteps)

            if (step % (timesteps * 0.2) == 0 and step != 0) or step == timesteps - 1:
                self.evaluate_checkpoint(step, timesteps, maxi, spawn_algo, despawn_algo)
                graph_plots(None, self.name, None, self.loss_plot, self.loss_plot5, self.loss_plot6, None)
        graph_plots(None, self.name, None, self.loss_plot, self.loss_plot5, self.loss_plot6, None)
        for _ in range(2):
            self.evaluate_checkpoint(timesteps - 1, timesteps, maxi, spawn_algo, despawn_algo)
        print("Total Reward:", total_reward)
        print("Total Apples:", self.env.total_apples)
        return self.max_ratio

    @abstractmethod
    def run(self, timesteps, spawn_algo, despawn_algo):
        raise NotImplementedError

#         elif "AC" in name:
#         for nummer, netwk in enumerate(network_list):
#             torch.save(netwk.function.state_dict(),
#                        path + "/" + name + "_" + str(nummer) + "_it_" + str(
#                            iteration) + ".pt")
#     else:
#     torch.save(network_list[0].function.state_dict(),
#                path + "/" + name + "_cen_it_" + str(iteration) + ".pt")
# maxi = max(maxi, val)
# return maxi, ratio
