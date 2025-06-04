from abc import abstractmethod
from orchard.algorithms import single_apple_spawn, single_apple_despawn
from plots import graph_plots
from eval_network import eval_network
from orchard.environment import *
from helpers import generate_sample_states
from config import get_config


class Algorithm:
    def __init__(self, batch_size, alpha, name):
        self.discount = get_config()["discount"]
        self.agents_list = []
        self.env = None  # Orchard environment
        self.alpha = alpha  # Learning rate

        self.batch_size = batch_size  # Batch size for learning

        self.name = f"{name}_{get_config()["num_agents"]}_{get_config()["orchard_length"]}"  # Name of the experiment

        # Plots for evaluating value of sample states
        self.loss_plot = []
        self.loss_plot5 = []
        self.loss_plot6 = []

        # Network(s) used for eval_network at the middle and end of training
        self.network_for_eval = []

    def create_env(self, spawn_apples=None, despawn_apples=None):
        if not spawn_apples or not despawn_apples:
            self.env = Orchard(get_config()["orchard_length"], get_config()["num_agents"], get_config()["S"], get_config()["phi"], self.agents_list, one=True, spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn)
        else:
            self.env = Orchard(get_config()["orchard_length"], get_config()["num_agents"], get_config()["S"], get_config()["phi"], self.agents_list, one=True, spawn_algo=spawn_apples, despawn_algo=despawn_apples)
        self.env.initialize(self.agents_list)

    @abstractmethod
    def update_actor(self):
        raise NotImplementedError

    @abstractmethod
    def update_critic(self):
        raise NotImplementedError

    @abstractmethod
    def collect_observation(self, step, timesteps):
        raise NotImplementedError

    def train_batch(self):
        self.update_actor()
        self.update_critic()

    def log_progress(self, sample_state, sample_state5, sample_state6):
        v_value = self.agents_list[0].evaluate_interface(sample_state["agents"], sample_state["apples"], self.agents_list, sample_state["poses"])
        v_value5 = self.agents_list[0].evaluate_interface(sample_state5["agents"], sample_state5["apples"], self.agents_list, sample_state5["poses"])
        v_value6 = self.agents_list[0].evaluate_interface(sample_state6["agents"], sample_state6["apples"], self.agents_list, sample_state6["poses"])
        print("P", v_value)
        self.loss_plot.append(v_value.item())
        self.loss_plot5.append(v_value5.item())
        self.loss_plot6.append(v_value6.item())

    @abstractmethod
    def update_lr(self, step, timesteps):
        raise NotImplementedError

    def evaluate_checkpoint(self, step, timesteps, maxi):
        if (step % (timesteps * 0.5) == 0 and step != 0) or step == timesteps - 1:
            print("=====Eval at", step, "steps======")
            eval_network(self.name, maxi, len(self.agents_list), self.network_for_eval, side_length=get_config()["orchard_length"], iteration=99)
            print("=====Completed Evaluation=====")

    def train(self, timesteps):
        self.create_env()
        sample_state, sample_state5, sample_state6 = generate_sample_states(self.env.length, len(self.agents_list))
        maxi = 0
        total_reward = 0
        for step in range(timesteps):
            for _ in range(self.batch_size):
                self.collect_observation(step, timesteps)
            self.train_batch()

            if step % (0.1 * timesteps) == 0:
                self.log_progress(sample_state, sample_state5, sample_state6)
            self.update_lr(step, timesteps)

            if (step % (timesteps * 0.5) == 0 and step != 0) or step == timesteps - 1:
                self.evaluate_checkpoint(step, timesteps, maxi)
        graph_plots(None, self.name, None, self.loss_plot, self.loss_plot5, self.loss_plot6, None)
        print("Total Reward:", total_reward)
        print("Total Apples:", self.env.total_apples)

    @abstractmethod
    def run(self, timesteps):
        raise NotImplementedError
