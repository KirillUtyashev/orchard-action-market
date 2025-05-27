from agents.communicating_agent import CommAgent
from config import get_config
from models.value_function import VNetwork
import torch
from agents.simple_agent import SimpleAgent
from train_value_function import training_loop_c, training_loop_d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
The LEARNING file. This serves as an execution file for both centralized and decentralizing learning.
"""


def central(config):
    network = VNetwork(config["orchard_length"], 0.0008, config["discount"])
    agents_list = []
    for i in range(config["num_agents"]):
        agents_list.append(SimpleAgent(policy="value_function", num="f"))
        agents_list[i].policy_value = network
    training_loop_c(agents_list, config["orchard_length"], config["S"], config["phi"], discount=config["discount"], timesteps=6250)


def decentralized(config):
    agents_list = []
    for _ in range(config["num_agents"]):
        agents_list.append(CommAgent(policy="value_function", num="f"))

    training_loop_d(agents_list, config["orchard_length"], config["S"], config["phi"], 0.001, discount=config["discount"], timesteps=2500)


if __name__ == "__main__":
    import random
    import numpy as np
    random.seed(10)
    np.random.seed(10)
    # central(get_config())
    decentralized(get_config())
