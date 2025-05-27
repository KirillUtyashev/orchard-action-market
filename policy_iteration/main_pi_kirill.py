import os

from actor_critic_kirill import train_ac_value, train_ac_rate, train_ac_beta, train_ac_binary
from agents.actor_critic_agent import ACAgent
from agents.actor_critic_alloc_agent import ACAgent as ACAAgent
from config import CHECKPOINT_DIR
from models.value_function import VNetwork
from policies.random_policy import random_policy_1d
from models.actor_dc_1d import ActorNetwork
from value_function_learning.train_value_function import training_loop_d
from config import get_config
import torch

torch.set_default_dtype(torch.float64)

"""
The main training loop for *Policy Iteration*. Includes provisions for training in the middle of an iteration (i.e. after Q training, but before AC training).
"""


def policy_iteration(approach="value"):
    num_agents = get_config()["num_agents"]
    orchard_size = get_config()["orchard_length"]
    name = "AC-" + str(num_agents) + "_" + str(orchard_size)
    alpha = 0.001

    agents_list = []
    # if approach == "rate":
    #     for i in range(num_agents):
    #         agents_list.append(ACAAgent(policy=random_policy_1d, num=i, num_agents=num_agents))
    # else:
    for i in range(num_agents):
        agents_list.append(ACAgent(policy="baseline", num=i, num_agents=num_agents))
    path = os.path.join(CHECKPOINT_DIR, name + "_" + approach)
    # We want to combine actor and critic into 1 step

    # Понедельник
    # Create a loop that does all the same things as value function learning but also incorporate actor network and change policy actions:
    # 1. Use softmax + temperature (decay too) for choosing actions
    # 2. Tweak Actor Network to make it similar to value function one

    # Вторник
    # Возможно придется посидеть на 4-20, чтобы понять как улучшить обучение:
    # Выписать параметры, которые могу менять, поменять их, сделать определенные гипотезы

    for nummer, agn in enumerate(agents_list):
        agn.policy_network = ActorNetwork(orchard_size, alpha, get_config()["discount"])
        agn.policy_value = VNetwork(get_config()["orchard_length"], alpha, get_config()["discount"])
        agn.policy = "learned_policy"

    # Perform actor-critic training
    if approach == "value":
        train_ac_value(orchard_size, num_agents, agents_list, name + "_" + approach, 0.99, 50000)
    elif approach == "beta":
        train_ac_beta(orchard_size, num_agents, agents_list, name + "_" + approach, get_config()["discount"], 600000)
    elif approach == "rate":
        train_ac_rate(orchard_size, num_agents, agents_list, name + "_" + approach, get_config()["discount"], 600000)
    elif approach == "binary":
        train_ac_binary(orchard_size, num_agents, agents_list, name + "_" + approach, get_config()["discount"], 600000)


if __name__ == "__main__":
    """
    Call the policy iteration function (from main_pi.py).
    Commences training.
    """

    approach_ = "value"

    policy_iteration(approach_)
