import os

from actor_critic import train_ac_value, train_ac_rate, train_ac_beta, train_ac_binary
from agents.actor_critic_agent import ACAgent
from agents.actor_critic_alloc_agent import ACAgent as ACAAgent
from config import CHECKPOINT_DIR
from policies.random_policy import random_policy_1d
from models.actor_dc_1d import ActorNetwork
from value_function_learning.train_value_function import training_loop_d
from config import get_config
import torch

torch.set_default_dtype(torch.float64)

"""
The main training loop for *Policy Iteration*. Includes provisions for training in the middle of an iteration (i.e. after Q training, but before AC training).
"""


# def eval_network_dece(name, num_agents, discount, side_length, experiment, iteration, prefix, approach):
#     print(prefix, name, approach)
#     network_list = []
#     a_list = []
#     for ii in range(num_agents):
#         # print("A")
#         network = CNetwork(side_length, 0.001, discount)
#         network.function.load_state_dict(
#             torch.load(prefix + name + "_" + approach + "_decen_" + str(ii) + "_it_" + str(iteration) + ".pt"))
#
#         network_list.append(network)
#
#     for ii in range(num_agents):
#         trained_agent = CommAgent(policy="value_function", num=ii, num_agents=num_agents)
#         # print(trained_agent.num)
#         trained_agent.policy_value = network_list[ii]
#         a_list.append(trained_agent)
#     with torch.no_grad():
#         run_environment_1d(num_agents, random_policy_1d, side_length, None, None, name, experiment + "_" + str(iteration), agents_list=a_list,
#                            spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn, timesteps=30000)


def policy_iteration(approach="value"):
    num_agents = get_config()["num_agents"]
    orchard_size = get_config()["orchard_length"]
    name = "AC-" + str(num_agents) + "_" + str(orchard_size)
    alpha = 0.001

    agents_list = []
    if approach == "rate":
        for i in range(num_agents):
            agents_list.append(ACAAgent(policy=random_policy_1d, num=i, num_agents=num_agents))
    else:
        for i in range(num_agents):
            agents_list.append(ACAgent(policy=random_policy_1d, num=i, num_agents=num_agents))

    path = os.path.join(CHECKPOINT_DIR, name + "_" + approach)
    for iteration in range(2):
        # Re-initialize to get rid of some old data
        if iteration > 0:
            agents_list = []
            if approach == "rate":
                for i in range(num_agents):
                    agents_list.append(ACAAgent(policy="baseline", num=i, num_agents=num_agents))
            else:
                for i in range(num_agents):
                    agents_list.append(ACAgent(policy="baseline", num=i, num_agents=num_agents))
            for i in range(num_agents):
                # Here, we load the NN weights from the policy improvement from last iteration
                agents_list[i].behavior_net = ActorNetwork(orchard_size, alpha, get_config()["discount"])
                agents_list[i].behavior_net.function.load_state_dict(torch.load(path + "/" + name + "_" + approach + "_" + str(i) + "_it_" + str(iteration - 1) + ".pt"))

        # At this step, we have to learn the value functions --> for now, we learn it every time from scratch
        if iteration == 0:
            timesteps = 50000
        else:
            timesteps = 50000

        training_loop_d(agents_list, orchard_size, get_config()["S"], get_config()['phi'], 0.0005, discount=get_config()["discount"], timesteps=timesteps, iteration=iteration)

        # if not skip_decen or iteration > first_it:
        #     title = name + "_" + approach
        #     # Get decentralized value function
        #     if iteration == 0:
        #         training_loop_d(agents_list, orchard_size, S, phi, 0.0002, title, discount=0.99, timesteps=400000,
        #                         iteration=iteration)
        #     else:
        #         training_loop_d(agents_list, orchard_size, S, phi, 0.0005, title, discount=0.99, timesteps=800000, iteration=iteration)
        # else:
        #     for nummer, agn in enumerate(agents_list):
        #         if nummer == 0:
        #             print("Loading: " + prefix + name + "_" + approach + "_decen_" + str(nummer) + "_it_" + str(iteration) + ".pt")
        #         agn.policy_value = CNetwork(orchard_size, alpha, discount)
        #         agn.policy_value.function.load_state_dict(
        #             torch.load(prefix + name + "_" + approach + "_decen_" + str(nummer) + "_it_" + str(iteration) + ".pt"))

        # At this point, we have updated the critic and ready to do the policy improvement step
        # First, we initialize ActorNetworks -> if it's not the first iteration, we load previously learned weights
        # if it's the first iteration, we use the blank network because our actions are random for now
        for nummer, agn in enumerate(agents_list):
            agn.policy_network = ActorNetwork(orchard_size, alpha, get_config()["discount"])
            agn.policy = "learned_policy"
            if iteration > 0:
                agn.policy_network.function.state_dict(torch.load(path + "/" + name + "_" + approach + "_" + str(nummer) + "_it_" + str(iteration - 1) + ".pt"))

        # Perform actor-critic training
        if approach == "value":
            train_ac_value(orchard_size, num_agents, agents_list, name + "_" + approach, 0.99, 50000, iteration=iteration)
        elif approach == "beta":
            train_ac_beta(orchard_size, num_agents, agents_list, name + "_" + approach, discount, 600000, iteration=iteration)
        elif approach == "rate":
            train_ac_rate(orchard_size, num_agents, agents_list, name + "_" + approach, discount, 600000, iteration=iteration)
        elif approach == "binary":
            train_ac_binary(orchard_size, num_agents, agents_list, name + "_" + approach, discount, 600000, iteration=iteration)


if __name__ == "__main__":
    """
    Call the policy iteration function (from main_pi.py).
    Commences training.
    """

    approach_ = "value"

    policy_iteration(approach_)
