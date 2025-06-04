import os

import torch

from config import CHECKPOINT_DIR
from agents.actor_critic_agent import ACAgent
from agents.communicating_agent import CommAgent
from agents.simple_agent import SimpleAgent
from main import run_environment_1d
from orchard.algorithms import single_apple_despawn, single_apple_spawn
from policies.random_policy import random_policy_1d


def eval_network(name, maxi, num_agents, network_list, side_length=10,
                 iteration=99, discount=0.99):
    a_list = []
    if "DC" in name:
        for ii in range(num_agents):
            trained_agent = CommAgent(policy="value_function")
            trained_agent.policy_value = network_list[ii]
            a_list.append(trained_agent)
    elif "AC" in name:
        for ii in range(num_agents):
            trained_agent = ACAgent(policy="learned_policy")
            trained_agent.policy_network = network_list[ii]
            a_list.append(trained_agent)
    else:
        for ii in range(num_agents):
            trained_agent = SimpleAgent(policy="learned_policy")
            trained_agent.policy_network = network_list[0]
            a_list.append(trained_agent)

    with torch.no_grad():
        val, ratio = run_environment_1d(num_agents, random_policy_1d,
                                        side_length, None, None, name,
                                        agents_list=a_list,
                                        spawn_algo=single_apple_spawn,
                                        despawn_algo=single_apple_despawn,
                                        timesteps=20000)
    if val > maxi:
        print("saving best")
        path = os.path.join(CHECKPOINT_DIR, name)
        if not os.path.isdir(path):
            os.makedirs(path)
        if "DC" in name:
            for nummer, netwk in enumerate(network_list):
                torch.save(netwk.function.state_dict(),
                           path + "/" + name + "_decen_" + str(
                               nummer) + "_it_" + str(iteration) + ".pt")
        elif "AC" in name:
            for nummer, netwk in enumerate(network_list):
                torch.save(netwk.function.state_dict(),
                           path + "/" + name + "_" + str(nummer) + "_it_" + str(
                               iteration) + ".pt")
        else:
            torch.save(network_list[0].function.state_dict(),
                       path + "/" + name + "_cen_it_" + str(iteration) + ".pt")
    maxi = max(maxi, val)
    return maxi, ratio
