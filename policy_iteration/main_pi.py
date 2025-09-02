from actor_critic.actor_critic import train_ac_value, train_ac_rate, train_ac_beta, train_ac_binary
from agents.actor_critic_agent import ACAgent
from agents.actor_critic_alloc_agent import ACAgent as ACAAgent
from agents.communicating_agent import CommAgent
from main import eval_performance
from models.simple_connected_multiple import SCMNetwork
from policies.random_policy import random_policy_1d
from models.simple_connected_multiple_dc import SCMNetwork, SimpleConnectedMultiple
from models.actor_network import ActorNetwork
# from models.simple_connected_multiple_dc_altinput import SCMNetwork, SimpleConnectedMultiple
# from models.actor_dc_1d_altinput import ActorNetwork
from orchard.algorithms import single_apple_spawn, single_apple_despawn
from train_decentral import training_loop as training_loop_d

import torch

torch.set_default_dtype(torch.float64)

"""
The main training loop for *Policy Iteration*. Includes provisions for training in the middle of an iteration (i.e. after Q training, but before AC training).
"""
def eval_network(name, discount, side_length, experiment, iteration, num_agents, prefix, approach, S=None, phi=None):
    network_list = []
    a_list = []
    for ii in range(num_agents):
        # print("A")
        network = ActorNetwork(side_length, 0.001, discount)
        network.function.load_state_dict(
            torch.load(prefix + name + "_" + approach + "_" + str(ii) + "_it_" + str(iteration) + ".pt"))

        network_list.append(network)

    for ii in range(num_agents):
        trained_agent = ACAgent(policy="learned_policy", num=ii, num_agents=num_agents)
        # print(trained_agent.num)
        trained_agent.policy_network = network_list[ii]
        a_list.append(trained_agent)
    with torch.no_grad():
        eval_performance(num_agents, random_policy_1d, side_length, S, phi, name, experiment + "_" + str(iteration), agents_list=a_list,
                         spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn, timesteps=30000)

def eval_network_dece(name, num_agents, discount, side_length, experiment, iteration, prefix, approach):
    print(prefix, name, approach)
    network_list = []
    a_list = []
    for ii in range(num_agents):
        # print("A")
        network = SCMNetwork(side_length, 0.001, discount)
        network.function.load_state_dict(
            torch.load(prefix + name + "_" + approach + "_decen_" + str(ii) + "_it_" + str(iteration) + ".pt"))

        network_list.append(network)

    for ii in range(num_agents):
        trained_agent = CommAgent(policy="value_function", num=ii, num_agents=num_agents)
        # print(trained_agent.num)
        trained_agent.policy_value = network_list[ii]
        a_list.append(trained_agent)
    with torch.no_grad():
        eval_performance(num_agents, random_policy_1d, side_length, None, None, name, experiment + "_" + str(iteration), agents_list=a_list,
                         spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn, timesteps=30000)


def policy_iteration(approach="value", name="D-2_5", num_agents=2, orchard_size=5, first_it=0, skip_decen=False):
    #approach = "value" # or "rate", "beta", "projection"
    #name = "D-2_5_altinput"

    name = "AC-" + str(num_agents) + "_" + str(orchard_size)
    #num_agents = 2
    #orchard_size = 5
    alpha = 0.001
    discount = 0.99
    S = None
    phi = None

    agents_list = []
    if approach == "rate":
        for i in range(num_agents):
            agents_list.append(ACAAgent(policy=random_policy_1d, num=i, num_agents=num_agents))
    else:
        for i in range(num_agents):
            agents_list.append(ACAgent(policy=random_policy_1d, num=i, num_agents=num_agents))

    #skip_decen = True

    for iteration in range(first_it, 5):
        prefix = "policyitchk/" + name + "_" + approach + "/"
        print(approach + "================ ITERATION " + str(iteration) + " DECENTRALIZED =====================")
        if iteration > 0:  # Re-initialize to get rid of some old data
            agents_list = []
            if approach == "rate":
                for i in range(num_agents):
                    agents_list.append(ACAAgent(policy="baseline", num=i, num_agents=num_agents))
                    agents_list[i].basic_network = ActorNetwork(orchard_size, alpha, discount)
                    if i == 0:
                        print("Loading: " + prefix + name + "_" + approach + "_" + str(i) + "_it_" + str(iteration-1) + ".pt")
                    agents_list[i].basic_network.function.load_state_dict(torch.load(prefix + name + "_" + approach + "_" + str(i) + "_it_" + str(iteration-1) + ".pt"))
            else:
                for i in range(num_agents):
                    agents_list.append(ACAgent(policy="baseline", num=i, num_agents=num_agents))
                    agents_list[i].basic_network = ActorNetwork(orchard_size, alpha, discount)
                    if i == 0:
                        print("Loading: " + prefix + name + "_" + approach + "_" + str(i) + "_it_" + str(
                            iteration - 1) + ".pt")
                    agents_list[i].basic_network.function.load_state_dict(
                        torch.load(prefix + name + "_" + approach + "_" + str(i) + "_it_" + str(iteration-1) + ".pt"))

        title = name + "_iteration_" + str(iteration)

        if not skip_decen or iteration > first_it:
            title = name + "_" + approach
            # Get decentralized value function
            if iteration == 0:
                training_loop_d(agents_list, orchard_size, S, phi, 0.0002, title, discount=0.99, timesteps=400000,
                                iteration=iteration)
            else:
                training_loop_d(agents_list, orchard_size, S, phi, 0.0005, title, discount=0.99, timesteps=800000, iteration=iteration)
            # for nummer, agn in enumerate(agents_list):
            #     torch.save(agn.policy_value.function.state_dict(),
            #                prefix + name + "_" + approach + "_decen_" + str(nummer) + "_it_" + str(iteration) + ".pt")
        else:
            for nummer, agn in enumerate(agents_list):
                if nummer == 0:
                    print("Loading: " + prefix + name + "_" + approach + "_decen_" + str(nummer) + "_it_" + str(iteration) + ".pt")
                agn.policy_value = SCMNetwork(orchard_size, alpha, discount)
                agn.policy_value.function.load_state_dict(
                    torch.load(prefix + name + "_" + approach + "_decen_" + str(nummer) + "_it_" + str(iteration) + ".pt"))
        print(approach + "================ ITERATION " + str(iteration) + " ACTOR-CRITIC =====================")

        for nummer, agn in enumerate(agents_list):
            if iteration > 0:
                #agn.policy_network = agn.basic_network
                agn.policy_network = ActorNetwork(orchard_size, alpha, discount)
                agn.policy = "learned_policy"
        # Perform actor-critic training
        if approach == "value":
            train_ac_value(orchard_size, num_agents, agents_list, name + "_" + approach, discount, 600000, iteration=iteration)
        elif approach == "beta":
            train_ac_beta(orchard_size, num_agents, agents_list, name + "_" + approach, discount, 600000, iteration=iteration)
        elif approach == "rate":
            train_ac_rate(orchard_size, num_agents, agents_list, name + "_" + approach, discount, 600000, iteration=iteration)
        elif approach == "binary":
            train_ac_binary(orchard_size, num_agents, agents_list, name + "_" + approach, discount, 600000, iteration=iteration)


        for nummer, agn in enumerate(agents_list):
            torch.save(agn.policy_network.function.state_dict(), prefix + name + "_" + approach + "_" + str(nummer) + "_it_" + str(iteration) + ".pt")
        print("Saved Models; Evaluating Network")
        eval_network(name, discount, orchard_size, approach, iteration, len(agents_list), prefix, approach)
