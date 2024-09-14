from agents.communicating_agent import CommAgent
from models.simple_connected_multiple_dc import SCMNetwork
from policies.random_policy import random_policy_1d
from train_decentral import training_loop as training_loop_d
from agents.actor_critic_agent import ACAgent
from agents.actor_critic_alloc_agent import ACAgent as ACAAgent
from models.actor_dc_1d import ActorNetwork

from main import run_environment_1d
from orchard.algorithms import single_apple_spawn, single_apple_despawn

import torch

def eval_network(name, discount, side_length, experiment, iteration):
    network_list = []
    a_list = []
    for ii in range(num_agents):
        # print("A")
        network = ActorNetwork(side_length, 0.001, discount)
        network.function.load_state_dict(
            torch.load(prefix + name + "_" + approach + "_" + str(i) + "_it_" + str(iteration) + ".pt"))

        network_list.append(network)

    for ii in range(num_agents):
        trained_agent = ACAgent(policy="learned_policy", num=ii, num_agents=num_agents)
        # print(trained_agent.num)
        trained_agent.policy_network = network_list[ii]
        a_list.append(trained_agent)
    with torch.no_grad():
        run_environment_1d(num_agents, random_policy_1d, side_length, S, phi, name, experiment + "_" + str(iteration), agents_list=a_list,
                           spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn, timesteps=30000)



approach = "decentral" # or "rate", "D-4_10_beta", "D-4_10_projection"
name = "D-4_10"

num_agents = 4
orchard_size = 10
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

skip_decen = True

for iteration in range(2, 5):
    #prefix = "policyitchk/" + name + "_" + "value" + "/"
    prefix = "policyitchk/" + name + "_decen/"
    print(approach + "================ ITERATION " + str(iteration) + " DECENTRALIZED =====================")
    if not skip_decen or iteration > 0: # Re-initialize to get rid of some old data
        agents_list = []

        for i in range(num_agents):
            agents_list.append(CommAgent(policy="value_function2", num=i, num_agents=num_agents))
            agents_list[i].policy_value2 = SCMNetwork(orchard_size, alpha, discount)
            agents_list[i].agents_list = agents_list
            #agents_list[i].policy_value2.function.load_state_dict(
            #    torch.load(prefix + name + "_" + "value" + "_decen_" + str(i) + "_it_" + str(iteration-1) + ".pt"))
            agents_list[i].policy_value2.function.load_state_dict(
                torch.load(prefix + name + "_" + "decen" + "_decen_" + str(i) + "_it_" + str(iteration - 1) + ".pt"))

    title = name + "_iteration_" + str(iteration)

    if not skip_decen or iteration > 0:
        title = name + "_decen"
        # Get decentralized value function
        training_loop_d(agents_list, orchard_size, S, phi, 0.0005, title, discount=0.99, timesteps=1600000, iteration=iteration)
        # for nummer, agn in enumerate(agents_list):
        #     torch.save(agn.policy_value.function.state_dict(),
        #                prefix + name + "_" + approach + "_decen_" + str(nummer) + "_it_" + str(iteration) + ".pt")
    else:
        for nummer, agn in enumerate(agents_list):
            agn.policy_value = SCMNetwork(orchard_size, alpha, discount)
            agn.policy_value.function.load_state_dict(
                torch.load(prefix + name + "_" + approach + "_decen_" + str(nummer) + "_it_" + str(iteration) + ".pt"))
    # print(approach + "================ ITERATION " + str(iteration) + " ACTOR-CRITIC =====================")
#
    # # Perform actor-critic training
    # if approach == "value":
    #     train_ac_value(orchard_size, num_agents, agents_list, name + "_" + approach, discount, 700000)
    # elif approach == "beta":
    #     train_ac_beta(orchard_size, num_agents, agents_list, name + "_" + approach, discount, 700000)
    # elif approach == "rate":
    #     train_ac_rate(orchard_size, num_agents, agents_list, name + "_" + approach, discount, 700000)
    # elif approach == "binary":
    #     train_ac_binary(orchard_size, num_agents, agents_list, name + "_" + approach, discount, 700000)


    # for nummer, agn in enumerate(agents_list):
    #     torch.save(agn.policy_network.function.state_dict(), prefix + name + "_" + approach + "_" + str(nummer) + "_it_" + str(iteration) + ".pt")
    print("Saved Models; Evaluating Network")
    eval_network(name, discount, orchard_size, approach, iteration)
