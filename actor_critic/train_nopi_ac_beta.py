import time

from main import run_environment_1d
from models.simple_connected_multiple_blind import SCMBNetwork
from models.simple_connected_multiple import CNetwork
from orchard.environment import *
import numpy as np
import matplotlib.pyplot as plt
import orchard.environment_one
import random
from policies.nearest_uniform import replace_agents_1d
from policies.random_policy import random_policy_1d, random_policy
from policies.nearest import nearest_1d, nearest
from metrics.metrics import append_metrics, plot_metrics, append_positional_metrics, plot_agent_specific_metrics
from agents.simple_agent import SimpleAgent
from models.simple_connected import SimpleConnected
from models.simple_connected_multiple_dc import SCMNetwork, SimpleConnectedMultiple
from models.simple_connected_multiple import CNetwork as SCMNetwork_Central
from models.actor_dc_1d import ActorNetwork
#from models.actor_dc_1d_complex import ActorNetwork
from orchard.algorithms import single_apple_spawn, single_apple_despawn

from tests.benchmark_loss import estimate_value_function

import torch
import torch.nn as nn
import torch.optim as optim
torch.set_default_dtype(torch.float64)
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

one_plot = {}
two_plot = {}
def setup_plots(dictn, plot):
    for param_tensor in dictn:
        #print(dictn[param_tensor].size())
        plot[param_tensor] = []
        if dictn[param_tensor].dim() > 1 and "weight" in param_tensor: # and "1" not in param_tensor:
            for id in range(5):
                plot[param_tensor + str(id)] = []
def add_to_plots(dictn, timestep, plot):
    for param_tensor in dictn:
        if dictn[param_tensor].dim() > 1 and "weight" in param_tensor: # and "1" not in param_tensor:
            for id in range(5):
                plot[param_tensor + str(id)].append(dictn[param_tensor].cpu().flatten()[id])
            #else:
            #    plot[param_tensor].append(dictn[param_tensor].cpu().flatten()[0])

graph = 0
loss_plot = []
loss_plot1 = []
loss_plot2 = []
colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
def graph_plots(dictn, name, plot):
    global graph
    graph += 1
    plt.figure("plots" + str(graph) + name, figsize=(10, 5))
    num = 0
    for param_tensor in dictn:
        num += 1
        if num == 10:
            break
        if dictn[param_tensor].dim() > 1 and "weight" in param_tensor: # and "1" not in param_tensor:
            for id in range(5):
                if id == 0:
                    plt.plot(plot[param_tensor + str(id)], color=colours[num], label="Tensor " + str(num))
                else:
                    plt.plot(plot[param_tensor + str(id)], color=colours[num])
        # else:
        #     plt.plot(plot[param_tensor], label=str(num))
    plt.legend()
    plt.title("Model Parameters during Training, iteration " + str(graph))
    plt.savefig("ParamsActor_" + name + str(graph) + ".png")

    plt.figure("loss" + str(graph), figsize=(10, 5))
    plt.plot(loss_plot)
    plt.plot(loss_plot1)
    plt.plot(loss_plot2)
    #print(loss_plot2)
    plt.title("Value Function for Sample State, iteration " + str(graph))
    plt.savefig("ValActor_" + name + str(graph) + ".png")

# sample_state = {
#     "agents": np.array([[1], [0], [0], [0], [3], [0], [0], [0], [0], [0]]),
#     "apples": np.array([[0], [1], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "pos": [np.array([4, 0]), np.array([0, 0]), np.array([4, 0]), np.array([4, 0])]
# }
# sample_state5 = {
#     "agents": np.array([[0], [0], [1], [0], [0], [0], [0], [0], [0], [0]]),
#     "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "pos": [np.array([2, 0]), np.array([4, 0]), np.array([4, 0]), np.array([4, 0])]
# }
# sample_state6 = {
#     "agents": np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "pos": [np.array([4, 0]), np.array([4, 0]), np.array([4, 0]), np.array([4, 0])]
# }

sample_state = {
    "agents": np.array([[0], [2], [0], [0], [0]]),
    "apples": np.array([[1], [0], [0], [0], [0]]),
    "pos": [np.array([1, 0]), np.array([1, 0])]
}
sample_state5 = {
    "agents": np.array([[0], [0], [2], [0], [0]]),
    "apples": np.array([[1], [0], [0], [0], [0]]),
    "pos": [np.array([2, 0]), np.array([2, 0])]
}
sample_state6 = {
    "agents": np.array([[0], [0], [0], [0], [2]]),
    "apples": np.array([[1], [0], [0], [0], [0]]),
    "pos": [np.array([4, 0]), np.array([4, 0])]
}

def training_loop(agents_list, orchard_length, S, phi, alpha, name, discount=0.99, epsilon=0, timesteps=100000, iteration=99, altname=None):
    print(orchard_length)
    print(len(agents_list))
    env = Orchard(orchard_length, len(agents_list), S, phi, one=True, spawn_algo=single_apple_spawn,
                  despawn_algo=single_apple_despawn)  # initialize env
    env.initialize(agents_list)  # attach agents to env
    print("Experiment", name)
    v_network_list = []
    p_network_list = []

    #network1 = SCMNetwork_Central(orchard_length, alpha, discount)
    #network1.function.load_state_dict(torch.load(name + ".pt"))
    for agn in range(len(agents_list)):
        network1 = CNetwork(orchard_length, 0.0002, discount)
        agents_list[agn].policy_value = network1
        v_network_list.append(network1)

        network2 = ActorNetwork(orchard_length, alpha, discount, num=agn)
        agents_list[agn].policy_network = network2
        p_network_list.append(network2)
        agents_list[agn].policy = "learned_policy"

        network2.critic = network1


        # network2 = ActorNetwork(orchard_length, alpha, discount, num=agn)
        # agents_list[agn].policy_network = network2
        # p_network_list.append(network2)

    total_reward = 0

    """ Plotting Setup """
    setup_plots(p_network_list[0].function.state_dict(), one_plot)
    global loss_plot
    loss_plot = []
    maxi = 0
    """"""
    for i in range(timesteps):

        agent = random.randint(0, env.n - 1)  # Choose random agent
        old_pos = agents_list[agent].position.copy()  # Get position of this agent
        state = env.get_state()  # Get the current state (a COPY)

        #action = agents_list[agent].get_action(state, discount)

        epsilon = 0
        chance = random.random()  # Epsilon-greedy policy. Epsilon is zero currently.
        if chance < epsilon:
            action = random_policy_1d(state, old_pos)
        else:
            action = agents_list[agent].get_action(state, discount)  # Get action.

        reward, new_position = env.main_step(agents_list[agent].position.copy(),
                                             action)  # Take the action. Observe reward, and get new position
        agents_list[agent].position = new_position.copy()  # Save new position.
        total_reward += reward  # Add to reward.

        new_state = env.get_state()

        sp_state = {
            "agents": state["agents"].copy(),
            "apples": state["apples"].copy(),
            "pos": old_pos.copy()
        }
        sp_new_state = {
            "agents": new_state["agents"].copy(),
            "apples": new_state["apples"].copy(),
            "pos": agents_list[agent].position.copy()
        }
        # p_network_list[agent].train(sp_state, sp_new_state, reward, action, agents_list)
        p_network_list[agent].add_experience(sp_state, sp_new_state, reward, action, agents_list)
        # if i == 1000:
        #     start = time.time()
        # if i == 2000:
        #     end = time.time()
        #     print(end - start)
        if i % 1000 == 0:
            add_to_plots(p_network_list[0].function.state_dict(), i, one_plot)
            v_value = agents_list[1].policy_network.get_function_output(sample_state["agents"], sample_state["apples"], pos=sample_state["pos"][0])
            v_value1 = agents_list[1].policy_network.get_function_output(sample_state5["agents"], sample_state5["apples"],
                                                                        pos=sample_state5["pos"][1])
            v_value2 = agents_list[0].policy_network.get_function_output(sample_state6["agents"],
                                                                         sample_state6["apples"],
                                                                         pos=sample_state6["pos"][0])
            # v_value = agents_list[0].get_comm_value_function(sample_state["agents"], sample_state["apples"], agents_list, debug=True, agent_poses=sample_state["pos"])
            # v_value1 = agents_list[0].get_comm_value_function(sample_state5["agents"], sample_state5["apples"], agents_list, debug=True, agent_poses=sample_state5["pos"])
            # v_value2 = agents_list[0].get_comm_value_function(sample_state6["agents"], sample_state6["apples"], agents_list, debug=True, agent_poses=sample_state6["pos"])
            if i % 20000 == 0:

                print("A", v_value)
            #add_to_plots(agents_list[1].value.state_dict(), i, two_plot)
            loss_plot.append(v_value[0])
            loss_plot1.append(v_value1[0])
            loss_plot2.append(v_value2[0])
            for ntwk in p_network_list:
                ntwk.train_multiple_with_critic(agents_list)

        if i % 20000 == 0 and i != 0:
            print("At timestep", i)
            # for numbering, network in enumerate(p_network_list):
            #     print("Avg Norm for" + str(numbering) + ":", network.vs / i)
        # was: 300000
        if i == 50000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.001
        if i == 100000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.0005
        # was: 500000
        if i == 200000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.0001
        # was: 700000
        if i == 300000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.00005
        if i == 740000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.00005
        if i == 860000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.00002
        if i == 1000000:
            for network in p_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.00001
        """
        Critic LR
        """
        if i == 50000:
            for network in v_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.0001
        if i == 150000:
            for network in v_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.00005
        if i == 250000:
            for network in v_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.00001
        if i == 400000:
            for network in v_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.000005
        if i == 600000:
            for network in v_network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.000002
        #if i == 800000:
        #    for network in p_network_list:
        #        for g in network.optimizer.param_groups:
        #            g['lr'] = 0.00001
        #if i == 1000000:
        #    for network in p_network_list:
        #        for g in network.optimizer.param_groups:
         #           g['lr'] = 0.000005
        # if i == 150000:
        #     for network in p_network_list:
        #         for g in network.optimizer.param_groups:
        #             g['lr'] = 0.00001
        # if i == 200000:
        #     for network in p_network_list:
        #         for g in network.optimizer.param_groups:
        #             g['lr'] = 0.000005
        # if i == 250000:
        #     for network in p_network_list:
        #         for g in network.optimizer.param_groups:
        #             g['lr'] = 0.000003
        # if i == 300000:
        #     for network in p_network_list:
        #         for g in network.optimizer.param_groups:
        #             g['lr'] = 0.000001
        # if i == 150000:
        #     for network in p_network_list:
        #         for g in network.optimizer.param_groups:
        #             g['lr'] = 0.00001
        # if i == 200000:
        #     for network in p_network_list:
        #         for g in network.optimizer.param_groups:
        #             g['lr'] = 0.000003
        if i % 40000 == 0 and i != 0:
            print("=====Eval at", i, "steps======")
            fname = name
            if altname is not None:
                fname = altname
            for numbering, network in enumerate(p_network_list):
                torch.save(network.function.state_dict(), fname + "_Actor3_" + str(numbering) + ".pt")
                torch.save(network.optimizer.state_dict(), fname + "_Actor3_" + str(numbering) + "_optimizer.pt")
            maxi = eval_network(fname, discount, maxi, p_network_list, v_network_list, iteration=iteration, num_agents=len(agents_list), side_length=orchard_length)
            print("=====Completed Evaluation=====")
    fname = name
    if altname is not None:
        fname = altname
    for numbering, network in enumerate(p_network_list):
        torch.save(network.function.state_dict(), fname + "_Actor3_" + str(numbering) + ".pt")
    graph_plots(p_network_list[0].function.state_dict(), fname, one_plot)
    print("Total Reward:", total_reward)
    print("Total Apples:", env.total_apples)



from agents.actor_critic_agent import ACAgent
"""
An evaluation every x steps that saves the checkpoint in case we pass the best performance
"""
def eval_network(name, discount, maxi, p_network_list, v_network_list, num_agents=4, side_length=10, iteration=99):
    network_list = []
    a_list = []

    for ii in range(num_agents):
        trained_agent = ACAgent(policy="learned_policy", num=ii, num_agents=num_agents)
        # print(trained_agent.num)
        trained_agent.policy_network = p_network_list[ii]
        a_list.append(trained_agent)
    with torch.no_grad():
        val = run_environment_1d(num_agents, random_policy_1d, side_length, None, None, "AC", "test", agents_list=a_list,
                           spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn, timesteps=30000)
    if val > maxi and iteration != 99:
        print("saving best")
        for nummer, netwk in enumerate(p_network_list):
            torch.save(netwk.function.state_dict(),
                       "policyitchk/" + name + "/" + name + "_" + str(nummer) + "_it_" + str(iteration) + ".pt")
        for nummer, netwk in enumerate(v_network_list):
            torch.save(netwk.function.state_dict(),
                       "policyitchk/" + name + "/" + name + "_decen_" + str(nummer) + "_it_" + str(iteration) + ".pt")

    maxi = max(maxi, val)
    return maxi


if __name__ == "__main__":
    side_length = 5
    num_agents = 2

    S = None
    phi = None

    agents_list = []
    for i in range(num_agents):
        agents_list.append(ACAgent(policy=random_policy_1d, num=i, num_agents=num_agents, is_beta_agent=1))
        #agents_list[i].policy = "learned_policy"

    for i in range(1):
        print("loop", i)

        training_loop(agents_list, side_length, S, phi, 0.0013, "D-2_5_True_Beta", discount=0.99, timesteps=2000000, iteration=25)

        if i > 0:
            print("Total Same Actions:", agents_list[0].same_actions)
            agents_list[0].same_actions = 0










