from models.simple_connected_multiple_blind import SCMBNetwork
from models.simple_connected_multiple import SCMNetwork
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
from models.decentralized_simple_connected_multiple import SCMNetwork, SimpleConnectedMultiple
from models.simple_connected_multiple_blind import SCMBNetwork
from orchard.algorithms import single_apple_spawn, single_apple_despawn

from tests.benchmark_loss import estimate_value_function

import torch
import torch.nn as nn
import torch.optim as optim
torch.set_default_dtype(torch.float64)

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
    plt.savefig("params_" + name + str(graph) + "_4PM.png")

    plt.figure("loss" + str(graph), figsize=(10, 5))
    plt.plot(loss_plot)
    plt.plot(loss_plot1)
    plt.plot(loss_plot2)
    # print(loss_plot2)
    plt.title("Value Function for Sample State, iteration " + str(graph))
    plt.savefig("Val_" + name + str(graph) + ".png")

sample_state = {
    "agents": np.array([[0], [1], [0], [0], [0]]),
    "apples": np.array([[1], [0], [0], [0], [0]])
}

sample_state2 = {
    "agents": np.array([[0], [0], [1], [0], [0]]),
    "apples": np.array([[1], [0], [0], [0], [0]])
}

sample_state3 = {
    "agents": np.array([[0], [0], [1], [0], [0]]),
    "apples": np.array([[0], [0], [0], [1], [0]])
}

sample_state = {
    "agents": np.array([[0], [1], [0], [0], [0], [0], [0], [0], [0], [0]]),
    "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
}
sample_state5 = {
    "agents": np.array([[0], [0], [1], [0], [0], [0], [0], [0], [0], [0]]),
    "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
}
sample_state6 = {
    "agents": np.array([[0], [0], [0], [0], [1], [0], [0], [0], [0], [0]]),
    "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
}

def get_closest_left_right_1d(mat, agent_pos):
    mat = list(mat)
    left = -1
    right = -1
    pos, count = agent_pos, mat[agent_pos]
    while pos > -1:
        if count > 0:
            left = agent_pos - pos
            break
        else:
            pos -= 1
            count = mat[pos]
    pos, count = agent_pos, mat[agent_pos]
    while pos < len(mat):
        if count > 0:
            right = pos - agent_pos
            break
        else:
            pos += 1
            if pos >= len(mat):
                break
            count = mat[pos]
    return left, right

def unwrap_state(state):
    return state["agents"].copy(), state["apples"].copy()

def convert_input(state, agent_pos):
    a, b = unwrap_state(state)
    #print(list(a.flatten()), list(b.flatten()), agent_pos)

    a[agent_pos[0], agent_pos[1]] -= 1
    a = a.flatten()
    b = b.flatten()
    #print(list(a), list(b), agent_pos)
    left1, right1 = get_closest_left_right_1d(b, agent_pos[0])
    left2, right2 = get_closest_left_right_1d(a, agent_pos[0])
    arr = [left1, right1]
    arr1 = [left2, right2]
    #print(arr, arr1)
    return [np.array(arr), np.array(arr1)]



def training_loop(agents_list, orchard_length, S, phi, alpha, name, altinput=False, discount=0.99, epsilon=0, timesteps=100000):
    print(orchard_length)
    print(len(agents_list))
    env = Orchard(orchard_length, len(agents_list), S, phi, one=True, spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn) # initialize env
    env.initialize(agents_list) # attach agents to env
    print("Experiment", name)
    network_list = []
    for i in range(len(agents_list)):
        if altinput:
            network = SCMBNetwork(orchard_length, alpha, discount)
        else:
            network = SCMNetwork(orchard_length, alpha, discount)
        network_list.append(network)
        agents_list[i].policy_value = network
    total_reward = 0

    """ Plotting Setup """
    setup_plots(network_list[0].function.state_dict(), one_plot)
    global loss_plot
    loss_plot = []
    """"""

    for i in range(timesteps):
        agent = random.randint(0, env.n - 1)  # Choose random agent
        old_pos = agents_list[agent].position.copy()  # Get position of this agent
        state = env.get_state()  # Get the current state (a COPY)

        #chance = random.random()  # Epsilon-greedy policy. Epsilon is zero currently.
        #if chance < epsilon:
        #    action = random_policy_1d(state, old_pos)
        #else:
        action = agents_list[agent].get_action(state)  # Get action.

        agent_poses = []
        for agp in agents_list:
            agent_poses.append(agp.position.copy())

        reward, new_position = env.main_step(agents_list[agent].position.copy(), action)  # Take the action. Observe reward, and get new position
        agents_list[agent].position = new_position.copy()  # Save new position.
        total_reward += reward  # Add to reward.

        new_state = env.get_state()

        """ Some asserts to make sure things make sense """
        old_state_test = state["agents"].copy()
        old_state_test[old_pos[0], old_pos[1]] -= 1
        old_state_test[new_position[0], new_position[1]] += 1
        assert np.array_equal(old_state_test, new_state["agents"])
        if reward > 0:
            assert np.sum(state["apples"]) >= np.sum(new_state["apples"])
        """"""
        if not altinput:
            for each_agent in range(len(agents_list)):
                if each_agent == agent:
                    state["pos"] = old_pos
                    new_state["pos"] = agents_list[agent].position
                    network_list[agent].train(state, new_state, reward)                             # <- for NORMAL
                else:
                    state["pos"] = agents_list[each_agent].position
                    new_state["pos"] = agents_list[each_agent].position
                    network_list[each_agent].train(state, new_state, 0)
        else:
            state1 = convert_input(state, old_pos)  # <- for ALT INPUT
            state2 = convert_input(env.get_state(), agents_list[agent].position)  # <- for ALT INPUT
            network_list[agent].train(state1, state2, reward)  # <- for ALT INPUT

        """ For Plotting """
        if i % 1000 == 0:
            add_to_plots(network_list[0].function.state_dict(), i, one_plot)
            # add_to_plots(agents_list[1].value.state_dict(), i, two_plot)
            #loss_plot.append(np.average(np.array(mini_loss_plot)))
            #mini_loss_plot = []
            # v_value = agents_list[0].get_value_function(sample_state["agents"], sample_state["apples"])
            # v_value1 = agents_list[0].get_value_function(sample_state5["agents"], sample_state5["apples"])
            # v_value2 = agents_list[0].get_value_function(sample_state6["agents"], sample_state6["apples"])
            v_value = agents_list[0].get_comm_value_function(sample_state["agents"], sample_state["apples"], agents_list)
            v_value1 = agents_list[0].get_comm_value_function(sample_state5["agents"], sample_state5["apples"], agents_list)
            v_value2 = agents_list[0].get_comm_value_function(sample_state6["agents"], sample_state6["apples"], agents_list)
            print("P", v_value)
            # add_to_plots(agents_list[1].value.state_dict(), i, two_plot)
            loss_plot.append(v_value[0])
            loss_plot1.append(v_value1[0])
            loss_plot2.append(v_value2[0])
        if i % 20000 == 0:
            print("At timestep", i)
            #print("At timestep", i)
        if i == 65000:
            print("Decreasing LR")
            for network in network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.00001
        if i == 100000:
            print("Decreasing LR")
            for network in network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.000002
    graph_plots(network_list[0].function.state_dict(), name, one_plot)
    print("Total Reward:", total_reward)
    print("Total Apples:", env.total_apples)
    for i, network in enumerate(network_list):
        torch.save(network.function.state_dict(), name + "_agent_" + str(i) + ".pt")


"""
SINGLE AGENT TRIALS
"""
# side_length = 5
# num_agents = 1
# S = np.zeros((side_length, 1))
# for i in range(side_length):
#     S[i] = 0.2
# phi = 0.2
# discount = 0.99

"""
"REGULAR" TRIALS

side_length = 20
num_agents = 5
S = np.zeros((side_length, 1))
for i in range(side_length):
    S[i] = 0.05
phi = 0.05
discount = 0.99
"""





#agents_list = initialize_agents(num_agents, nearest)
#training_loop(agents_list, side_length, S, phi, 0.001, "nearest")
#
# run_environment_1d(num_agents, random_policy_1d, side_length, S, phi, "Nearest", "Uniform", timesteps=500000)
#
# agents_list = initialize_agents(num_agents, random_policy_1d)
# training_loop(agents_list, side_length, S, phi, 0.001, "random")


#run_environment_1d(num_agents, random_policy_1d, side_length, S, phi, "Random", "Uniform")
