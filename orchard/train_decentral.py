from agents.communicating_agent import CommAgent
from models.simple_connected_multiple_blind import SCMBNetwork
#from models.simple_connected_multiple import SCMNetwork
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
# TEMP:
# from models.simple_connected_multiple_dc import SCMNetwork, SimpleConnectedMultiple
from models.simple_connected_multiple_dc_altinput import SCMNetwork, SimpleConnectedMultiple
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
    plt.savefig("Params_" + name + str(graph) + ".png")

    plt.figure("loss" + str(graph), figsize=(10, 5))
    plt.plot(loss_plot)
    plt.plot(loss_plot1)
    plt.plot(loss_plot2)
    #print(loss_plot2)
    plt.title("Value Function for Sample State, iteration " + str(graph))
    plt.savefig("Val_" + name + str(graph) + ".png")

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
"""
sample_state = {
    "agents": np.array([[0], [1], [0], [0], [0], [0], [0], [0], [0], [0]]),
    "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
    "pos": np.array([1, 0])
}
sample_state5 = {
    "agents": np.array([[0], [0], [1], [0], [0], [0], [0], [0], [0], [0]]),
    "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
    "pos": np.array([2, 0])
}
sample_state6 = {
    "agents": np.array([[0], [0], [0], [0], [1], [0], [0], [0], [0], [0]]),
    "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
    "pos": np.array([4, 0])
}
"""
# sample_state = {
#     "agents": np.array([[0], [1], [0], [0], [3], [0], [0], [0], [0], [0]]),
#     "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "pos": [np.array([1, 0]), np.array([4, 0]), np.array([4, 0]), np.array([4, 0])]
# }
# sample_state5 = {
#     "agents": np.array([[0], [0], [1], [0], [3], [0], [0], [0], [0], [0]]),
#     "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "pos": [np.array([2, 0]), np.array([4, 0]), np.array([4, 0]), np.array([4, 0])]
# }
# sample_state6 = {
#     "agents": np.array([[0], [0], [0], [0], [4], [0], [0], [0], [0], [0]]),
#     "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "pos": [np.array([4, 0]), np.array([4, 0]), np.array([4, 0]), np.array([4, 0])]
# }

# sample_state = {
#     "agents": np.array([[0], [1], [0], [0], [0], [3], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "pos": [np.array([1, 0]), np.array([5, 0]), np.array([5, 0]), np.array([5, 0])]
# }
# sample_state5 = {
#     "agents": np.array([[0], [0], [1], [0], [0], [3], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "pos": [np.array([2, 0]), np.array([5, 0]), np.array([5, 0]), np.array([5, 0])]
# }
# sample_state6 = {
#     "agents": np.array([[0], [0], [0], [0], [1], [3], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "pos": [np.array([4, 0]), np.array([5, 0]), np.array([5, 0]), np.array([5, 0])]
# }

# sample_state = {
#     "agents": np.array([[0], [1], [0], [0], [1], [0], [0], [0], [0], [0]]),
#     "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "pos": [np.array([1, 0]), np.array([4, 0])]
# }
# sample_state5 = {
#     "agents": np.array([[0], [0], [1], [0], [1], [0], [0], [0], [0], [0]]),
#     "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "pos": [np.array([2, 0]), np.array([4, 0])]
# }
# sample_state6 = {
#     "agents": np.array([[0], [0], [0], [0], [2], [0], [0], [0], [0], [0]]),
#     "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#     "pos": [np.array([4, 0]), np.array([4, 0])]
# }



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

def get_possible_states(state, agent_pos):
    apples = state["apples"].copy()
    agents = state["agents"].copy()
    length = agents.size

    apples1 = apples.copy()
    agents1 = agents.copy()
    new_pos1 = [np.clip(agent_pos[0] - 1, 0, length-1), agent_pos[1]]
    agents1[agent_pos[0], agent_pos[1]] -= 1
    agents1[new_pos1[0], new_pos1[1]] += 1
    if apples1[new_pos1[0], new_pos1[1]] > 0:
        apples1[new_pos1[0], new_pos1[1]] -= 1
    state1 = {
        "apples": apples1,
        "agents": agents1
    }

    apples2 = apples.copy()
    agents2 = agents.copy()
    new_pos2 = [np.clip(agent_pos[0] + 1, 0, length - 1), agent_pos[1]]
    agents2[agent_pos[0], agent_pos[1]] -= 1
    agents2[new_pos2[0], new_pos2[1]] += 1
    if apples2[new_pos2[0], new_pos2[1]] > 0:
        apples2[new_pos2[0], new_pos2[1]] -= 1
    state2 = {
        "apples": apples2,
        "agents": agents2
    }

    if apples[agent_pos[0], agent_pos[1]] > 0:
        apples[agent_pos[0], agent_pos[1]] -= 1

    state3 = {
        "apples": apples,
        "agents": agents
    }
    #print(list(state1["apples"].flatten()), list(state1["agents"].flatten()))
    #print(list(state2["apples"].flatten()), list(state2["agents"].flatten()))
    #print(list(state3["apples"].flatten()), list(state3["agents"].flatten()))
    return state1, state2, state3

sample_state1 = {
        "agents": np.array(
            [[1], [0], [1], [0], [0]]),
        "apples": np.array(
            [[0], [1], [0], [0], [0]])
    }

debugging = False
def training_loop(agents_list, orchard_length, S, phi, alpha, name, discount=0.99, epsilon=0, timesteps=100000, iteration=99):
    print(orchard_length)
    print(len(agents_list))
    env = Orchard(orchard_length, len(agents_list), S, phi, one=True, spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn) # initialize env
    env.initialize(agents_list) # attach agents to env

    print("Experiment", name)
    network_list = []
    for agn in range(len(agents_list)):
        network = SCMNetwork(orchard_length, alpha, discount)
        agents_list[agn].policy_value = network
        network_list.append(network)
    total_reward = 0

    import time
    start = time.time()

    """ Plotting Setup """
    setup_plots(network_list[0].function.state_dict(), one_plot)
    global loss_plot
    loss_plot = []
    """"""
    if debugging:
        print(env.get_state())
        for i in agents_list:
            print(i.position)
    old_state_ag = None
    old_state_ap = None
    maxi = 0
    for i in range(timesteps):

        agent = random.randint(0, env.n - 1)  # Choose random agent
        old_pos = agents_list[agent].position.copy()  # Get position of this agent
        state = env.get_state()  # Get the current state (a COPY)
        if debugging:
            print(agent)
            print(list(state["agents"].flatten()), list(state["apples"].flatten()))
            print(list(old_pos))
            assert state["agents"][old_pos[0], old_pos[1]] >= 1
            if old_state_ag is not None:
                assert np.array_equal(old_state_ap, state["apples"])
                assert np.array_equal(old_state_ag, state["agents"])

        epsilon = 0.1
        chance = random.random()  # Epsilon-greedy policy. Epsilon is zero currently.
        if chance < epsilon:
            action = random_policy_1d(state, old_pos)
        else:
            action = agents_list[agent].get_action(state, discount)  # Get action.

        reward, new_position = env.main_step(agents_list[agent].position.copy(), action)  # Take the action. Observe reward, and get new position
        agents_list[agent].position = new_position.copy()  # Save new position.
        total_reward += reward  # Add to reward.

        new_state = env.get_state()

        """ Some asserts to make sure things make sense """
        if debugging:
            old_state_test = state["agents"].copy()
            old_state_test[old_pos[0], old_pos[1]] -= 1
            old_state_test[new_position[0], new_position[1]] += 1
            assert np.array_equal(old_state_test, new_state["agents"])
            if reward > 0:
                assert np.sum(state["apples"]) >= np.sum(new_state["apples"])
            old_state_ag = new_state["agents"].copy()
            old_state_ap = new_state["apples"].copy()

            print(list(new_state["agents"].flatten()), list(new_state["apples"].flatten()))
            # Check if the state is physically possible
            one, two, three = get_possible_states(state, old_pos)
            assert np.array_equal(new_state["agents"], one["agents"]) \
                or np.array_equal(new_state["agents"], two["agents"]) \
                or np.array_equal(new_state["agents"], three["agents"])

            assert np.sum(new_state["apples"].flatten()) <= 1
            if i % int(env.length / 2) == 0:
                assert np.sum(new_state["apples"].flatten()) == 1

            if state["apples"][agents_list[agent].position[0], agents_list[agent].position[1]] >= 1:
                assert reward == 1
            else:
                assert reward == 0
        """"""
        #print(list(env.get_state()["agents"].flatten()), list(env.get_state()["apples"].flatten()))
        #network.train(state, env.get_state(), reward, agents_list[agent], agent_poses)   # <- for BLIND
        #state1 = convert_input(state, old_pos)
        #state2 = convert_input(env.get_state(), agents_list[agent].position)
        for each_agent in range(len(agents_list)):
            if each_agent == agent:
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
                network_list[agent].train(sp_state, sp_new_state, reward)
            else:
                sp_state = {
                    "agents": state["agents"].copy(),
                    "apples": state["apples"].copy(),
                    "pos": agents_list[each_agent].position.copy()
                }
                sp_new_state = {
                    "agents": new_state["agents"].copy(),
                    "apples": new_state["apples"].copy(),
                    "pos": agents_list[each_agent].position.copy()
                }
                network_list[each_agent].train(sp_state, sp_new_state, 0)                             # <- for NORMAL
        #network.train(state1, state2, reward)  # <- for NORMAL

        """ For Plotting """
        # if i == 1000:
        #     start = time.time()
        # if i == 2000:
        #     end = time.time()
        #     print(end - start)
        if i % 2500 == 0 and i != 0:
            add_to_plots(network_list[0].function.state_dict(), i, one_plot)
            v_value = agents_list[0].get_comm_value_function(sample_state["agents"], sample_state["apples"], agents_list, debug=True, agent_poses=sample_state["pos"])
            #v_value1 = agents_list[0].get_comm_value_function(sample_state5["agents"], sample_state5["apples"], agents_list, debug=True, agent_poses=sample_state5["pos"])
            #v_value2 = agents_list[0].get_comm_value_function(sample_state6["agents"], sample_state6["apples"], agents_list, debug=True, agent_poses=sample_state6["pos"])
            if i % 5000 == 0:
                print("P", v_value)
            #add_to_plots(agents_list[1].value.state_dict(), i, two_plot)
            loss_plot.append(v_value[0])
            #loss_plot1.append(v_value1[0])
            #loss_plot2.append(v_value2[0])
            #mini_loss_plot = []
        if i % 20000 == 0:
            print("At timestep", i)
            #print("At timestep", i)
        if i == 50000:
            for network in network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.0001
        if i == 150000:
            for network in network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.00005
        if i == 250000:
            for network in network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.00001
        if i == 400000:
            for network in network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.000005
        if i == 600000:
            for network in network_list:
                for g in network.optimizer.param_groups:
                    g['lr'] = 0.000002
        # if i == 150000:
        #     for network in network_list:
        #         for g in network.optimizer.param_groups:
        #             g['lr'] = 0.0001 / 2
        # if i == 300000:
        #     for network in network_list:
        #         for g in network.optimizer.param_groups:
        #             g['lr'] = 0.00005 / 2
        # if i == 500000:
        #     for network in network_list:
        #         for g in network.optimizer.param_groups:
        #             g['lr'] = 0.00001
        # if i == 1000000:
        #     for network in network_list:
        #         for g in network.optimizer.param_groups:
        #             g['lr'] = 0.000003
        # if i == 400000:
        #     for network in network_list:
        #         for g in network.optimizer.param_groups:
        #             g['lr'] = 0.00001
        if (i % 50000 == 0 and i != 0) or i == timesteps - 1:
            print("=====Eval at", i, "steps======")
            fname = name
            maxi = eval_network(fname, maxi, discount, network_list, num_agents=len(agents_list), side_length=orchard_length, iteration=iteration)
            print("=====Completed Evaluation=====")
    for numbering, network in enumerate(network_list):
        torch.save(network.function.state_dict(), name + "_" + str(numbering) + ".pt")
    graph_plots(network_list[0].function.state_dict(), name, one_plot)
    print("Total Reward:", total_reward)
    print("Total Apples:", env.total_apples)

from main import run_environment_1d
def eval_network(name, maxi, discount, network_list, num_agents=4, side_length=10, iteration=99):
    a_list = []

    for ii in range(num_agents):
        trained_agent = CommAgent(policy="value_function", num=ii, num_agents=num_agents)
        # print(trained_agent.num)
        trained_agent.policy_value = network_list[ii]
        a_list.append(trained_agent)
    with torch.no_grad():
        val = run_environment_1d(num_agents, random_policy_1d, side_length, None, None, "DC", "test", agents_list=a_list,
                           spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn, timesteps=20000)
    if val > maxi and iteration != 99:
        print("saving best")
        for nummer, netwk in enumerate(network_list):
            torch.save(netwk.function.state_dict(),
                       "policyitchk/" + name + "/" + name + "_decen_" + str(nummer) + "_it_" + str(iteration) + ".pt")

    maxi = max(maxi, val)
    return maxi


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
