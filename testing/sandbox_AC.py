import numpy as np
from policies.random_policy import random_policy_1d, random_policy
from orchard.algorithms import single_apple_spawn, single_apple_despawn, single_apple_spawn_malicious

from main import run_environment_1d

import torch
torch.set_default_dtype(torch.float64)



side_length = 5
num_agents = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
A-C Sandbox File for Measuring Performance.

"""

"""
"Regular" Agent Trials
"""
side_length = 5
num_agents = 2
S = np.zeros((side_length, 1))
for i in range(side_length):
    S[i] = 0.04
phi = 0.2
discount = 0.99

"""
"""

a_list = []

from agents.actor_critic_alloc_agent import ACAgent
from models.actor_dc_1d import ActorNetwork
experiment_name = "policyitchk/D-2_5_comp1_value/D-2_5_comp1_value"

network_list = []
for i in range(num_agents):
    print("Initializing Agents")
    network = ActorNetwork(side_length, 0.001, discount)
    # network.function.load_state_dict(torch.load("../" + experiment_name + "_Actor1_" + str(i) + ".pt"))
    network.function.load_state_dict(torch.load("../" + experiment_name + "_" + str(i) + "_it_2.pt"))
    #network.function.load_state_dict(torch.load("../" + experiment_name + "_Actor3_" + str(i) + ".pt"))
    #network.function.load_state_dict(torch.load("../" + experiment_name + "_Actor_BETA_" + str(i) + ".pt"))
    #network.function.load_state_dict(torch.load("../" + experiment_name + "_Actor_BETA_ALPHA_" + str(i) + ".pt"))
    for param in network.function.parameters():
        print(param.data)
    network_list.append(network)

for i in range(num_agents):

    trained_agent = ACAgent(policy="learned_policy", num=i, debug=False)
    print(trained_agent.num)
    trained_agent.policy_network = network_list[i]
    trained_agent.basic_network = network_list[i]
    a_list.append(trained_agent)

# saved_graph_name = experiment_name
saved_graph_name = "AC-Value-2-5"
import time
start = time.time()
run_environment_1d(num_agents, random_policy_1d, side_length, S, phi, "AC", saved_graph_name, agents_list=a_list, spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn, timesteps=30000)
end = time.time()

print(end - start)
sample_state = {
    "agents": np.array([[0], [1], [0], [0], [3], [0], [0], [0], [0], [0]]),
    "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
    "pos": [np.array([1, 0]), np.array([4, 0]), np.array([4, 0]), np.array([4, 0])]
}
sample_state1 = {
    "agents": np.array([[0], [1], [0], [0], [3], [0], [0], [0], [0], [0]]),
    "apples": np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
    "pos": [np.array([1, 0]), np.array([4, 0]), np.array([4, 0]), np.array([4, 0])]
}
a = np.array([sample_state["agents"].flatten(), sample_state1["agents"].flatten()])
b = np.array([sample_state["apples"].flatten(), sample_state1["apples"].flatten()])
c = np.array([[1], [4], [4]])
print(a)
#v_value = a_list[0].policy_network.get_function_output(sample_state["agents"], sample_state["apples"],
#                                                            pos=sample_state["pos"][0])

with torch.no_grad():
    for _ in range(1000):
        v_value = a_list[0].policy_value.get_value_function(sample_state["agents"], sample_state["apples"],
                                                                    pos=sample_state["pos"][0])
        v_value = a_list[1].policy_value.get_value_function(sample_state["agents"], sample_state["apples"],
                                                            pos=sample_state["pos"][1])
        v_value = a_list[2].policy_value.get_value_function(sample_state["agents"], sample_state["apples"],
                                                            pos=sample_state["pos"][2])
        v_value = a_list[3].policy_value.get_value_function(sample_state["agents"], sample_state["apples"],
                                                            pos=sample_state["pos"][3])

print(v_value)



