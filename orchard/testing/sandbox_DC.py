import numpy as np
import random
from policies.random_policy import random_policy_1d, random_policy
from orchard.algorithms import single_apple_spawn, single_apple_despawn, single_apple_spawn_malicious

from main import run_environment_1d

import torch
torch.set_default_dtype(torch.float64)

random.seed(4327423786)

side_length = 5
num_agents = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Decentralized Agent Sandbox
"""

"""
"Regular" Agent Trials
"""
side_length = 5 #20
num_agents = 2 #5
S = np.zeros((side_length, 1))
for i in range(side_length):
    S[i] = 0.04
phi = 0.2
discount = 0.99

"""
"""

experiment_name = "policyitchk/D-2_5_comp1_value/D-2_5_comp1_value_decen"


a_list = []
from models.simple_connected_multiple_dc import SCMNetwork
from agents.communicating_agent import CommAgent

network_list = []

for i in range(num_agents):
    network = SCMNetwork(side_length, 0.001, discount)
    network.function.load_state_dict(torch.load("../" + experiment_name + "_" + str(i) + "_it_2.pt"))
    #network.function.load_state_dict(torch.load("../" + experiment_name + "_" + str(i) + ".pt"))
    network_list.append(network)

for i in range(num_agents):
    trained_agent = CommAgent(policy="value_function", num=i, debug=False)
    print(trained_agent.num)
    trained_agent.policy_value = network_list[i]
    a_list.append(trained_agent)
#
#
import time
start = time.time()
run_environment_1d(num_agents, random_policy_1d, side_length, S, phi, "Decentralized", experiment_name, agents_list=a_list, spawn_algo=single_apple_spawn, despawn_algo=single_apple_despawn, timesteps=20000)
end = time.time()
# state = {
#         "agents": np.array(
#             [[1], [0], [0], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
#         "apples": np.array(
#             [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [0], [0]])
#     }
#
# a_list[0].position = np.array([0, 0])
# a_list[1].position = np.array([3, 0])
# # a_list[2].position = np.array([5, 0])
# # a_list[3].position = np.array([7, 0])
# #
state = {
        "agents": np.array(
            [[0], [0], [1], [3], [0], [0], [0], [0], [0], [0]]),
        "apples": np.array(
            [[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
    }
# a_list[0].position = np.array([2, 0])
# a_list[1].position = np.array([4, 0])
# print(a_list[0].get_comm_value_function(state["agents"], state["apples"], a_list, debug=True,
 #                                        agent_poses=[np.array([1, 0]), np.array([4, 0]), np.array([4, 0]), np.array([4, 0])]))
# print(a_list[1].get_comm_value_function(state["agents"], state["apples"], a_list))





