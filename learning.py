from orchard.environment import *
import numpy as np
import matplotlib.pyplot as plt
import orchard.environment_one
import random
from policies.nearest_uniform import replace_agents_1d
from policies.random_policy import random_policy_1d, random_policy
from policies.nearest import nearest_1d, nearest
from metrics.metrics import append_metrics, plot_metrics, append_positional_metrics, plot_agent_specific_metrics

import torch
import torch.nn as nn
import torch.optim as optim

from train_central import training_loop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
side_length = 20
num_agents = 5
S = np.zeros((side_length, 1))
for i in range(side_length):
    S[i] = 0.02
phi = 0.02
discount = 0.9

"""

"""
TRIAL SETTINGS
"""
side_length = 5
num_agents = 2
S = np.zeros((side_length, 1))
for i in range(side_length):
    S[i] = 0.04
phi = 0.2
discount = 0.99

from models.simple_connected_multiple import SimpleConnectedMultiple
from models.qnet_connected import CentralizedQNetwork
from agents.simple_agent import SimpleAgent
from agents.communicating_agent import CommAgent
from train_decentral import training_loop as training_loop_d


# CENTRALIZED LEARN ING

agents_list = []
for i in range(num_agents):
    agents_list.append(SimpleAgent(policy=random_policy_1d, num="f"))


for i in range(1):
    print("loop", i)
    # From experience: around 350000 - 400000 is good
    training_loop(agents_list, side_length, S, phi, 0.0002, "C-RANDOM_2_5", discount=0.99, timesteps=800000)

    if i > 0:
        print("Total Same Actions:", agents_list[0].same_actions)
        agents_list[0].same_actions = 0
"""
# DECENTRALIZED LEARNING
agents_list = []
for i in range(num_agents):
    agents_list.append(CommAgent(policy=random_policy_1d, num="f"))

for i in range(1):
    print("loop", i)

    training_loop_d(agents_list, side_length, S, phi, 0.0001, "D-RANDOM_4_10_99", discount=0.99, timesteps=800000)

    if i > 0:
        print("Total Same Actions:", agents_list[0].same_actions)
        agents_list[0].same_actions = 0

"""
# num_agents = 2
# agents_list = []
# for i in range(num_agents):
#     agents_list.append(SimpleAgent(policy=nearest, num="f"))
# for i in range(1):
#     print("loop", i)
#     training_loop(agents_list, side_length, S, phi, 0.0001, "simple_2_agent_length_20", discount=0.99, timesteps=300000)
#     if i > 0:
#         print("Total Same Actions:", agents_list[0].same_actions)
#         agents_list[0].same_actions = 0
# num_agents = 5
# agents_list = []
# for i in range(num_agents):
#     agents_list.append(SimpleAgent(policy=nearest, num="f"))
#
# for i in range(1):
#     print("loop", i)
#
#     training_loop(agents_list, side_length, S, phi, 0.0001, "simple_5_agent_length_20", discount=0.99, timesteps=300000)
#
#     if i > 0:
#         print("Total Same Actions:", agents_list[0].same_actions)
#         agents_list[0].same_actions = 0
#

# agents_list = []
# for i in range(num_agents):
#     agents_list.append(SimpleAgent(policy=nearest, num=i))
#
# epsilon = 0.5
# for i in range(10):
#     print("loop", i)
#
#     training_loop(agents_list, side_length, S, phi, 0.0001, "epsilon", epsilon=epsilon, timesteps=50000)
#     epsilon -= 0.04
#     if epsilon < 0.05:
#         epsilon = 0.05
#
#
#     for agent in agents_list:
#         agent.copy_value_function()
#         agent.policy = "value_function"
#






