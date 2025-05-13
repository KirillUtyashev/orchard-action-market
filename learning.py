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
The LEARNING file. This serves as an execution file for both centralized and decentralizing learning.
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
from agents.simple_agent import SimpleAgent
from agents.communicating_agent import CommAgent
from train_decentral import training_loop as training_loop_d

"""
# CENTRALIZED LEARNING

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







