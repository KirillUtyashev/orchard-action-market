import numpy as np
from policies.random_policy import random_policy
from policies.nearest import nearest
from models.connected import Connected
from models.simple_connected import SimpleConnected
from models.actor_dc_1d import ActorNetwork
from models.qnet import QNetwork
from models.unet import UNetwork
import torch.nn.functional as F

from models.util import ReplayBuffer

import random

import torch
import torch.nn as nn
import torch.optim as optim
torch.set_default_dtype(torch.float64)

action_vectors = [
            np.array([-1, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([0, -1]),
            np.array([0, 0])
        ]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ACAgent:
    def __init__(self, policy=random_policy, model=SimpleConnected, debug=False, num=0, num_agents=1, is_beta_agent=0, is_alpha_agent=0, is_projecting=0):
        self.position = np.array([0, 0])
        self.policy = policy
        self.policy_value = None
        self.basic_network = None

        self.policy_network = None

        self.debug = debug

        self.same_actions = 0

        self.scheduler = None
        self.num = num

        self.agent_rates = np.zeros(num_agents)
        for i in range(self.agent_rates.size):
            self.agent_rates[i] = 1

        self.alphas = np.zeros(num_agents)
        self.alpha_agents = np.zeros(num_agents)
        self.beta = 0
        self.times = 0
        self.avg_alpha = None
        self.beta_factor = 0.999
        if is_beta_agent:
            print("Beta Agent")
            if is_alpha_agent:
                print("& Alpha Agent")
        self.beta_agent = is_beta_agent
        self.alpha_agent = is_alpha_agent
        self.is_projecting = is_projecting

        # projection
        self.proj_beta = 0
        self.proj_alpha = 0

    def get_comm_value_function(self, a, b, agents_list, new_pos=None, debug=False, agent_poses=None):
        sum = 0
        if debug:
            assert agent_poses is not None
            for num, agent in enumerate(agents_list):
                sum += agent.policy_value.get_value_function(a, b, np.array(agent_poses[num]))
        else:
            assert new_pos is not None
            for agent in agents_list:
                if agent.num == self.num:
                    sum += agent.policy_value.get_value_function(a, b, new_pos)
                else:
                    sum += agent.policy_value.get_value_function(a, b, agent.position)
        return sum

    def get_value_function(self, a, b):
        f = self.policy_value.get_value_function(a, b, self.position)
        return f

    def get_value_function_bin(self, a, b, pos=None):
        assert self.avg_alpha is not None
        if pos is None:
            pos = self.position
        v = self.policy_value.get_value_function(a, b, pos)[0]
        val = (v - self.avg_alpha) / self.avg_alpha
        bound = 0.885
        val = ((val + 1) / 2) / bound
        if val > 1:
            val = 1
        elif val < 0:
            val = 0
        return np.array([val])

    def get_value_function_override(self, a, b, pos):
        f = self.policy_value.get_value_function(a, b, pos)
        return f

    def get_learned_action(self, state):
        a = state["agents"]
        b = state["apples"]

        actions = [0, 1, 4]
        #print(list(a.flatten()), list(b.flatten()), self.position)
        output = self.policy_network.get_function_output(a, b, self.position)
        #output = np.exp(output)
        #print(output)
        action = random.choices(actions, weights=output)[0]
        return action

    def get_base_action(self, state):
        a = state["agents"]
        b = state["apples"]

        actions = [0, 1, 4]
        #print(list(a.flatten()), list(b.flatten()), self.position)
        output = self.basic_network.get_function_output(a, b, self.position)
        #output = np.exp(output)
        #print(output)
        action = random.choices(actions, weights=output)[0]
        return action

    def update_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def get_action(self, state, discount=0.99, agent_poses=None, agents_list=None, device=device):
        if self.policy == "learned_policy":
            return self.get_learned_action(state)
        elif self.policy == "baseline":
            return self.get_base_action(state)
        else:
            return self.policy(state, self.position)



