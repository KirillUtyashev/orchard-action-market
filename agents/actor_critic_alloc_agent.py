import numpy as np
from policies.random_policy import random_policy
from policies.nearest import nearest
from models.value_function import MainNet

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


def ten(c):
    return torch.from_numpy(c).to(device).double()


def calculate_ir(a, b, pos, action):
    new_pos = np.clip(pos + action_vectors[action], [0, 0], a.shape-np.array([1, 1]))
    agents = a.copy()
    apples = b.copy()
    agents[new_pos[0], new_pos[1]] += 1
    agents[pos[0], pos[1]] -= 1
    if apples[new_pos[0], new_pos[1]] > 0:
        apples[new_pos[0], new_pos[1]] -= 1
        return 1, agents, apples, new_pos
    else:
        return 0, agents, apples, new_pos


def onehot(a, pos):
    length = a.size
    posr = np.zeros(length)
    posr[pos[0]] = 1
    return posr


def evaluate_value_function(a, b, pos, function, device):
    pos = ten(onehot(a, pos), device)
    a = ten(a, device)
    b = ten(b, device)

    return function(a.flatten(), b.flatten(), pos)


class ACAgent:
    def __init__(self, policy=random_policy, model=MainNet, debug=False, num=0, num_agents=1):
        self.position = np.array([0, 0])
        self.policy = policy
        self.policy_value = None

        self.basic_network = None
        self.policy_network = None

        self.debug = debug

        self.same_actions = 0

        self.scheduler = None
        self.num = num

        self.alphas = np.zeros(num_agents)
        self.alpha_agents = np.zeros(num_agents)
        self.agent_rates = np.zeros(num_agents)
        for i in range(self.agent_rates.size):
            self.agent_rates[i] = 1
        self.beta = 0
        self.times = 0
        self.avg_alpha = None

    def set_alloc_rates(self, rates, max=1):
        # assert np.sum(rates) <= max
        self.agent_rates = np.array(rates)

    def get_comm_value_function(self, a, b, agents_list, new_pos=None, debug=False, agent_poses=None):
        sum = 0
        if debug:
            assert agent_poses is not None
            for num, agent in enumerate(agents_list):
                sum += agent.policy_value.get_sum_value(a, b, np.array(agent_poses[num])) * agents_list[num].agent_rates[self.num]
        else:
            assert new_pos is not None
            for num, agent in enumerate(agents_list):
                if agent.num == self.num:
                    sum += agent.policy_value.get_sum_value(a, b, new_pos) * self.agent_rates[num]
                else:
                    sum += agent.policy_value.get_sum_value(a, b, agent.position) * self.agent_rates[num]
        return sum

    def get_value_function(self, a, b):
        f = self.policy_value.get_sum_value(a, b, self.position)
        return f

    def get_value_function_override(self, a, b, pos):
        f = self.policy_value.get_sum_value(a, b, pos)
        return f

    def get_learned_action(self, state):
        a = state["agents"]
        b = state["apples"]

        actions = [0, 1, 4]
        output = self.policy_network.get_function_output(a, b, self.position)

        action = random.choices(actions, weights=output)[0]
        return action

    def get_base_action(self, state):
        a = state["agents"]
        b = state["apples"]

        actions = [0, 1, 4]
        output = self.basic_network.get_function_output(a, b, self.position)
        action = random.choices(actions, weights=output)[0]
        return action

    def update_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def get_best_action(self, state, discount, agents_list):
        a = state["agents"]
        b = state["apples"]

        action = 0
        best_val = 0
        if self.debug:
            print("==========Making Decision for Agent "+str(self.num)+"===========")
            print(list(a.flatten()), list(b.flatten()))

        for act in range(len(action_vectors)):
            val, new_a, new_b, new_pos = calculate_ir(a, b, self.position, act)
            rew = val
            val += discount * self.get_comm_value_function(new_a, new_b, agents_list, new_pos=new_pos)
            if self.debug:
                print("Action " + str(action_vectors[act]) + " has expected value " + str(val) + "; immediate reward " + str(rew))
            if val > best_val:
                action = act
                best_val = val

        if self.debug:
            print("Ultimately chose " + str(action_vectors[action]) + " with expected value " + str(best_val) + ".")

        return action

    def get_action(self, state, discount=0.99, agent_poses=None, agents_list=None, device=device):
        if self.policy == "learned_policy":
            return self.get_learned_action(state)
        elif self.policy == "baseline":
            return self.get_base_action(state)
        elif self.policy == "value_function":
            return self.get_best_action(state, discount, agents_list)
        else:
            return self.policy(state, self.position)



