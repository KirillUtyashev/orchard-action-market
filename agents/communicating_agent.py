import numpy as np
from policies.random_policy import random_policy
from policies.nearest import nearest

import torch
import torch.nn as nn
import torch.optim as optim

"""
The "Communicating Agent" - The decentralized agent that has its own value functions. Retrieves Q-values from other agents in the list.
"""
torch.set_default_dtype(torch.float64)

action_vectors = [
            np.array([-1, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([0, -1]),
            np.array([0, 0])
        ]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class CommAgent:
    def __init__(self, policy=random_policy, model=None, debug=False, num=0, num_agents=1):
        self.position = np.array([0, 0])
        self.policy = policy
        self.policy_value = None
        self.policy_value2 = None

        self.debug = debug

        self.same_actions = 0

        self.scheduler = None
        self.num = num

        self.alphas = np.zeros(num_agents)
        self.alpha_agents = np.zeros(num_agents)
        self.beta = 0
        self.times = 0
        self.agents_list = None

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

    def get_comm_value_function_alt(self, a, b, agents_list, new_pos=None, debug=False, agent_poses=None):
        sum = 0
        if debug:
            assert agent_poses is not None
            for num, agent in enumerate(agents_list):
                sum += agent.policy_value2.get_value_function(a, b, np.array(agent_poses[num]))
        else:
            assert new_pos is not None
            for agent in agents_list:
                if agent.num == self.num:
                    sum += agent.policy_value2.get_value_function(a, b, new_pos)
                else:
                    sum += agent.policy_value2.get_value_function(a, b, agent.position)
        return sum

    def get_value_function(self, a, b):
        f = self.policy_value.get_value_function(a, b, self.position)
        return f

    def get_best_action(self, state, discount, agents_list):
        a = state["agents"]
        b = state["apples"]

        action = 0
        best_val = 0
        if self.debug:
            print("==========Making Decision for Agent "+str(self.num)+"===========")
            print(list(a.flatten()), list(b.flatten()))

        #for act in range(len(action_vectors)):
        for act in [0, 1, 4]:
            val, new_a, new_b, new_pos = calculate_ir(a, b, self.position, act)
            rew = val
            val += discount * self.get_comm_value_function(new_a, new_b, agents_list, new_pos=new_pos)
            if self.debug:
                print("Action " + str(action_vectors[act]) + " has expected value " + str(val) + "; immediate reward " + str(rew))
            if val > best_val:
                action = act
                best_val = val

        # self.same_actions += (action == nearest(state, self.position))
        if self.debug:
            print("Ultimately chose " + str(action_vectors[action]) + " with expected value " + str(best_val) + ".")

        return action

    def get_best_action_alt(self, state, discount, agents_list):
        a = state["agents"]
        b = state["apples"]

        action = 0
        best_val = 0
        if self.debug:
            print("==========Making Decision for Agent " + str(self.num) + "===========")
            print(list(a.flatten()), list(b.flatten()))

        for act in [0, 1, 4]:
            val, new_a, new_b, new_pos = calculate_ir(a, b, self.position, act)
            rew = val
            val += discount * self.get_comm_value_function_alt(new_a, new_b, agents_list, new_pos=new_pos)
            if self.debug:
                print("Action " + str(action_vectors[act]) + " has expected value " + str(
                    val) + "; immediate reward " + str(rew))
            if val > best_val:
                action = act
                best_val = val

        # self.same_actions += (action == nearest(state, self.position))
        if self.debug:
            print("Ultimately chose " + str(action_vectors[action]) + " with expected value " + str(best_val) + ".")

        return action

    def update_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def get_action(self, state, discount=0.99, agent_poses=None, agents_list=None, device=device):
        if self.policy == "value_function":
            if agents_list is None:
                agents_list = self.agents_list
            return self.get_best_action(state, discount, agents_list)
        if self.policy == "value_function2":
            if agents_list is None:
                agents_list = self.agents_list
            return self.get_best_action_alt(state, discount, agents_list)
        else:
            return self.policy(state, self.position)



