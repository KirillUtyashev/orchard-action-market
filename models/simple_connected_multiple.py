import numpy as np
from policies.random_policy import random_policy
from policies.nearest import nearest
from models.connected import Connected
from models.simple_connected import SimpleConnected
from models.qnet import QNetwork
from models.unet import UNetwork


from models.util import ReplayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
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

def unwrap_state(state):
    return state["agents"].copy(), state["apples"].copy()

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
def convert_input(a, b, agent_pos):
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

class SimpleConnectedMultiple(nn.Module):
    def __init__(self, oned_size): # we work with 1-d size here.
        super(SimpleConnectedMultiple, self).__init__()
        # self.layer1 = nn.Linear(oned_size * 2, 64)
        # self.layer2 = nn.Linear(64, 64)
        # self.layer4 = nn.Linear(64, 1)
        #self.layer1 = nn.Linear(oned_size * 2, 256)

        # FIRST INPUT TYPE MODEL
        if oned_size == 5:
            self.layer1 = nn.Conv1d(1, 6, 3, 1)
            #self.layer2 = nn.Linear(48, 64) # 48 for an input dimension of 10 (i.e. oned size is 5)
            self.layer2 = nn.Linear(48, 64)
            #self.layer2 = nn.Linear(64, 64)
            self.layer3 = nn.Linear(64, 64)
            self.layer4 = nn.Linear(64, 1)
        elif oned_size == 10:
            self.layer1 = nn.Conv1d(1, 6, 3, 1)
            # self.layer2 = nn.Linear(48, 64) # 48 for an input dimension of 10 (i.e. oned size is 5)
            self.layer2 = nn.Linear(108, 128)
            # self.layer2 = nn.Linear(64, 64)
            self.layer3 = nn.Linear(128, 128)
            # self.layer5 = nn.Linear(256, 128)
            self.layer4 = nn.Linear(128, 1)

        #self.layer2 = nn.Linear(108, 128)
        ## self.layer2 = nn.Linear(64, 64)
        #self.layer3 = nn.Linear(128, 128)
        #self.layer4 = nn.Linear(128, 1)

        # SECOND INPUT TYPE MODEL
        # self.layer1 = nn.Conv1d(1, 6, 3, 1)
        # self.layer2 = nn.Linear(12, 32)
        # self.layer3 = nn.Linear(32, 32)
        # self.layer4 = nn.Linear(32, 1)

        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
        torch.nn.init.xavier_uniform_(self.layer4.weight)
        print("Initialized Neural Network")

    def forward(self, a, b, pos):
        x = torch.cat((a.flatten(), b.flatten()))
        #print(x)
        #print(x)
        x = x.view(1, -1)
        x = F.leaky_relu(self.layer1(x))
        x = x.flatten()
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        return self.layer4(x)

counter = 0
total_reward = 0
class SCMNetwork():
    def __init__(self, oned_size, alpha, discount):
        self.function = SimpleConnectedMultiple(oned_size)
        self.function.to(device)
        self.optimizer = optim.AdamW(self.function.parameters(), lr=alpha, amsgrad=True)
        self.alpha = alpha
        self.discount = discount
        print(self.alpha)
        print(self.discount)

    def get_value_function(self, a, b, pos=None):
        # state1 = convert_input(a, b, pos)
        # a, b = state1[0], state1[1]
        with torch.no_grad():
            val = self.function(ten(a), ten(b), None).cpu().numpy()
        #return self.function(ten(a), ten(b), None).detach().cpu().numpy()
        return val

    def get_value_function2(self, state):
        #a, b = unwrap_state(state)
        a, b = state[0], state[1]
        return self.function(ten(a), ten(b), None).detach().cpu().numpy()

    def train(self, state, new_state, reward):
        new_pos = None
        old_pos = None

        debug = False
        if debug:
            print("=========TRAINING=========")
            print(list(state["agents"].flatten()), list(state["apples"].flatten()))
            print(list(new_state["agents"].flatten()), list(new_state["apples"].flatten()))
            print(reward)
        a, b = unwrap_state(state)
        new_a, new_b = unwrap_state(new_state)

        approx = self.function(ten(a), ten(b), old_pos)
        with torch.no_grad():
            target = reward + self.discount * self.function(ten(new_a), ten(new_b), new_pos)
        #target = 1 + self.discount * self.function(ten(new_a), ten(new_b), new_pos)
        #criterion = torch.nn.L1Loss()
        criterion = torch.nn.MSELoss()
        self.optimizer.zero_grad()

        loss = criterion(approx, target)
        loss.backward()
        self.optimizer.step()

        # global counter
        # global total_reward
        # total_reward += reward
        # counter += 1
        # if counter % 10000 == 0:
        #     print(total_reward, counter)



