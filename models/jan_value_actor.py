import numpy as np
from policies.random_policy import random_policy
from policies.nearest import nearest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_dtype(torch.float64)

"""
The VALUE FUNCTION network in the MARL environment.
"""

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

class SimpleConnectedMultiple(nn.Module):
    def __init__(self, oned_size): # we work with 1-d size here.
        super(SimpleConnectedMultiple, self).__init__()

        if oned_size == 10:
            self.layer1 = nn.Conv1d(1, 6, 3, 1)
            self.layer2 = nn.Linear(114, 128)
            self.layer3 = nn.Linear(128, 128)
            self.layer4 = nn.Linear(128, 1)
        elif oned_size == 5:
            self.layer1 = nn.Conv1d(1, 6, 3, 1)
            self.layer2 = nn.Linear(54, 64)
            self.layer3 = nn.Linear(64, 64)
            self.layer4 = nn.Linear(64, 1)
        elif oned_size == 20:
            self.layer1 = nn.Conv1d(1, 6, 3, 1)
            self.layer2 = nn.Linear(234, 128)
            self.layer3 = nn.Linear(128, 128)
            self.layer4 = nn.Linear(128, 1)
        else:
            outl = 6 * ((oned_size * 2 + 1) - 2)
            self.layer1 = nn.Conv1d(1, 6, 3, 1)
            self.layer2 = nn.Linear(outl, 512)
            self.layer3 = nn.Linear(512, 256)
            self.layer4 = nn.Linear(256, 1)

        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
        torch.nn.init.xavier_uniform_(self.layer4.weight)
        print("Initialized Neural Network")

    def forward(self, a, b, pos):
        x = torch.cat((a.flatten(), b.flatten(), pos.flatten()))
        x = x.view(1, -1)
        x = F.leaky_relu(self.layer1(x))
        x = x.flatten()
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        #x = F.leaky_relu(self.layer5(x))
        return self.layer4(x)

counter = 0
total_reward = 0
class ValueNetwork():
    def __init__(self, oned_size, alpha, discount, num=None):
        self.function = SimpleConnectedMultiple(oned_size)
        self.function.to(device)
        self.optimizer = optim.AdamW(self.function.parameters(), lr=alpha, amsgrad=True)
        self.alpha = alpha
        self.discount = discount
        self.num = num

    def get_value_function(self, a, b, pos):
        pose = np.array([pos[0]])
        with torch.no_grad():
            val = self.function(ten(a), ten(b), ten(pose)).cpu().numpy()

        return val

    def train(self, state, new_state, reward):
        old_pos = np.array([state["pos"][0]])
        new_pos = np.array([new_state["pos"][0]])

        debug = False
        if debug:
            print("=========TRAINING=========")
            print(list(state["agents"].flatten()), list(state["apples"].flatten()))
            print(list(new_state["agents"].flatten()), list(new_state["apples"].flatten()))
            print(old_pos, new_pos)
            print(reward)
        a, b = unwrap_state(state)
        new_a, new_b = unwrap_state(new_state)

        approx = self.function(ten(a), ten(b), ten(old_pos))
        with torch.no_grad():
            target = reward + self.discount * self.function(ten(new_a), ten(new_b), ten(new_pos))
        # target = 1 + self.discount * self.function(ten(new_a), ten(new_b), new_pos)
        # criterion = torch.nn.L1Loss()
        criterion = torch.nn.MSELoss()
        self.optimizer.zero_grad()

        loss = criterion(approx, target)
        loss.backward()
        self.optimizer.step()

        t, ap = target.detach().cpu().numpy(), approx.detach().cpu().numpy()

        return t, ap

    def train_with_learned_util(self, state, new_state, reward, old_pos, new_pos, target, att):
        old_pos = np.array([old_pos[0]])
        new_pos = np.array([new_pos[0]])

        debug = False
        if debug:
            print("=========TRAINING=========")
            print(list(state["agents"].flatten()), list(state["apples"].flatten()))
            print(list(new_state["agents"].flatten()), list(new_state["apples"].flatten()))
            print(old_pos, new_pos)
            print(reward)
        a, b = unwrap_state(state)
        new_a, new_b = unwrap_state(new_state)
        approx = self.function(ten(a), ten(b), ten(old_pos))
        with torch.no_grad():
            q = self.discount * self.function(ten(new_a), ten(new_b), ten(new_pos))
            if q < 0:
                q = ten([0])
            target = reward + q
        print(approx, target, reward)

        criterion = torch.nn.MSELoss()
        self.optimizer.zero_grad()
        loss = criterion(approx, target) # * att
        loss.backward()
        self.optimizer.step()
        return target.detach().cpu(), approx.detach().cpu()

    def just_get_q_v(self, state, new_state, reward, old_pos, new_pos):
        old_pos = np.array([old_pos[0]])
        new_pos = np.array([new_pos[0]])

        debug = False
        if debug:
            print("=========TRAINING=========")
            print(list(state["agents"].flatten()), list(state["apples"].flatten()))
            print(list(new_state["agents"].flatten()), list(new_state["apples"].flatten()))
            print(old_pos, new_pos)
            print(reward)
        a, b = unwrap_state(state)
        new_a, new_b = unwrap_state(new_state)

        with torch.no_grad():
            approx = self.function(ten(a), ten(b), ten(old_pos))
            target = reward + self.discount * self.function(ten(new_a), ten(new_b), ten(new_pos))

        return target.detach().cpu(), approx.detach().cpu()
