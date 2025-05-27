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

N = 5
     # 1 agent per 5 spaces
def convert_altinput(a, b, agent_pos):
    """
    Convert to a view around the agent.
    :param a:
    :param b:
    :param agent_pos:
    :return:
    """

    ap = np.concatenate(([-1, -1, -1, -1, -1], a.flatten(), [-1, -1, -1, -1, -1]))
    bp = np.concatenate(([-1, -1, -1, -1, -1], b.flatten(), [-1, -1, -1, -1, -1]))

    leftmost = agent_pos[0]-N
    rightmost = agent_pos[0] + N
    true_a = ap[leftmost+N: rightmost+N+1]
    true_b = bp[leftmost + N: rightmost + N + 1]

    return true_a, true_b, agent_pos[0] / len(a.flatten())

class SimpleConnectedMultiple(nn.Module):
    def __init__(self, oned_size): # we work with 1-d size here.
        super(SimpleConnectedMultiple, self).__init__()
        # self.layer1 = nn.Linear(oned_size * 2 + 1, 64)
        # self.layer2 = nn.Linear(64, 64)
        # # self.layer3 = nn.Linear(64, 64)
        # self.layer4 = nn.Linear(64, 1)
        # #self.layer1 = nn.Linear(oned_size * 2, 256)

        oned_size = (N*2 + 1)

        if oned_size == 10:
            # FIRST INPUT TYPE MODEL
            self.layer1 = nn.Conv1d(1, 6, 3, 1)
            #self.layer2 = nn.Linear(48, 64) # 48 for an input dimension of 10 (i.e. oned size is 5)
            self.layer2 = nn.Linear(114, 128)
            #self.layer2 = nn.Linear(64, 64)
            self.layer3 = nn.Linear(128, 128)
            #self.layer5 = nn.Linear(256, 128)
            self.layer4 = nn.Linear(128, 1)
        elif oned_size == 5:
            # FIRST INPUT TYPE MODEL (5 size)
            self.layer1 = nn.Conv1d(1, 6, 3, 1)
            # self.layer2 = nn.Linear(48, 64) # 48 for an input dimension of 10 (i.e. oned size is 5)
            self.layer2 = nn.Linear(54, 64)
            # self.layer2 = nn.Linear(64, 64)
            self.layer3 = nn.Linear(64, 64)
            self.layer4 = nn.Linear(64, 1)
        else:
            outl = 6 * ((oned_size * 2 + 1) - 2)
            self.layer1 = nn.Conv1d(1, 6, 3, 1)
            # self.layer2 = nn.Linear(48, 64) # 48 for an input dimension of 10 (i.e. oned size is 5)
            self.layer2 = nn.Linear(outl, 256)
            # self.layer2 = nn.Linear(64, 64)
            self.layer3 = nn.Linear(256, 128)
            self.layer4 = nn.Linear(128, 1)

        # # FIRST INPUT TYPE MODEL （iteration 0, works for random)
        # self.layer1 = nn.Conv1d(1, 6, 3, 1)
        # # self.layer2 = nn.Linear(48, 64) # 48 for an input dimension of 10 (i.e. oned size is 5)
        # self.layer2 = nn.Linear(114, 128)
        # # self.layer2 = nn.Linear(64, 64)
        # self.layer3 = nn.Linear(128, 128)
        # #self.layer5 = nn.Linear(128, 128)
        # self.layer4 = nn.Linear(128, 1)

        # SECOND INPUT TYPE MODEL
        # self.layer1 = nn.Conv1d(1, 6, 3, 1)
        # self.layer2 = nn.Linear(12, 32)
        # self.layer3 = nn.Linear(32, 32)
        # self.layer4 = nn.Linear(32, 1)

        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
        #torch.nn.init.xavier_uniform_(self.layer5.weight)
        torch.nn.init.xavier_uniform_(self.layer4.weight)
        print("Initialized Neural Network")

    def forward(self, a, b, pos):
        x = torch.cat((a.flatten(), b.flatten(), pos.flatten()))
        #print(x)
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
        # for p in self.function.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
        self.optimizer = optim.AdamW(self.function.parameters(), lr=alpha, amsgrad=True)
        self.alpha = alpha
        self.discount = discount
        self.num = num

    def get_value_function(self, a, b, pos):
        a, b, agpos_old = convert_altinput(a, b, pos)
        with torch.no_grad():
            val = self.function(ten(a), ten(b), ten(pos)).cpu().numpy()
        return val

    def train(self, state, new_state, reward, old_pos, new_pos):
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
        a, b, agpos_old = convert_altinput(a, b, old_pos)
        new_a, new_b, agpos_new = convert_altinput(new_a, new_b, new_pos)
        approx = self.function(ten(a), ten(b), ten(old_pos))
        with torch.no_grad():
            target = reward + self.discount * self.function(ten(new_a), ten(new_b), ten(new_pos))
            target = torch.clamp(target, 0)
            #target = 1 + self.discount * self.function(ten(new_a), ten(new_b), ten(new_pos))
        criterion = torch.nn.MSELoss()
        self.optimizer.zero_grad()
        #print(approx, target)
        loss = criterion(approx, target)
        loss.backward()
        self.optimizer.step()

        t, ap = target.detach().cpu().numpy(), approx.detach().cpu().numpy()
        if t > 1000:
            print(t, ap, reward)
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
