import numpy as np

from config import DEVICE
from helpers import convert_position, ten, unwrap_state
from models.main_net import MainNet
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

class ValueNetwork():
    def __init__(self, oned_size, alpha, discount, num=None):
        self.function = MainNet(oned_size + 1, 1)
        self.function.to(DEVICE)
        self.optimizer = optim.AdamW(self.function.parameters(), lr=alpha, amsgrad=True)
        self.alpha = alpha
        self.discount = discount
        self.num = num

    def get_value_function(self, a, b, pos):
        x = np.concatenate([a, b, convert_position(pos)], axis=0)
        res = ten(x, DEVICE)
        res = res.view(1, -1)
        with torch.no_grad():
            val = self.function(res).cpu().numpy()

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
