import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helpers import ten, unwrap_state
from config import DEVICE
from helpers import convert_position
torch.set_default_dtype(torch.float64)

action_vectors = [
    np.array([-1, 0]),
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([0, -1]),
    np.array([0, 0])
]


class ValueNet(nn.Module):
    def __init__(self, input_dim, outl):
        super().__init__()
        # --- 1D conv: from 1 channel → 6 channels, kernel=3, padding=1 to keep length ==
        self.layer1 = nn.Conv1d(1, 6, kernel_size=3, stride=1)
        if input_dim == 10:
            self.layer2 = nn.Linear(outl, 128)
            self.layer3 = nn.Linear(128, 128)
            self.layer4 = nn.Linear(128, 1)
        elif input_dim == 5:
            self.layer2 = nn.Linear(outl, 64)
            self.layer3 = nn.Linear(64, 64)
            self.layer4 = nn.Linear(64, 1)
        elif input_dim == 20:
            self.layer2 = nn.Linear(outl, 256)
            self.layer3 = nn.Linear(256, 256)
            self.layer4 = nn.Linear(256, 1)
        else:
            self.layer2 = nn.Linear(outl, 512)
            self.layer3 = nn.Linear(512, 256)
            self.layer4 = nn.Linear(256, 1)

        # Xavier initialization
        for m in [self.layer1, self.layer2, self.layer3, self.layer4]:
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # x: [B, state_dim] → reshape to [B,1,state_dim] for Conv1d
        x = x.unsqueeze(1)
        x = F.leaky_relu(self.layer1(x))        # [B, 6, L]
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.layer2(x))          # [B, 128]
        x = F.leaky_relu(self.layer3(x))          # [B, 128]
        return self.layer4(x)


class VNetwork:
    def __init__(self, input_dim, alpha, discount):
        self.function = ValueNet(input_dim, 6 * ((input_dim * 2) - 2)).to(DEVICE)
        self.target = ValueNet(input_dim, 6 * ((input_dim * 2) - 2)).to(DEVICE)
        self.target.load_state_dict(self.function.state_dict())
        self.optimizer = optim.AdamW(self.function.parameters(), lr=alpha,
                                     amsgrad=True)
        self.alpha = alpha
        self.discount = discount
        self.batch_states = []
        self.batch_new_states = []
        self.batch_actions = []
        self.batch_rewards = []

    def get_value_function(self, x):
        res = ten(x, DEVICE)
        res = res.view(1, -1)
        with torch.no_grad():
            val = self.function(res).cpu().numpy()
        return val

    def get_value_function_v(self, As, Bs, poses):
        poses = poses[:, 0]
        with torch.no_grad():
            val = self.function(ten(As), ten(Bs), ten(poses)).cpu().numpy()
        # return self.function(ten(As), ten(Bs), ten(poses)).detach().cpu().numpy()
        return val

    def get_value_function2(self, state):
        # a, b = unwrap_state(state)
        a, b = state[0], state[1]
        return self.function(ten(a, DEVICE), ten(b, DEVICE), None).detach().cpu().numpy()

    def get_trainable_adv(self, state, new_state, old_pos, new_pos, reward):
        a, b = unwrap_state(state)
        new_a, new_b = unwrap_state(new_state)
        q = reward + self.discount * self.function(ten(new_a, DEVICE), ten(new_b, DEVICE),
                                                   ten(new_pos, DEVICE))
        # print(old_pos, a, b)
        v = self.function(ten(a, DEVICE), ten(b, DEVICE), ten(old_pos, DEVICE))

        # print(q)
        return q - v

    def get_adv_and_train(self, state, new_state, old_pos, new_pos, reward):
        a, b = unwrap_state(state)
        new_a, new_b = unwrap_state(new_state)

        approx = self.function(ten(a, DEVICE), ten(b, DEVICE),
                               ten(old_pos, DEVICE))
        with torch.no_grad():
            target = reward + self.discount * self.function(ten(new_a, DEVICE),
                                                            ten(new_b, DEVICE),
                                                            ten(new_pos,
                                                                DEVICE))

        criterion = torch.nn.MSELoss()
        self.optimizer.zero_grad()

        loss = criterion(approx, target)
        loss.backward()
        self.optimizer.step()

        return target.detach(), approx.detach()

    def train(self):
        states = ten(np.stack(self.batch_states, axis=0).squeeze(), DEVICE)
        states = states.view(states.size(0), -1)
        approx = self.function(states)               # shape [B]

        approx = approx.squeeze(1)
        # 2) Build TD‐target: y = r + γ·V_target(s')·(1 – done)
        with torch.no_grad():
            next_states = ten(np.stack(self.batch_new_states, axis=0).squeeze(), DEVICE)
            next_states = next_states.view(next_states.size(0), -1)
            target = self.target(next_states)   # [B]
            y = ten(np.array(self.batch_rewards), DEVICE) + self.discount * target.squeeze(1)

        # 3) Compute MSE loss & backpropagate
        criterion = torch.nn.MSELoss()
        self.optimizer.zero_grad()
        loss = criterion(approx, y)
        loss.backward()
        self.optimizer.step()

        # 4) Soft‐update the target network
        for p, p_t in zip(self.function.parameters(), self.target.parameters()):
            p_t.data.mul_(1 - 0.01)
            p_t.data.add_(0.01 * p.data)

    def add_experience(self, state, new_state, action, reward):
        self.batch_states.append(state)
        self.batch_new_states.append(new_state)
        self.batch_actions.append(action)
        self.batch_rewards.append(reward)
