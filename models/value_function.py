import numpy as np
import torch
from models.main_net import MainNet
import torch.optim as optim
from helpers import convert_position, ten, unwrap_state
from config import DEVICE
torch.set_default_dtype(torch.float64)

action_vectors = [
    np.array([-1, 0]),
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([0, -1]),
    np.array([0, 0])
]


class VNetwork:
    def __init__(self, input_dim, alpha, discount):
        self.function = MainNet(input_dim, 1).to(DEVICE)
        self.optimizer = optim.AdamW(self.function.parameters(), lr=alpha,
                                     amsgrad=True)
        self.alpha = alpha
        self.discount = discount

        self.batch_states = []
        self.batch_new_states = []
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
            val = self.function(ten(As, DEVICE), ten(Bs, DEVICE), ten(poses, DEVICE)).cpu().numpy()
        return val

    def get_value_function2(self, state):
        a, b = state[0], state[1]
        return self.function(ten(a, DEVICE), ten(b, DEVICE), None).detach().cpu().numpy()

    def get_trainable_adv(self, state, new_state, old_pos, new_pos, reward):
        a, b = unwrap_state(state)
        new_a, new_b = unwrap_state(new_state)
        q = reward + self.discount * self.function(ten(new_a, DEVICE), ten(new_b, DEVICE),
                                                   ten(new_pos, DEVICE))
        v = self.function(ten(a, DEVICE), ten(b, DEVICE), ten(old_pos, DEVICE))

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
        if len(self.batch_states) == 0:  # Meaning that agent didn't act / collect any observations in this batch
            return

        states = ten(np.stack(self.batch_states, axis=0).squeeze(), DEVICE)
        states = states.view(states.size(0), -1)
        approx = self.function(states)               # shape [B]
        approx = approx.squeeze(1)
        # 2) Build TD‐target: y = r + γ·V_target(s')·(1 – done)
        with torch.no_grad():
            next_states = ten(np.stack(self.batch_new_states, axis=0).squeeze(), DEVICE)
            next_states = next_states.view(next_states.size(0), -1)
            target = self.function(next_states)   # [B]
            y = ten(np.array(self.batch_rewards), DEVICE) + self.discount * target.squeeze(1)

        # 3) Compute MSE loss & backpropagate
        criterion = torch.nn.MSELoss()
        self.optimizer.zero_grad()
        loss = criterion(approx, y)
        loss.backward()
        self.optimizer.step()

        self.batch_states = []
        self.batch_new_states = []
        self.batch_rewards = []
        return loss.item()

    def add_experience(self, state, new_state, reward):
        self.batch_states.append(state)
        self.batch_new_states.append(new_state)
        self.batch_rewards.append(reward)
