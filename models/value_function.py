import numpy as np
import torch
from models.main_net import MainNet
import torch.optim as optim
from helpers import convert_position, ten, unwrap_state
from config import DEVICE
torch.set_default_dtype(torch.float64)

class VNetwork:
    def __init__(self, input_dim, alpha, discount, hidden_dim=128, num_layers=4):
        self.function = MainNet(input_dim, 1, hidden_dim, num_layers).to(DEVICE)
        self.optimizer = optim.AdamW(self.function.parameters(), lr=alpha,
                                     amsgrad=True)
        self.alpha = alpha
        self.discount = discount

        self.batch_states = []
        self.batch_new_states = []
        self.batch_rewards = []

        self._input_dim = input_dim * 2

    def get_value_function(self, x):
        res = ten(x, DEVICE)
        res = res.view(1, -1)
        with torch.no_grad():
            val = self.function(res).cpu().numpy()
        return val

    def get_input_dim(self):
        return self._input_dim

    def train(self):
        if len(self.batch_states) == 0:  # Meaning that agent didn't act / collect any observations in this batch
            return None

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
        criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer.zero_grad()
        loss = criterion(approx, y)
        loss.backward()
        self.optimizer.step()

        # self.optimizer.zero_grad()
        self.batch_states = []
        self.batch_new_states = []
        self.batch_rewards = []
        return loss.item()

    def add_experience(self, state, new_state, reward):
        self.batch_states.append(state)
        self.batch_new_states.append(new_state)
        self.batch_rewards.append(reward)
