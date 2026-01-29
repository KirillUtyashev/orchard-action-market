import numpy as np
import torch
from utils import ten
from config import DEVICE
from models.network import NetworkWrapper

torch.set_default_dtype(torch.float64)


class VNetwork(NetworkWrapper):
    def __init__(
        self, input_dim, output_dim, alpha, discount, hidden_dim=128, num_layers=4
    ):
        super().__init__(input_dim, output_dim, alpha, discount, hidden_dim, num_layers)
        self.batch_rewards = []

    def get_value_function(self, x):
        self.model.eval()
        with torch.no_grad():
            t = ten(x, DEVICE).view(1, -1)
            val = self.model(t)
        return val.detach().cpu().numpy()

    def get_input_dim(self):
        return self._input_dim

    def train(self):
        states = ten(np.stack(self.batch_states, axis=0), DEVICE).reshape(len(self.batch_states), -1)
        next_states = ten(np.stack(self.batch_new_states, axis=0), DEVICE).reshape(len(self.batch_new_states), -1)
        rewards = ten(np.asarray(self.batch_rewards, dtype=np.float64), DEVICE).reshape(-1)

        approx = self.model(states).reshape(-1)
        with torch.no_grad():
            target = self.model(next_states).reshape(-1)
            y = rewards + self.discount * target

        # ensure shapes match exactly for MSELoss
        assert approx.shape == y.shape

        loss = torch.nn.MSELoss(reduction="mean")(approx, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self.batch_states.clear()
        self.batch_new_states.clear()
        self.batch_rewards.clear()
        return loss.item()

    def add_experience(self, state, new_state, reward):
        self.batch_states.append(state)
        self.batch_new_states.append(new_state)
        self.batch_rewards.append(reward)
