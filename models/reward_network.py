import numpy as np
import torch
from helpers.helpers import ten
from config import DEVICE
from models.network import Network
torch.set_default_dtype(torch.float64)


class RewardNetwork(Network):
    def __init__(self, input_dim, output_dim, alpha, discount, hidden_dim=128, num_layers=4):
        super().__init__(input_dim, output_dim, alpha, discount, hidden_dim, num_layers)
        self.batch_rewards = []
        self.loss_history = []

    def get_value_function(self, x):
        res = ten(x, DEVICE)
        res = res.view(1, -1)
        with torch.no_grad():
            val = self.function(res).cpu().numpy()
        return val

    def get_input_dim(self):
        return self._input_dim

    def train(self):
        if len(self.batch_states) == 0:
            return None  # nothing to do

        # States: [B, *] -> [B, input_dim]
        states_np = np.stack(self.batch_states, axis=0)  # don't squeeze here
        states = ten(states_np, DEVICE)
        states = states.view(states.size(0), -1)

        # Targets: [B]
        targets = ten(np.asarray(self.batch_rewards, dtype=np.float64), DEVICE).view(-1)

        preds = self.function(states).view(-1)  # robust to [B] or [B,1] outputs

        criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer.zero_grad()
        loss = criterion(preds, targets)
        loss.backward()
        self.optimizer.step()

        # Clear buffers
        self.batch_states.clear()
        self.batch_rewards.clear()
        # Save loss value
        loss_val = float(loss.detach().cpu().item())
        self.loss_history.append(loss_val)

        return loss_val

    def add_experience(self, state, reward):
        self.batch_states.append(state)
        self.batch_rewards.append(reward)

