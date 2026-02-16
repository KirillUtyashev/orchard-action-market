import numpy as np
import torch
from utils import ten
from config import DEVICE
from debug.code.network import NetworkWrapper

torch.set_default_dtype(torch.float64)


class VNetwork(NetworkWrapper):
    def __init__(
            self, input_dim, output_dim, alpha, discount, hidden_dim=128, num_layers=4, num_training_steps=10000, schedule=False
    ):
        super().__init__(input_dim, output_dim, alpha, discount, hidden_dim, num_layers, schedule, num_training_steps)
        self.batch_rewards = []
        self.batch_discounts = []
        self.theoretical_vals = []

    def get_value_function(self, x):
        res = ten(x, DEVICE)
        res = res.view(1, -1)
        with torch.no_grad():
            val = self.model(res).cpu().numpy().item()
        return val

    def get_input_dim(self):
        return self._input_dim

    def train(self):
        states = ten(np.stack(self.batch_states, axis=0).squeeze(), DEVICE)
        states = states.view(states.size(0), -1)
        approx = self.model(states)
        approx = approx.squeeze(1)

        with torch.no_grad():
            next_states = ten(np.stack(self.batch_new_states, axis=0).squeeze(), DEVICE)
            next_states = next_states.view(next_states.size(0), -1)
            target = self.model(next_states)

            # Use per-transition discount factors
            discounts = ten(np.array(self.batch_discounts), DEVICE)
            y = ten(np.array(self.batch_rewards), DEVICE) + discounts * target.squeeze(1)

        criterion = torch.nn.MSELoss(reduction="mean")
        self.optimizer.zero_grad()
        loss = criterion(approx, y)
        loss.backward()
        self.optimizer.step()
        self._after_update()

        self.batch_states = []
        self.batch_new_states = []
        self.batch_rewards = []
        self.batch_discounts = []  # Clear discount buffer
        return loss.item()

    def add_experience(self, state, new_state, reward, discount_factor, theoretical_val=None):
        self.batch_states.append(state)
        self.batch_new_states.append(new_state)
        self.batch_rewards.append(reward)
        self.batch_discounts.append(discount_factor)  # Store per-transition discount
        if theoretical_val:
            self.theoretical_vals.append(theoretical_val)

    def train_supervised(self):
        # 1) Batch states -> tensor [B, obs_dim]
        states_np = np.stack(self.batch_states, axis=0).squeeze()
        states = ten(states_np, DEVICE)
        states = states.view(states.size(0), -1)

        # 2) Forward pass: approx V(s)
        approx = self.model(states)          # [B, 1] (assumed)
        approx = approx.squeeze(1)           # [B]

        # 3) Supervised targets: theoretical V*(s) (or your analytic V(s))
        #    Compute per-state, ignore next states entirely.
        with torch.no_grad():
            y_np = np.array(
                self.theoretical_vals
            )
            y = ten(y_np, DEVICE)            # [B]

        # 4) Loss + update
        criterion = torch.nn.MSELoss(reduction="mean")  # mean squared error
        self.optimizer.zero_grad()
        loss = criterion(approx, y)
        loss.backward()
        self.optimizer.step()
        self._after_update()

        # 5) Clear batch buffers
        self.batch_states = []
        self.batch_new_states = []   # optional: keep if other code expects it
        self.batch_rewards = []      # optional: keep if other code expects it
        self.theoretical_vals = []

        return loss.item()

    def train_reward_supervised(self):
        # states: [B, obs_dim]
        states = ten(np.stack(self.batch_states, 0).squeeze(), DEVICE)
        states = states.view(states.size(0), -1)

        # y: [B] (or [B,1] depending on your net output)
        y = ten(np.array(self.batch_rewards), DEVICE).view(-1)

        pred = self.model(states).squeeze(-1)  # make it [B]

        loss = torch.nn.functional.mse_loss(pred, y)  # standard regression loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.batch_states = []
        self.batch_rewards = []
        return loss.item()
