import numpy as np
import torch
from helpers.helpers import ten
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
        res = ten(x, DEVICE)
        res = res.view(1, -1)
        with torch.no_grad():
            val = self.function(res).cpu().numpy()
        return val

    def get_input_dim(self):
        return self._input_dim

    def train(self) -> float:
        """Train value function using TD(0). This is not tabular, and instead uses a neural network to
        approximate the value function because the state space is too large.

        Returns:
            float: The Mean Squared Error loss computed for the training batch.
        """
        states = ten(np.stack(self.batch_states, axis=0).squeeze(), DEVICE)
        states = states.view(states.size(0), -1)
        value_of_this_state_prediction: torch.Tensor = self.function(
            states
        )  # shape [B]
        value_of_this_state_prediction = value_of_this_state_prediction.squeeze(1)
        with torch.no_grad():
            next_states = ten(np.stack(self.batch_new_states, axis=0).squeeze(), DEVICE)
            next_states = next_states.view(next_states.size(0), -1)
            value_of_next_state_prediction = self.function(next_states)
            rewards_tensor = ten(np.array(self.batch_rewards), DEVICE)
            td_target = (
                rewards_tensor
                + self.discount * value_of_next_state_prediction.squeeze(1)
            )

        # 3) Compute MSE loss & backpropagate
        criterion = torch.nn.MSELoss(reduction="mean")
        self.optimizer.zero_grad()
        loss: torch.Tensor = criterion(value_of_this_state_prediction, td_target)
        loss.backward()
        # this moves a little towards td_target, just like tabular td(0)
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
