import numpy as np
import torch
from config import DEVICE
from models.cnn import CNN, CNNCentralized
from typing_extensions import override

from utils import ten_float


class ValueCNNCentralized(CNNCentralized):
    """Centralized CNN for learning the value function (critique).
    Uses TD learning for training.

    Note a single "time step" is an index i. So when training, we must use the same i for that instance of batch_states, batch_new_states, and batch_rewards.

    Parameters:
        batch_states: List of states in the current training batch. Index corresponds to time step.
        batch_new_states: List of next states in the current training batch. Index corresponds to time step.
        batch_rewards: List of rewards in the current training batch. Index corresponds to time step.
        discount: Discount factor for future rewards.
    """

    def __init__(
        self,
        height: int,
        width: int,
        alpha: float,
        discount: float,
        mlp_hidden_features: int = 128,
        num_mlp_hidden_layers: int = 1,
    ):
        super().__init__(
            height=height,
            width=width,
            alpha=alpha,
            mlp_hidden_features=mlp_hidden_features,
            num_mlp_hidden_layers=num_mlp_hidden_layers,
        )
        self.batch_new_states = []
        self.discount = discount

    def add_experience(self, state, new_state, reward):
        """Adds an experience tuple of (state, new_state, reward) to the training batch."""
        self.batch_states.append(state)
        self.batch_new_states.append(new_state)
        self.batch_rewards.append(reward)

    @override
    def train_batch(self):
        """
        Trains the network using TD learning.
        """
        if len(self.batch_states) == 0:
            return None  # Nothing to train on

        # 1) Prepare batches and get current state value predictions
        states_np = np.stack(self.batch_states, axis=0)
        states = ten_float(states_np, DEVICE)

        # The model's prediction for V(s)
        approx = self.forward(states).squeeze(1)

        # 2) Build the TD-target: y = r + γ * V(s')
        with torch.no_grad():
            next_states_np = np.stack(self.batch_new_states, axis=0)
            next_states = ten_float(next_states_np, DEVICE)

            # The model's prediction for V(s')
            target_v = self.forward(next_states).squeeze(1)

            rewards_tensor = ten_float(np.array(self.batch_rewards), DEVICE)

            # The TD-target
            y = rewards_tensor + self.discount * target_v

        # 3) Compute MSE loss and backpropagate
        criterion = torch.nn.MSELoss()
        self.optimizer.zero_grad()
        loss = criterion(approx, y)
        loss.backward()
        self.optimizer.step()

        # 4) Clear the buffers
        self.batch_states.clear()
        self.batch_new_states.clear()
        self.batch_rewards.clear()

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def get_value_function(self, processed_state: np.ndarray) -> float:
        """
        Takes a processed state (call state_to_nn_input) and returns its value.
        """
        # Convert to a PyTorch tensor and add a batch dimension of 1
        state_tensor = ten_float(processed_state, DEVICE).unsqueeze(0)

        with torch.no_grad():
            value = self.forward(state_tensor)

        return value.item()


class ValueCNNDecentralized(CNN):
    @override
    def train_batch(self):
        pass  # TODO Implement
