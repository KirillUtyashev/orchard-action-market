# --- START OF FILE models/value_cnn.py ---

import numpy as np
import torch
from config import DEVICE, STATES_DIR
from helpers.states_logger import HtmlDataLogger
from models.cnn import CNNCentralized, CNNDecentralized, CNN
from typing_extensions import override
from utils import ten_float


class ValueTrainer:
    """
    A helper class that encapsulates TD-learning logic for a CNN.
    This class is composed within a CNN and is not a Mixin.
    """

    def __init__(self, network: CNN, discount: float, batch_size: int = 0):
        # Store a direct reference to the network we are training.
        self.network = network
        self.discount = discount
        self.data_logger = (
            HtmlDataLogger(STATES_DIR / "log.html", batch_size)
            if batch_size > 0
            else None
        )

        # The trainer now explicitly manages the buffers on the network object.
        self.network.batch_new_states = []

    def add_experience(self, state, new_state, reward):
        """Adds an experience to the network's buffers and logs it."""
        if self.data_logger:
            self.data_logger.log_experience(state, new_state, reward)
        self.network.batch_states.append(state)
        self.network.batch_new_states.append(new_state)
        self.network.batch_rewards.append(reward)

    def train_batch(self):
        """Trains the referenced network using TD learning."""
        net = self.network  # Use a short alias for clarity
        if len(net.batch_states) == 0:
            return None

        # --- All operations are now explicitly on `net` ---
        states = ten_float(np.stack(net.batch_states, axis=0), DEVICE)
        approx = net.forward(states).squeeze(1)

        with torch.no_grad():
            next_states = ten_float(np.stack(net.batch_new_states, axis=0), DEVICE)
            target_v = net.forward(next_states).squeeze(1)
            rewards = ten_float(np.array(net.batch_rewards, dtype=np.float32), DEVICE)
            y = rewards + self.discount * target_v

        criterion = torch.nn.MSELoss()
        net.optimizer.zero_grad()
        loss = criterion(approx, y)
        loss.backward()
        net.optimizer.step()

        # Clear the buffers on the network object
        net.batch_states.clear()
        net.batch_new_states.clear()
        net.batch_rewards.clear()

        loss_val = loss.item()
        net.loss_history.append(loss_val)
        return loss_val


# --- Concrete Classes using Composition ---


class ValueCNNCentralized(CNNCentralized):
    """A centralized CNN for learning a value function."""

    def __init__(
        self,
        height: int,
        width: int,
        alpha: float,
        discount: float,
        batch_size: int = 0,
        **kwargs
    ):
        super().__init__(height=height, width=width, alpha=alpha, **kwargs)
        # Create and hold an instance of the trainer. Pass `self` as the network.
        self.trainer = ValueTrainer(self, discount=discount, batch_size=batch_size)

    # Delegate the training and experience calls to the trainer helper.
    def add_experience(self, state, new_state, reward):
        self.trainer.add_experience(state, new_state, reward)

    def train_batch(self):
        return self.trainer.train_batch()

    def get_value_function(self, processed_state: np.ndarray) -> float:
        state_tensor = ten_float(processed_state, DEVICE).unsqueeze(0)
        with torch.no_grad():
            value = self.forward(state_tensor)
        return value.item()


class ValueCNNDecentralized(CNNDecentralized):
    """A decentralized CNN for learning a value function."""

    def __init__(
        self,
        height: int,
        width: int,
        alpha: float,
        discount: float,
        batch_size: int = 0,
        **kwargs
    ):
        super().__init__(height=height, width=width, alpha=alpha, **kwargs)
        self.trainer = ValueTrainer(self, discount=discount, batch_size=batch_size)

    def add_experience(self, state, new_state, reward):
        self.trainer.add_experience(state, new_state, reward)

    def train_batch(self):
        return self.trainer.train_batch()

    def get_value_function(self, processed_state: np.ndarray) -> float:
        state_tensor = ten_float(processed_state, DEVICE).unsqueeze(0)
        with torch.no_grad():
            value = self.forward(state_tensor)
        return value.item()
