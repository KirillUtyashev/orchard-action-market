# --- START OF FILE models/value_cnn.py ---

import numpy as np
import torch
from config import DEVICE
from helpers.states_logger import HtmlDataLogger
from models.cnn import CNNCentralized, CNNDecentralized, CNN
from typing_extensions import override
from utils import ten_float
import random
import collections
from collections import namedtuple

Transition = namedtuple("Transition", ("state", "new_state", "reward"))


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = collections.deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ValueTrainer:
    """
    A helper class that encapsulates TD-learning logic for a CNN.
    This class is composed within a CNN.
    """

    def __init__(
        self,
        network: CNN,
        discount: float,
    ):
        # Store a direct reference to the network we are training.
        self.network = network
        self.discount = discount

        # The trainer now explicitly manages the buffers on the network object.
        self.network.batch_new_states = []

    def add_experience(self, state, new_state, reward):
        """Adds an experience to the network's buffers and logs it."""
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


class ValueCNNCentralized(CNNCentralized):
    """A centralized CNN for learning a value function."""

    def __init__(
        self,
        height: int,
        width: int,
        alpha: float,
        discount: float,
        mlp_hidden_features: int,
        mlp_hidden_layers: int,
        replay_buffer_capacity: int = 10000,
    ):
        super().__init__(height, width, alpha, mlp_hidden_features, mlp_hidden_layers)
        # Create and hold an instance of the trainer. Pass `self` as the network.
        self.trainer = ValueTrainer(self, discount=discount)
        
        self.memory = ReplayBuffer(replay_buffer_capacity)
        self.target_net = CNNCentralized(height, width, alpha, mlp_hidden_features, mlp_hidden_layers)
        self.target_net.load_state_dict(self.state_dict())
        self.target_net.eval()  # Target network is only for evaluation/inference

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
        mlp_hidden_features: int,
        mlp_hidden_layers: int,
        **kwargs
    ):
        super().__init__(height, width, alpha, mlp_hidden_features, mlp_hidden_layers)
        self.trainer = ValueTrainer(self, discount=discount)

    def add_experience(self, state, new_state, reward):
        self.trainer.add_experience(state, new_state, reward)

    def train_batch(self):
        return self.trainer.train_batch()

    def get_value_function(self, processed_state: np.ndarray) -> float:
        state_tensor = ten_float(processed_state, DEVICE).unsqueeze(0)
        with torch.no_grad():
            value = self.forward(state_tensor)
        return value.item()
