# --- START OF FILE models/value_cnn.py ---

import copy
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
        self.discount = discount
        # Create the target network, which is a clone of the policy network
        self.memory = ReplayBuffer(replay_buffer_capacity)
        target_net = CNNCentralized(
            height, width, alpha, mlp_hidden_features, mlp_hidden_layers
        )
        policy_net_state_dict = self.state_dict()

        target_net.load_state_dict(policy_net_state_dict)
        self.target_net = target_net
        self.target_net.eval()

    # Delegate the training and experience calls to the trainer helper.
    def add_experience(self, state, new_state, reward):
        self.memory.push(state, new_state, reward)

    def train_batch(self, batch_size: int):
        """Trains the network on a random batch of experiences from the replay buffer."""
        if len(self.memory) < batch_size:
            return None  # Not enough experiences in memory to train yet

        # Sample a batch of transitions from the replay buffer
        transitions = self.memory.sample(batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for details)
        batch = Transition(*zip(*transitions))

        states = ten_float(np.stack(batch.state, axis=0), DEVICE)
        approx = self.forward(states).squeeze(1)

        with torch.no_grad():
            next_states = ten_float(np.stack(batch.new_state, axis=0), DEVICE)
            # Use the target_net to calculate the value of the next state
            target_v = self.target_net(next_states).squeeze(1)
            rewards = ten_float(np.array(batch.reward, dtype=np.float32), DEVICE)
            y = rewards + self.discount * target_v

        criterion = torch.nn.MSELoss()
        self.optimizer.zero_grad()
        loss = criterion(approx, y)
        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def update_target_net(self):
        """Hard update of the target network's weights."""
        self.target_net.load_state_dict(self.state_dict(), strict=False)

    def get_value_function(self, processed_state: np.ndarray) -> float:
        state_tensor = ten_float(processed_state, DEVICE).unsqueeze(0)
        with torch.no_grad():
            value = self.forward(state_tensor)
        return value.item()


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
