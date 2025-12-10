"""
Base Value Model (V2)

Abstract base class for all value models (centralized and decentralized).
Provides shared functionality:
- Replay buffer management
- train_batch for TD(0) learning
- Target network updates
- Loss history tracking

Subclasses must implement:
- raw_state_to_nn_input(state, acting_agent_idx) -> np.ndarray
- get_value(state, acting_agent_idx) -> float
- add_experience(state, next_state, reward, acting_agent_idx, next_acting_agent_idx)
"""

from abc import ABC, abstractmethod
from collections import namedtuple, deque
from typing import Optional, List
import random

import numpy as np
import torch
import torch.nn as nn

from tadd_helpers.env_functions import State

# Transition tuple for replay buffer
Transition = namedtuple("Transition", ("state", "next_state", "reward"))


class ReplayBuffer:
    """Simple replay buffer for experience replay."""

    def __init__(self, capacity: int):
        self.memory: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, next_state: np.ndarray, reward: float):
        """Add a transition to the buffer."""
        self.memory.append(Transition(state, next_state, reward))

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions."""
        return random.sample(self.memory, batch_size)

    def clear(self):
        """Clear all stored transitions."""
        self.memory.clear()

    def __len__(self) -> int:
        return len(self.memory)


class BaseValueModelV2(nn.Module, ABC):
    """
    Abstract base class for value models.

    Subclasses must:
    1. Initialize self.policy_net, self.target_net, self.optimizer in __init__
    2. Implement raw_state_to_nn_input, get_value, add_experience
    """

    def __init__(
        self,
        discount: float,
        replay_buffer_capacity: int,
        device: torch.device,
    ):
        """
        Args:
            discount: Gamma for TD learning (0 for reward learning)
            replay_buffer_capacity: Maximum transitions to store
            device: Torch device (cuda or cpu)
        """
        super().__init__()
        self.discount = discount
        self.device = device
        self.memory = ReplayBuffer(replay_buffer_capacity)
        self.loss_history: List[float] = []

        # These must be initialized by subclasses
        self.policy_net: nn.Module
        self.target_net: nn.Module
        self.optimizer: torch.optim.Optimizer

    @abstractmethod
    def raw_state_to_nn_input(self, state: State, acting_agent_idx: int) -> np.ndarray:
        """
        Convert environment state to neural network input.

        Args:
            state: Current environment state
            acting_agent_idx: Index of the agent that is/will act

        Returns:
            Numpy array suitable for the network (will be converted to tensor)
        """
        raise NotImplementedError

    @abstractmethod
    def get_value(self, state: State, acting_agent_idx: int) -> float:
        """
        Get predicted value for a state.

        Args:
            state: Environment state
            acting_agent_idx: Index of the acting agent

        Returns:
            Predicted value as a float
        """
        raise NotImplementedError

    @abstractmethod
    def add_experience(
        self,
        state: State,
        next_state: State,
        reward: float,
        acting_agent_idx: int,
        next_acting_agent_idx: int,
    ) -> None:
        """
        Add experience tuple to replay buffer.

        Args:
            state: Current state s_t
            next_state: Next state s_{t+1}
            reward: Reward received r_t
            acting_agent_idx: Who acted at time t (for encoding s_t)
            next_acting_agent_idx: Who will act at time t+1 (for encoding s_{t+1})
        """
        raise NotImplementedError

    def train_batch(self, batch_size: int) -> Optional[float]:
        """
        Train on a batch sampled from replay buffer.

        Uses TD(0) target: r + gamma * target_net(s')
        For reward learning (gamma=0), target is just r.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Loss value, or None if buffer has insufficient samples
        """
        if len(self.memory) < batch_size:
            return None

        # Sample batch
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # Convert to tensors
        states = torch.tensor(
            np.stack(batch.state), dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            np.stack(batch.next_state), dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor(
            np.array(batch.reward), dtype=torch.float32, device=self.device
        )

        # Forward pass
        self.policy_net.train()
        curr_v = self.policy_net(states).squeeze(1)

        # Compute targets
        with torch.no_grad():
            if self.discount == 0:
                # Reward learning: target is just reward
                target_v = rewards
            else:
                # TD(0): target = r + gamma * V(s')
                next_v = self.target_net(next_states).squeeze(1)
                target_v = rewards + self.discount * next_v

        # Compute loss and update
        loss = nn.MSELoss()(curr_v, target_v)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def update_target_net(self) -> None:
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
