"""
Base Value Model (V2) - DEBUG VERSION
"""

from abc import ABC, abstractmethod
from collections import namedtuple, deque
from typing import Optional, List
import random

import numpy as np
import torch
import torch.nn as nn

from tadd_helpers.env_functions import State

Transition = namedtuple("Transition", ("state", "next_state", "reward"))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.memory: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, next_state: np.ndarray, reward: float):
        self.memory.append(Transition(state, next_state, reward))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory.clear()

    def __len__(self) -> int:
        return len(self.memory)


class BaseValueModelV2(nn.Module, ABC):
    def __init__(
        self,
        discount: float,
        replay_buffer_capacity: int,
        device: torch.device,
    ):
        super().__init__()
        self.discount = discount
        self.device = device
        self.memory = ReplayBuffer(replay_buffer_capacity)

        self.policy_net: nn.Module
        self.target_net: nn.Module
        self.optimizer: torch.optim.Optimizer

    def _safe_init_weights(self):
        """
        Universal safe initialization:
        1. Kaiming Normal for all hidden layers (avoids vanishing gradients).
        2. Small Uniform for the FINAL Linear layer (safety against initial spikes).
        """
        with torch.no_grad():
            # 1. Standard Kaiming Init for everything
            for m in self.policy_net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None: 
                        nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None: 
                        nn.init.constant_(m.bias, 0.0)
            
            # 2. Re-init ONLY the final Linear layer to be small
            # We iterate to find the very last Linear layer in the whole net
            all_linears = [m for m in self.policy_net.modules() if isinstance(m, nn.Linear)]
            if all_linears:
                last_layer = all_linears[-1]
                nn.init.uniform_(last_layer.weight, -0.01, 0.01)
                if last_layer.bias is not None:
                    nn.init.constant_(last_layer.bias, 0.0)
                print(f"[INIT] BaseValueModelV2: Scaled last Linear layer {last_layer}")

    def init_scheduler(self, max_steps: int, min_lr: float = 1e-7):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps, eta_min=min_lr
        )

    def step_scheduler(self):
        if hasattr(self, "scheduler") and self.scheduler:
            self.scheduler.step()

    def get_current_lr(self) -> float:
        if hasattr(self, "optimizer"):
            return self.optimizer.param_groups[0]["lr"]
        return 0.0

    @abstractmethod
    def raw_state_to_nn_input(self, state: State, acting_agent_idx: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_value(self, state: State, acting_agent_idx: int) -> float:
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
        raise NotImplementedError

    def update_target_net(self) -> None:
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Debug: verify copy worked
        for name, param in self.target_net.named_parameters():
            if torch.isnan(param).any():
                print(f"[WARNING] NaN in target_net.{name} after update!")

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
