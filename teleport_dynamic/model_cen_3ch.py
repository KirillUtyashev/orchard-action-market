"""
Centralized Value Models (3-Channel Input)

Input Channels:
    0: Apples - apple count at each position
    1: Agents - agent count at each position
    2: Actor  - one-hot indicating acting agent's position

These models predict the centralized value V(s) where reward is 1 if actor is on apple, 0 otherwise.
"""

import torch
import torch.nn as nn
import numpy as np

from tadd_helpers.env_functions import State
from teleport_dynamic.base_value_model import BaseValueModelV2


class Clamp(nn.Module):
    def forward(self, x):
        return torch.clamp(x, -1000.0, 1000.0)


class ValueCNNCentralized3Ch(BaseValueModelV2):
    """
    CNN-based centralized value model.

    Architecture: Conv layers -> Flatten -> MLP head -> scalar output
    """

    def __init__(
        self,
        height: int,
        width: int,
        lr: float,
        discount: float,
        mlp_hidden_dim: int,
        mlp_num_layers: int,
        conv_channels: list,
        kernel_size: int,
        device: torch.device,
        replay_buffer_capacity: int = 10000,
    ):
        super().__init__(discount, replay_buffer_capacity, device)

        self.height = height
        self.width = width
        self.input_channels = 3  # Apples, Agents, Actor

        # Handle string input from Papermill
        if isinstance(conv_channels, str):
            import ast

            conv_channels = ast.literal_eval(conv_channels)

        # Build conv layers
        conv_layers = []
        in_c = self.input_channels
        for out_c in conv_channels:
            pad = (kernel_size - 1) // 2
            conv_layers.append(nn.Conv2d(in_c, out_c, kernel_size, padding=pad))
            conv_layers.append(nn.ReLU())
            in_c = out_c
        self.conv_net = nn.Sequential(*conv_layers)

        # Calculate flattened dimension
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, height, width)
            conv_out = self.conv_net(dummy)
        flat_dim = int(np.prod(conv_out.shape[1:]))

        # Build MLP head
        head_layers = []
        in_d = flat_dim
        for _ in range(mlp_num_layers):
            head_layers.append(nn.Linear(in_d, mlp_hidden_dim))
            head_layers.append(nn.ReLU())
            in_d = mlp_hidden_dim
        head_layers.append(nn.Linear(in_d, 1))
        head_layers.append(Clamp())
        self.head = nn.Sequential(*head_layers)

        # Assemble full network
        self.policy_net = nn.Sequential(self.conv_net, nn.Flatten(), self.head).to(
            self.device
        )

        self.target_net = nn.Sequential(self.conv_net, nn.Flatten(), self.head).to(
            self.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=lr, amsgrad=True
        )

    def raw_state_to_nn_input(self, state: State, acting_agent_idx: int) -> np.ndarray:
        """
        Convert state to 3-channel tensor.

        Args:
            state: Current environment state
            acting_agent_idx: Index of the agent that is/will act

        Returns:
            3xHxW numpy array [Apples, Agents, Actor]
        """
        H, W = state.H, state.L

        c_apples = state.apples.astype(np.float32)
        c_agents = state.agents.astype(np.float32)

        c_actor = np.zeros((H, W), dtype=np.float32)
        r, c = state.agent_position(acting_agent_idx)
        c_actor[r, c] = 1.0

        return np.stack([c_apples, c_agents, c_actor])

    def get_value(self, state: State, acting_agent_idx: int) -> float:
        """Get predicted value for a state."""
        arr = self.raw_state_to_nn_input(state, acting_agent_idx)
        t = torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(t).item()

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
        s_arr = self.raw_state_to_nn_input(state, acting_agent_idx)
        ns_arr = self.raw_state_to_nn_input(next_state, next_acting_agent_idx)
        self.memory.push(s_arr, ns_arr, reward)


class ValueMLPCentralized3Ch(BaseValueModelV2):
    """
    MLP-based centralized value model.

    Architecture: Flatten -> MLP layers -> scalar output
    """

    def __init__(
        self,
        height: int,
        width: int,
        lr: float,
        discount: float,
        hidden_dim: int,
        num_layers: int,
        device: torch.device,
        replay_buffer_capacity: int = 100000,
    ):
        super().__init__(discount, replay_buffer_capacity, device)

        self.height = height
        self.width = width
        self.input_channels = 3
        input_dim = self.input_channels * height * width

        # Build MLP
        layers = []
        in_d = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU())
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, 1))
        layers.append(Clamp())
        self.mlp = nn.Sequential(*layers)

        self.policy_net = nn.Sequential(nn.Flatten(), self.mlp).to(self.device)
        self.target_net = nn.Sequential(nn.Flatten(), self.mlp).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=lr, amsgrad=True
        )

    def raw_state_to_nn_input(self, state: State, acting_agent_idx: int) -> np.ndarray:
        """
        Convert state to flattened 3-channel vector.

        Args:
            state: Current environment state
            acting_agent_idx: Index of the agent that is/will act

        Returns:
            1D numpy array of shape (3*H*W,)
        """
        H, W = state.H, state.L

        c_apples = state.apples.astype(np.float32)
        c_agents = state.agents.astype(np.float32)

        c_actor = np.zeros((H, W), dtype=np.float32)
        r, c = state.agent_position(acting_agent_idx)
        c_actor[r, c] = 1.0

        return np.stack([c_apples, c_agents, c_actor]).flatten()

    def get_value(self, state: State, acting_agent_idx: int) -> float:
        """Get predicted value for a state."""
        arr = self.raw_state_to_nn_input(state, acting_agent_idx)
        t = torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(t).item()

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
        s_arr = self.raw_state_to_nn_input(state, acting_agent_idx)
        ns_arr = self.raw_state_to_nn_input(next_state, next_acting_agent_idx)
        self.memory.push(s_arr, ns_arr, reward)
