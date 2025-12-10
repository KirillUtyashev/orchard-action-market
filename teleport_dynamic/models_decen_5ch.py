"""
Decentralized Value Models (5-Channel Input)

Input Channels:
    0: Apples    - apple count at each position
    1: Others    - count of OTHER agents at each position
    2: Self      - one-hot indicating THIS agent's position
    3: Self_Act  - one-hot if THIS agent is the actor (at self's position)
    4: Other_Act - one-hot if ANOTHER agent is the actor (at their position)

Each agent has its own model instance with self_agent_idx fixed at initialization.
These models predict the decentralized value V^(i)(s) for agent i.
"""

import torch
import torch.nn as nn
import numpy as np

from tadd_helpers.env_functions import State
from teleport_dynamic.base_value_model import BaseValueModelV2


class Clamp(nn.Module):
    def forward(self, x):
        return torch.clamp(x, -1000.0, 1000.0)


class ValueCNNDecentralized5Ch(BaseValueModelV2):
    """
    CNN-based decentralized value model for a specific agent.

    Architecture: Conv layers -> Flatten -> MLP head -> scalar output

    Each agent i has its own instance with self_agent_idx=i.
    """

    def __init__(
        self,
        height: int,
        width: int,
        self_agent_idx: int,
        lr: float,
        discount: float,
        mlp_hidden_dim: int,
        mlp_num_layers: int,
        conv_channels: list,
        kernel_size: int,
        device: torch.device,
        replay_buffer_capacity: int = 10000,
    ):
        """
        Args:
            height: Grid height
            width: Grid width
            self_agent_idx: The agent index this model belongs to (fixed)
            lr: Learning rate
            discount: Gamma for TD learning
            mlp_hidden_dim: Hidden dimension for MLP head
            mlp_num_layers: Number of MLP layers in head
            conv_channels: List of conv channel sizes, e.g. [32, 64]
            kernel_size: Convolution kernel size
            device: Torch device
            replay_buffer_capacity: Size of replay buffer
        """
        super().__init__(discount, replay_buffer_capacity, device)

        self.height = height
        self.width = width
        self.self_agent_idx = self_agent_idx
        self.input_channels = 5  # Apples, Others, Self, Self_Act, Other_Act

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
        Convert state to 5-channel tensor from this agent's perspective.

        Args:
            state: Current environment state
            acting_agent_idx: Index of the agent that is/will act

        Returns:
            5xHxW numpy array [Apples, Others, Self, Self_Act, Other_Act]
        """
        H, W = state.H, state.L
        i = self.self_agent_idx

        # Channel 0: Apples
        c_apples = state.apples.astype(np.float32)

        # Channel 1: Others (all agents except self)
        c_others = np.zeros((H, W), dtype=np.float32)
        for agent_id in state._agents:
            if agent_id != i:
                r, c = state.agent_position(agent_id)
                c_others[r, c] += 1.0

        # Channel 2: Self position
        c_self = np.zeros((H, W), dtype=np.float32)
        self_r, self_c = state.agent_position(i)
        c_self[self_r, self_c] = 1.0

        # Channel 3: Self_Act (1 at self's position if self is the actor)
        c_self_act = np.zeros((H, W), dtype=np.float32)

        # Channel 4: Other_Act (1 at actor's position if actor is not self)
        c_other_act = np.zeros((H, W), dtype=np.float32)

        if acting_agent_idx == i:
            c_self_act[self_r, self_c] = 1.0
        else:
            actor_r, actor_c = state.agent_position(acting_agent_idx)
            c_other_act[actor_r, actor_c] = 1.0

        return np.stack([c_apples, c_others, c_self, c_self_act, c_other_act])

    def get_value(self, state: State, acting_agent_idx: int) -> float:
        """Get predicted value for a state from this agent's perspective."""
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
            reward: Reward received by THIS agent (r^(i)_t)
            acting_agent_idx: Who acted at time t (for encoding s_t)
            next_acting_agent_idx: Who will act at time t+1 (for encoding s_{t+1})
        """
        s_arr = self.raw_state_to_nn_input(state, acting_agent_idx)
        ns_arr = self.raw_state_to_nn_input(next_state, next_acting_agent_idx)
        self.memory.push(s_arr, ns_arr, reward)


class ValueMLPDecentralized5Ch(BaseValueModelV2):
    """
    MLP-based decentralized value model for a specific agent.

    Architecture: Flatten -> MLP layers -> scalar output

    Each agent i has its own instance with self_agent_idx=i.
    """

    def __init__(
        self,
        height: int,
        width: int,
        self_agent_idx: int,
        lr: float,
        discount: float,
        hidden_dim: int,
        num_layers: int,
        device: torch.device,
        replay_buffer_capacity: int = 100000,
    ):
        """
        Args:
            height: Grid height
            width: Grid width
            self_agent_idx: The agent index this model belongs to (fixed)
            lr: Learning rate
            discount: Gamma for TD learning
            hidden_dim: Hidden dimension for MLP layers
            num_layers: Number of hidden layers
            device: Torch device
            replay_buffer_capacity: Size of replay buffer
        """
        super().__init__(discount, replay_buffer_capacity, device)

        self.height = height
        self.width = width
        self.self_agent_idx = self_agent_idx
        self.input_channels = 5
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
        Convert state to flattened 5-channel vector from this agent's perspective.

        Args:
            state: Current environment state
            acting_agent_idx: Index of the agent that is/will act

        Returns:
            1D numpy array of shape (5*H*W,)
        """
        H, W = state.H, state.L
        i = self.self_agent_idx

        # Channel 0: Apples
        c_apples = state.apples.astype(np.float32)

        # Channel 1: Others (all agents except self)
        c_others = np.zeros((H, W), dtype=np.float32)
        for agent_id in state._agents:
            if agent_id != i:
                r, c = state.agent_position(agent_id)
                c_others[r, c] += 1.0

        # Channel 2: Self position
        c_self = np.zeros((H, W), dtype=np.float32)
        self_r, self_c = state.agent_position(i)
        c_self[self_r, self_c] = 1.0

        # Channel 3: Self_Act (1 at self's position if self is the actor)
        c_self_act = np.zeros((H, W), dtype=np.float32)

        # Channel 4: Other_Act (1 at actor's position if actor is not self)
        c_other_act = np.zeros((H, W), dtype=np.float32)

        if acting_agent_idx == i:
            c_self_act[self_r, self_c] = 1.0
        else:
            actor_r, actor_c = state.agent_position(acting_agent_idx)
            c_other_act[actor_r, actor_c] = 1.0

        return np.stack([c_apples, c_others, c_self, c_self_act, c_other_act]).flatten()

    def get_value(self, state: State, acting_agent_idx: int) -> float:
        """Get predicted value for a state from this agent's perspective."""
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
            reward: Reward received by THIS agent (r^(i)_t)
            acting_agent_idx: Who acted at time t (for encoding s_t)
            next_acting_agent_idx: Who will act at time t+1 (for encoding s_{t+1})
        """
        s_arr = self.raw_state_to_nn_input(state, acting_agent_idx)
        ns_arr = self.raw_state_to_nn_input(next_state, next_acting_agent_idx)
        self.memory.push(s_arr, ns_arr, reward)
