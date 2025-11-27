# --- START OF FILE models/value_mlp.py ---

import numpy as np
import torch
import torch.nn as nn
from config import DEVICE
from tadd_helpers.env_functions import State
from models.value_cnn_new import BaseValueModel
from typing_extensions import override


class MLPBaseNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ValueMLPCentralized(BaseValueModel):
    def __init__(
        self,
        height: int,
        width: int,
        lr: float,
        discount: float,
        hidden_dim: int = 512,
        num_layers: int = 4,
        replay_buffer_capacity: int = 100000,
    ):
        super().__init__(discount, replay_buffer_capacity)

        # Input: Flatten(Apples) + Flatten(Agents) = 2 * H * W
        input_dim = 2 * height * width

        self.policy_net = MLPBaseNet(input_dim, hidden_dim, num_layers).to(DEVICE)
        self.target_net = MLPBaseNet(input_dim, hidden_dim, num_layers).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=lr, amsgrad=True
        )

    def raw_state_to_nn_input(self, state: State, **kwargs) -> np.ndarray:
        # Returns 1D numpy array
        return np.concatenate([state.apples.flatten(), state.agents.flatten()]).astype(
            np.float32
        )


class ValueMLPDecentralized(BaseValueModel):
    def __init__(
        self,
        height: int,
        width: int,
        lr: float,
        discount: float,
        hidden_dim: int = 512,
        num_layers: int = 4,
        replay_buffer_capacity: int = 100000,
    ):
        super().__init__(discount, replay_buffer_capacity)

        # Input: Flatten(Apples) + Flatten(Others) + Flatten(Self) = 3 * H * W
        input_dim = 3 * height * width

        self.policy_net = MLPBaseNet(input_dim, hidden_dim, num_layers).to(DEVICE)
        self.target_net = MLPBaseNet(input_dim, hidden_dim, num_layers).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=lr, amsgrad=True
        )

    def raw_state_to_nn_input(self, state: State, **kwargs) -> np.ndarray:
        agent_pos = kwargs.get("agent_pos")
        if agent_pos is None:
            raise ValueError("Decentralized MLP requires 'agent_pos' kwarg.")

        agents_map = state.agents
        apples_map = state.apples

        # Create Self Map
        self_map = np.zeros_like(agents_map)
        self_map[agent_pos[0], agent_pos[1]] = 1.0

        # Create Others Map
        others_map = agents_map - self_map

        # Returns 1D numpy array
        return np.concatenate(
            [apples_map.flatten(), others_map.flatten(), self_map.flatten()]
        ).astype(np.float32)

    @override
    def add_experience(self, state: State, next_state: State, reward: float, **kwargs):
        # Explicit extraction
        agent_pos = kwargs.get("agent_pos")
        agent_pos_next = kwargs.get("agent_pos_next")

        # Validation
        if agent_pos is None or agent_pos_next is None:
            raise ValueError(
                "Decentralized add_experience requires 'agent_pos' and 'agent_pos_next'"
            )

        # 1. Current State (Perspective: agent_pos)
        s_input = self.raw_state_to_nn_input(state, agent_pos=agent_pos)

        # 2. Next State (Perspective: agent_pos_next)
        ns_input = self.raw_state_to_nn_input(next_state, agent_pos=agent_pos_next)

        self.memory.push(s_input, ns_input, reward)
