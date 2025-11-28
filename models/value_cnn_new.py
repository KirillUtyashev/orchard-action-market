from typing_extensions import override
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import collections
from collections import namedtuple
from config import DEVICE
from utils import ten_float
from tadd_helpers.env_functions import State

Transition = namedtuple("Transition", ("state", "new_state", "reward"))


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = collections.deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class BaseValueModel(nn.Module):
    def __init__(self, discount: float, replay_buffer_capacity: int):
        super().__init__()
        self.discount = discount
        self.memory = ReplayBuffer(replay_buffer_capacity)
        self.loss_history = []
        # Placeholders
        self.policy_net: nn.Module
        self.target_net: nn.Module
        self.optimizer: torch.optim.Optimizer

    def raw_state_to_nn_input(self, state: State, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def get_value(self, state: State, **kwargs) -> float:
        arr = self.raw_state_to_nn_input(state, **kwargs)
        t = ten_float(arr, DEVICE).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(t).item()

    def add_experience(self, state: State, next_state: State, reward: float, **kwargs):
        s = self.raw_state_to_nn_input(state, **kwargs)
        ns = self.raw_state_to_nn_input(next_state, **kwargs)
        self.memory.push(s, ns, reward)

    def train_batch(self, batch_size: int) -> float:
        if len(self.memory) < batch_size:
            return 0.0
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        states = ten_float(np.stack(batch.state), DEVICE)
        next_states = ten_float(np.stack(batch.new_state), DEVICE)
        rewards = ten_float(np.array(batch.reward), DEVICE)

        curr_v = self.policy_net(states).squeeze(1)
        with torch.no_grad():
            if self.discount == 0:
                target_v = rewards
            else:
                next_v = self.target_net(next_states).squeeze(1)
                target_v = rewards + self.discount * next_v

        loss = nn.MSELoss()(curr_v, target_v)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=1.0
        )  # Gradient Clipping
        self.optimizer.step()
        l = loss.item()
        self.loss_history.append(l)
        return l

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# =============================================================================
# ARCHITECTURE 1: DEEP STANDARD CNN (Flatten) - "The Professor's Choice"
# =============================================================================
class CNNDeepStandard(nn.Module):
    def __init__(
        self,
        input_channels,
        height,
        width,
        conv_channels,
        hidden_dim,
        num_layers,
        kernel_size,
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_c = input_channels

        # Ensure at least 3 layers for Receptive Field
        actual_channels = list(conv_channels)
        while len(actual_channels) < 3:
            actual_channels.append(actual_channels[-1])

        for out_c in actual_channels:
            padding = (kernel_size - 1) // 2
            self.conv_layers.append(
                nn.Conv2d(in_c, out_c, kernel_size, padding=padding)
            )
            self.conv_layers.append(nn.ReLU())
            in_c = out_c

        flat_dim = in_c * height * width

        layers = []
        in_dim = flat_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, x):
        for l in self.conv_layers:
            x = l(x)
        x = torch.flatten(x, 1)
        return self.mlp_head(x)


class ValueCNNCentralizedStandard(BaseValueModel):
    def __init__(
        self,
        height,
        width,
        lr,
        discount,
        hidden_dim,
        num_layers,
        conv_channels,
        kernel_size,
    ):
        super().__init__(discount, 100000)
        self.policy_net = CNNDeepStandard(
            2, height, width, conv_channels, hidden_dim, num_layers, kernel_size
        ).to(DEVICE)
        self.target_net = CNNDeepStandard(
            2, height, width, conv_channels, hidden_dim, num_layers, kernel_size
        ).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=lr, amsgrad=True
        )

    def raw_state_to_nn_input(self, state, **kwargs):
        return np.stack([state.apples, state.agents], axis=0).astype(np.float32)


# =============================================================================
# ARCHITECTURE 2: COORD CONV (Tunable) - "The Sanity Check"
# =============================================================================
class CNNCoordConv(nn.Module):
    def __init__(
        self,
        input_channels,
        height,
        width,
        conv_channels,
        hidden_dim,
        num_layers,
        kernel_size=1,
    ):
        super().__init__()
        self.height = height
        self.width = width

        # Buffer coordinates
        yy, xx = torch.meshgrid(
            torch.arange(height), torch.arange(width), indexing="ij"
        )
        self.register_buffer("yy", yy.float() / (height - 1))
        self.register_buffer("xx", xx.float() / (width - 1))

        # --- CONV LAYERS (1x1 typically) ---
        layers = []
        # Input = Original Channels + 2 Coordinate Channels
        in_c = input_channels + 2

        # Use provided channels or default
        actual_channels = list(conv_channels)
        if not actual_channels:
            actual_channels = [hidden_dim, hidden_dim]

        for out_c in actual_channels:
            padding = (kernel_size - 1) // 2
            layers.append(nn.Conv2d(in_c, out_c, kernel_size, padding=padding))
            layers.append(nn.ReLU())
            in_c = out_c

        self.conv_stack = nn.Sequential(*layers)

        # --- FLATTEN + MLP HEAD ---
        # No MaxPool. We flatten the grid.
        # This allows the MLP to see ALL pixels at once and suppress global noise instantly.
        flat_dim = in_c * height * width

        head_layers = []
        in_dim = flat_dim
        for _ in range(num_layers):
            head_layers.append(nn.Linear(in_dim, hidden_dim))
            head_layers.append(nn.ReLU())
            in_dim = hidden_dim
        head_layers.append(nn.Linear(in_dim, 1))

        self.mlp_head = nn.Sequential(*head_layers)

    def forward(self, x):
        batch_size = x.shape[0]

        # Explicit expansion
        yy = self.yy.expand(batch_size, 1, self.height, self.width)  # type: ignore
        xx = self.xx.expand(batch_size, 1, self.height, self.width)  # type: ignore

        # Append Coords
        x = torch.cat([x, yy, xx], dim=1)

        # Conv
        x = self.conv_stack(x)

        # Flatten
        x = torch.flatten(x, 1)

        # MLP
        return self.mlp_head(x)


class ValueCNNCoord(BaseValueModel):
    def __init__(
        self,
        height,
        width,
        lr,
        discount,
        hidden_dim,
        num_layers,
        conv_channels,
        kernel_size,
    ):
        super().__init__(discount, 100000)
        # Pass all tunable parameters to the CoordConv Net
        self.policy_net = CNNCoordConv(
            2, height, width, conv_channels, hidden_dim, num_layers, kernel_size
        ).to(DEVICE)
        self.target_net = CNNCoordConv(
            2, height, width, conv_channels, hidden_dim, num_layers, kernel_size
        ).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=lr, amsgrad=True
        )

    def raw_state_to_nn_input(self, state, **kwargs):
        return np.stack([state.apples, state.agents], axis=0).astype(np.float32)


class ValueCNNDecentralizedStandard(BaseValueModel):
    def __init__(
        self,
        height,
        width,
        lr,
        discount,
        hidden_dim,
        num_layers,
        conv_channels,
        kernel_size,
    ):
        super().__init__(discount, 100000)
        # Input Channels = 3 (Apples, Others, Self)
        self.policy_net = CNNDeepStandard(
            3, height, width, conv_channels, hidden_dim, num_layers, kernel_size
        ).to(DEVICE)
        self.target_net = CNNDeepStandard(
            3, height, width, conv_channels, hidden_dim, num_layers, kernel_size
        ).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=lr, amsgrad=True
        )

    def raw_state_to_nn_input(self, state: State, **kwargs):
        agent_pos = kwargs.get("agent_pos")
        if agent_pos is None:
            raise ValueError("Decentralized model requires 'agent_pos'")

        # 1. Apples
        map_apples = state.apples.astype(np.float32)

        # 2. Self Map (One Hot)
        map_self = np.zeros(state.apples.shape, dtype=np.float32)
        map_self[agent_pos[0], agent_pos[1]] = 1.0

        # 3. Others Map (Global Agents - Self)
        map_others = state.agents.astype(np.float32) - map_self

        return np.stack([map_apples, map_others, map_self], axis=0)

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


class ValueCNNCoordDecentralized(BaseValueModel):
    def __init__(
        self,
        height,
        width,
        lr,
        discount,
        hidden_dim,
        num_layers,
        conv_channels,
        kernel_size,
    ):
        super().__init__(discount, 100000)
        # Input Channels = 3 (Apples, Others, Self)
        self.policy_net = CNNCoordConv(
            3, height, width, conv_channels, hidden_dim, num_layers, kernel_size
        ).to(DEVICE)
        self.target_net = CNNCoordConv(
            3, height, width, conv_channels, hidden_dim, num_layers, kernel_size
        ).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=lr, amsgrad=True
        )

    def raw_state_to_nn_input(self, state: State, **kwargs):
        agent_pos = kwargs.get("agent_pos")
        if agent_pos is None:
            raise ValueError("Decentralized model requires 'agent_pos'")

        map_apples = state.apples.astype(np.float32)
        map_self = np.zeros(state.apples.shape, dtype=np.float32)
        map_self[agent_pos[0], agent_pos[1]] = 1.0
        map_others = state.agents.astype(np.float32) - map_self

        return np.stack([map_apples, map_others, map_self], axis=0)

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
