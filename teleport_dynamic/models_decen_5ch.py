"""
Decentralized Value Models (5-Channel Input) - DEBUG VERSION
"""

import torch
import torch.nn as nn
import numpy as np
import copy

from tadd_helpers.env_functions import State
from teleport_dynamic.base_value_model import BaseValueModelV2


def check_nan_params(net, name="network"):
    """Check all parameters for NaN/Inf"""
    for pname, param in net.named_parameters():
        if torch.isnan(param).any():
            print(f"  [NaN CHECK] NaN in {name}.{pname}")
            return True
        if torch.isinf(param).any():
            print(f"  [NaN CHECK] Inf in {name}.{pname}")
            return True
    return False

class ValueCNNDecentralized5Ch(BaseValueModelV2):
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
        super().__init__(discount, replay_buffer_capacity, device)

        self.height = height
        self.width = width
        self.self_agent_idx = self_agent_idx
        self.input_channels = 5

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

        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, height, width)
            conv_out = self.conv_net(dummy)
        flat_dim = int(np.prod(conv_out.shape[1:]))

        head_layers = []
        in_d = flat_dim
        for _ in range(mlp_num_layers):
            head_layers.append(nn.Linear(in_d, mlp_hidden_dim))
            head_layers.append(nn.ReLU())
            in_d = mlp_hidden_dim
        head_layers.append(nn.Linear(in_d, 1))
        self.head = nn.Sequential(*head_layers)

        # === BUILD ON CPU ===
        self.policy_net = nn.Sequential(self.conv_net, nn.Flatten(), self.head)
        
        # Initialize on CPU
        self._safe_init_weights()
        
        # CHECK BEFORE MOVE
        print(f"[DEBUG] Checking policy_net BEFORE .to(device)")
        if check_nan_params(self.policy_net, "policy_net_BEFORE_MOVE"):
            raise ValueError("NaN BEFORE move to CUDA!")
        
        # Deepcopy on CPU
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()
        
        # Move to device
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        
        # CHECK AFTER MOVE
        print(f"[DEBUG] Checking policy_net AFTER .to(device)")
        if check_nan_params(self.policy_net, "policy_net_AFTER_MOVE"):
            raise ValueError("NaN AFTER move to CUDA!")

        self.optimizer = torch.optim.Adam(
    self.policy_net.parameters(), lr=lr
        )
        
        print(f"  [INIT OK] Agent {self_agent_idx} CNN initialized")
        
        # Check immediately after init
        for pname, param in self.policy_net.named_parameters():
            if torch.isnan(param).any():
                print(f"[INIT DEBUG] NaN in {pname} immediately after init ON CPU")
                print(f"  param device: {param.device}")
                print(f"  param shape: {param.shape}")
    
    def _test_forward_pass(self):
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, self.height, self.width, device=self.device)
            out_policy = self.policy_net(dummy)
            out_target = self.target_net(dummy)
            
            if torch.isnan(out_policy).any() or torch.isnan(out_target).any():
                raise ValueError(f"NaN in forward pass!")
            if torch.isinf(out_policy).any() or torch.isinf(out_target).any():
                raise ValueError(f"Inf in forward pass!")

    def raw_state_to_nn_input(self, state: State, acting_agent_idx: int) -> np.ndarray:
        H, W = state.H, state.L
        i = self.self_agent_idx

        c_apples = state.apples.astype(np.float32)
        
        c_others = np.zeros((H, W), dtype=np.float32)
        for agent_id in state._agents:
            if agent_id != i:
                r, c = state.agent_position(agent_id)
                c_others[r, c] += 1.0

        c_self = np.zeros((H, W), dtype=np.float32)
        self_r, self_c = state.agent_position(i)
        c_self[self_r, self_c] = 1.0

        c_self_act = np.zeros((H, W), dtype=np.float32)
        c_other_act = np.zeros((H, W), dtype=np.float32)

        if acting_agent_idx == i:
            c_self_act[self_r, self_c] = 1.0
        else:
            actor_r, actor_c = state.agent_position(acting_agent_idx)
            c_other_act[actor_r, actor_c] = 1.0

        return np.stack([c_apples, c_others, c_self, c_self_act, c_other_act])

    def get_value(self, state: State, acting_agent_idx: int) -> float:
        arr = self.raw_state_to_nn_input(state, acting_agent_idx)
        t = torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(t).item()

    def add_experience(
        self, state: State, next_state: State, reward: float,
        acting_agent_idx: int, next_acting_agent_idx: int,
    ) -> None:
        s_arr = self.raw_state_to_nn_input(state, acting_agent_idx)
        ns_arr = self.raw_state_to_nn_input(next_state, next_acting_agent_idx)
        self.memory.push(s_arr, ns_arr, reward)


class ValueMLPDecentralized5Ch(BaseValueModelV2):
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
        super().__init__(discount, replay_buffer_capacity, device)

        self.height = height
        self.width = width
        self.self_agent_idx = self_agent_idx
        self.input_channels = 5
        input_dim = self.input_channels * height * width

        layers = []
        in_d = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU())
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, 1))
        self.mlp = nn.Sequential(*layers)
        
        self.policy_net = nn.Sequential(nn.Flatten(), self.mlp)
        self._safe_init_weights()
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()
        
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        self.optimizer = torch.optim.Adam(
    self.policy_net.parameters(), lr=lr
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels * height * width, device=self.device)
            _ = self.policy_net(dummy)
            _ = self.target_net(dummy)
        print(f"  [INIT OK] Agent {self_agent_idx} MLP initialized")


    def raw_state_to_nn_input(self, state: State, acting_agent_idx: int) -> np.ndarray:
        H, W = state.H, state.L
        i = self.self_agent_idx

        c_apples = state.apples.astype(np.float32)
        c_others = np.zeros((H, W), dtype=np.float32)
        for agent_id in state._agents:
            if agent_id != i:
                r, c = state.agent_position(agent_id)
                c_others[r, c] += 1.0

        c_self = np.zeros((H, W), dtype=np.float32)
        self_r, self_c = state.agent_position(i)
        c_self[self_r, self_c] = 1.0

        c_self_act = np.zeros((H, W), dtype=np.float32)
        c_other_act = np.zeros((H, W), dtype=np.float32)

        if acting_agent_idx == i:
            c_self_act[self_r, self_c] = 1.0
        else:
            actor_r, actor_c = state.agent_position(acting_agent_idx)
            c_other_act[actor_r, actor_c] = 1.0

        return np.stack([c_apples, c_others, c_self, c_self_act, c_other_act]).flatten()

    def get_value(self, state: State, acting_agent_idx: int) -> float:
        arr = self.raw_state_to_nn_input(state, acting_agent_idx)
        t = torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(t).item()

    def add_experience(
        self, state: State, next_state: State, reward: float,
        acting_agent_idx: int, next_acting_agent_idx: int,
    ) -> None:
        s_arr = self.raw_state_to_nn_input(state, acting_agent_idx)
        ns_arr = self.raw_state_to_nn_input(next_state, next_acting_agent_idx)
        self.memory.push(s_arr, ns_arr, reward)
