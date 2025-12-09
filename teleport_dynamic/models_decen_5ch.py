import torch
import torch.nn as nn
import numpy as np
from config import DEVICE
from tadd_helpers.env_functions import State
from models.value_cnn_new import BaseValueModel
from typing_extensions import override


class ValueCNNDecentralized5Ch(BaseValueModel):
    def __init__(
        self, height, width, lr, discount, conv_layers_config, head_layers_config
    ):
        # Inherit with buffer size 100k
        super().__init__(discount, 100000)

        self.input_channels = 5

        # 1. Conv Body
        layers = []
        in_c = 5
        for out_c, k, s in conv_layers_config:
            pad = (k - 1) // 2
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=pad))
            layers.append(nn.ReLU())
            in_c = out_c
        self.conv_net = nn.Sequential(*layers)

        # 2. Infer Flatten Dim (Strict Init)
        with torch.no_grad():
            dummy = torch.zeros(1, 5, height, width)
            out = self.conv_net(dummy)
        flat_dim = int(np.prod(out.shape[1:]))

        # 3. Head
        head = []
        in_d = flat_dim
        for h in head_layers_config:
            head.append(nn.Linear(in_d, h))
            head.append(nn.ReLU())
            in_d = h
        head.append(nn.Linear(in_d, 1))
        self.head = nn.Sequential(*head)

        # 4. Init Nets
        self.policy_net = nn.Sequential(self.conv_net, nn.Flatten(), self.head).to(
            DEVICE
        )
        self.target_net = nn.Sequential(self.conv_net, nn.Flatten(), self.head).to(
            DEVICE
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=lr, amsgrad=True
        )

    @override
    def raw_state_to_nn_input(self, state: State, **kwargs) -> np.ndarray:
        """
        kwargs:
            acting_agent_idx (int): Index of the agent that was teleported.
            self_agent_idx (int): Index of the "self" agent for whom we compute V
        """
        acting_idx = kwargs.get("acting_agent_idx")
        self_idx = kwargs.get("self_agent_idx")
        if acting_idx is None or self_idx is None:
            raise ValueError("Model requires acting_agent_idx and self_agent_idx")

        H, W = state.H, state.L

        # 1. Apples
        c_apples = state.apples.astype(np.float32)

        # 2. Others & 3. Self
        c_others = np.zeros((H, W), dtype=np.float32)
        c_self = np.zeros((H, W), dtype=np.float32)

        for i in range(len(state._agents)):
            r, c = state.agent_position(i)
            if i == self_idx:
                c_self[r, c] = 1.0
            else:
                c_others[r, c] += 1.0

        # 4. Self Act & 5. Other Act
        c_self_act = np.zeros((H, W), dtype=np.float32)
        c_other_act = np.zeros((H, W), dtype=np.float32)
        actor_pos = state.agent_position(acting_idx)

        if acting_idx == self_idx:
            c_self_act[actor_pos[0], actor_pos[1]] = 1.0
        else:
            c_other_act[actor_pos[0], actor_pos[1]] = 1.0

        return np.stack([c_apples, c_others, c_self, c_self_act, c_other_act])


class ValueMLPDecentralized5Ch(BaseValueModel):
    def __init__(self, height, width, lr, discount, hidden_dims):
        super().__init__(discount, 100000)

        input_dim = 5 * height * width

        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ReLU())
            in_d = h
        layers.append(nn.Linear(in_d, 1))

        self.policy_net = nn.Sequential(*layers).to(DEVICE)
        self.target_net = nn.Sequential(*layers).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=lr, amsgrad=True
        )

    def raw_state_to_nn_input(self, state: State, **kwargs) -> np.ndarray:
        # Duplicated logic from CNN to keep class self-contained
        acting_idx = kwargs.get("acting_agent_idx")
        self_idx = kwargs.get("self_agent_idx")

        H, W = state.H, state.L
        c_apples = state.apples.astype(np.float32)
        c_others = np.zeros((H, W), dtype=np.float32)
        c_self = np.zeros((H, W), dtype=np.float32)
        for i in range(len(state._agents)):
            r, c = state.agent_position(i)
            if i == self_idx:
                c_self[r, c] = 1.0
            else:
                c_others[r, c] += 1.0
        c_self_act = np.zeros((H, W), dtype=np.float32)
        c_other_act = np.zeros((H, W), dtype=np.float32)
        actor_pos = state.agent_position(acting_idx)
        if acting_idx == self_idx:
            c_self_act[actor_pos[0], actor_pos[1]] = 1.0
        else:
            c_other_act[actor_pos[0], actor_pos[1]] = 1.0

        stack = np.stack([c_apples, c_others, c_self, c_self_act, c_other_act])
        return stack.flatten()
