from abc import abstractmethod

import torch
from torch import optim

from config import DEVICE
from models.main_net import MainNet


class Network:
    def __init__(self, input_dim, output_dim, alpha, discount, hidden_dim=128, num_layers=4):
        self.function = MainNet(input_dim, output_dim, hidden_dim, num_layers).to(DEVICE)
        self.optimizer = optim.AdamW(self.function.parameters(), lr=alpha,
                                     amsgrad=True)
        self.alpha = alpha
        self.discount = discount

        self.batch_states = []
        self.batch_new_states = []

        self._input_dim = input_dim

    def get_input_dim(self):
        return self._input_dim

    def export_net_state(self):
        return {
            "weights": self.function.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def import_net_state(self, blob, device=DEVICE):
        self.function.load_state_dict(blob["weights"])
        if blob.get("optimizer") is not None:
            self.optimizer.load_state_dict(blob["optimizer"])
            # move optimizer tensors to correct device
            for st in self.optimizer.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(device)

    @abstractmethod
    def train(self):
        raise NotImplementedError
