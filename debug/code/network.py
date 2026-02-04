from abc import ABC, abstractmethod

import torch
from torch import optim

from debug.code.config import DEVICE
from debug.code.main_net import MainNet


class NetworkWrapper(ABC):
    def __init__(
        self, input_dim, output_dim, alpha, discount, hidden_dim=128, num_layers=4
    ):
        self.model = MainNet(input_dim, output_dim, hidden_dim, num_layers).to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=alpha, amsgrad=True)
        self.alpha = alpha
        self.discount = discount

        self.batch_states = []
        self.batch_new_states = []

        self._input_dim = input_dim

    def get_input_dim(self):
        return self._input_dim

    def export_net_state(self):
        return {
            "weights": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def import_net_state(self, blob, device=DEVICE):
        self.model.load_state_dict(blob["weights"])
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

    def set_train_mode(self) -> None:
        self.model.train()

    def set_eval_mode(self) -> None:
        self.model.eval()
