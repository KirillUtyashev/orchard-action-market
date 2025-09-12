from abc import abstractmethod

import torch
from torch import optim

from config import DEVICE
from models.main_net import MainNet


class NetworkWrapper:
    """Wrapper class for neural networks, and it contains the actual neural net

    Attributes:
        function (torch.nn.Module): The neural network model.
        optimizer (optim.Optimizer): The optimizer for training the network.
        alpha (float): The learning rate.
        discount (float): The discount factor for future rewards.
        batch_states (list): List to store states for batch training.
        batch_new_states (list): List to store new states for batch training.
        _input_dim (int): The input dimension for the network.
    """

    function: torch.nn.Module
    optimizer: optim.Optimizer
    alpha: float
    discount: float
    batch_states: list
    batch_new_states: list
    _input_dim: int

    def __init__(
        self, input_dim, output_dim, alpha, discount, hidden_dim=128, num_layers=4
    ):
        self.function = MainNet(input_dim, output_dim, hidden_dim, num_layers).to(
            DEVICE
        )
        self.optimizer = optim.AdamW(self.function.parameters(), lr=alpha, amsgrad=True)
        self.alpha = alpha
        self.discount = discount

        self.batch_states = []
        self.batch_new_states = []

        self._input_dim = input_dim * 2

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
