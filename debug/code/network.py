from abc import ABC, abstractmethod

import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from debug.code.config import DEVICE
from debug.code.main_net import MainNet


class NetworkWrapper(ABC):
    def __init__(
        self, input_dim, output_dim, alpha, discount, hidden_dim=128, num_layers=4, num_training_steps=10000, schedule=False
    ):
        self.model = MainNet(input_dim, output_dim, hidden_dim, num_layers).to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=alpha, amsgrad=True)
        self.alpha = alpha
        self.discount = discount

        self.batch_states = []
        self.batch_new_states = []

        self._input_dim = input_dim

        self._lr_step = 0
        if schedule:
            self.scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda s: linear_decay_factor(s, num_training_steps),
            )  # lr = base_lr * lr_lambda(step) [web:231]
        else:
            self.scheduler = None

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

    def _after_update(self):
        self._lr_step += 1
        if self.scheduler:
            self.scheduler.step()  # updates optimizer.param_groups[i]["lr"] [web:231]


def linear_decay_factor(step: int, total_steps: int):
    step = min(max(step, 0), total_steps)
    return 1.0 - (step / max(1, total_steps))  # scales base lr [web:231]
