from abc import ABC, abstractmethod

import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from debug.code.config import DEVICE
from debug.code.main_net import MainNet


def linear_decay_then_hold_factor(step: int, decay_steps: int, min_factor: float) -> float:
    """
    Returns a multiplicative factor f(step) such that:
      lr(step) = base_lr * f(step)
    Linearly decays from 1.0 to min_factor over `decay_steps`,
    then stays at min_factor forever.
    """
    decay_steps = max(1, int(decay_steps))
    step = max(0, int(step))

    # Clamp min_factor into [0, 1] to avoid weird configs
    min_factor = float(min_factor)
    if min_factor < 0.0:
        min_factor = 0.0
    if min_factor > 1.0:
        min_factor = 1.0

    if step >= decay_steps:
        return min_factor

    t = step / decay_steps  # in [0, 1)
    return 1.0 + t * (min_factor - 1.0)


class NetworkWrapper(ABC):
    def __init__(
            self,
            input_dim,
            output_dim,
            alpha,
            discount,
            hidden_dim=128,
            num_layers=4,
            # scheduler params
            schedule=False,
            decay_steps=1_000_000,     # linear decay duration
            min_lr=1e-6,               # final LR after decay (then held)
    ):
        self.model = MainNet(input_dim, output_dim, hidden_dim, num_layers).to(DEVICE)
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=alpha, amsgrad=True)
        self.optimizer = optim.SGD(self.model.parameters(), lr=alpha)
        self.alpha = alpha
        self.discount = discount

        self.batch_states = []
        self.batch_new_states = []

        self._input_dim = input_dim

        self.decay_steps = int(decay_steps * 0.8)
        self.hold_steps = int(decay_steps * 0.2)
        self.min_lr = float(min_lr)

        self._lr_step = 0
        if schedule:
            # LambdaLR multiplies base_lr by lr_lambda(step)
            min_factor = (self.min_lr / self.alpha) if self.alpha > 0 else 0.0
            self.scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda s: linear_decay_then_hold_factor(s, self.decay_steps, min_factor),
            )
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
            self.scheduler.step()  # call after optimizer.step() [web:52]
