from abc import ABC, abstractmethod

import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from debug.code.encoders import BaseEncoder, EncoderOutput, GridEncoder
from debug.code.enums import DEVICE, W
from debug.code.main_net import CNNMainNet, MainNet


def linear_decay_then_hold_factor(step: int, decay_steps: int, min_factor: float) -> float:
    decay_steps = max(1, int(decay_steps))
    step = max(0, int(step))
    min_factor = float(min_factor)
    min_factor = max(0.0, min(1.0, min_factor))
    if step >= decay_steps:
        return min_factor
    return 1.0 + (step / decay_steps) * (min_factor - 1.0)


class NetworkWrapper(ABC):
    def __init__(
            self,
            encoder: BaseEncoder,
            output_dim: int,
            alpha: float,
            discount: float,
            mlp_dims: tuple[int, ...] = (128, 128),
            schedule: bool = False,
            decay_steps: int = 1_000_000,
            min_lr: float = 1e-6,
            conv_channels: list[int] = None,
            kernel_size: int = 3,
    ):
        self.encoder = encoder
        self._is_cnn = isinstance(encoder, GridEncoder)

        if self._is_cnn:
            self.model = CNNMainNet(
                grid_shape=(encoder.grid_channels(), encoder.H, encoder.W),
                scalar_dim=encoder.scalar_dim(),
                output_dim=output_dim,
                conv_channels=conv_channels or [32, 64],
                kernel_size=kernel_size,
                mlp_dims=mlp_dims,
            ).to(DEVICE)
        else:
            self.model = MainNet(encoder.output_dim(), output_dim, mlp_dims).to(DEVICE)

        self.model = self.model.float()
        self.optimizer = optim.SGD(self.model.parameters(), lr=alpha)
        self.alpha = alpha
        self.discount = discount

        self.batch_states = []
        self.batch_new_states = []

        self.decay_steps = int(decay_steps)
        self.min_lr = float(min_lr)
        self._lr_step = 0

        if schedule:
            min_factor = (self.min_lr / self.alpha) if self.alpha > 0 else 0.0
            self.scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda s: linear_decay_then_hold_factor(s, self.decay_steps, min_factor),
            )
        else:
            self.scheduler = None

    # -----------------------------------------------------------------------
    # Unchanged
    # -----------------------------------------------------------------------

    def export_net_state(self):
        return {
            "weights":   self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def import_net_state(self, blob, device=DEVICE):
        self.model.load_state_dict(blob["weights"])
        if blob.get("optimizer") is not None:
            self.optimizer.load_state_dict(blob["optimizer"])
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

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def _after_update(self):
        self._lr_step += 1
        if self.scheduler:
            self.scheduler.step()
