from abc import ABC, abstractmethod

import torch
from torch import optim
from torch.optim.lr_scheduler import LinearLR

from debug.code.nn.encoders import BaseEncoder, GridEncoder
from debug.code.core.enums import DEVICE
from debug.code.nn.main_net import CNNMainNet, MainNet


class NetworkWrapper(ABC):
    def __init__(
            self,
            encoder: BaseEncoder,
            output_dim: int,
            alpha: float,
            discount: float,
            mlp_dims: tuple[int, ...] = (128, 128),
            schedule: bool = False,
            decay_steps: int = 1_000_000,   # number of optimizer updates to reach end LR
            min_lr: float = 1e-6,           # final LR after decay_steps
            conv_channels: list[int] | None = None,
            kernel_size: int = 3,
            momentum: float = 0.0,
            use_mlp: bool = True,
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
                use_mlp=use_mlp,
            ).to(DEVICE)
        else:
            self.model = MainNet(encoder.output_dim(), output_dim, mlp_dims * 2).to(DEVICE)

        self.model = self.model.float()

        self.alpha = float(alpha)
        self.discount = float(discount)
        self.decay_steps = int(decay_steps)
        self.min_lr = float(min_lr)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.alpha, momentum=momentum)

        self.batch_states = []
        self.batch_new_states = []

        # Scheduler: linear decay from alpha -> min_lr over decay_steps optimizer steps.
        self.scheduler = None
        if schedule:
            # LinearLR uses multiplicative factors. Convert min_lr to a factor of alpha.
            end_factor = (self.min_lr / self.alpha) if self.alpha > 0 else 0.0
            end_factor = max(0.0, min(1.0, float(end_factor)))

            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=end_factor,
                total_iters=max(1, self.decay_steps),
            )

    # -----------------------------------------------------------------------
    # Save / load
    # -----------------------------------------------------------------------

    def export_net_state(self):
        return {
            "weights": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (self.scheduler.state_dict() if self.scheduler else None),
        }

    def import_net_state(self, blob, device=DEVICE, *, load_optimizer_state: bool = True):
        self.model.load_state_dict(blob["weights"])

        opt_state = blob.get("optimizer")
        if load_optimizer_state and opt_state is not None:
            self.optimizer.load_state_dict(opt_state)
            for st in self.optimizer.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(device)

        sch_state = blob.get("scheduler")
        if load_optimizer_state and self.scheduler is not None and sch_state is not None:
            self.scheduler.load_state_dict(sch_state)

    # -----------------------------------------------------------------------
    # Modes / LR
    # -----------------------------------------------------------------------

    @abstractmethod
    def train(self):
        raise NotImplementedError

    def set_train_mode(self) -> None:
        self.model.train()

    def set_eval_mode(self) -> None:
        self.model.eval()

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def get_value_function_batch(self, enc):
        self.model.eval()
        with torch.no_grad():
            out = self.model(enc)
        self.model.train()
        return out.reshape(-1).detach().cpu().numpy()

    # -----------------------------------------------------------------------
    # Call this *after* optimizer.step()
    # -----------------------------------------------------------------------

    def _after_update(self) -> None:
        """
        Use in training loop:

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._after_update()
        """
        if self.scheduler is not None:
            self.scheduler.step()
