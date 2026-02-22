"""ValueNetwork: owns its optimizer, supports train_step for TD updates."""

from __future__ import annotations

import torch
import torch.nn as nn

import orchard.encoding as encoding
from orchard.enums import ModelType
from orchard.schedule import compute_schedule_value
from orchard.datatypes import EncoderOutput, EnvConfig, ModelConfig, ScheduleConfig


class ValueNetwork(nn.Module):
    """Value function approximator. Owns its optimizer.

    train_step processes one transition. Override in subclasses for
    different TD methods (e.g. TD(λ)).
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        env_cfg: EnvConfig,
        lr_schedule: ScheduleConfig,
        total_steps: int,
    ) -> None:
        super().__init__()
        self._lr_schedule = lr_schedule
        self._total_steps = total_steps
        self._step_count: int = 0

        if model_cfg.model_type == ModelType.MLP:
            input_dim = encoding.get_input_dim()
            self.net = self._build_mlp(input_dim, model_cfg.mlp_dims)
        elif model_cfg.model_type == ModelType.CNN:
            channels = encoding.get_input_dim()
            conv, flat_dim = self._build_conv(
                channels, model_cfg.conv_specs, env_cfg.height, env_cfg.width
            )
            mlp_head = self._build_mlp(flat_dim, model_cfg.mlp_dims)
            self.net = nn.Sequential(conv, nn.Flatten(), mlp_head)
        else:
            raise ValueError(f"Unknown model type: {model_cfg.model_type}")

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule.start)

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
        """Build MLP: input → [hidden+ReLU]* → Linear(1)."""
        layers: list[nn.Module] = []
        d = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(d, hd))
            layers.append(nn.ReLU())
            d = hd
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_conv(
        in_channels: int,
        conv_specs: tuple[tuple[int, int], ...] | None,
        height: int,
        width: int,
    ) -> tuple[nn.Sequential, int]:
        """Build conv layers. Returns (conv_sequential, flat_output_dim)."""
        if conv_specs is None:
            conv_specs = ((32, 3), (64, 3))

        layers: list[nn.Module] = []
        c = in_channels
        for out_c, ks in conv_specs:
            padding = ks // 2
            layers.append(nn.Conv2d(c, out_c, kernel_size=ks, padding=padding))
            layers.append(nn.ReLU())
            c = out_c

        flat_dim = c * height * width
        return nn.Sequential(*layers), flat_dim

    def forward(self, encoder_output: EncoderOutput) -> torch.Tensor:
        """Returns scalar value estimate."""
        if encoder_output.scalar is not None:
            return self.net(encoder_output.scalar).squeeze(-1)
        elif encoder_output.grid is not None:
            x = encoder_output.grid
            if x.dim() == 3:
                x = x.unsqueeze(0)
            out = self.net(x).squeeze(-1)
            if out.dim() == 1 and out.size(0) == 1:
                return out.squeeze(0)
            return out
        else:
            raise ValueError("EncoderOutput has neither scalar nor grid")

    def train_step(
        self,
        s_enc: EncoderOutput,
        reward: float,
        discount: float,
        s_next_enc: EncoderOutput,
    ) -> float:
        """One semi-gradient TD(0) step. Returns loss (float)."""
        with torch.no_grad():
            target = reward + discount * self.forward(s_next_enc)
        pred = self.forward(s_enc)
        loss = (pred - target) ** 2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._step_count += 1
        self._update_lr()
        return loss.item()

    def _update_lr(self) -> None:
        """Update LR based on schedule config and internal step count."""
        new_lr = compute_schedule_value(
            self._lr_schedule, self._step_count, self._total_steps
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def get_weight_norms(self) -> dict[str, float]:
        """Per-layer weight norms for detailed logging."""
        norms: dict[str, float] = {}
        for name, param in self.named_parameters():
            if "weight" in name:
                norms[name] = param.data.norm().item()
        return norms

    def get_grad_norms(self) -> dict[str, float]:
        """Per-layer gradient norms for detailed logging."""
        norms: dict[str, float] = {}
        for name, param in self.named_parameters():
            if param.grad is not None and "weight" in name:
                norms[name] = param.grad.norm().item()
        return norms


def create_networks(
    model_cfg: ModelConfig,
    env_cfg: EnvConfig,
    lr_schedule: ScheduleConfig,
    total_steps: int,
) -> list[ValueNetwork]:
    """Create N value networks (one per agent), each with its own optimizer."""
    return [
        ValueNetwork(model_cfg, env_cfg, lr_schedule, total_steps)
        for _ in range(env_cfg.n_agents)
    ]
