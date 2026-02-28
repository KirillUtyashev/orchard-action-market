"""ValueNetwork: owns its optimizer, supports train_step for TD updates."""

from __future__ import annotations

import torch
import torch.nn as nn

import orchard.encoding as encoding
from orchard.enums import ModelType
from orchard.schedule import compute_schedule_value
from orchard.datatypes import EncoderOutput, EnvConfig, ModelConfig, NStepTransition, ScheduleConfig


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
        nstep: int = 1,
    ) -> None:
        super().__init__()
        self._lr_schedule = lr_schedule
        self._total_steps = total_steps
        self._step_count: int = 0
        self._env_step: int = 0

        if model_cfg.model_type == ModelType.MLP:
            input_dim = encoding.get_scalar_dim()
            self.net = self._build_mlp(input_dim, model_cfg.mlp_dims)
        elif model_cfg.model_type == ModelType.CNN:
            channels = encoding.get_grid_channels()
            scalar_extra = encoding.get_scalar_dim()
            self.conv, conv_flat_dim = self._build_conv(
                channels, model_cfg.conv_specs, env_cfg.height, env_cfg.width
            )
            self.flatten = nn.Flatten()
            self.net = self._build_mlp(conv_flat_dim + scalar_extra, model_cfg.mlp_dims)
        else:
            raise ValueError(f"Unknown model type: {model_cfg.model_type}")

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule.start)
        
        self._nstep = nstep
        self._nstep_buffer: list[NStepTransition] = []
        self._nstep_latest_next: EncoderOutput | None = None

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
        h, w = height, width
        for out_c, ks in conv_specs:
            padding = ks // 2
            layers.append(nn.Conv2d(c, out_c, kernel_size=ks, padding=padding))
            layers.append(nn.ReLU())
            h = (h + 2 * padding - ks) // 1 + 1
            w = (w + 2 * padding - ks) // 1 + 1
            c = out_c

        flat_dim = c * h * w
        return nn.Sequential(*layers), flat_dim

    def forward(self, encoder_output: EncoderOutput) -> torch.Tensor:
        """Returns scalar value estimate."""
        if encoder_output.grid is not None:
            x = encoder_output.grid
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = self.flatten(self.conv(x))
            if encoder_output.scalar is not None:
                s = encoder_output.scalar
                if s.dim() == 1:
                    s = s.unsqueeze(0)
                x = torch.cat([x, s], dim=-1)
            out = self.net(x).squeeze(-1)
            if out.dim() == 1 and out.size(0) == 1:
                return out.squeeze(0)
            return out
        elif encoder_output.scalar is not None:
            return self.net(encoder_output.scalar).squeeze(-1)
        else:
            raise ValueError("EncoderOutput has neither scalar nor grid")

    def train_step(self, s_enc, reward, discount, s_next_enc, env_step: int = 0) -> float:
        self._env_step = env_step
        self._nstep_buffer.append(NStepTransition(s_enc, reward, discount))
        self._nstep_latest_next = s_next_enc
        if len(self._nstep_buffer) >= self._nstep:
            return self._do_nstep_update()
        return 0.0

    def _do_nstep_update(self) -> float:
        """Compute n-step return for buffer[0], update, pop."""
        assert self._nstep_latest_next is not None

        with torch.no_grad():
            G = self.forward(self._nstep_latest_next).item()
        for i in range(len(self._nstep_buffer) - 1, -1, -1):
            t = self._nstep_buffer[i]
            G = t.reward + t.discount * G

        s_enc = self._nstep_buffer[0].s_enc
        self._nstep_buffer.pop(0)

        with torch.no_grad():
            target = torch.tensor(G)
        pred = self.forward(s_enc)
        loss = (pred - target) ** 2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._step_count += 1
        self._update_lr()
        return loss.item()

    def flush_nstep(self) -> float:
        """Flush buffer with truncated returns. Call at reset/end. Returns total loss."""
        total_loss = 0.0
        while self._nstep_buffer:
            total_loss += self._do_nstep_update()
        self._nstep_latest_next = None
        return total_loss
    
    def _update_lr(self) -> None:
        new_lr = compute_schedule_value(
            self._lr_schedule, self._env_step, self._total_steps
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
    nstep: int = 1,
) -> list[ValueNetwork]:
    return [
        ValueNetwork(model_cfg, env_cfg, lr_schedule, total_steps, nstep)
        for _ in range(env_cfg.n_agents)
    ]
