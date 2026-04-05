"""ValueNetwork: value function approximator with backward-view TD(λ)."""

from __future__ import annotations

import torch
import torch.nn as nn

import orchard.encoding as encoding
from orchard.actor_critic import PolicyNetwork
from orchard.enums import Activation, LearningType, WeightInit
from orchard.datatypes import EncoderOutput, EnvConfig, ModelConfig, TrainConfig
from orchard.schedule import compute_schedule_value


class ValueNetwork(nn.Module):
    """Value function approximator with backward-view TD(λ).

    Owns its eligibility traces. The caller is responsible for computing
    the learning rate and passing it to td_step.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        env_cfg: EnvConfig,
        td_lambda: float = 0.0,
    ) -> None:
        super().__init__()

        channels = encoding.get_grid_channels()
        scalar_extra = encoding.get_scalar_dim()
        self.conv, conv_flat_dim = self._build_conv(
            channels, model_cfg.conv_specs,
            encoding.get_grid_height(), encoding.get_grid_width(),
            model_cfg.activation,
        )
        self.flatten = nn.Flatten()
        self.net = self._build_mlp(
            conv_flat_dim + scalar_extra, model_cfg.mlp_dims, model_cfg.activation,
        )

        if model_cfg.weight_init == WeightInit.ZERO_BIAS:
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                    nn.init.zeros_(m.bias)

        # --- TD(λ) state ---
        self._td_lambda = td_lambda
        self._gamma_prev: float = 0.0
        self._traces: dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters():
            self._traces[name] = torch.zeros_like(param.data)

    @staticmethod
    def _build_mlp(
        input_dim: int, hidden_dims: tuple[int, ...],
        activation: Activation = Activation.RELU,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        d = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(d, hd))
            if activation == Activation.LEAKY_RELU:
                layers.append(nn.LeakyReLU())
            elif activation == Activation.RELU:
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
        activation: Activation = Activation.RELU,
    ) -> tuple[nn.Sequential, int]:
        if conv_specs is None:
            conv_specs = ((32, 3), (64, 3))

        layers: list[nn.Module] = []
        c = in_channels
        h, w = height, width
        for out_c, ks in conv_specs:
            padding = ks // 2
            layers.append(nn.Conv2d(c, out_c, kernel_size=ks, padding=padding))
            if activation == Activation.LEAKY_RELU:
                layers.append(nn.LeakyReLU())
            elif activation == Activation.RELU:
                layers.append(nn.ReLU())
            h = (h + 2 * padding - ks) // 1 + 1
            w = (w + 2 * padding - ks) // 1 + 1
            c = out_c

        flat_dim = c * h * w
        return nn.Sequential(*layers), flat_dim

    def forward(self, encoder_output: EncoderOutput) -> torch.Tensor:
        """Returns scalar value estimate."""
        if encoder_output.grid is not None:
            return self.forward_raw(encoder_output.grid, encoder_output.scalar)
        elif encoder_output.scalar is not None:
            return self.net(encoder_output.scalar).squeeze(-1)
        else:
            raise ValueError("EncoderOutput has neither scalar nor grid")

    def forward_raw(self, grid: torch.Tensor, scalar: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass on raw tensors (for vmap compatibility).

        Args:
            grid: (C, H, W) or (B, C, H, W)
            scalar: (S,) or (B, S) or None
        """
        x = grid
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.flatten(self.conv(x))
        if scalar is not None:
            s = scalar
            if s.dim() == 1:
                s = s.unsqueeze(0)
            x = torch.cat([x, s], dim=-1)
        out = self.net(x).squeeze(-1)
        if out.dim() == 1 and out.size(0) == 1:
            return out.squeeze(0)
        return out

    def td_step(
        self, s_enc: EncoderOutput, reward: float,
        discount: float, s_next_enc: EncoderOutput, alpha: float,
    ) -> float:
        """One step of backward-view semi-gradient TD(λ). Returns δ.

        z ← γ_prev · λ · z + ∇V(s)
        δ = r + γ · V(s') − V(s)
        θ ← θ + α · δ · z
        """
        # Forward V(s) with gradient, then compute ∇V(s)
        v_s = self.forward(s_enc)
        for p in self.parameters():
            p.grad = None
        v_s.backward()

        with torch.no_grad():
            # z ← γ_prev · λ · z + ∇V(s)
            for name, param in self.named_parameters():
                self._traces[name].mul_(self._gamma_prev * self._td_lambda).add_(param.grad)

            # δ = r + γ · V(s') − V(s)
            delta = reward + discount * self.forward(s_next_enc).item() - v_s.item()

            # θ ← θ + α · δ · z
            for name, param in self.named_parameters():
                param.data.add_(self._traces[name], alpha=alpha * delta)

        self._gamma_prev = discount
        return delta

    def reset_traces(self) -> None:
        """Zero all eligibility traces and γ_prev. Call at episode/reset boundaries."""
        for name in self._traces:
            self._traces[name].zero_()
        self._gamma_prev = 0.0

    def get_weight_norms(self) -> dict[str, float]:
        norms: dict[str, float] = {}
        for name, param in self.named_parameters():
            if "weight" in name:
                norms[name] = param.data.norm().item()
        return norms

    def get_grad_norms(self) -> dict[str, float]:
        norms: dict[str, float] = {}
        for name, param in self.named_parameters():
            if param.grad is not None and "weight" in name:
                norms[name] = param.grad.norm().item()
        return norms


def create_networks(
    model_cfg: ModelConfig,
    env_cfg: EnvConfig,
    train_cfg: TrainConfig,
) -> list[ValueNetwork]:
    n_networks = 1 if train_cfg.learning_type == LearningType.CENTRALIZED else env_cfg.n_agents
    return [
        ValueNetwork(model_cfg, env_cfg, td_lambda=train_cfg.td_lambda)
        for _ in range(n_networks)
    ]


def create_actor_networks(
    model_cfg: ModelConfig,
    env_cfg: EnvConfig,
    train_cfg: TrainConfig,
) -> list[PolicyNetwork]:
    n_networks = 1 if train_cfg.learning_type == LearningType.CENTRALIZED else env_cfg.n_agents
    actor_lr_cfg = train_cfg.actor_lr or train_cfg.lr
    init_lr = compute_schedule_value(actor_lr_cfg, 0, train_cfg.total_steps)
    return [
        PolicyNetwork(model_cfg, env_cfg, lr=init_lr)
        for _ in range(n_networks)
    ]
