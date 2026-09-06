"""Policy network for orchard actor-critic experiments."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import orchard.encoding as encoding
from orchard.actor_critic.action_space import (
    action_to_policy_index,
    full_action_head_dim,
    policy_index_to_action,
)
from orchard.datatypes import EncoderOutput, EnvConfig, ModelConfig
from orchard.enums import Action, Activation, WeightInit


def _stack_encoder_outputs(outputs: list[EncoderOutput]) -> EncoderOutput:
    if not outputs:
        raise ValueError("Cannot stack an empty list of encoder outputs.")

    first = outputs[0]
    grid = None
    scalar = None
    if first.grid is not None:
        grid = torch.stack([out.grid for out in outputs], dim=0)
    if first.scalar is not None:
        scalar = torch.stack([out.scalar for out in outputs], dim=0)
    return EncoderOutput(scalar=scalar, grid=grid)


class PolicyNetwork(nn.Module):
    """Masked policy network over orchard's fixed action head."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        env_cfg: EnvConfig,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()

        self.env_cfg = env_cfg
        self.output_dim = full_action_head_dim(env_cfg)

        channels = encoding.get_grid_channels()
        scalar_extra = encoding.get_scalar_dim()
        self.conv, conv_flat_dim = self._build_conv(
            channels,
            model_cfg.conv_specs,
            encoding.get_grid_height(),
            encoding.get_grid_width(),
            model_cfg.activation,
        )
        self.flatten = nn.Flatten()
        self.net = self._build_mlp(
            conv_flat_dim + scalar_extra,
            model_cfg.mlp_dims,
            self.output_dim,
            model_cfg.activation,
        )

        if model_cfg.weight_init == WeightInit.ZERO_BIAS:
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)) and module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=float(lr))

        self.batch_states: list[EncoderOutput] = []
        self.batch_actions: list[int] = []
        self.batch_advantages: list[float] = []
        self.batch_legal_masks: list[np.ndarray] = []

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        activation: Activation = Activation.RELU,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(dim, hidden_dim))
            if activation == Activation.LEAKY_RELU:
                layers.append(nn.LeakyReLU())
            elif activation == Activation.RELU:
                layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
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
        channels = in_channels
        h, w = height, width
        for out_channels, kernel_size in conv_specs:
            padding = kernel_size // 2
            layers.append(nn.Conv2d(channels, out_channels, kernel_size=kernel_size, padding=padding))
            if activation == Activation.LEAKY_RELU:
                layers.append(nn.LeakyReLU())
            elif activation == Activation.RELU:
                layers.append(nn.ReLU())
            h = (h + 2 * padding - kernel_size) // 1 + 1
            w = (w + 2 * padding - kernel_size) // 1 + 1
            channels = out_channels

        return nn.Sequential(*layers), channels * h * w

    def forward(self, encoder_output: EncoderOutput) -> torch.Tensor:
        if encoder_output.grid is not None:
            return self.forward_raw(encoder_output.grid, encoder_output.scalar)
        if encoder_output.scalar is not None:
            logits = self.net(encoder_output.scalar)
            if logits.dim() == 2 and logits.size(0) == 1:
                return logits.squeeze(0)
            return logits
        raise ValueError("EncoderOutput has neither scalar nor grid.")

    def forward_raw(self, grid: torch.Tensor, scalar: torch.Tensor | None = None) -> torch.Tensor:
        x = grid
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.flatten(self.conv(x))
        if scalar is not None:
            s = scalar
            if s.dim() == 1:
                s = s.unsqueeze(0)
            x = torch.cat([x, s], dim=-1)
        logits = self.net(x)
        if logits.dim() == 2 and logits.size(0) == 1:
            return logits.squeeze(0)
        return logits

    @staticmethod
    def _mask_tensor(legal_mask, batch_size: int | None = None) -> torch.Tensor:
        mask = torch.as_tensor(legal_mask, dtype=torch.bool)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if batch_size is not None and mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1)
        return mask

    def _masked_logits(self, logits: torch.Tensor, legal_mask) -> torch.Tensor:
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        mask = self._mask_tensor(legal_mask, batch_size=int(logits.shape[0])).to(logits.device)
        if mask.shape != logits.shape:
            raise ValueError(f"Legal mask shape {tuple(mask.shape)} does not match logits {tuple(logits.shape)}.")
        if (~mask).all(dim=1).any():
            raise ValueError("Each action mask must allow at least one action.")
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    def get_action_probabilities_tensor(self, enc: EncoderOutput, legal_mask) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(enc)
            masked_logits = self._masked_logits(logits, legal_mask)
            probs = F.softmax(masked_logits, dim=1)
        self.train()
        return probs.squeeze(0).detach()

    def get_action_probabilities(self, enc: EncoderOutput, legal_mask) -> np.ndarray:
        return self.get_action_probabilities_tensor(enc, legal_mask).cpu().numpy()

    def sample_action(self, enc: EncoderOutput, legal_mask) -> tuple[Action, np.ndarray]:
        probs = self.get_action_probabilities(enc, legal_mask)
        action_idx = int(np.random.choice(len(probs), p=probs))
        return policy_index_to_action(action_idx), probs

    def add_experience(
        self,
        state: EncoderOutput,
        legal_mask,
        action: Action | int,
        advantage: float,
    ) -> None:
        self.batch_states.append(state)
        self.batch_actions.append(
            action_to_policy_index(action) if isinstance(action, Action) else int(action)
        )
        self.batch_advantages.append(float(advantage))
        self.batch_legal_masks.append(np.asarray(legal_mask, dtype=bool))

    def train_batch(self) -> dict[str, float] | None:
        if not self.batch_states:
            return None

        sample_count = len(self.batch_states)
        states = _stack_encoder_outputs(self.batch_states)
        legal_masks = np.stack(self.batch_legal_masks, axis=0)

        logits = self.forward(states)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        device = logits.device
        actions = torch.as_tensor(self.batch_actions, dtype=torch.long, device=device)
        advantages = torch.as_tensor(self.batch_advantages, dtype=torch.float32, device=device)
        masked_logits = self._masked_logits(logits, legal_masks)
        log_probs = F.log_softmax(masked_logits, dim=1)
        probs = log_probs.exp()
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(probs * log_probs).sum(dim=1)
        loss = -(advantages * action_log_probs).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics = {
            "loss": float(loss.item()),
            "advantage_mean": float(advantages.mean().item()),
            "entropy_mean": float(entropy.mean().item()),
            "sample_count": float(sample_count),
        }

        self.batch_states = []
        self.batch_actions = []
        self.batch_advantages = []
        self.batch_legal_masks = []
        return metrics

    def get_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

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


__all__ = ["PolicyNetwork", "_stack_encoder_outputs"]
