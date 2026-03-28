"""ValueNetwork: owns its optimizer, supports train_step for TD updates."""

from __future__ import annotations

import torch
import torch.nn as nn

import orchard.encoding as encoding
from orchard.enums import Activation, ModelType, TrainMethod, WeightInit
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
        td_lambda: float = 0.0,
        train_method: TrainMethod = TrainMethod.NSTEP,
    ) -> None:
        super().__init__()
        self._lr_schedule = lr_schedule
        self._total_steps = total_steps
        self._step_count: int = 0
        self._env_step: int = 0

        if model_cfg.model_type == ModelType.MLP:
            input_dim = encoding.get_scalar_dim()
            self.net = self._build_mlp(input_dim, model_cfg.mlp_dims, model_cfg.activation)
        elif model_cfg.model_type == ModelType.CNN:
            channels = encoding.get_grid_channels()
            scalar_extra = encoding.get_scalar_dim()
            self.conv, conv_flat_dim = self._build_conv(
                channels, model_cfg.conv_specs, encoding.get_grid_height(), encoding.get_grid_width(), model_cfg.activation
            )
            self.flatten = nn.Flatten()
            self.net = self._build_mlp(conv_flat_dim + scalar_extra, model_cfg.mlp_dims, model_cfg.activation)
        else:
            raise ValueError(f"Unknown model type: {model_cfg.model_type}")
        
        if model_cfg.weight_init == WeightInit.ZERO_BIAS:
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        # NOTE: for n-step adam converges a lot faster, so be careful here.
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr_schedule.start) #self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule.start)
        
        self._nstep = nstep
        self._nstep_buffer: list[NStepTransition] = []
        self._nstep_latest_next: EncoderOutput | None = None
        self._nstep = nstep
        self._nstep_buffer: list[NStepTransition] = []
        self._nstep_latest_next: EncoderOutput | None = None

        # --- TD(λ) state ---
        self._td_lambda = td_lambda
        self._train_method = train_method
        self._gamma_prev: float = 0.0
        self._traces: dict[str, torch.Tensor] = {}
        if train_method == TrainMethod.BACKWARD_VIEW:
            for name, param in self.named_parameters():
                self._traces[name] = torch.zeros_like(param.data)
        

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...], activation: Activation = Activation.RELU) -> nn.Sequential:
        layers: list[nn.Module] = []
        d = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(d, hd))
            if activation == Activation.LEAKY_RELU:
                layers.append(nn.LeakyReLU())
            elif activation == Activation.RELU:
                layers.append(nn.ReLU())
            # else no activation for None
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
        """Build conv layers. Returns (conv_sequential, flat_output_dim)."""
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
        Returns:
            scalar value(s)
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

    def train_step(self, s_enc, reward, discount, s_next_enc, env_step: int = 0) -> float:
        if self._train_method == TrainMethod.BACKWARD_VIEW:
            return self.td_lambda_step(s_enc, reward, discount, s_next_enc, env_step)
        return self._nstep_train_step(s_enc, reward, discount, s_next_enc, env_step)
    
    def _nstep_train_step(self, s_enc, reward, discount, s_next_enc, env_step: int = 0) -> float:
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

    def td_lambda_step(self, s_enc, reward, discount, s_next_enc, env_step: int = 0) -> float:
        """One step of backward-view semi-gradient TD(λ) with variable discount.
        
        Arguments:
            s_enc: EncoderOutput for state s_t^a (actor's perspective)
            reward: r_{t+2} observed after taking action at s_t^a
            discount: γ_{t+2} observed after taking action at s_t^a (can be variable per step)
            s_next_enc: EncoderOutput for next state s_{t+1}^a (actor's perspective)
            env_step: current environment step count (for learning rate scheduling)
        """
        self._env_step = env_step

        # 1. Forward pass on s_t^a (with gradient tracking)
        v_s = self.forward(s_enc)

        # 2. Backward pass: get ∇_θ V̂(s_t^a) into param.grad
        self.optimizer.zero_grad()
        v_s.backward()

        # 3. Update eligibility traces: z = γ_prev * λ * z + grad
        with torch.no_grad():
            for name, param in self.named_parameters():
                assert isinstance(param.grad, torch.Tensor)
                self._traces[name].mul_(self._gamma_prev * self._td_lambda).add_(param.grad)

        # 4. Forward pass on s_{t+1}^a (no gradient)
        with torch.no_grad():
            v_s_next = self.forward(s_next_enc).item()

        # 5. TD error: δ = r + γ_{t+2} * V̂(s_{t+1}^a) - V̂(s_t^a)
        v_s_val = v_s.item()
        delta = reward + discount * v_s_next - v_s_val

        # 6. Get learning rate from schedule
        alpha = compute_schedule_value(self._lr_schedule, env_step, self._total_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = alpha
        # 7. Manual SGD: θ += α * δ * z
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.data.add_(self._traces[name], alpha=alpha * delta)

        # 8. Store current discount as γ_prev for next step's trace decay
        self._gamma_prev = discount

        self._step_count += 1
        return delta ** 2

    def reset_traces(self) -> None:
        """Zero all eligibility traces and γ_prev. Call at episode/reset boundaries."""
        for name in self._traces:
            self._traces[name].zero_()
        self._gamma_prev = 0.0
        
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
    td_lambda: float = 0.0,
    train_method: TrainMethod = TrainMethod.NSTEP,
    n_networks: int | None = None, # None means decentralized. Centralized uses 1 network.
) -> list[ValueNetwork]:
    if n_networks is None:
        n_networks = env_cfg.n_agents
    return [
        ValueNetwork(model_cfg, env_cfg, lr_schedule, total_steps, nstep, td_lambda=td_lambda, train_method=train_method)
        for _ in range(n_networks)
    ]
