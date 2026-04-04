"""Batched GPU training for N decentralized networks using vmap.

Stacks all N networks' parameters and eligibility traces into (N, *shape)
tensors on GPU. Uses torch.func.vmap + grad_and_value + functional_call to
execute all N forward+backward passes in one parallel kernel launch.

The algorithm is IDENTICAL to ValueNetwork.td_step — same TD(λ)
backward view, same eligibility traces, same manual SGD. Only the execution
strategy changes: N sequential CPU calls → 1 batched GPU call.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch.func import functional_call, grad_and_value, stack_module_state, vmap

from orchard.datatypes import EncoderOutput
from orchard.model import ValueNetwork


class _VmapForwardWrapper(nn.Module):
    """Pure-function wrapper for functional_call compatibility.

    Takes (grid, scalar) as raw tensors. Delegates to ValueNetwork.forward_raw
    so the forward logic is defined in exactly one place.
    """

    def __init__(self, base_net: ValueNetwork) -> None:
        super().__init__()
        self.conv = base_net.conv
        self.flatten = base_net.flatten
        self.net = base_net.net

    def forward(self, grid: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        return ValueNetwork.forward_raw(self, grid, scalar)


class BatchedTrainer:
    """Batched GPU training for N decentralized TD(λ) networks.

    Lifecycle:
        1. __init__: stacks params + traces, moves to device.
        2. td_lambda_step_batched(): one batched training step.
        3. forward_batched(): batched no-grad inference (action selection).
        4. reset_traces(): zero traces + gamma_prev at episode boundaries.
        5. sync_to_networks(): push params back to CPU networks (for eval/ckpt).
    """

    def __init__(
        self,
        networks: list[ValueNetwork],
        td_lambda: float,
        device: str = "cuda",
    ) -> None:
        self.n = len(networks)
        self.networks = networks
        self._td_lambda = td_lambda
        self.device = torch.device(device)

        # Create wrapper modules with identical structure
        wrappers = [_VmapForwardWrapper(net) for net in networks]

        # Stack params from all N wrappers ON CPU first: {name: (N, *shape)}
        params, buffers = stack_module_state(wrappers)

        # Now create base on device (deepcopy to avoid sharing params with networks)
        import copy
        self._base = copy.deepcopy(wrappers[0]).to(self.device)

        # Move stacked params to device
        self._params = {k: v.to(self.device).detach() for k, v in params.items()}
        self._buffers = {k: v.to(self.device).detach() for k, v in buffers.items()}

        # Eligibility traces: same structure, initialized to zero
        self._traces = {k: torch.zeros_like(v) for k, v in self._params.items()}

        # Per-network gamma_prev (all start at 0.0)
        self._gamma_prev = torch.zeros(self.n, device=self.device)

        # Build the vmapped grad_and_value function (cached)
        def _f(params, buffers, grid, scalar):
            return functional_call(self._base, (params, buffers), (grid, scalar))

        self._vmap_grad_and_value = vmap(grad_and_value(_f))
        self._vmap_forward = vmap(_f)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def td_lambda_step_batched(
        self,
        grids_t: torch.Tensor,       # (N, C, H, W) — s_t encodings
        scalars_t: torch.Tensor,      # (N, S)
        rewards: torch.Tensor,        # (N,)
        discount: float,              # scalar, shared across agents
        grids_next: torch.Tensor,     # (N, C, H, W) — s_{t+1} encodings
        scalars_next: torch.Tensor,   # (N, S)
        alpha: float,
    ) -> float:
        """One batched TD(λ) backward-view step for all N networks.

        z ← γ_prev · λ · z + ∇V(s)
        δ = r + γ · V(s') − V(s)
        θ ← θ + α · δ · z

        Returns sum of squared TD errors (for loss logging).
        """
        # Move inputs to device
        grids_t = grids_t.to(self.device)
        scalars_t = scalars_t.to(self.device)
        rewards = rewards.to(self.device)
        grids_next = grids_next.to(self.device)
        scalars_next = scalars_next.to(self.device)

        # Forward + backward: all N grads and values
        grads, v_s = self._vmap_grad_and_value(
            self._params, self._buffers, grids_t, scalars_t
        )

        # z ← γ_prev · λ · z + ∇V(s)
        for name in self._params:
            gp = self._gamma_prev
            for _ in range(self._traces[name].dim() - 1):
                gp = gp.unsqueeze(-1)
            self._traces[name] = gp * self._td_lambda * self._traces[name] + grads[name]

        # δ = r + γ · V(s') − V(s)
        with torch.no_grad():
            v_next = self._vmap_forward(
                self._params, self._buffers, grids_next, scalars_next
            )
        deltas = rewards + discount * v_next - v_s.detach()

        # θ ← θ + α · δ · z
        with torch.no_grad():
            for name in self._params:
                d = deltas
                for _ in range(self._params[name].dim() - 1):
                    d = d.unsqueeze(-1)
                self._params[name] = self._params[name] + alpha * d * self._traces[name]

        self._gamma_prev.fill_(discount)
        return (deltas ** 2).sum().item()

    # ------------------------------------------------------------------
    # Inference (action selection)
    # ------------------------------------------------------------------
    def forward_batched(
        self,
        grids: torch.Tensor,    # (N, B, C, H, W) — B after-states per network
        scalars: torch.Tensor,  # (N, B, S)
    ) -> torch.Tensor:
        """Batched no-grad forward for action selection.

        Uses vmap over N networks, each processing B states via standard
        batched conv/linear ops. No Python loop over N.

        Args:
            grids:   (N, B, C, H, W) — N networks, B candidate after-states each
            scalars: (N, B, S)
        Returns:
            values:  (N, B)
        """
        grids = grids.to(self.device)
        scalars = scalars.to(self.device)

        def _f(params, buffers, grid_batch, scalar_batch):
            # Each vmap lane: grid_batch (B, C, H, W), scalar_batch (B, S)
            return functional_call(
                self._base, (params, buffers), (grid_batch, scalar_batch),
            )

        with torch.no_grad():
            return vmap(_f)(self._params, self._buffers, grids, scalars)  # (N, B)

    def forward_single_batched(
        self,
        grids: torch.Tensor,    # (N, C, H, W) — one state per network
        scalars: torch.Tensor,  # (N, S)
    ) -> torch.Tensor:
        """Batched no-grad forward, one input per network. Returns (N,)."""
        grids = grids.to(self.device)
        scalars = scalars.to(self.device)
        with torch.no_grad():
            return self._vmap_forward(self._params, self._buffers, grids, scalars)

    # ------------------------------------------------------------------
    # Trace management
    # ------------------------------------------------------------------
    def reset_traces(self) -> None:
        """Zero all eligibility traces and gamma_prev. Call at episode/reset."""
        for name in self._traces:
            self._traces[name].zero_()
        self._gamma_prev.zero_()

    # ------------------------------------------------------------------
    # Sync with individual networks
    # ------------------------------------------------------------------
    def sync_to_networks(self) -> None:
        """Push stacked GPU params back to individual CPU networks.

        Call before eval, checkpointing, or any code that reads
        individual network weights.
        """
        for i, net in enumerate(self.networks):
            # Extract this network's params from stacked tensors
            net_params = {k: v[i].detach().cpu() for k, v in self._params.items()}
            # The wrapper has same submodule structure as ValueNetwork (conv, net)
            # so param names match: conv.0.weight, conv.0.bias, net.0.weight, etc.
            conv_sd = {}
            net_sd = {}
            for k, v in net_params.items():
                if k.startswith("conv."):
                    conv_sd[k[len("conv."):]] = v
                elif k.startswith("net."):
                    net_sd[k[len("net."):]] = v
            net.conv.load_state_dict(conv_sd)
            net.net.load_state_dict(net_sd)

    def sync_from_networks(self) -> None:
        """Pull params from individual CPU networks to stacked GPU tensors.

        Call after loading a checkpoint or externally modifying network weights.
        """
        wrappers = [_VmapForwardWrapper(net) for net in self.networks]
        params, buffers = stack_module_state(wrappers)
        self._params = {k: v.to(self.device).detach() for k, v in params.items()}
        self._buffers = {k: v.to(self.device).detach() for k, v in buffers.items()}
