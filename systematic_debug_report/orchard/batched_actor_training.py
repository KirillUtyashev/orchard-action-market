"""Batched GPU training for decentralized actor networks using vmap.

Stacks all actor networks' parameters into leading-dimension tensors and
executes one batched policy-gradient update per round-robin cycle. Each actor
lane receives a fixed two-slot batch: move decision plus optional pick
decision. Missing slots are masked out.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.func import functional_call, grad_and_value, stack_module_state, vmap

from orchard.actor_critic import PolicyNetwork


class _VmapPolicyWrapper(nn.Module):
    """Pure-function wrapper for PolicyNetwork.forward_raw."""

    def __init__(self, base_net: PolicyNetwork) -> None:
        super().__init__()
        self.conv = base_net.conv
        self.flatten = base_net.flatten
        self.net = base_net.net

    def forward(self, grid: torch.Tensor, scalar: torch.Tensor | None) -> torch.Tensor:
        return PolicyNetwork.forward_raw(self, grid, scalar)


class BatchedActorTrainer:
    """Batched policy-gradient updates for decentralized actor networks."""

    _MAX_SAMPLES_PER_ACTOR = 2

    def __init__(
        self,
        networks: list[PolicyNetwork],
        device: str = "cuda",
    ) -> None:
        self.n = len(networks)
        self.networks = networks
        self.device = torch.device(device)

        wrappers = [_VmapPolicyWrapper(net) for net in networks]
        params, buffers = stack_module_state(wrappers)

        self._base = copy.deepcopy(wrappers[0]).to(self.device)
        self._params = {k: v.to(self.device).detach() for k, v in params.items()}
        self._buffers = {k: v.to(self.device).detach() for k, v in buffers.items()}

        def _loss_fn(
            params,
            buffers,
            grid_batch,
            scalar_batch,
            legal_mask_batch,
            actions_batch,
            advantages_batch,
            sample_mask_batch,
        ):
            logits = functional_call(
                self._base,
                (params, buffers),
                (grid_batch, scalar_batch),
            )
            masked_logits = self._masked_logits(logits, legal_mask_batch)
            log_probs = F.log_softmax(masked_logits, dim=1)
            probs = log_probs.exp()
            action_log_probs = log_probs.gather(1, actions_batch.unsqueeze(1)).squeeze(1)
            entropy = -(probs * log_probs).sum(dim=1)

            weights = sample_mask_batch.to(dtype=logits.dtype)
            denom = weights.sum().clamp_min(1.0)
            loss = (-(advantages_batch * action_log_probs) * weights).sum() / denom
            advantage_mean = (advantages_batch * weights).sum() / denom
            entropy_mean = (entropy * weights).sum() / denom
            sample_count = weights.sum()
            return loss, (advantage_mean, entropy_mean, sample_count)

        self._vmap_grad_and_value = vmap(grad_and_value(_loss_fn, has_aux=True))

    @staticmethod
    def _masked_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if legal_mask.shape != logits.shape:
            raise ValueError(
                f"Legal mask shape {tuple(legal_mask.shape)} does not match logits {tuple(logits.shape)}."
            )
        return logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)

    def _clear_network_batches(self) -> None:
        for net in self.networks:
            net.batch_states = []
            net.batch_actions = []
            net.batch_advantages = []
            net.batch_legal_masks = []

    def _pack_batches_from_networks(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        first_state = None
        first_legal_mask = None
        for net in self.networks:
            if net.batch_states:
                first_state = net.batch_states[0]
                first_legal_mask = net.batch_legal_masks[0]
                break
        if first_state is None:
            return None
        if first_state.grid is None:
            raise NotImplementedError("Batched actor training currently requires grid-based encoder outputs.")

        grid_shape = tuple(first_state.grid.shape)
        scalar_shape = tuple(first_state.scalar.shape) if first_state.scalar is not None else None
        assert first_legal_mask is not None
        action_dim = len(first_legal_mask)

        grids = torch.zeros(
            (self.n, self._MAX_SAMPLES_PER_ACTOR, *grid_shape),
            dtype=first_state.grid.dtype,
            device=self.device,
        )
        scalars = None
        if scalar_shape is not None:
            scalars = torch.zeros(
                (self.n, self._MAX_SAMPLES_PER_ACTOR, *scalar_shape),
                dtype=first_state.scalar.dtype,
                device=self.device,
            )
        legal_masks = torch.zeros(
            (self.n, self._MAX_SAMPLES_PER_ACTOR, action_dim),
            dtype=torch.bool,
            device=self.device,
        )
        legal_masks[..., 0] = True
        actions = torch.zeros(
            (self.n, self._MAX_SAMPLES_PER_ACTOR),
            dtype=torch.long,
            device=self.device,
        )
        advantages = torch.zeros(
            (self.n, self._MAX_SAMPLES_PER_ACTOR),
            dtype=torch.float32,
            device=self.device,
        )
        sample_mask = torch.zeros(
            (self.n, self._MAX_SAMPLES_PER_ACTOR),
            dtype=torch.float32,
            device=self.device,
        )

        for actor_id, net in enumerate(self.networks):
            n_samples = len(net.batch_states)
            if n_samples > self._MAX_SAMPLES_PER_ACTOR:
                raise ValueError(
                    f"Actor {actor_id} accumulated {n_samples} samples; expected at most "
                    f"{self._MAX_SAMPLES_PER_ACTOR} per round-robin cycle."
                )
            for slot in range(n_samples):
                state = net.batch_states[slot]
                if state.grid is None:
                    raise NotImplementedError(
                        "Batched actor training currently requires grid-based encoder outputs."
                    )
                grids[actor_id, slot] = state.grid.to(self.device)
                if scalars is not None and state.scalar is not None:
                    scalars[actor_id, slot] = state.scalar.to(self.device)
                legal_masks[actor_id, slot] = torch.as_tensor(
                    net.batch_legal_masks[slot],
                    dtype=torch.bool,
                    device=self.device,
                )
                actions[actor_id, slot] = int(net.batch_actions[slot])
                advantages[actor_id, slot] = float(net.batch_advantages[slot])
                sample_mask[actor_id, slot] = 1.0

        if scalars is None:
            raise NotImplementedError("Batched actor training currently requires scalar encoder features.")
        return grids, scalars, legal_masks, actions, advantages, sample_mask

    def train_batch_batched(self, alpha: float) -> dict[str, float] | None:
        packed = self._pack_batches_from_networks()
        if packed is None:
            return None
        grids, scalars, legal_masks, actions, advantages, sample_mask = packed

        grads, (losses, (advantage_means, entropy_means, sample_counts)) = self._vmap_grad_and_value(
            self._params,
            self._buffers,
            grids,
            scalars,
            legal_masks,
            actions,
            advantages,
            sample_mask,
        )

        with torch.no_grad():
            for name in self._params:
                self._params[name] = (self._params[name] - alpha * grads[name]).detach()

        self.sync_to_networks()
        self._clear_network_batches()

        total_samples = float(sample_counts.sum().item())
        if total_samples <= 0.0:
            return None
        sample_weights = sample_counts / sample_counts.sum().clamp_min(1.0)
        return {
            "loss": float((losses * sample_weights).sum().item()),
            "advantage_mean": float((advantage_means * sample_weights).sum().item()),
            "entropy_mean": float((entropy_means * sample_weights).sum().item()),
            "sample_count": total_samples,
        }

    def sync_to_networks(self) -> None:
        for i, net in enumerate(self.networks):
            device = next(net.parameters()).device
            net_params = {k: v[i].detach().to(device) for k, v in self._params.items()}
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
        wrappers = [_VmapPolicyWrapper(net) for net in self.networks]
        params, buffers = stack_module_state(wrappers)
        self._params = {k: v.to(self.device).detach() for k, v in params.items()}
        self._buffers = {k: v.to(self.device).detach() for k, v in buffers.items()}
