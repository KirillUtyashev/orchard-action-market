"""Shared masked actor for the learned-actor path.

One network serves every agent. The encoder input already carries the acting
agent's one-hot id and the pick-phase flag, so a single masked head covers both
phases: the 5 move logits are legal in the move phase, and STAY plus the
eligible pick logits are legal in the pick phase. A single head is used because
the existing PolicyNetwork already has one and the action indices are shared
between phases, so no extra routing code is needed.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from orchard.actor_critic.policy_network import PolicyNetwork, _stack_encoder_outputs
from orchard.datatypes import EncoderOutput

_EPS = 1e-8


class SharedActor(PolicyNetwork):
    """PolicyNetwork with a batched policy-gradient update and entropy bonus."""

    def update(
        self,
        states: list[EncoderOutput],
        legal_masks: np.ndarray,
        actions: list[int],
        advantages: torch.Tensor,
        entropy_coef: float,
        normalize_advantage: bool,
    ) -> dict[str, float]:
        """One SGD step on a batch. `advantages` must already be detached."""
        if advantages.requires_grad:
            raise ValueError("Actor advantages must be detached from the critic graph.")

        batch = _stack_encoder_outputs(states)
        logits = self.forward(batch)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        device = logits.device
        advantages = advantages.to(device=device, dtype=torch.float32)
        if normalize_advantage:
            # RMS scaling keeps sign and mean information and is well defined
            # for a batch of one, unlike a z-score.
            advantages = advantages / (advantages.pow(2).mean().sqrt() + _EPS)

        action_idx = torch.as_tensor(actions, dtype=torch.long, device=device)
        log_probs = F.log_softmax(self._masked_logits(logits, legal_masks), dim=1)
        probs = log_probs.exp()
        chosen = log_probs.gather(1, action_idx.unsqueeze(1)).squeeze(1)
        # Masked entries have prob 0; clamp avoids 0 * -inf-like terms.
        entropy = -(probs * log_probs.clamp(min=-1e4)).sum(dim=1)
        loss = -(chosen * advantages).mean() - float(entropy_coef) * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "advantage_mean": float(advantages.mean().item()),
            "entropy_mean": float(entropy.mean().item()),
            "sample_count": float(len(actions)),
        }


__all__ = ["SharedActor"]
