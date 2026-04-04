import numpy as np
import torch
import torch.nn.functional as F

from debug.code.core.enums import DEVICE
from debug.code.nn.encoders import BaseEncoder, EncoderOutput, stack_encoder_outputs
from debug.code.nn.network import NetworkWrapper

torch.set_default_dtype(torch.float64)


class PolicyNetwork(NetworkWrapper):
    def __init__(
        self,
        encoder: BaseEncoder,
        output_dim: int,
        alpha: float,
        discount: float,
        mlp_dims: tuple[int, ...] = (128, 128),
        num_training_steps: int = 10_000,
        schedule: bool = False,
        conv_channels: list[int] | None = None,
        kernel_size: int = 3,
        use_mlp: bool = True,
    ):
        super().__init__(
            encoder,
            output_dim,
            alpha,
            discount,
            mlp_dims=mlp_dims,
            schedule=schedule,
            decay_steps=num_training_steps,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            use_mlp=use_mlp,
        )
        self.batch_actions: list[int] = []
        self.batch_advantages: list[float] = []
        self.batch_legal_masks: list[np.ndarray] = []

    @staticmethod
    def _mask_tensor(legal_mask, batch_size: int | None = None) -> torch.Tensor:
        mask = torch.as_tensor(legal_mask, device=DEVICE, dtype=torch.bool)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if batch_size is not None and mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1)
        return mask

    def _masked_logits(self, logits: torch.Tensor, legal_mask) -> torch.Tensor:
        mask = self._mask_tensor(legal_mask, batch_size=int(logits.shape[0]))
        if mask.shape != logits.shape:
            raise ValueError(f"Legal mask shape {tuple(mask.shape)} does not match logits {tuple(logits.shape)}.")
        masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        if (~mask).all(dim=1).any():
            raise ValueError("Each action mask must allow at least one action.")
        return masked

    def get_action_probabilities(self, enc: EncoderOutput, legal_mask) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(enc)
            masked_logits = self._masked_logits(logits, legal_mask)
            probs = F.softmax(masked_logits, dim=1)
        self.model.train()
        return probs.squeeze(0).detach().cpu().numpy()

    def sample_action(self, enc: EncoderOutput, legal_mask) -> tuple[int, np.ndarray]:
        probs = self.get_action_probabilities(enc, legal_mask)
        action_idx = int(np.random.choice(len(probs), p=probs))
        return action_idx, probs

    def add_experience(self, state: EncoderOutput, legal_mask, action: int, advantage: float) -> None:
        self.batch_states.append(state)
        self.batch_actions.append(int(action))
        self.batch_advantages.append(float(advantage))
        self.batch_legal_masks.append(np.asarray(legal_mask, dtype=bool))

    def train(self):
        if not self.batch_states:
            return None

        states = stack_encoder_outputs(self.batch_states)
        actions = torch.as_tensor(self.batch_actions, device=DEVICE, dtype=torch.long)
        advantages = torch.as_tensor(self.batch_advantages, device=DEVICE, dtype=torch.float32)
        legal_masks = np.stack(self.batch_legal_masks, axis=0)

        logits = self.model(states)
        masked_logits = self._masked_logits(logits, legal_masks)
        log_probs = F.log_softmax(masked_logits, dim=1)
        probs = log_probs.exp()
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(probs * log_probs).sum(dim=1)
        loss = -(advantages * action_log_probs).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._after_update()

        metrics = {
            "loss": float(loss.item()),
            "advantage_mean": float(advantages.mean().item()),
            "entropy_mean": float(entropy.mean().item()),
        }

        self.batch_states = []
        self.batch_new_states = []
        self.batch_actions = []
        self.batch_advantages = []
        self.batch_legal_masks = []
        return metrics
