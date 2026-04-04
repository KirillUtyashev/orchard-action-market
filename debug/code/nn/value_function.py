import numpy as np
import torch

from debug.code.nn.encoders import BaseEncoder, EncoderOutput
from debug.code.core.enums import DEVICE
from debug.code.nn.network import NetworkWrapper

torch.set_default_dtype(torch.float64)


class VNetwork(NetworkWrapper):
    def __init__(
            self,
            encoder: BaseEncoder,
            output_dim: int,
            alpha: float,
            discount: float,
            lam: float = 0.9,  # <-- TD(lambda)
            mlp_dims: tuple[int, ...] = (128, 128),
            num_training_steps: int = 10_000,
            schedule: bool = False,
            conv_channels: list[int] = None,
            kernel_size: int = 3,
            reward_learning: bool = False,
            supervised: bool = False,
            use_mlp: bool = True,
    ):
        super().__init__(
            encoder, output_dim, alpha, discount,
            mlp_dims=mlp_dims,
            use_mlp=use_mlp,
            schedule=schedule,
            decay_steps=num_training_steps,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
        )

        self.reward_learning = bool(reward_learning)
        self.supervised = bool(supervised)
        self.lam = float(lam)

        self.batch_rewards = []
        self.batch_discounts = []
        self.batch_true_values = []

        # eligibility traces over parameters (initialized lazily or via reset_traces())
        self._traces = None
        self.reset_traces()

    def reset_traces(self):
        """Call at episode start (and when done=True), like Sutton & Barto Fig. 9.1."""
        self._traces = {}
        for p in self.model.parameters():
            if p.requires_grad:
                self._traces[p] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def get_value_function(self, enc: EncoderOutput) -> float:
        self.model.eval()
        with torch.no_grad():
            out = self.model(enc)
        self.model.train()
        return float(out.squeeze())

    def td_lambda_update(self, state, next_state, reward, discount) -> float:
        # 1. Compute V(s) with grad tracking
        v = self.model(state).squeeze()

        # 2. Compute V(s') without grad tracking
        with torch.no_grad():
            v_next = self.model(next_state).squeeze()
            delta = reward + discount * v_next - v.detach()

        # 3. Get grad_w V(s)
        self.optimizer.zero_grad(set_to_none=True)
        v.backward()

        # 4. Update eligibility traces: e <- gamma*lambda*e + grad
        with torch.no_grad():
            gl = float(discount) * self.lam
            for p in self.model.parameters():
                if not p.requires_grad or p.grad is None:
                    continue
                self._traces[p].mul_(gl).add_(p.grad)

            # 5. Replace gradients with TD(lambda) pseudo-gradients
            for p in self.model.parameters():
                if not p.requires_grad:
                    continue
                p.grad = (-delta * self._traces[p]).clone()

        # 6. Let SGD apply the update
        self.optimizer.step()

        return float(delta.item())

    def train_lambda(self):
        """
        Drop-in replacement that consumes the stored batch sequentially.
        IMPORTANT: TD(lambda) is order-dependent; do NOT shuffle this batch.
        If your batch spans multiple episodes, you must also store `done` flags
        and pass them into td_lambda_update(...) to reset traces correctly.
        """
        if len(self.batch_states) == 0:
            return 0.0

        # If your existing code calls train() once per episode, traces are already fine.
        # If train() is called mid-episode, do NOT reset traces here.
        total_abs_delta = 0.0

        batch = list(zip(
            self.batch_states,
            self.batch_new_states,
            self.batch_rewards,
            self.batch_discounts,
        ))

        for state, next_state, reward, discount in batch:
            # done=False here because your stored experience doesn't include terminals.
            # If you have terminals, add self.batch_dones and pass done accordingly.
            delta = self.td_lambda_update(state, next_state, reward, discount)
            total_abs_delta += abs(delta)

        self._after_update()

        self.batch_states = []
        self.batch_new_states = []
        self.batch_rewards = []
        self.batch_discounts = []
        self.batch_true_values = []

        return total_abs_delta / len(batch)

    def train_td0(self):
        if len(self.batch_states) == 0:
            return 0.0

        total_loss = 0.0
        batch = list(zip(self.batch_states, self.batch_new_states, self.batch_rewards, self.batch_discounts))

        for state, next_state, reward, discount in batch:
            self.optimizer.zero_grad()

            with torch.no_grad():
                target = reward + discount * self.model(next_state).squeeze()

            pred = self.model(state).squeeze()
            loss = (pred - target) ** 2
            total_loss += float(loss.item())

            loss.backward()
            self.optimizer.step()

        self._after_update()

        self.batch_states = []
        self.batch_new_states = []
        self.batch_rewards = []
        self.batch_discounts = []
        self.batch_true_values = []

        return total_loss / len(batch)

    def train(self):
        if self.reward_learning:
            return self.reward_supervised()
        if self.supervised:
            return self.train_supervised()
        if self.lam == -1:
            return self.train_td0()
        else:
            return self.train_lambda()

    def reward_supervised(self):
        if len(self.batch_states) == 0:
            return 0.0

        total_loss = 0.0
        batch = list(zip(self.batch_states, self.batch_rewards))

        for state, reward in batch:
            self.optimizer.zero_grad()
            pred = self.model(state).squeeze()
            target = torch.as_tensor(float(reward), device=DEVICE, dtype=pred.dtype)
            loss = (pred - target) ** 2
            total_loss += float(loss.item())
            loss.backward()
            self.optimizer.step()

        self._after_update()

        self.batch_states = []
        self.batch_new_states = []
        self.batch_rewards = []
        self.batch_discounts = []
        self.batch_true_values = []
        return total_loss / len(batch)

    def train_reward_supervised(self):
        return self.reward_supervised()

    def add_experience(
            self,
            state,
            new_state,
            reward,
            discount_factor,
            theoretical_val=None,
            true_value=None,
    ):
        self.batch_states.append(state)
        self.batch_new_states.append(new_state)
        self.batch_rewards.append(reward)
        self.batch_discounts.append(discount_factor)  # Store per-transition discount
        if self.supervised:
            if true_value is None:
                raise ValueError("In supervised mode, add_experience requires true_value.")
            self.batch_true_values.append(float(true_value))
        elif theoretical_val is not None:
            self.batch_true_values.append(float(theoretical_val))

    def train_supervised(self):
        if len(self.batch_states) == 0:
            return 0.0
        if len(self.batch_true_values) != len(self.batch_states):
            raise RuntimeError(
                f"Supervised targets count ({len(self.batch_true_values)}) "
                f"does not match batch size ({len(self.batch_states)})."
            )

        total_loss = 0.0
        batch = list(zip(self.batch_states, self.batch_true_values))

        for state, true_value in batch:
            self.optimizer.zero_grad()
            pred = self.model(state).squeeze()
            target = torch.as_tensor(float(true_value), device=DEVICE, dtype=pred.dtype)
            loss = (pred - target) ** 2
            total_loss += float(loss.item())
            loss.backward()
            self.optimizer.step()

        self._after_update()

        self.batch_states = []
        self.batch_new_states = []
        self.batch_rewards = []
        self.batch_discounts = []
        self.batch_true_values = []
        return total_loss / len(batch)
