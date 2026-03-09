import numpy as np
import torch

from debug.code.encoders import BaseEncoder, EncoderOutput
from debug.code.helpers import ten
from debug.code.enums import DEVICE
from debug.code.network import NetworkWrapper

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
    ):
        super().__init__(
            encoder, output_dim, alpha, discount,
            mlp_dims=mlp_dims,
            schedule=schedule,
            decay_steps=num_training_steps,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
        )

        self.lam = float(lam)

        self.batch_rewards = []
        self.batch_discounts = []
        self.theoretical_vals = []

        # eligibility traces over parameters (initialized lazily or via reset_traces())
        self._traces = None
        self.reset_traces()

    def reset_traces(self):
        """Call at episode start (and when done=True), like Sutton & Barto Fig. 9.1."""
        self._traces = {}
        for p in self.model.parameters():
            if p.requires_grad:
                self._traces[p] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def _apply_td_lambda_update(self, delta: torch.Tensor):
        """w <- w + alpha * delta * e  (expects e already updated)."""
        a = float(self.alpha)
        d = float(delta)
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            e = self._traces.get(p, None)
            if e is None:
                continue
            p.add_(a * d * e)

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

        return total_loss / len(batch)

    def train(self):
        if self.lam == -1:
            return self.train_td0()
        else:
            return self.train_lambda()

    def add_experience(self, state, new_state, reward, discount_factor, theoretical_val=None):
        self.batch_states.append(state)
        self.batch_new_states.append(new_state)
        self.batch_rewards.append(reward)
        self.batch_discounts.append(discount_factor)  # Store per-transition discount
        if theoretical_val:
            self.theoretical_vals.append(theoretical_val)

    def train_supervised(self):
        # 1) Batch states -> tensor [B, obs_dim]
        states_np = np.stack(self.batch_states, axis=0).squeeze()
        states = ten(states_np, DEVICE)
        states = states.view(states.size(0), -1)

        # 2) Forward pass: approx V(s)
        approx = self.model(states)          # [B, 1] (assumed)
        approx = approx.squeeze(1)           # [B]

        # 3) Supervised targets: theoretical V*(s) (or your analytic V(s))
        #    Compute per-state, ignore next states entirely.
        with torch.no_grad():
            y_np = np.array(
                self.theoretical_vals
            )
            y = ten(y_np, DEVICE)            # [B]

        # 4) Loss + update
        criterion = torch.nn.MSELoss(reduction="mean")  # mean squared error
        self.optimizer.zero_grad()
        loss = criterion(approx, y)
        loss.backward()
        self.optimizer.step()
        self._after_update()

        # 5) Clear batch buffers
        self.batch_states = []
        self.batch_new_states = []   # optional: keep if other code expects it
        self.batch_rewards = []      # optional: keep if other code expects it
        self.theoretical_vals = []

        return loss.item()

    def train_reward_supervised(self):
        states = ten(np.stack(self.batch_states, 0), DEVICE)  # (B, 5, H, W)
        y = ten(np.array(self.batch_rewards), DEVICE).view(-1)

        pred = self.model(states).squeeze(-1)  # (B,)

        loss = torch.nn.functional.mse_loss(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.batch_states = []
        self.batch_rewards = []
        return loss.item()
