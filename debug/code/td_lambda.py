import numpy as np
import torch
from collections import deque

from utils import ten
from debug.code.enums import DEVICE
from debug.code.network import NetworkWrapper

torch.set_default_dtype(torch.float64)


class TDLambda(NetworkWrapper):
    def __init__(
            self,
            input_dim, output_dim, alpha, discount,
            hidden_dim=128, num_layers=4,
            num_training_steps=10000, schedule=False,
            lambda_coeff: float = 0.0,
            batch_size: int = 1000,
            train_every: int = 1,
    ):
        super().__init__(
            input_dim, output_dim, alpha, discount,
            hidden_dim, num_layers, schedule, num_training_steps
        )

        self.lambda_coeff = float(lambda_coeff)
        self.batch_size = int(batch_size)
        self.train_every = int(train_every)

        # Sliding window buffers (contiguous transitions only!)
        self.batch_states = deque(maxlen=self.batch_size)
        self.batch_new_states = deque(maxlen=self.batch_size)
        self.batch_rewards = deque(maxlen=self.batch_size)
        self.batch_discounts = deque(maxlen=self.batch_size)  # NEW

        self._num_appended = 0

    def get_value_function(self, x):
        res = ten(x, DEVICE)
        res = res.view(1, -1)
        with torch.no_grad():
            val = self.model(res).cpu().numpy().item()
        return val

    def get_input_dim(self):
        return self._input_dim

    def ready_to_train(self) -> bool:
        return len(self.batch_states) == self.batch_size

    def add_experience(self, state, new_state, reward, discount_factor: float):
        self.batch_states.append(state)
        self.batch_new_states.append(new_state)
        self.batch_rewards.append(reward)
        self.batch_discounts.append(float(discount_factor))  # NEW
        self._num_appended += 1

    def train(self):
        if not self.ready_to_train():
            return None

        if self.train_every > 1 and (self._num_appended % self.train_every) != 0:
            return None

        states_b = list(self.batch_states)
        new_states_b = list(self.batch_new_states)
        rewards_b = list(self.batch_rewards)
        discounts_b = list(self.batch_discounts)  # NEW

        return self._train_td_lambda_forward(states_b, new_states_b, rewards_b, discounts_b)

    def _train_td_lambda_forward(self, states_b, new_states_b, rewards_b, discounts_b):
        """
        Forward-view TD(lambda) on one contiguous window of transitions.

        Assumes new_states_b[t] is the true next state of states_b[t],
        so the window corresponds to s0->s1->...->sB.
        """
        B = len(states_b)
        assert B == self.batch_size
        assert len(new_states_b) == B
        assert len(rewards_b) == B
        assert len(discounts_b) == B  # NEW

        lam = float(self.lambda_coeff)

        states = ten(np.stack(states_b, axis=0).squeeze(), DEVICE).view(B, -1)
        rewards = ten(np.asarray(rewards_b, dtype=np.float64), DEVICE).view(B)
        discounts = ten(np.asarray(discounts_b, dtype=np.float64), DEVICE).view(B)  # NEW

        # Build state sequence s0..sB (length B+1)
        seq_states_np = [states_b[0]] + list(new_states_b)  # [s0, s1, ..., sB]
        seq_states = ten(np.stack(seq_states_np, axis=0).squeeze(), DEVICE).view(B + 1, -1)

        approx = self.model(states).squeeze(1)  # [B]

        with torch.no_grad():
            v_seq = self.model(seq_states).squeeze(1)  # [B+1]
            targets = torch.empty(B, device=DEVICE, dtype=v_seq.dtype)

            g_next = v_seq[-1]  # start from V(s_B)
            for t in range(B - 1, -1, -1):
                d_t = discounts[t].item()
                g_next = rewards[t] + d_t * ((1.0 - lam) * v_seq[t + 1] + lam * g_next)
                targets[t] = g_next

        criterion = torch.nn.MSELoss(reduction="mean")
        self.optimizer.zero_grad()
        loss = criterion(approx, targets)
        loss.backward()
        self.optimizer.step()
        self._after_update()

        return float(loss.item())
