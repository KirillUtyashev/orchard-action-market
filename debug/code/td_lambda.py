import numpy as np
import torch
from collections import deque

from utils import ten
from config import DEVICE
from debug.code.network import NetworkWrapper

torch.set_default_dtype(torch.float64)


class TDLambda(NetworkWrapper):
    def __init__(
            self,
            input_dim, output_dim, alpha, discount,
            hidden_dim=128, num_layers=4,
            num_training_steps=10000, schedule=False,
            lambda_coeff: float = 0.0,      # 0 -> TD0, 1 -> MC over window horizon
            batch_size: int = 1000,
            train_every: int = 1,           # train every k appended transitions once warm
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

        self._num_appended = 0

    def ready_to_train(self) -> bool:
        return len(self.batch_states) == self.batch_size

    def add_experience(self, state, new_state, reward):
        self.batch_states.append(state)
        self.batch_new_states.append(new_state)
        self.batch_rewards.append(reward)
        self._num_appended += 1

    def train(self):
        # Warmup: don’t train until the sliding window is full
        if not self.ready_to_train():
            return None

        # Optional: train less frequently
        if self.train_every > 1 and (self._num_appended % self.train_every) != 0:
            return None

        states_b = list(self.batch_states)
        new_states_b = list(self.batch_new_states)
        rewards_b = list(self.batch_rewards)

        return self._train_td_lambda_forward(states_b, new_states_b, rewards_b)

    def _train_td_lambda_forward(self, states_b, new_states_b, rewards_b):
        """
        Forward-view TD(lambda) on one contiguous window of transitions.

        IMPORTANT: assumes new_states_b[t] is the true next state of states_b[t],
        so the window corresponds to s0->s1->...->sB. [file:1]
        """
        B = len(states_b)
        assert B == self.batch_size
        assert len(new_states_b) == B
        assert len(rewards_b) == B

        lam = float(self.lambda_coeff)
        gamma = float(self.discount)

        # Tensorize current states and rewards
        states = ten(np.stack(states_b, axis=0).squeeze(), DEVICE).view(B, -1)
        rewards = ten(np.asarray(rewards_b, dtype=np.float64), DEVICE).view(B)

        # Build state sequence s0..sB (length B+1) so we can get V(s_{t+1}) and V(s_B)
        seq_states_np = [states_b[0]] + list(new_states_b)  # [s0, s1, ..., sB]
        seq_states = ten(np.stack(seq_states_np, axis=0).squeeze(), DEVICE).view(B + 1, -1)

        # Current predictions V(s_t)
        approx = self.model(states).squeeze(1)  # [B]

        # Targets: truncated lambda-returns via backward recursion
        with torch.no_grad():
            v_seq = self.model(seq_states).squeeze(1)  # [B+1] = [V(s0), ..., V(sB)]
            targets = torch.empty(B, device=DEVICE, dtype=v_seq.dtype)

            g_next = v_seq[-1]  # starts from V(s_B)
            for t in range(B - 1, -1, -1):
                # G_t^λ = r_t + γ[(1-λ)V(s_{t+1}) + λ G_{t+1}^λ]  [file:1]
                g_next = rewards[t] + gamma * ((1.0 - lam) * v_seq[t + 1] + lam * g_next)
                targets[t] = g_next

        criterion = torch.nn.MSELoss(reduction="mean")
        self.optimizer.zero_grad()
        loss = criterion(approx, targets)
        loss.backward()
        self.optimizer.step()
        self._after_update()

        return float(loss.item())
