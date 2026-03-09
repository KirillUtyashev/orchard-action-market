# models/torchrl_critic.py
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from debug.code.core.enums import DEVICE


class EligibilityCritic:
    """
    Online semi-gradient TD(lambda) with eligibility traces (backward view).
    No trajectory window needed; updates every transition.
    """
    def __init__(self, model: nn.Module, alpha, gamma: float,
                 lambda_coeff: float = 0.9, num_training_steps=10000):
        self.device = DEVICE
        self.model = model.to(self.device)
        self.gamma = float(gamma)
        self.lmbda = float(lambda_coeff)

        self.optimizer = optim.SGD(self.model.parameters(), lr=alpha)

        self._lr_step = 0
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda s: linear_decay_factor(s, num_training_steps),
        )  # lr = base_lr * lr_lambda(step)

        # Parameter-space eligibility traces: one tensor per parameter
        self._init_traces()

    def _after_update(self):
        self._lr_step += 1
        self.scheduler.step()  # updates optimizer.param_groups[i]["lr"] [web:231]

    def _init_traces(self):
        self.traces = []
        for p in self.model.parameters():
            self.traces.append(torch.zeros_like(p, device=self.device))

    def reset_trajectory(self):
        # Reset traces at episode boundary
        for e in self.traces:
            e.zero_()

    @torch.no_grad()
    def get_value_function(self, obs):
        obs = obs.to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.model(obs)  # [B, 1]

    def push_transition(self, obs, next_obs, reward, done=False, terminated=False):
        obs = obs.to(self.device)
        next_obs = next_obs.to(self.device)
        r = torch.as_tensor(reward, device=self.device, dtype=torch.get_default_dtype()).view(1, 1)

        end = bool(done) or bool(terminated)
        not_end = 0.0 if end else 1.0

        # ---- 1) Compute TD error (semi-gradient: do NOT backprop through bootstrap target) ----
        v = self.model(obs.view(1, -1))                        # requires grad
        with torch.no_grad():
            v_next = self.model(next_obs.view(1, -1))          # stop-grad target
            target = r + self.gamma * not_end * v_next
        delta = (target - v).detach()                          # scalar TD error (no grad)

        # ---- 2) Get grad of V(s_t) w.r.t params ----
        self.optimizer.zero_grad(set_to_none=True)
        v.backward()  # fills p.grad with ∇θ V(s_t)

        # ---- 3) Update eligibility traces: e <- gamma*lambda*e + grad ----
        with torch.no_grad():
            gl = self.gamma * self.lmbda
            for p, e in zip(self.model.parameters(), self.traces):
                if p.grad is None:
                    continue
                e.mul_(gl).add_(p.grad)  # accumulating trace in parameter space

        # ---- 4) Apply parameter update: θ <- θ + α * delta * e ----
        # Optimizers do θ <- θ - lr * grad, so set grad = -(delta * e).
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            d = float(delta.item())
            for p, e in zip(self.model.parameters(), self.traces):
                if p.requires_grad:
                    p.grad = (-d) * e

        self.optimizer.step()
        self._after_update()

        # ---- 5) Reset traces at episode end ----
        if end:
            self.reset_trajectory()

        # Optional: return squared TD error as a scalar loss-like diagnostic
        return float((delta * delta).item())


def linear_decay_factor(step: int, total_steps: int):
    return 1
    # step = min(max(step, 0), total_steps)
    # return 1.0 - (step / max(1, total_steps))  # scales base lr [web:231]
