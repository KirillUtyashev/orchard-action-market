# models/torchrl_critic.py
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch import optim
from torchrl.modules import ValueOperator
from torchrl.objectives import LossModule
from torchrl.objectives.value import TD0Estimator, TDLambdaEstimator
from debug.code.network import linear_decay_then_hold_factor
from torch.optim.lr_scheduler import LambdaLR


class VNet(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.net(obs)


class TD0ValueLoss(LossModule):
    def __init__(self, value_op: ValueOperator, gamma: float):
        super().__init__()
        self.value_op = value_op
        self.est = TD0Estimator(gamma=gamma, value_network=value_op)

    def forward(self, td: TensorDict) -> TensorDict:
        if len(td.batch_size) == 1:
            td = td.unsqueeze(0)  # [B, T]

        pred_all = self.value_op(td.clone())["state_value"]  # [B, T, 1]
        with torch.no_grad():
            targ_all = self.est.value_estimate(td)           # [B, T, 1]

        # Standard: train on all timesteps in the window
        loss = F.mse_loss(pred_all, targ_all)  # mean over B,T,1 by default
        return TensorDict({"loss_value": loss}, batch_size=[])


class TDLambdaValueLoss(LossModule):
    def __init__(self, value_op: ValueOperator, gamma: float, lambda_coeff: float):
        super().__init__()
        self.value_op = value_op
        self.est = TDLambdaEstimator(gamma=gamma, lmbda=lambda_coeff, value_network=value_op)

    def forward(self, td: TensorDict) -> TensorDict:
        if len(td.batch_size) == 1:
            td = td.unsqueeze(0)  # [B, T]

        pred_all = self.value_op(td.clone())["state_value"]  # [B, T, 1]
        with torch.no_grad():
            targ_all = self.est.value_estimate(td)           # [B, T, 1]

        # Standard: train on all timesteps in the window
        loss = F.mse_loss(pred_all, targ_all)
        return TensorDict({"loss_value": loss}, batch_size=[])


class TorchRLCritic:
    def __init__(
            self,
            model: torch.nn.Module,
            alpha: float,
            gamma: float,
            traj_len: int,
            lambda_coeff: float = 1.0,
            device="cpu",
            # scheduler params
            schedule: bool = True,
            num_training_steps: int = 1_000_000,  # total updates you expect to run
            min_lr: float = 1e-6,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=alpha)

        # ---- scheduler (80% decay, 20% hold) ----
        self.alpha = float(alpha)
        self.min_lr = float(min_lr)
        self.decay_steps = int(0.8 * num_training_steps)
        self.hold_steps = int(0.2 * num_training_steps)  # informational

        self._lr_step = 0
        if schedule:
            min_factor = (self.min_lr / self.alpha) if self.alpha > 0 else 0.0
            self.scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda s: linear_decay_then_hold_factor(s, self.decay_steps, min_factor),
            )
        else:
            self.scheduler = None

        # ---- the rest of your init (unchanged) ----
        self.value_op = ValueOperator(self.model)
        self.traj_len = max(int(traj_len), 1)
        self.buf = deque(maxlen=self.traj_len)

        if self.traj_len <= 1:
            self.loss_mod = TD0ValueLoss(self.value_op, gamma=gamma)
        else:
            self.loss_mod = TDLambdaValueLoss(self.value_op, gamma=gamma, lambda_coeff=lambda_coeff)

    def _after_update(self):
        self._lr_step += 1
        if self.scheduler is not None:
            # Call AFTER optimizer.step() to avoid skipping the first LR value.
            self.scheduler.step()  # updates optimizer.param_groups[i]["lr"]

    @torch.no_grad()
    def get_value_function(self, obs):
        """
        obs: torch tensor [obs_dim] or [B, obs_dim]
        returns: torch tensor [1, 1] or [B, 1] (state_value)
        """
        obs = obs.to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # [1, obs_dim]

        td = TensorDict({"observation": obs}, batch_size=[obs.shape[0]], device=self.device)
        td = self.value_op(td)  # writes "state_value" by default [page:19]
        return td["state_value"]

    def push_transition(self, obs, next_obs, reward, done=False, terminated=False):
        """
        obs, next_obs: torch tensors [obs_dim] (already processed by your controller)
        reward: float or 0-d torch tensor
        """
        obs = obs.to(self.device)
        next_obs = next_obs.to(self.device)
        reward = torch.as_tensor(reward, device=self.device, dtype=torch.get_default_dtype())

        self.buf.append((obs, next_obs, reward, bool(done), bool(terminated)))

        # Train once we have a full window
        if len(self.buf) == self.traj_len:
            return self._update_from_buffer()
        return None

    def reset_trajectory(self):
        self.buf.clear()

    def _update_from_buffer(self):
        T = len(self.buf)

        obs = torch.stack([b[0] for b in self.buf], dim=0).unsqueeze(0)        # [1, T, obs_dim]
        next_obs = torch.stack([b[1] for b in self.buf], dim=0).unsqueeze(0)        # [1, T, obs_dim]
        reward = torch.stack([b[2] for b in self.buf], dim=0).view(1, T, 1)       # [1, T, 1]
        done = torch.tensor([b[3] for b in self.buf], device=self.device, dtype=torch.bool).view(1, T, 1)
        term = torch.tensor([b[4] for b in self.buf], device=self.device, dtype=torch.bool).view(1, T, 1)

        # TorchRL estimators expect the transition to be under "next": reward/done/terminated/obs. [page:12]
        td = TensorDict(
            {
                "observation": obs,
                "next": {
                    "observation": next_obs,
                    "reward": reward,
                    "done": done,
                    "terminated": term,
                },
            },
            batch_size=[1, T],
            device=self.device,
        )

        loss_td = self.loss_mod(td)  # returns {"loss_value": ...} (LossModule convention) [page:15]
        loss = loss_td["loss_value"]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._after_update()

        return float(loss.detach().cpu())

    @torch.no_grad()
    def explicit_lambda_return_t0(self, reward, done, terminated, v_next, gamma, lmbda):
        # reward: [1, T, 1] contains r_{t+1}; v_next: [1, T, 1] contains V(s_{t+1})
        T = reward.shape[1]
        not_end = (~(done.bool() | terminated.bool())).float()  # [1, T, 1]

        # Build n-step returns G_0^(n) for n=1..T, truncated by done/terminated.
        G = []
        disc = 1.0
        ret = 0.0
        alive = 1.0
        for n in range(1, T + 1):
            # add reward at step n (i.e., reward[:, n-1])
            ret = ret + disc * alive * reward[:, n-1, 0]
            alive = alive * not_end[:, n-1, 0]
            disc = disc * gamma
            # bootstrap from V(s_n) = v_next[:, n-1]
            G_n = ret + disc * alive * v_next[:, n-1, 0]
            G.append(G_n)  # each is [1]
        # weights: (1-l) l^{n-1} for n=1..T-1, plus tail weight l^{T-1} on n=T
        weights = [(1 - lmbda) * (lmbda ** (n - 1)) for n in range(1, T)]
        tail_w = (lmbda ** (T - 1))
        out = sum(w * G[n-1] for n, w in enumerate(weights, start=1)) + tail_w * G[T-1]
        return out  # [1]
