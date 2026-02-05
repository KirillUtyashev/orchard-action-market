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

        # Only train on the oldest state in the window (t=0)
        pred0 = pred_all[:, 0, :]  # [B, 1]
        targ0 = targ_all[:, 0, :]  # [B, 1]

        loss = F.mse_loss(pred0, targ0)
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
            targ_all = self.est.value_estimate(td)           # [B, T, 1] [web:44]

        # Only train on the oldest state in the window (t=0)
        pred0 = pred_all[:, 0, :]  # [B, 1]
        targ0 = targ_all[:, 0, :]  # [B, 1]

        loss = F.mse_loss(pred0, targ0)
        return TensorDict({"loss_value": loss}, batch_size=[])


class TorchRLCritic:
    """
    Drop-in-ish replacement for your VNetwork when exp_config.train_config.use_library=True.
    You push transitions; it buffers traj_len steps; when full it runs TD(0) or TD(lambda).
    """
    def __init__(self, model: torch.nn.Module, alpha, gamma: float,
                 traj_len: int, lambda_coeff: float = 1, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=alpha, amsgrad=True)

        self.value_op = ValueOperator(self.model)  # default: in=["observation"], out=["state_value"] [page:14]

        self.traj_len = max(int(traj_len), 1)
        self.buf = deque(maxlen=self.traj_len)

        if self.traj_len <= 1:
            self.loss_mod = TD0ValueLoss(self.value_op, gamma=gamma)
        else:
            self.loss_mod = TDLambdaValueLoss(self.value_op, gamma=gamma, lambda_coeff=lambda_coeff)

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

        obs      = torch.stack([b[0] for b in self.buf], dim=0).unsqueeze(0)        # [1, T, obs_dim]
        next_obs = torch.stack([b[1] for b in self.buf], dim=0).unsqueeze(0)        # [1, T, obs_dim]
        reward   = torch.stack([b[2] for b in self.buf], dim=0).view(1, T, 1)       # [1, T, 1]
        done     = torch.tensor([b[3] for b in self.buf], device=self.device).view(1, T, 1)
        term     = torch.tensor([b[4] for b in self.buf], device=self.device).view(1, T, 1)

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
        # if isinstance(self.loss_mod, TDLambdaValueLoss):
        #     with torch.no_grad():
        #         # Predictions on current states: V(s_t) for t=0..T-1
        #         pred_all = self.value_op(td.clone())["state_value"]  # [1, T, 1]
        #
        #         # Bootstrap sequence: v_next[t] = V(s_{t+1}) for t=0..T-1
        #         td_next = TensorDict({"observation": next_obs}, batch_size=[1, T], device=self.device)
        #         v_next = self.value_op(td_next)["state_value"]       # [1, T, 1] [web:94]
        #
        #         # TorchRL's TD(lambda) targets for each t (for comparison)
        #         targ_all = self.loss_mod.est.value_estimate(td)      # [1, T, 1] [web:44]
        #
        #         # Explicit finite-horizon (truncated) lambda-return target for t=0 only
        #         g0 = self.explicit_lambda_return_t0(
        #             reward=reward,          # [1, T, 1] rewards for transitions s_t -> s_{t+1}
        #             done=done,              # [1, T, 1]
        #             terminated=term,        # [1, T, 1]
        #             v_next=v_next,          # [1, T, 1] = V(s_{t+1})
        #             gamma=float(self.loss_mod.est.gamma),
        #             lmbda=float(self.loss_mod.est.lmbda),
        #         )  # returns shape [1]
        #
        #         pred0 = float(pred_all[0, 0, 0].cpu())
        #         targ0 = float(targ_all[0, 0, 0].cpu())
        #         g0f  = float(g0[0].cpu())
        #
        #         print(f"t=0 pred={pred0:.6f} targ(est)={targ0:.6f} targ(explicit)={g0f:.6f}")
        #

        loss_td = self.loss_mod(td)  # returns {"loss_value": ...} (LossModule convention) [page:15]
        loss = loss_td["loss_value"]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
