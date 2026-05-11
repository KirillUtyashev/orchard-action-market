"""Grid-based encoders for the general φ/R framework.

GeneralDecEncoder:  dec, T+3 channels, 3 scalars
GeneralCenEncoder:  cen, T+N+1 channels, N+1 scalars
EverythingEncoder:  cen and dec, T+N+1 channels, N+1 scalars; raw binary positions only
"""

from __future__ import annotations

import numpy as np
import torch

from orchard.encoding.base import GridEncoder
from orchard.datatypes import EncoderOutput, Grid, State


class GeneralDecEncoder(GridEncoder):
    """Decentralized encoder implementing the general φ/R observation spec.

    Grid channels (T+3):
      0..T-1  — task value: φ(i,κ)·Σ_j R(i,j)·r'_j at each task-κ cell
      T       — self position: 1.0 at agent i's cell
      T+1     — weighted teammate positions: Σ_j R(i,j)·1[p_j=cell]
      T+2     — actor position (R-weighted): R(i,c)·1[p_c=cell]

    Scalars (3):
      0 — 1[i = c]  (am I the actor?)
      1 — R(c,i)·1[i ≠ c]  (relatedness to actor, if not actor)
      2 — 1[pick_phase]·1[c=i or R(c,i)>0]  (pick pending and I'm involved?)
    """

    def __init__(self, env_cfg, phi: np.ndarray, relatedness: np.ndarray,
                 category_rewards: np.ndarray) -> None:
        super().__init__(env_cfg)
        T = env_cfg.n_task_types
        N = env_cfg.n_agents
        self._T = T
        self._N = N

        # task_value[i, kappa] = phi[i,kappa] * sum_j R[i,j] * r'[kappa,j]
        # = phi[i,kappa] * (R[i,:] @ category_rewards[kappa,:])
        rel_t = torch.from_numpy(relatedness).float()   # (N, N)
        phi_t = torch.from_numpy(phi).float()          # (N, T)
        cr_t = torch.from_numpy(category_rewards).float()  # (T, N)
        # R @ cr^T  → (N, T):  row i = sum_j R[i,j]*r'[kappa,j] for each kappa
        rel_x_cr = rel_t @ cr_t.T                      # (N, T)
        self._task_value: torch.Tensor = phi_t * rel_x_cr   # (N, T)

        # Relatedness matrix for encoding
        self._rel_t: torch.Tensor = rel_t              # (N, N)
        self._phi_t: torch.Tensor = phi_t              # (N, T)

        # Per-agent: which task channels are non-zero (for efficient masking)
        # phi_mask[i, kappa] = True if phi[i, kappa] > 0
        self._phi_mask: torch.Tensor = phi_t > 0       # (N, T) bool

    def grid_channels(self) -> int:
        return self._T + 3

    def scalar_dim(self) -> int:
        return 3

    # ------------------------------------------------------------------
    # Helper: build task value channels for all agents at once
    # ------------------------------------------------------------------
    def _build_task_grids(
        self, task_positions: tuple, task_types: tuple | None,
    ) -> torch.Tensor:
        """Returns (N, T, H, W) task-value grids."""
        N, T = self._N, self._T
        h, w = self.env_cfg.height, self.env_cfg.width
        grids = torch.zeros(N, T, h, w, dtype=torch.float32)
        if not task_positions or task_types is None:
            return grids
        rows = torch.tensor([p.row for p in task_positions], dtype=torch.long)
        cols = torch.tensor([p.col for p in task_positions], dtype=torch.long)
        types = torch.tensor(task_types, dtype=torch.long)
        # task_value[:, types] → (N, n_tasks)
        vals = self._task_value[:, types]  # (N, n_tasks)
        for k in range(len(task_positions)):
            tau = int(task_types[k])
            r, c = int(rows[k]), int(cols[k])
            grids[:, tau, r, c] = vals[:, k]
        return grids

    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        T, N = self._T, self._N
        h, w = self.env_cfg.height, self.env_cfg.width
        actor = state.actor

        grid = torch.zeros(T + 3, h, w, dtype=torch.float32)

        # Ch 0..T-1: task value channels for agent i
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                v = float(self._task_value[agent_idx, tau])
                if v != 0.0:
                    grid[tau, pos.row, pos.col] = v

        # Ch T: self position
        r, c = state.agent_positions[agent_idx]
        grid[T, r, c] = 1.0

        # Ch T+1: weighted teammate positions Σ_j R(i,j)·1[p_j=cell]
        for j, pos in enumerate(state.agent_positions):
            rel = float(self._rel_t[agent_idx, j])
            if j != agent_idx and rel > 0:
                grid[T + 1, pos.row, pos.col] += rel

        # Ch T+2: R(i,c)·1[p_c=cell]
        rel_to_actor = float(self._rel_t[agent_idx, actor])
        actor_pos = state.agent_positions[actor]
        grid[T + 2, actor_pos.row, actor_pos.col] = rel_to_actor

        # Scalars
        is_actor = 1.0 if agent_idx == actor else 0.0
        rel_from_actor = float(self._rel_t[actor, agent_idx]) if agent_idx != actor else 0.0
        involved = (agent_idx == actor) or (rel_from_actor > 0)
        pick_signal = 1.0 if (state.pick_phase and involved) else 0.0
        scalar = torch.tensor([is_actor, rel_from_actor, pick_signal], dtype=torch.float32)

        return EncoderOutput(grid=grid, scalar=scalar)

    def encode_batch_for_actions(
        self, state: State, agent_idx: int, after_states: list[State],
    ) -> EncoderOutput:
        T, N = self._T, self._N
        h, w = self.env_cfg.height, self.env_cfg.width
        n = len(after_states)
        actor = state.actor
        is_actor = (agent_idx == actor)
        rel_to_actor = float(self._rel_t[agent_idx, actor])
        rel_from_actor = float(self._rel_t[actor, agent_idx]) if not is_actor else 0.0
        involved = is_actor or (rel_from_actor > 0)

        # Build base grid (task channels + static agent positions)
        base = torch.zeros(T + 3, h, w, dtype=torch.float32)

        # Task channels (same for all actions unless task was picked)
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                v = float(self._task_value[agent_idx, tau])
                if v != 0.0:
                    base[tau, pos.row, pos.col] = v

        if not is_actor:
            # Self stays put
            r, c = state.agent_positions[agent_idx]
            base[T, r, c] = 1.0
            # Teammates (excluding actor) stay put
            for j, pos in enumerate(state.agent_positions):
                rel = float(self._rel_t[agent_idx, j])
                if j != agent_idx and j != actor and rel > 0:
                    base[T + 1, pos.row, pos.col] += rel
        else:
            # All other agents stay put (actor=self moves)
            for j, pos in enumerate(state.agent_positions):
                rel = float(self._rel_t[agent_idx, j])
                if j != agent_idx and rel > 0:
                    base[T + 1, pos.row, pos.col] += rel

        grids = torch.zeros(n, T + 3, h, w, dtype=torch.float32)
        scalars = torch.zeros(n, 3, dtype=torch.float32)
        scalars[:, 0] = 1.0 if is_actor else 0.0
        scalars[:, 1] = rel_from_actor

        for k, s_after in enumerate(after_states):
            grids[k] = base.clone()
            actor_pos = s_after.agent_positions[actor]

            if is_actor:
                grids[k, T, actor_pos.row, actor_pos.col] = 1.0
            else:
                rel = float(self._rel_t[agent_idx, actor])
                if rel > 0:
                    grids[k, T + 1, actor_pos.row, actor_pos.col] += rel

            grids[k, T + 2, actor_pos.row, actor_pos.col] = rel_to_actor

            if s_after.pick_phase and involved:
                scalars[k, 2] = 1.0

            # If tasks changed (pick happened), refresh task channels
            if s_after.task_positions != state.task_positions:
                grids[k, :T] = 0.0
                if s_after.task_positions and s_after.task_types is not None:
                    for pos, tau in zip(s_after.task_positions, s_after.task_types):
                        v = float(self._task_value[agent_idx, tau])
                        if v != 0.0:
                            grids[k, tau, pos.row, pos.col] = v

        return EncoderOutput(grid=grids, scalar=scalars)

    def encode_all_agents(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        N, T = self._N, self._T
        h, w = self.env_cfg.height, self.env_cfg.width
        actor = state.actor
        grids = torch.zeros(N, T + 3, h, w, dtype=torch.float32)

        # Ch 0..T-1: task value channels for all agents simultaneously
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                vals = self._task_value[:, tau]   # (N,)
                grids[:, tau, pos.row, pos.col] = vals

        # Ch T: self positions
        self_grids = torch.zeros(N, h, w, dtype=torch.float32)
        for i, pos in enumerate(state.agent_positions):
            self_grids[i, pos.row, pos.col] = 1.0
        grids[:, T] = self_grids

        # Ch T+1: Σ_j R(i,j)·1[p_j=cell] — weighted agent count (excl self)
        # rel_t (N,N) @ self_grids.view(N,-1) → (N, H*W)
        all_weighted = torch.mm(self._rel_t, self_grids.view(N, -1)).view(N, h, w)
        # subtract self contribution R(i,i)*1[p_i=cell] = 1*self_grid[i]
        self_contribution = self_grids  # R(i,i)=1 always
        grids[:, T + 1] = all_weighted - self_contribution

        # Ch T+2: R(i,actor)·1[p_actor=cell]
        actor_pos = state.agent_positions[actor]
        rel_col = self._rel_t[:, actor]   # (N,) — R(i, actor) for each i
        grids[:, T + 2, actor_pos.row, actor_pos.col] = rel_col

        # Scalars (N, 3)
        scalars = torch.zeros(N, 3, dtype=torch.float32)
        scalars[actor, 0] = 1.0
        # scalar[1] = R(actor, i) * 1[i != actor]
        rel_from_actor = self._rel_t[actor].clone()  # (N,)
        rel_from_actor[actor] = 0.0
        scalars[:, 1] = rel_from_actor
        # scalar[2] = pick_phase * (i==actor or R(actor,i)>0)
        if state.pick_phase:
            involved = (rel_from_actor > 0)
            involved[actor] = True
            scalars[involved, 2] = 1.0

        return grids, scalars

    def encode_all_agents_for_actions(
        self, state: State, after_states: list[State],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N, T = self._N, self._T
        B = len(after_states)
        h, w = self.env_cfg.height, self.env_cfg.width
        actor = state.actor

        grids = torch.zeros(N, B, T + 3, h, w, dtype=torch.float32)

        # Task channels — same for all actions (broadcast over B), updated per pick
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                vals = self._task_value[:, tau]   # (N,)
                grids[:, :, tau, pos.row, pos.col] = vals.unsqueeze(1)

        # Self positions (broadcast over B)
        self_grids = torch.zeros(N, h, w, dtype=torch.float32)
        for i, pos in enumerate(state.agent_positions):
            self_grids[i, pos.row, pos.col] = 1.0
        grids[:, :, T] = self_grids.unsqueeze(1)

        # Ch T+1: weighted others (broadcast, then update for actor movement)
        all_weighted = torch.mm(self._rel_t, self_grids.view(N, -1)).view(N, h, w)
        grids[:, :, T + 1] = (all_weighted - self_grids).unsqueeze(1)

        # Per-action: update actor position in Ch T, Ch T+1, Ch T+2
        old_actor_pos = state.agent_positions[actor]
        actor_rows = torch.tensor([s.agent_positions[actor].row for s in after_states], dtype=torch.long)
        actor_cols = torch.tensor([s.agent_positions[actor].col for s in after_states], dtype=torch.long)

        # Ch T: actor's self grid varies per action (vectorized scatter)
        grids[actor, :, T] = 0.0
        actor_self = torch.zeros(B, h, w, dtype=torch.float32)
        actor_self[torch.arange(B), actor_rows, actor_cols] = 1.0
        grids[actor, :, T] = actor_self

        # Ch T+1: fix for agents that see actor as teammate (excluding actor itself)
        old_ar, old_ac = old_actor_pos.row, old_actor_pos.col
        moved_mask = (actor_rows != old_ar) | (actor_cols != old_ac)
        moved_indices = moved_mask.nonzero(as_tuple=True)[0]
        if len(moved_indices) > 0:
            moved_rows = actor_rows[moved_indices]
            moved_cols = actor_cols[moved_indices]
            for i in range(N):
                if i == actor:
                    continue
                rel = float(self._rel_t[i, actor])
                if rel > 0:
                    grids[i, moved_indices, T + 1, old_ar, old_ac] -= rel
                    grids[i, moved_indices, T + 1, moved_rows, moved_cols] += rel

        # Ch T+2: R(i, actor) * 1[p_actor=cell] per action (broadcast over N and B)
        rel_col = self._rel_t[:, actor]   # (N,)
        grids[:, :, T + 2] = rel_col.view(N, 1, 1, 1) * actor_self.unsqueeze(0)

        # Fix task channels for pick after-states
        changed_task_indices = [k for k, s in enumerate(after_states)
                                 if s.task_positions != state.task_positions]
        for k in changed_task_indices:
            grids[:, k, :T] = 0.0
            s = after_states[k]
            if s.task_positions and s.task_types is not None:
                for pos, tau in zip(s.task_positions, s.task_types):
                    vals = self._task_value[:, tau]
                    grids[:, k, tau, pos.row, pos.col] = vals

        # Scalars (N, B, 3)
        scalars = torch.zeros(N, B, 3, dtype=torch.float32)
        scalars[actor, :, 0] = 1.0
        rel_from_actor = self._rel_t[actor].clone()  # (N,)
        rel_from_actor[actor] = 0.0
        scalars[:, :, 1] = rel_from_actor.unsqueeze(1)
        pick_phase_mask = torch.tensor([s.pick_phase for s in after_states], dtype=torch.bool)
        if pick_phase_mask.any():
            involved = (rel_from_actor > 0)
            involved[actor] = True
            scalars[involved, :, 2] = pick_phase_mask.float().unsqueeze(0)

        return grids, scalars


class GeneralCenEncoder(GridEncoder):
    """Centralized encoder implementing the general φ/R observation spec.

    Grid channels (T+N+1):
      0..T-1      — optimal task value: max_k[φ(k,κ)·Σ_j R(k,j)·r'_j] at each κ cell
      T..T+N-1    — per-agent position: channel T+j has 1.0 at agent j's cell
      T+N         — actor position: 1.0 at actor's cell

    Scalars (N+1):
      0..N-1  — one-hot actor identity
      N       — 1[pick_phase]
    """

    def __init__(self, env_cfg, phi: np.ndarray, relatedness: np.ndarray,
                 category_rewards: np.ndarray) -> None:
        super().__init__(env_cfg)
        T = env_cfg.n_task_types
        N = env_cfg.n_agents
        self._T = T
        self._N = N

        # opt_val[kappa] = max_k(phi[k,kappa] * sum_j R[k,j] * r'[kappa,j])
        rel_t = torch.from_numpy(relatedness).float()
        phi_t = torch.from_numpy(phi).float()
        cr_t = torch.from_numpy(category_rewards).float()
        per_agent_val = phi_t * (rel_t @ cr_t.T)   # (N, T): value if agent k picks kappa
        opt_val, _ = per_agent_val.max(dim=0)       # (T,): best possible value per category
        self._opt_val: torch.Tensor = opt_val        # (T,)

    def grid_channels(self) -> int:
        return self._T + self._N + 1

    def scalar_dim(self) -> int:
        return self._N + 1

    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        T, N = self._T, self._N
        h, w = self.env_cfg.height, self.env_cfg.width
        C = T + N + 1
        grid = torch.zeros(C, h, w, dtype=torch.float32)

        # Task channels
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                grid[tau, pos.row, pos.col] = float(self._opt_val[tau])

        # Per-agent position channels
        for j, pos in enumerate(state.agent_positions):
            grid[T + j, pos.row, pos.col] = 1.0

        # Actor position channel
        actor_pos = state.agent_positions[state.actor]
        grid[T + N, actor_pos.row, actor_pos.col] = 1.0

        # Scalar: one-hot actor + pick_phase
        scalar = torch.zeros(N + 1, dtype=torch.float32)
        scalar[state.actor] = 1.0
        if state.pick_phase:
            scalar[N] = 1.0

        return EncoderOutput(grid=grid, scalar=scalar)

    def encode_batch_for_actions(
        self, state: State, agent_idx: int, after_states: list[State],
    ) -> EncoderOutput:
        T, N = self._T, self._N
        h, w = self.env_cfg.height, self.env_cfg.width
        C = T + N + 1
        n = len(after_states)
        actor = state.actor

        base = torch.zeros(C, h, w, dtype=torch.float32)
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                base[tau, pos.row, pos.col] = float(self._opt_val[tau])
        for j, pos in enumerate(state.agent_positions):
            if j != actor:
                base[T + j, pos.row, pos.col] = 1.0

        grids = torch.zeros(n, C, h, w, dtype=torch.float32)
        for k, s_after in enumerate(after_states):
            grids[k] = base.clone()
            actor_pos = s_after.agent_positions[actor]
            grids[k, T + actor, actor_pos.row, actor_pos.col] = 1.0
            grids[k, T + N, actor_pos.row, actor_pos.col] = 1.0
            if s_after.task_positions != state.task_positions:
                grids[k, :T] = 0.0
                if s_after.task_positions and s_after.task_types is not None:
                    for pos, tau in zip(s_after.task_positions, s_after.task_types):
                        grids[k, tau, pos.row, pos.col] = float(self._opt_val[tau])

        scalar_base = torch.zeros(N + 1, dtype=torch.float32)
        scalar_base[actor] = 1.0
        scalars = scalar_base.unsqueeze(0).expand(n, -1).clone()
        for k, s_after in enumerate(after_states):
            if s_after.pick_phase:
                scalars[k, N] = 1.0

        return EncoderOutput(grid=grids, scalar=scalars)

    def encode_all_agents(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        """Single shared encoding, wrapped in N=1 dim for GpuTrainer."""
        out = self.encode(state, agent_idx=0)
        assert out.grid is not None and out.scalar is not None
        return out.grid.unsqueeze(0), out.scalar.unsqueeze(0)

    def encode_all_agents_for_actions(
        self, state: State, after_states: list[State],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.encode_batch_for_actions(state, agent_idx=0, after_states=after_states)
        assert out.grid is not None and out.scalar is not None
        return out.grid.unsqueeze(0), out.scalar.unsqueeze(0)


class EverythingEncoder(GridEncoder):
    """Raw-binary encoder compatible with both centralized and decentralized learning.

    The encoding contains no pre-calculated φ, R, or reward values. Agents must
    learn the underlying structure purely from the team reward signal.

    Grid channels (T+N+1):
      0..T-1      — task presence: 1 if a task of type κ exists at (r, l)
      T..T+N-1    — per-agent position: channel T+j has 1.0 at agent j's cell
      T+N         — actor position: 1.0 at the acting agent's cell

    Scalars (N+1):
      0..N-1  — one-hot actor identity (e_c)
      N       — 1[pick_phase]

    Works for centralized (n_networks=1) and decentralized (n_networks=N).
    encode_all_agents returns (n_networks, C, H, W): a single encoding
    broadcast to n_networks copies so each network receives an identical view.
    """

    def __init__(self, env_cfg, n_networks: int) -> None:
        super().__init__(env_cfg)
        self._T = env_cfg.n_task_types
        self._N = env_cfg.n_agents
        self._n_networks = n_networks

    def grid_channels(self) -> int:
        return self._T + self._N + 1

    def scalar_dim(self) -> int:
        return self._N + 1

    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        T, N = self._T, self._N
        h, w = self.env_cfg.height, self.env_cfg.width
        C = T + N + 1
        grid = torch.zeros(C, h, w, dtype=torch.float32)

        # Ch 0..T-1: binary task presence by category
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                grid[tau, pos.row, pos.col] = 1.0

        # Ch T..T+N-1: per-agent positions
        for j, pos in enumerate(state.agent_positions):
            grid[T + j, pos.row, pos.col] = 1.0

        # Ch T+N: actor position
        actor_pos = state.agent_positions[state.actor]
        grid[T + N, actor_pos.row, actor_pos.col] = 1.0

        # Scalars: one-hot actor identity + pick_phase flag
        scalar = torch.zeros(N + 1, dtype=torch.float32)
        scalar[state.actor] = 1.0
        if state.pick_phase:
            scalar[N] = 1.0

        return EncoderOutput(grid=grid, scalar=scalar)

    def encode_batch_for_actions(
        self, state: State, agent_idx: int, after_states: list[State],
    ) -> EncoderOutput:
        T, N = self._T, self._N
        h, w = self.env_cfg.height, self.env_cfg.width
        C = T + N + 1
        B = len(after_states)
        actor = state.actor

        # Build base: task channels + all non-actor agent channels (static across actions)
        base = torch.zeros(C, h, w, dtype=torch.float32)
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                base[tau, pos.row, pos.col] = 1.0
        for j, pos in enumerate(state.agent_positions):
            if j != actor:
                base[T + j, pos.row, pos.col] = 1.0

        # Broadcast base to (B, C, H, W)
        grids = base.unsqueeze(0).expand(B, -1, -1, -1).clone()

        # Vectorized actor position update across all B actions
        actor_rows = torch.tensor(
            [s.agent_positions[actor].row for s in after_states], dtype=torch.long
        )
        actor_cols = torch.tensor(
            [s.agent_positions[actor].col for s in after_states], dtype=torch.long
        )
        b_idx = torch.arange(B, dtype=torch.long)
        grids[b_idx, T + actor, actor_rows, actor_cols] = 1.0
        grids[b_idx, T + N, actor_rows, actor_cols] = 1.0

        # Refresh task channels for pick after-states (task list changed)
        changed = [k for k, s in enumerate(after_states)
                   if s.task_positions != state.task_positions]
        for k in changed:
            grids[k, :T] = 0.0
            s = after_states[k]
            if s.task_positions and s.task_types is not None:
                for pos, tau in zip(s.task_positions, s.task_types):
                    grids[k, tau, pos.row, pos.col] = 1.0

        # Scalars (B, N+1): base one-hot actor, then per-action pick_phase
        scalar_base = torch.zeros(N + 1, dtype=torch.float32)
        scalar_base[actor] = 1.0
        scalars = scalar_base.unsqueeze(0).expand(B, -1).clone()
        for k, s in enumerate(after_states):
            if s.pick_phase:
                scalars[k, N] = 1.0

        return EncoderOutput(grid=grids, scalar=scalars)

    def encode_all_agents(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        """Single encoding broadcast to n_networks copies for GPU Trainer."""
        out = self.encode(state, agent_idx=0)
        assert out.grid is not None and out.scalar is not None
        grid = out.grid.unsqueeze(0).expand(self._n_networks, -1, -1, -1).clone()
        scalar = out.scalar.unsqueeze(0).expand(self._n_networks, -1).clone()
        return grid, scalar

    def encode_all_agents_for_actions(
        self, state: State, after_states: list[State],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T, N = self._T, self._N
        h, w = self.env_cfg.height, self.env_cfg.width
        C = T + N + 1
        B = len(after_states)
        actor = state.actor

        # Build base (B, C, H, W) — same as encode_batch_for_actions but vectorized
        base = torch.zeros(C, h, w, dtype=torch.float32)
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                base[tau, pos.row, pos.col] = 1.0
        for j, pos in enumerate(state.agent_positions):
            if j != actor:
                base[T + j, pos.row, pos.col] = 1.0

        grids = base.unsqueeze(0).expand(B, -1, -1, -1).clone()

        actor_rows = torch.tensor(
            [s.agent_positions[actor].row for s in after_states], dtype=torch.long
        )
        actor_cols = torch.tensor(
            [s.agent_positions[actor].col for s in after_states], dtype=torch.long
        )
        b_idx = torch.arange(B, dtype=torch.long)
        grids[b_idx, T + actor, actor_rows, actor_cols] = 1.0
        grids[b_idx, T + N, actor_rows, actor_cols] = 1.0

        changed = [k for k, s in enumerate(after_states)
                   if s.task_positions != state.task_positions]
        for k in changed:
            grids[k, :T] = 0.0
            s = after_states[k]
            if s.task_positions and s.task_types is not None:
                for pos, tau in zip(s.task_positions, s.task_types):
                    grids[k, tau, pos.row, pos.col] = 1.0

        scalar_base = torch.zeros(N + 1, dtype=torch.float32)
        scalar_base[actor] = 1.0
        scalars = scalar_base.unsqueeze(0).expand(B, -1).clone()
        for k, s in enumerate(after_states):
            if s.pick_phase:
                scalars[k, N] = 1.0

        # Expand (B, C, H, W) → (n_networks, B, C, H, W)
        grid_out = grids.unsqueeze(0).expand(self._n_networks, -1, -1, -1, -1).clone()
        scalar_out = scalars.unsqueeze(0).expand(self._n_networks, -1, -1).clone()
        return grid_out, scalar_out
