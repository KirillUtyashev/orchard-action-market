"""Grid-based encoders implementing the spec's raw-binary observation inputs.

EverythingEncoder:   cen and dec, T+N+1 channels, N+1 scalars; raw binary positions
FilteredDecEncoder:  dec, |R_i|+|W_i|+1 channels, |W_i|+1 scalars; raw binary, masked
                     to the tasks (R_i) and agents (W_i) network i can see (spec §4.2)
"""

from __future__ import annotations

import numpy as np
import torch

from orchard.encoding.base import GridEncoder
from orchard.datatypes import EncoderOutput, Grid, State
from orchard.env.base import circular_task_distance, task_center_for_agent


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


class FilteredDecEncoder(GridEncoder):
    """Decentralized raw-binary encoder, masked to what network i can see (spec §4.2).

    Like EverythingEncoder it contains no pre-computed φ/R/reward values — only
    raw binary positions — but network i only observes:
      - the tasks it cares about, R_i = {k : d(center_i,k) ≤ R^R}, |R_i| = min(T, 2R^R+1)
      - agent position channels. For T=N this keeps the old local W_i window;
        for T != N it keeps all agents so the shape is stable.
    Here center_i is agent i's base task in task-type space. The kept task ids
    are ordered circularly around that center.

    Grid channels (|R_i| + |W_i| + 1):
      0..|R_i|-1            — task presence: 1 where a task of kept type k_t exists
      |R_i|..|R_i|+|W_i|-1  — per-agent position: 1 at kept agent j_s's cell
      |R_i|+|W_i|           — actor position: 1 at the actor's cell

    Scalars (|W_i| + 1):
      0..|W_i|-1  — one-hot of the actor within W_i (all zero if actor ∉ W_i)
      |W_i|       — 1[pick_phase]

    |R_i| and |W_i| are the same for every i (circular symmetry), so the per-agent
    encodings stack into fixed-shape (N, ...) tensors for the GPU/vmap trainer.
    Decentralized only: n_networks must equal N.
    """

    def __init__(self, env_cfg, n_networks: int) -> None:
        super().__init__(env_cfg)
        T = env_cfg.n_task_types
        N = env_cfg.n_agents
        assert n_networks == N, (
            f"FilteredDecEncoder is decentralized: n_networks ({n_networks}) must equal N ({N})."
        )
        self._T = T
        self._N = N
        self._n_networks = n_networks

        RR = env_cfg.relatedness_width
        PR = env_cfg.proficiency_width
        KR = min(T, 2 * RR + 1)            # |R_i|: task types inside interest window
        KW = min(N, 2 * (RR + PR) + 1) if T == N else N
        self._KR = KR
        self._KW = KW
        task_centers = [task_center_for_agent(i, N, T) for i in range(N)]

        # Per-network task-space windows + inverse maps to local channel indices.
        # task_local[i, tau] = channel of task tau in R_i, or -1 if tau not in agent i's interests.
        # agent_local[i, j]  = channel of agent j in W_i. For T != N we keep all agents visible.
        task_local = torch.full((N, T), -1, dtype=torch.long)
        agent_local = torch.full((N, N), -1, dtype=torch.long)
        for i, center in enumerate(task_centers):
            kept_tasks = [tau for tau in range(T) if circular_task_distance(center, tau, T) <= RR]
            kept_tasks.sort(key=lambda tau: ((tau - center) % T))
            for c, tau in enumerate(kept_tasks[:KR]):
                task_local[i, tau] = c

            if T == N:
                for c in range(KW):
                    j = (i - (RR + PR) + c) % N
                    agent_local[i, j] = c
            else:
                for j in range(N):
                    agent_local[i, j] = j
        self._task_local = task_local
        self._agent_local = agent_local

    def grid_channels(self) -> int:
        return self._KR + self._KW + 1

    def scalar_dim(self) -> int:
        return self._KW + 1

    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        i = agent_idx
        KR, KW = self._KR, self._KW
        h, w = self.env_cfg.height, self.env_cfg.width
        grid = torch.zeros(KR + KW + 1, h, w, dtype=torch.float32)

        # Task channels (only tasks in R_i)
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                c = int(self._task_local[i, tau])
                if c >= 0:
                    grid[c, pos.row, pos.col] = 1.0

        # Agent channels (only agents in W_i)
        for j, pos in enumerate(state.agent_positions):
            c = int(self._agent_local[i, j])
            if c >= 0:
                grid[KR + c, pos.row, pos.col] = 1.0

        # Actor position channel (always marks the actor's cell)
        actor_pos = state.agent_positions[state.actor]
        grid[KR + KW, actor_pos.row, actor_pos.col] = 1.0

        # Scalars: one-hot actor within W_i (zero if actor ∉ W_i) + pick_phase
        scalar = torch.zeros(KW + 1, dtype=torch.float32)
        ac = int(self._agent_local[i, state.actor])
        if ac >= 0:
            scalar[ac] = 1.0
        if state.pick_phase:
            scalar[KW] = 1.0

        return EncoderOutput(grid=grid, scalar=scalar)

    def encode_batch_for_actions(
        self, state: State, agent_idx: int, after_states: list[State],
    ) -> EncoderOutput:
        i = agent_idx
        KR, KW = self._KR, self._KW
        h, w = self.env_cfg.height, self.env_cfg.width
        C = KR + KW + 1
        B = len(after_states)
        actor = state.actor

        # Static base: kept task channels + kept non-actor agent channels
        base = torch.zeros(C, h, w, dtype=torch.float32)
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                c = int(self._task_local[i, tau])
                if c >= 0:
                    base[c, pos.row, pos.col] = 1.0
        for j, pos in enumerate(state.agent_positions):
            if j == actor:
                continue
            c = int(self._agent_local[i, j])
            if c >= 0:
                base[KR + c, pos.row, pos.col] = 1.0

        grids = base.unsqueeze(0).expand(B, -1, -1, -1).clone()
        actor_c = int(self._agent_local[i, actor])  # -1 if actor ∉ W_i
        for k, s in enumerate(after_states):
            apos = s.agent_positions[actor]
            if actor_c >= 0:
                grids[k, KR + actor_c, apos.row, apos.col] = 1.0
            grids[k, KR + KW, apos.row, apos.col] = 1.0
            # Refresh task channels for pick after-states (task list changed)
            if s.task_positions != state.task_positions:
                grids[k, :KR] = 0.0
                if s.task_positions and s.task_types is not None:
                    for pos, tau in zip(s.task_positions, s.task_types):
                        c = int(self._task_local[i, tau])
                        if c >= 0:
                            grids[k, c, pos.row, pos.col] = 1.0

        scalars = torch.zeros(B, KW + 1, dtype=torch.float32)
        if actor_c >= 0:
            scalars[:, actor_c] = 1.0
        for k, s in enumerate(after_states):
            if s.pick_phase:
                scalars[k, KW] = 1.0

        return EncoderOutput(grid=grids, scalar=scalars)

    def encode_all_agents(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        N, KR, KW = self._N, self._KR, self._KW
        h, w = self.env_cfg.height, self.env_cfg.width
        grids = torch.zeros(N, KR + KW + 1, h, w, dtype=torch.float32)
        scalars = torch.zeros(N, KW + 1, dtype=torch.float32)
        for i in range(N):
            out = self.encode(state, i)
            assert out.grid is not None and out.scalar is not None
            grids[i] = out.grid
            scalars[i] = out.scalar
        return grids, scalars

    def encode_all_agents_for_actions(
        self, state: State, after_states: list[State],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N, KR, KW = self._N, self._KR, self._KW
        B = len(after_states)
        h, w = self.env_cfg.height, self.env_cfg.width
        grids = torch.zeros(N, B, KR + KW + 1, h, w, dtype=torch.float32)
        scalars = torch.zeros(N, B, KW + 1, dtype=torch.float32)
        for i in range(N):
            out = self.encode_batch_for_actions(state, i, after_states)
            assert out.grid is not None and out.scalar is not None
            grids[i] = out.grid
            scalars[i] = out.scalar
        return grids, scalars
