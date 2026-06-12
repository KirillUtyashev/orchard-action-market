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
      - the tasks it cares about,  R_i = {k : d(i,k) ≤ R^R},     |R_i| = min(N, 2R^R+1)
      - the agents within reach,    W_i = {j : d(i,j) ≤ R^R+P^R}, |W_i| = min(N, 2(R^R+P^R)+1)
    where d is circular distance on the shared id ring (T=N). The kept ids are
    ordered circularly (k_0..k_{|R_i|-1}, j_0..j_{|W_i|-1}).

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
        assert T == N, "FilteredDecEncoder requires the shared id space T=N."
        assert n_networks == N, (
            f"FilteredDecEncoder is decentralized: n_networks ({n_networks}) must equal N ({N})."
        )
        self._T = T
        self._N = N
        self._n_networks = n_networks

        RR = env_cfg.relatedness_width
        PR = env_cfg.proficiency_width
        KR = min(N, 2 * RR + 1)            # |R_i|, same for all i
        KW = min(N, 2 * (RR + PR) + 1)     # |W_i|, same for all i
        self._KR = KR
        self._KW = KW

        # Per-network circular id windows + inverse maps to local channel indices.
        # task_local[i, tau]  = channel of task tau in R_i, or -1 if tau ∉ R_i.
        # agent_local[i, j]    = channel of agent j  in W_i, or -1 if j  ∉ W_i.
        task_local = torch.full((N, N), -1, dtype=torch.long)
        agent_local = torch.full((N, N), -1, dtype=torch.long)
        for i in range(N):
            for c in range(KR):
                k = (i - RR + c) % N
                task_local[i, k] = c
            for c in range(KW):
                j = (i - (RR + PR) + c) % N
                agent_local[i, j] = c
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

    # ------------------------------------------------------------------
    # Vectorized all-agents encoders.
    #
    # These produce output bit-identical to looping encode() /
    # encode_batch_for_actions() over the N networks (see
    # TestFilteredDecVectorizedEquivalence), but build every agent's masked
    # view in one shot with scatter ops instead of a Python N-loop. The N-loop
    # was the dominant per-step cost when models are tiny (CPU-bound encode,
    # idle GPU); collapsing it brings cost back near EverythingEncoder's.
    # ------------------------------------------------------------------
    def _scatter_task_channels(
        self, grids: torch.Tensor, task_positions, task_types,
    ) -> None:
        """Set task-presence channels (0..KR-1) of grids (N, C, H, W), per-agent masked.

        For every (viewer i, task t): channel = task_local[i, type_t]; if >= 0, mark a
        1.0 at the task's cell. Tasks of a type outside R_i are skipped (channel -1).
        """
        if not task_positions or task_types is None:
            return
        N = self._N
        taus = torch.as_tensor(task_types, dtype=torch.long)
        rows = torch.tensor([p.row for p in task_positions], dtype=torch.long)
        cols = torch.tensor([p.col for p in task_positions], dtype=torch.long)
        chan = self._task_local[:, taus]               # (N, M); -1 where type ∉ R_i
        M = taus.shape[0]
        vi = torch.arange(N).view(N, 1).expand(N, M)
        r = rows.view(1, M).expand(N, M)
        c = cols.view(1, M).expand(N, M)
        m = chan >= 0
        grids[vi[m], chan[m], r[m], c[m]] = 1.0

    def _scatter_agent_channels(
        self, grids: torch.Tensor, agent_positions, skip_actor: int | None = None,
    ) -> None:
        """Set agent-position channels (KR..KR+KW-1) of grids (N, C, H, W), per-agent masked.

        For every (viewer i, target j): channel = KR + agent_local[i, j]; if agent_local
        >= 0, mark a 1.0 at agent j's cell. Agents outside W_i are skipped. When
        skip_actor is given, that target column is omitted (its channel is written
        per-after-state elsewhere).
        """
        N, KR = self._N, self._KR
        arows = torch.tensor([p.row for p in agent_positions], dtype=torch.long)
        acols = torch.tensor([p.col for p in agent_positions], dtype=torch.long)
        chan_a = self._agent_local                     # (N_viewer, N_target); -1 where j ∉ W_i
        Nt = chan_a.shape[1]
        vi = torch.arange(N).view(N, 1).expand(N, Nt)
        r = arows.view(1, Nt).expand(N, Nt)
        c = acols.view(1, Nt).expand(N, Nt)
        m = chan_a >= 0
        if skip_actor is not None:
            m = m & (torch.arange(Nt).view(1, Nt) != skip_actor)
        grids[vi[m], KR + chan_a[m], r[m], c[m]] = 1.0

    def encode_all_agents(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        N, KR, KW = self._N, self._KR, self._KW
        h, w = self.env_cfg.height, self.env_cfg.width
        grids = torch.zeros(N, KR + KW + 1, h, w, dtype=torch.float32)

        self._scatter_task_channels(grids, state.task_positions, state.task_types)
        self._scatter_agent_channels(grids, state.agent_positions)  # all agents (incl. actor)

        # Actor-position channel: same cell for every viewer.
        apos = state.agent_positions[state.actor]
        grids[:, KR + KW, apos.row, apos.col] = 1.0

        # Scalars: one-hot actor within W_i (zero if actor ∉ W_i) + pick_phase.
        scalars = torch.zeros(N, KW + 1, dtype=torch.float32)
        ac = self._agent_local[:, state.actor]         # (N,)
        valid = ac >= 0
        scalars[valid, ac[valid]] = 1.0
        if state.pick_phase:
            scalars[:, KW] = 1.0
        return grids, scalars

    def encode_all_agents_for_actions(
        self, state: State, after_states: list[State],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N, KR, KW = self._N, self._KR, self._KW
        C = KR + KW + 1
        B = len(after_states)
        h, w = self.env_cfg.height, self.env_cfg.width
        actor = state.actor

        # Static base per viewer: current-state task channels + non-actor agent channels.
        base = torch.zeros(N, C, h, w, dtype=torch.float32)
        self._scatter_task_channels(base, state.task_positions, state.task_types)
        self._scatter_agent_channels(base, state.agent_positions, skip_actor=actor)

        grids = base.unsqueeze(1).expand(N, B, C, h, w).clone()  # (N, B, C, H, W)

        actor_c = self._agent_local[:, actor]          # (N,); -1 where actor ∉ W_i
        if B > 0:
            arows = torch.tensor([s.agent_positions[actor].row for s in after_states],
                                 dtype=torch.long)
            acols = torch.tensor([s.agent_positions[actor].col for s in after_states],
                                 dtype=torch.long)
            # Actor-position channel (KR+KW) for every (viewer, after-state).
            vi = torch.arange(N).view(N, 1).expand(N, B).reshape(-1)
            bi = torch.arange(B).view(1, B).expand(N, B).reshape(-1)
            rr = arows.view(1, B).expand(N, B).reshape(-1)
            cc = acols.view(1, B).expand(N, B).reshape(-1)
            grids[vi, bi, KR + KW, rr, cc] = 1.0
            # Actor's own agent channel (KR+actor_c[i]) where the actor is in W_i.
            valid = actor_c >= 0
            if bool(valid.any()):
                vi2 = torch.arange(N)[valid]           # (V,)
                ch2 = KR + actor_c[valid]              # (V,)
                V = vi2.shape[0]
                grids[vi2.view(V, 1).expand(V, B).reshape(-1),
                      torch.arange(B).view(1, B).expand(V, B).reshape(-1),
                      ch2.view(V, 1).expand(V, B).reshape(-1),
                      arows.view(1, B).expand(V, B).reshape(-1),
                      acols.view(1, B).expand(V, B).reshape(-1)] = 1.0

        # Refresh task channels for pick after-states whose task list changed.
        changed = [k for k, s in enumerate(after_states)
                   if s.task_positions != state.task_positions]
        for k in changed:
            grids[:, k, :KR] = 0.0
            s = after_states[k]
            self._scatter_task_channels(grids[:, k], s.task_positions, s.task_types)

        # Scalars (N, B, KW+1): one-hot actor within W_i (all B), then per-action pick_phase.
        scalars = torch.zeros(N, B, KW + 1, dtype=torch.float32)
        if B > 0:
            valid = actor_c >= 0
            if bool(valid.any()):
                scalars[torch.arange(N)[valid], :, actor_c[valid]] = 1.0
            pick_mask = torch.tensor([s.pick_phase for s in after_states], dtype=torch.bool)
            if bool(pick_mask.any()):
                scalars[:, pick_mask, KW] = 1.0
        return grids, scalars
