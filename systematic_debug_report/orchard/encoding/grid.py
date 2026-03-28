"""Grid encoder: multi-channel (C, H, W) tensor encoding."""

from __future__ import annotations

import torch

from orchard.encoding.base import BaseEncoder, GridEncoder
from orchard.datatypes import EncoderOutput, State


class BasicGridEncoder(GridEncoder):
    """Multi-channel grid encoding.

    Channels:
      0 — apples (1 per apple)
      1 — self position (1 at own cell)
      2 — other agents (1 per agent; may sum >1 if overlapping)
      3 — actor position (1 at actor's cell)
    Scalar: is_self_actor (1.0 if agent is current actor, else 0.0)
    """

    def grid_channels(self) -> int:
        return 4

    def scalar_dim(self) -> int:
        return 1

    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        h, w = self.env_cfg.height, self.env_cfg.width
        grid = torch.zeros(4, h, w, dtype=torch.float32)

        # Apples
        for ap in state.apple_positions:
            grid[0, ap.row, ap.col] += 1.0

        # Self
        r, c = state.agent_positions[agent_idx]
        grid[1, r, c] = 1.0

        # Other agents
        for i, pos in enumerate(state.agent_positions):
            if i != agent_idx:
                grid[2, pos.row, pos.col] += 1.0

        # Actor
        actor_pos = state.agent_positions[state.actor]
        grid[3, actor_pos.row, actor_pos.col] = 1.0

        # Scalar
        is_actor = 1.0 if agent_idx == state.actor else 0.0
        scalar = torch.tensor([is_actor], dtype=torch.float32)

        return EncoderOutput(grid=grid, scalar=scalar)

    def encode_batch_for_actions(self, state: State, agent_idx: int, after_states: list[State]) -> EncoderOutput:
        """Batch-encode after-states. Only actor position differs across them."""
        h, w = self.env_cfg.height, self.env_cfg.width
        n = len(after_states)
        actor = state.actor
        is_actor = (agent_idx == actor)

        # Base grid: channels constant across all actions
        base = torch.zeros(4, h, w, dtype=torch.float32)

        # Ch0: apples (always constant)
        for ap in state.apple_positions:
            base[0, ap.row, ap.col] += 1.0

        if is_actor:
            # Ch1 (self=actor) varies per action → leave blank
            # Ch2 (others): non-actor agents didn't move → constant
            for j, pos in enumerate(state.agent_positions):
                if j != agent_idx:
                    base[2, pos.row, pos.col] += 1.0
        else:
            # Ch1 (self): didn't move → constant
            r, c = state.agent_positions[agent_idx]
            base[1, r, c] = 1.0
            # Ch2 (others): all except self and actor are constant
            for j, pos in enumerate(state.agent_positions):
                if j != agent_idx and j != actor:
                    base[2, pos.row, pos.col] += 1.0
            # Actor's contribution to Ch2 varies → filled per action

        # Ch3 (actor) always varies → leave blank

        # Build per-action grids
        grids = torch.zeros(n, 4, h, w, dtype=torch.float32)
        for k, s_after in enumerate(after_states):
            grids[k] = base
            actor_pos = s_after.agent_positions[actor]
            if is_actor:
                grids[k, 1, actor_pos.row, actor_pos.col] = 1.0
            else:
                grids[k, 2, actor_pos.row, actor_pos.col] += 1.0
            grids[k, 3, actor_pos.row, actor_pos.col] = 1.0

        # Scalar (constant across actions)
        scalar_val = 1.0 if is_actor else 0.0
        scalars = torch.full((n, 1), scalar_val, dtype=torch.float32)

        return EncoderOutput(grid=grids, scalar=scalars)
    
class GridMLPEncoder(BaseEncoder):
    """Flattened CNN grid as MLP input (no convolutions).

    Identical information to BasicGridEncoder but flattened into a 1-D vector.
    Baseline to isolate convolutions vs. grid representation.

    Feature vector (length = 4*H*W + 1):
      channel 0 flattened: apples          (H*W values)
      channel 1 flattened: self position   (H*W values)
      channel 2 flattened: other agents    (H*W values)
      channel 3 flattened: actor position  (H*W values)
      scalar: is_actor
    """

    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        h, w = self.env_cfg.height, self.env_cfg.width
        grid = torch.zeros(4, h, w, dtype=torch.float32)

        for ap in state.apple_positions:
            grid[0, ap.row, ap.col] += 1.0

        r, c = state.agent_positions[agent_idx]
        grid[1, r, c] = 1.0

        for i, pos in enumerate(state.agent_positions):
            if i != agent_idx:
                grid[2, pos.row, pos.col] += 1.0

        actor_pos = state.agent_positions[state.actor]
        grid[3, actor_pos.row, actor_pos.col] = 1.0

        flat = grid.reshape(-1)
        is_actor = 1.0 if agent_idx == state.actor else 0.0
        scalar = torch.cat([flat, torch.tensor([is_actor])])

        return EncoderOutput(scalar=scalar)

    def scalar_dim(self) -> int:
        h, w = self.env_cfg.height, self.env_cfg.width
        return 4 * h * w + 1
    
class CentralizedGridEncoder(GridEncoder):
    """Centralized 3-channel grid encoding.

    Channels (count-valued, not binary):
      0 — apples (count at each cell)
      1 — all agents (count at each cell)
      2 — actor position (1 at actor's cell)
    No scalar features. agent_idx parameter is ignored.
    """

    def grid_channels(self) -> int:
        return 3

    def scalar_dim(self) -> int:
        return 0

    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        h, w = self.env_cfg.height, self.env_cfg.width
        grid = torch.zeros(3, h, w, dtype=torch.float32)

        # Apples (count)
        for ap in state.apple_positions:
            grid[0, ap.row, ap.col] += 1.0

        # All agents (count)
        for pos in state.agent_positions:
            grid[1, pos.row, pos.col] += 1.0

        # Actor
        actor_pos = state.agent_positions[state.actor]
        grid[2, actor_pos.row, actor_pos.col] = 1.0

        return EncoderOutput(grid=grid, scalar=None)

    def encode_batch_for_actions(self, state: State, agent_idx: int, after_states: list[State]) -> EncoderOutput:
        """Batch-encode after-states. Only actor position differs across them."""
        h, w = self.env_cfg.height, self.env_cfg.width
        n = len(after_states)
        actor = state.actor

        # Base grid
        base = torch.zeros(3, h, w, dtype=torch.float32)

        # Ch0: apples (constant)
        for ap in state.apple_positions:
            base[0, ap.row, ap.col] += 1.0

        # Ch1: non-actor agents (constant part)
        for j, pos in enumerate(state.agent_positions):
            if j != actor:
                base[1, pos.row, pos.col] += 1.0

        # Per-action grids
        grids = torch.zeros(n, 3, h, w, dtype=torch.float32)
        for k, s_after in enumerate(after_states):
            grids[k] = base
            actor_pos = s_after.agent_positions[actor]
            grids[k, 1, actor_pos.row, actor_pos.col] += 1.0
            grids[k, 2, actor_pos.row, actor_pos.col] = 1.0

        return EncoderOutput(grid=grids, scalar=None)

class NoRedundantAgentGridEncoder(GridEncoder):
    """Decentralized 4-channel grid encoding with actor removed from self/others.

    Identical to BasicGridEncoder except:
      - Ch1 (self): zeroed out when agent_idx == actor (actor channel already marks self)
      - Ch2 (others): actor excluded, leaving agents j where j != agent_idx AND j != actor
    The actor's position is still fully specified by Ch3.

    Channels:
      0 — apples (1 per apple)
      1 — self position (1 at own cell, BUT 0 if agent is the actor)
      2 — other agents excluding actor (count at each cell)
      3 — actor position (1 at actor's cell)
    Scalar: is_self_actor (1.0 if agent is current actor, else 0.0)
    """

    def grid_channels(self) -> int:
        return 4

    def scalar_dim(self) -> int:
        return 1

    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        h, w = self.env_cfg.height, self.env_cfg.width
        grid = torch.zeros(4, h, w, dtype=torch.float32)

        # Ch0: apples
        for ap in state.apple_positions:
            grid[0, ap.row, ap.col] += 1.0

        # Ch1: self — only if agent is NOT the actor
        if agent_idx != state.actor:
            r, c = state.agent_positions[agent_idx]
            grid[1, r, c] = 1.0

        # Ch2: others, excluding both self AND actor
        for j, pos in enumerate(state.agent_positions):
            if j != agent_idx and j != state.actor:
                grid[2, pos.row, pos.col] += 1.0

        # Ch3: actor
        actor_pos = state.agent_positions[state.actor]
        grid[3, actor_pos.row, actor_pos.col] = 1.0

        # Scalar
        is_actor = 1.0 if agent_idx == state.actor else 0.0
        scalar = torch.tensor([is_actor], dtype=torch.float32)

        return EncoderOutput(grid=grid, scalar=scalar)

    def encode_batch_for_actions(self, state: State, agent_idx: int, after_states: list[State]) -> EncoderOutput:
        """Batch-encode after-states. Only actor position differs across them."""
        h, w = self.env_cfg.height, self.env_cfg.width
        n = len(after_states)
        actor = state.actor
        is_actor = (agent_idx == actor)

        base = torch.zeros(4, h, w, dtype=torch.float32)

        # Ch0: apples (constant)
        for ap in state.apple_positions:
            base[0, ap.row, ap.col] += 1.0

        if is_actor:
            # Ch1 (self): zeroed because agent IS the actor → all zeros, constant
            # Ch2 (others): exclude self AND actor (same agent) → all other agents, constant
            for j, pos in enumerate(state.agent_positions):
                if j != agent_idx:
                    base[2, pos.row, pos.col] += 1.0
        else:
            # Ch1 (self): self is not actor, self didn't move → constant
            r, c = state.agent_positions[agent_idx]
            base[1, r, c] = 1.0
            # Ch2 (others): exclude self AND actor; non-actor others are constant
            for j, pos in enumerate(state.agent_positions):
                if j != agent_idx and j != actor:
                    base[2, pos.row, pos.col] += 1.0
            # Actor is excluded from Ch2, so no varying contribution

        # Ch3 (actor) varies per action → leave blank in base

        grids = torch.zeros(n, 4, h, w, dtype=torch.float32)
        for k, s_after in enumerate(after_states):
            grids[k] = base
            actor_pos = s_after.agent_positions[actor]
            grids[k, 3, actor_pos.row, actor_pos.col] = 1.0

        scalar_val = 1.0 if is_actor else 0.0
        scalars = torch.full((n, 1), scalar_val, dtype=torch.float32)

        return EncoderOutput(grid=grids, scalar=scalars)


class EgoCentricGridEncoder(GridEncoder):
    """Ego-centric decentralized encoding: agent always at grid center.

    Grid is (2H-1) x (2W-1). World position for ego-centric coord x' is:
        x_w = x_i + (x' - o), where o = (H-1, W-1) is the center.
    Out-of-bounds positions are filled with -1.

    Channels:
      0 — apples (1 if apple at x_w, -1 if OOB, 0 otherwise)
      1 — self (1 at center o, -1 if OOB, 0 otherwise)
      2 — other agents (count at x_w, -1 if OOB)
      3 — actor (1 if actor at x_w, -1 if OOB, 0 otherwise)
    Scalar: is_self_actor (1.0 if agent is current actor, else 0.0)
    """

    def grid_channels(self) -> int:
        return 4

    def scalar_dim(self) -> int:
        return 1

    def grid_height(self) -> int:
        return 2 * self.env_cfg.height - 1

    def grid_width(self) -> int:
        return 2 * self.env_cfg.width - 1

    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        H, W = self.env_cfg.height, self.env_cfg.width
        H_prime = 2 * H - 1
        W_prime = 2 * W - 1
        r_i, c_i = state.agent_positions[agent_idx]

        # Fill with -1 (OOB default)
        grid = torch.full((4, H_prime, W_prime), -1.0, dtype=torch.float32)

        # Center offset: world (r, c) → ego (r + or_off, c + oc_off)
        or_off = (H - 1) - r_i
        oc_off = (W - 1) - c_i

        # Zero in-bounds cells
        for r_w in range(H):
            for c_w in range(W):
                r_ego = r_w + or_off
                c_ego = c_w + oc_off
                grid[0, r_ego, c_ego] = 0.0
                grid[1, r_ego, c_ego] = 0.0
                grid[2, r_ego, c_ego] = 0.0
                grid[3, r_ego, c_ego] = 0.0

        # Ch0: apples
        for ap in state.apple_positions:
            grid[0, ap.row + or_off, ap.col + oc_off] = 1.0

        # Ch1: self — always at center
        grid[1, H - 1, W - 1] = 1.0

        # Ch2: other agents (exclude self)
        for j, pos in enumerate(state.agent_positions):
            if j != agent_idx:
                grid[2, pos.row + or_off, pos.col + oc_off] += 1.0

        # Ch3: actor
        actor_pos = state.agent_positions[state.actor]
        grid[3, actor_pos.row + or_off, actor_pos.col + oc_off] = 1.0

        # Scalar
        is_actor = 1.0 if agent_idx == state.actor else 0.0
        scalar = torch.tensor([is_actor], dtype=torch.float32)

        return EncoderOutput(grid=grid, scalar=scalar)

    def encode_batch_for_actions(self, state: State, agent_idx: int, after_states: list[State]) -> EncoderOutput:
        """Batch-encode after-states for ego-centric grid.

        When agent_idx IS the actor: the actor moves, so the ego center shifts
        per action — the entire grid must be rebuilt from scratch.
        When agent_idx is NOT the actor: the ego center is fixed, only the
        actor's position on Ch2/Ch3 varies.
        """
        H, W = self.env_cfg.height, self.env_cfg.width
        H_prime = 2 * H - 1
        W_prime = 2 * W - 1
        n = len(after_states)
        actor = state.actor
        is_actor = (agent_idx == actor)

        if is_actor:
            # Agent is actor and moves → ego center shifts per action.
            # Must build each grid from scratch.
            grids = torch.full((n, 4, H_prime, W_prime), -1.0, dtype=torch.float32)
            for k, s_after in enumerate(after_states):
                r_a, c_a = s_after.agent_positions[actor]
                off_r = (H - 1) - r_a
                off_c = (W - 1) - c_a
                # Zero in-bounds cells
                for r_w in range(H):
                    for c_w in range(W):
                        r_ego = r_w + off_r
                        c_ego = c_w + off_c
                        grids[k, 0, r_ego, c_ego] = 0.0
                        grids[k, 1, r_ego, c_ego] = 0.0
                        grids[k, 2, r_ego, c_ego] = 0.0
                        grids[k, 3, r_ego, c_ego] = 0.0
                # Ch0: apples
                for ap in state.apple_positions:
                    grids[k, 0, ap.row + off_r, ap.col + off_c] = 1.0
                # Ch1: self at center
                grids[k, 1, H - 1, W - 1] = 1.0
                # Ch2: others (exclude self=actor)
                for j, pos in enumerate(state.agent_positions):
                    if j != agent_idx:
                        grids[k, 2, pos.row + off_r, pos.col + off_c] += 1.0
                # Ch3: actor at center
                grids[k, 3, H - 1, W - 1] = 1.0

            scalars = torch.full((n, 1), 1.0, dtype=torch.float32)
            return EncoderOutput(grid=grids, scalar=scalars)
        else:
            # Agent is NOT actor → ego center is fixed
            r_i, c_i = state.agent_positions[agent_idx]
            or_off = (H - 1) - r_i
            oc_off = (W - 1) - c_i

            base = torch.full((4, H_prime, W_prime), -1.0, dtype=torch.float32)
            for r_w in range(H):
                for c_w in range(W):
                    r_ego = r_w + or_off
                    c_ego = c_w + oc_off
                    base[0, r_ego, c_ego] = 0.0
                    base[1, r_ego, c_ego] = 0.0
                    base[2, r_ego, c_ego] = 0.0
                    base[3, r_ego, c_ego] = 0.0

            # Ch0: apples (constant)
            for ap in state.apple_positions:
                base[0, ap.row + or_off, ap.col + oc_off] = 1.0

            # Ch1: self at center (constant)
            base[1, H - 1, W - 1] = 1.0

            # Ch2: others excluding actor (constant part)
            for j, pos in enumerate(state.agent_positions):
                if j != agent_idx and j != actor:
                    base[2, pos.row + or_off, pos.col + oc_off] += 1.0

            # Ch3 and actor's Ch2 contribution vary per action

            grids = torch.zeros(n, 4, H_prime, W_prime, dtype=torch.float32)
            for k, s_after in enumerate(after_states):
                grids[k] = base.clone()
                actor_pos = s_after.agent_positions[actor]
                grids[k, 2, actor_pos.row + or_off, actor_pos.col + oc_off] += 1.0
                grids[k, 3, actor_pos.row + or_off, actor_pos.col + oc_off] = 1.0

            scalars = torch.full((n, 1), 0.0, dtype=torch.float32)
            return EncoderOutput(grid=grids, scalar=scalars)


# ---------------------------------------------------------------------------
# Task specialization encoders (n_task_types > 1)
# ---------------------------------------------------------------------------

class TaskGridEncoder(GridEncoder):
    """Decentralized task-specialization encoding.

    Channels (T + 3 total):
      0..T-1 — task type channels: channel τ has 1.0 where type-τ task exists
      T      — self position (1.0 at agent i's cell)
      T+1    — other agents (count of agents j≠i at each cell)
      T+2    — actor position (1.0 at actor's cell)
    Scalar: is_self_actor ∈ {0, 1}. Dim = 1.
    """

    def __init__(self, env_cfg, use_vec_encode: bool = True) -> None:
        super().__init__(env_cfg)
        self._n_types = env_cfg.n_task_types
        self._use_vec = use_vec_encode

    def grid_channels(self) -> int:
        return self._n_types + 3

    def scalar_dim(self) -> int:
        return 2

    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        T = self._n_types
        h, w = self.env_cfg.height, self.env_cfg.width
        grid = torch.zeros(T + 3, h, w, dtype=torch.float32)

        if self._use_vec:
            # Task type channels (vectorized)
            if state.task_positions and state.task_types is not None:
                types_t = torch.tensor(state.task_types, dtype=torch.long)
                rows_t = torch.tensor([p.row for p in state.task_positions], dtype=torch.long)
                cols_t = torch.tensor([p.col for p in state.task_positions], dtype=torch.long)
                grid[types_t, rows_t, cols_t] = 1.0

            # Self
            r, c = state.agent_positions[agent_idx]
            grid[T, r, c] = 1.0

            # Others (vectorized scatter_add)
            other_idxs = [i for i in range(len(state.agent_positions)) if i != agent_idx]
            if other_idxs:
                o_rows = torch.tensor([state.agent_positions[i].row for i in other_idxs], dtype=torch.long)
                o_cols = torch.tensor([state.agent_positions[i].col for i in other_idxs], dtype=torch.long)
                flat_idx = o_rows * w + o_cols
                ones = torch.ones(len(other_idxs), dtype=torch.float32)
                grid[T + 1].view(-1).scatter_add_(0, flat_idx, ones)
        else:
            # Loop-based fallback (for timing baseline)
            if state.task_types is not None:
                for pos, tau in zip(state.task_positions, state.task_types):
                    grid[tau, pos.row, pos.col] = 1.0

            r, c = state.agent_positions[agent_idx]
            grid[T, r, c] = 1.0

            for i, pos in enumerate(state.agent_positions):
                if i != agent_idx:
                    grid[T + 1, pos.row, pos.col] += 1.0

        # Actor (same for both paths)
        actor_pos = state.agent_positions[state.actor]
        grid[T + 2, actor_pos.row, actor_pos.col] = 1.0

        is_actor = 1.0 if agent_idx == state.actor else 0.0
        p2_pending = 1.0 if state.phase2_pending else 0.0
        scalar = torch.tensor([is_actor, p2_pending], dtype=torch.float32)

        return EncoderOutput(grid=grid, scalar=scalar)

    def encode_batch_for_actions(self, state: State, agent_idx: int, after_states: list[State]) -> EncoderOutput:
        """Batch-encode after-states for all actions.

        Movement after-states: actor position varies, tasks constant.
        Pick after-states: actor position constant, task channels may vary.
        """
        T = self._n_types
        h, w = self.env_cfg.height, self.env_cfg.width
        n = len(after_states)
        actor = state.actor
        is_actor = (agent_idx == actor)

        if self._use_vec:
            # Build base grid with constant channels
            base = torch.zeros(T + 3, h, w, dtype=torch.float32)

            if state.task_positions and state.task_types is not None:
                types_t = torch.tensor(state.task_types, dtype=torch.long)
                rows_t = torch.tensor([p.row for p in state.task_positions], dtype=torch.long)
                cols_t = torch.tensor([p.col for p in state.task_positions], dtype=torch.long)
                base[types_t, rows_t, cols_t] = 1.0

            if is_actor:
                for j, pos in enumerate(state.agent_positions):
                    if j != agent_idx:
                        base[T + 1, pos.row, pos.col] += 1.0
            else:
                r, c = state.agent_positions[agent_idx]
                base[T, r, c] = 1.0
                for j, pos in enumerate(state.agent_positions):
                    if j != agent_idx and j != actor:
                        base[T + 1, pos.row, pos.col] += 1.0

            grids = torch.zeros(n, T + 3, h, w, dtype=torch.float32)
            for k, s_after in enumerate(after_states):
                grids[k] = base.clone()
                actor_pos = s_after.agent_positions[actor]

                if is_actor:
                    grids[k, T, actor_pos.row, actor_pos.col] = 1.0
                else:
                    grids[k, T + 1, actor_pos.row, actor_pos.col] += 1.0
                grids[k, T + 2, actor_pos.row, actor_pos.col] = 1.0

                if s_after.task_positions != state.task_positions:
                    grids[k, :T] = 0.0
                    if s_after.task_positions and s_after.task_types is not None:
                        at = torch.tensor(s_after.task_types, dtype=torch.long)
                        ar = torch.tensor([p.row for p in s_after.task_positions], dtype=torch.long)
                        ac = torch.tensor([p.col for p in s_after.task_positions], dtype=torch.long)
                        grids[k][at, ar, ac] = 1.0
        else:
            # Loop-based fallback
            grids = torch.zeros(n, T + 3, h, w, dtype=torch.float32)
            for k, s_after in enumerate(after_states):
                if s_after.task_types is not None:
                    for pos, tau in zip(s_after.task_positions, s_after.task_types):
                        grids[k, tau, pos.row, pos.col] = 1.0

                actor_pos = s_after.agent_positions[actor]
                if is_actor:
                    grids[k, T, actor_pos.row, actor_pos.col] = 1.0
                else:
                    r, c = state.agent_positions[agent_idx]
                    grids[k, T, r, c] = 1.0

                for j, pos in enumerate(s_after.agent_positions):
                    if j != agent_idx:
                        grids[k, T + 1, pos.row, pos.col] += 1.0

                grids[k, T + 2, actor_pos.row, actor_pos.col] = 1.0

        scalar_val = 1.0 if is_actor else 0.0
        scalars = torch.zeros((n, 2), dtype=torch.float32)
        scalars[:, 0] = scalar_val
        for k, s_after in enumerate(after_states):
            if s_after.phase2_pending:
                scalars[k, 1] = 1.0
        return EncoderOutput(grid=grids, scalar=scalars)

    def encode_all_agents(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode state for all N agents, returning stacked tensors.

        Returns:
            grids:   (N, C, H, W) — one grid per agent
            scalars: (N, S) — one scalar vector per agent

        O(N) — uses total_agents - self trick to avoid O(N²) inner loop.
        """
        T = self._n_types
        N = len(state.agent_positions)
        h, w = self.env_cfg.height, self.env_cfg.width
        C = T + 3

        grids = torch.zeros(N, C, h, w, dtype=torch.float32)

        # Task type channels — shared, broadcast to all N
        if state.task_positions and state.task_types is not None:
            types_t = torch.tensor(state.task_types, dtype=torch.long)
            rows_t = torch.tensor([p.row for p in state.task_positions], dtype=torch.long)
            cols_t = torch.tensor([p.col for p in state.task_positions], dtype=torch.long)
            task_grid = torch.zeros(T, h, w, dtype=torch.float32)
            task_grid[types_t, rows_t, cols_t] = 1.0
            grids[:, :T] = task_grid.unsqueeze(0)

        # Self channel (T): one-hot per agent — O(N)
        for i, pos in enumerate(state.agent_positions):
            grids[i, T, pos.row, pos.col] = 1.0

        # Others channel (T+1): total - self — O(N) via broadcast
        total_agents = torch.zeros(h, w, dtype=torch.float32)
        for pos in state.agent_positions:
            total_agents[pos.row, pos.col] += 1.0
        grids[:, T + 1] = total_agents.unsqueeze(0) - grids[:, T]

        # Actor channel (T+2) — shared, broadcast
        actor_pos = state.agent_positions[state.actor]
        grids[:, T + 2, actor_pos.row, actor_pos.col] = 1.0

        # Scalars
        scalars = torch.zeros(N, 2, dtype=torch.float32)
        scalars[state.actor, 0] = 1.0
        if state.phase2_pending:
            scalars[:, 1] = 1.0

        return grids, scalars

    def encode_all_agents_for_actions(
        self, state: State, after_states: list[State],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode all N agents × B after-states at once.

        Returns:
            grids:   (N, B, C, H, W)
            scalars: (N, B, S)

        O(N + B) — no O(N²) or O(N·B) Python loops.
        Uses total_agents - self trick for others channel.
        """
        T = self._n_types
        N = len(state.agent_positions)
        B = len(after_states)
        h, w = self.env_cfg.height, self.env_cfg.width
        C = T + 3
        actor = state.actor

        grids = torch.zeros(N, B, C, h, w, dtype=torch.float32)

        # --- Task channels (shared across N and B for movement) ---
        task_grid = torch.zeros(T, h, w, dtype=torch.float32)
        if state.task_positions and state.task_types is not None:
            types_t = torch.tensor(state.task_types, dtype=torch.long)
            rows_t = torch.tensor([p.row for p in state.task_positions], dtype=torch.long)
            cols_t = torch.tensor([p.col for p in state.task_positions], dtype=torch.long)
            task_grid[types_t, rows_t, cols_t] = 1.0
        grids[:, :, :T] = task_grid  # broadcasts (T, H, W) → (N, B, T, H, W)

        # --- Self one-hots at original positions: O(N) ---
        self_orig = torch.zeros(N, h, w, dtype=torch.float32)
        for i, pos in enumerate(state.agent_positions):
            self_orig[i, pos.row, pos.col] = 1.0
        total_orig = self_orig.sum(dim=0)  # (H, W)

        # --- Self channel (T): broadcast original positions over B ---
        grids[:, :, T] = self_orig.unsqueeze(1)  # (N, 1, H, W) → (N, B, H, W)

        # --- Others channel (T+1) ---
        # Actor: others = total - self[actor], constant across B
        grids[actor, :, T + 1] = (total_orig - self_orig[actor]).unsqueeze(0)
        # Non-actor: others_base = total - self[i] - self[actor], then + actor_new per action
        non_actor = [i for i in range(N) if i != actor]
        if non_actor:
            others_base = (total_orig.unsqueeze(0)
                           - self_orig[non_actor]
                           - self_orig[actor].unsqueeze(0))  # (N-1, H, W)
            grids[non_actor, :, T + 1] = others_base.unsqueeze(1)  # broadcast over B

        # --- Per-action adjustments: O(B) loop, no N loop ---
        for k, s_after in enumerate(after_states):
            actor_pos = s_after.agent_positions[actor]
            ar, ac_ = actor_pos.row, actor_pos.col

            # Override actor's self channel for this action
            grids[actor, k, T] = 0
            grids[actor, k, T, ar, ac_] = 1.0

            # Add actor's new position to non-actor others channels
            if non_actor:
                grids[non_actor, k, T + 1, ar, ac_] += 1.0

            # Actor channel — same for all agents
            grids[:, k, T + 2, ar, ac_] = 1.0

            # Handle pick after-states where tasks changed
            if s_after.task_positions != state.task_positions:
                grids[:, k, :T] = 0
                if s_after.task_positions and s_after.task_types is not None:
                    at = torch.tensor(s_after.task_types, dtype=torch.long)
                    a_r = torch.tensor([p.row for p in s_after.task_positions], dtype=torch.long)
                    a_c = torch.tensor([p.col for p in s_after.task_positions], dtype=torch.long)
                    task_k = torch.zeros(T, h, w, dtype=torch.float32)
                    task_k[at, a_r, a_c] = 1.0
                    grids[:, k, :T] = task_k  # broadcast

        # --- Scalars (N, B, 2) ---
        scalars = torch.zeros(N, B, 2, dtype=torch.float32)
        scalars[actor, :, 0] = 1.0
        for k, s_after in enumerate(after_states):
            if s_after.phase2_pending:
                scalars[:, k, 1] = 1.0

        return grids, scalars


class CentralizedTaskGridEncoder(GridEncoder):
    """Centralized task-specialization encoding.

    Channels (T + N + 1 total):
      0..T-1       — task type channels (same as dec)
      T..T+N-1     — per-agent position: channel T+j has 1.0 at agent j's cell
      T+N          — actor position (1.0 at actor's cell)
    Scalar: one-hot actor identity, length N.
    """

    def __init__(self, env_cfg, use_vec_encode: bool = True) -> None:
        super().__init__(env_cfg)
        self._n_types = env_cfg.n_task_types
        self._n_agents = env_cfg.n_agents
        self._use_vec = use_vec_encode

    def grid_channels(self) -> int:
        return self._n_types + self._n_agents + 1

    def scalar_dim(self) -> int:
        return self._n_agents + 1

    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        T = self._n_types
        N = self._n_agents
        h, w = self.env_cfg.height, self.env_cfg.width
        C = T + N + 1
        grid = torch.zeros(C, h, w, dtype=torch.float32)

        if self._use_vec:
            # Task type channels (vectorized)
            if state.task_positions and state.task_types is not None:
                types_t = torch.tensor(state.task_types, dtype=torch.long)
                rows_t = torch.tensor([p.row for p in state.task_positions], dtype=torch.long)
                cols_t = torch.tensor([p.col for p in state.task_positions], dtype=torch.long)
                grid[types_t, rows_t, cols_t] = 1.0
        else:
            if state.task_types is not None:
                for pos, tau in zip(state.task_positions, state.task_types):
                    grid[tau, pos.row, pos.col] = 1.0

        # Per-agent position channels (same for both paths — always loop, small N)
        for j, pos in enumerate(state.agent_positions):
            grid[T + j, pos.row, pos.col] = 1.0

        # Actor position channel
        actor_pos = state.agent_positions[state.actor]
        grid[T + N, actor_pos.row, actor_pos.col] = 1.0

        # Scalar: one-hot actor identity + phase2_pending
        scalar = torch.zeros(N + 1, dtype=torch.float32)
        scalar[state.actor] = 1.0
        if state.phase2_pending:
            scalar[N] = 1.0

        return EncoderOutput(grid=grid, scalar=scalar)

    def encode_batch_for_actions(self, state: State, agent_idx: int, after_states: list[State]) -> EncoderOutput:
        T = self._n_types
        N = self._n_agents
        h, w = self.env_cfg.height, self.env_cfg.width
        C = T + N + 1
        n = len(after_states)
        actor = state.actor

        if self._use_vec:
            # Base grid: tasks + non-actor agent channels
            base = torch.zeros(C, h, w, dtype=torch.float32)

            if state.task_positions and state.task_types is not None:
                types_t = torch.tensor(state.task_types, dtype=torch.long)
                rows_t = torch.tensor([p.row for p in state.task_positions], dtype=torch.long)
                cols_t = torch.tensor([p.col for p in state.task_positions], dtype=torch.long)
                base[types_t, rows_t, cols_t] = 1.0

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
                        at = torch.tensor(s_after.task_types, dtype=torch.long)
                        ar = torch.tensor([p.row for p in s_after.task_positions], dtype=torch.long)
                        ac = torch.tensor([p.col for p in s_after.task_positions], dtype=torch.long)
                        grids[k][at, ar, ac] = 1.0
        else:
            # Loop-based fallback
            grids = torch.zeros(n, C, h, w, dtype=torch.float32)
            for k, s_after in enumerate(after_states):
                if s_after.task_types is not None:
                    for pos, tau in zip(s_after.task_positions, s_after.task_types):
                        grids[k, tau, pos.row, pos.col] = 1.0

                for j, pos in enumerate(s_after.agent_positions):
                    grids[k, T + j, pos.row, pos.col] = 1.0

                actor_pos = s_after.agent_positions[actor]
                grids[k, T + N, actor_pos.row, actor_pos.col] = 1.0

        # Scalar: one-hot + phase2_pending (may vary across actions)
        scalar_base = torch.zeros(N + 1, dtype=torch.float32)
        scalar_base[actor] = 1.0
        scalars = scalar_base.unsqueeze(0).expand(n, -1).clone()
        for k, s_after in enumerate(after_states):
            if s_after.phase2_pending:
                scalars[k, N] = 1.0

        return EncoderOutput(grid=grids, scalar=scalars)