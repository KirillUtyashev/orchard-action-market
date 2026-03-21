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