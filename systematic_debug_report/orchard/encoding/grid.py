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