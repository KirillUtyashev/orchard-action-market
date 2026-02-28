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
            grid[0, ap.row, ap.col] = 1.0

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
            grid[0, ap.row, ap.col] = 1.0

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