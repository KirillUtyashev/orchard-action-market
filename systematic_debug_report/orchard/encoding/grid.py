"""Grid encoder: multi-channel (C, H, W) tensor encoding."""

from __future__ import annotations

import torch

from orchard.encoding.base import GridEncoder
from orchard.datatypes import EncoderOutput, State


class BasicGridEncoder(GridEncoder):
    """Multi-channel grid encoding.

    Channels:
      0 — self position (1 at own cell)
      1 — other agents (1 per agent; may sum >1 if overlapping)
      2 — apples (1 per apple)
      3 — actor marker (1 at actor's cell)
    """

    def grid_channels(self) -> int:
        return 4

    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        h, w = self.env_cfg.height, self.env_cfg.width
        grid = torch.zeros(4, h, w, dtype=torch.float32)

        # Self
        r, c = state.agent_positions[agent_idx]
        grid[0, r, c] = 1.0

        # Other agents
        for i, pos in enumerate(state.agent_positions):
            if i != agent_idx:
                grid[1, pos.row, pos.col] += 1.0

        # Apples
        for ap in state.apple_positions:
            grid[2, ap.row, ap.col] = 1.0

        # Actor
        actor_pos = state.agent_positions[state.actor]
        grid[3, actor_pos.row, actor_pos.col] = 1.0

        return EncoderOutput(grid=grid)
