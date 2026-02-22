"""Relative encoders: sorted relative distances from agent's perspective."""

from __future__ import annotations

import torch

from orchard.encoding.base import BaseEncoder
from orchard.datatypes import EncoderOutput, EnvConfig, State


class RelativeEncoder(BaseEncoder):
    """Sorted relative positions with absolute pos and actor indicator.

    Feature vector:
      For each apple (sorted by Manhattan dist, tie-break position):
          delta_row / height, delta_col / width
      For each OTHER agent (sorted by Manhattan dist, tie-break position):
          delta_row / height, delta_col / width, time_until_act / n_agents
      Scalars: time_until_self / n_agents, own_row / height, own_col / width, is_actor
    """

    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        my_r, my_c = state.agent_positions[agent_idx]
        h, w = self.env_cfg.height, self.env_cfg.width
        n = self.env_cfg.n_agents

        # --- apples ---
        apple_feats: list[tuple[int, tuple[int, int], float, float]] = []
        for ap in state.apple_positions:
            dr = ap.row - my_r
            dc = ap.col - my_c
            dist = abs(dr) + abs(dc)
            apple_feats.append((dist, (ap.row, ap.col), dr / h, dc / w))
        apple_feats.sort(key=lambda x: (x[0], x[1]))

        # --- other agents ---
        other_feats: list[tuple[int, tuple[int, int], float, float, float]] = []
        for i, pos in enumerate(state.agent_positions):
            if i == agent_idx:
                continue
            dr = pos.row - my_r
            dc = pos.col - my_c
            dist = abs(dr) + abs(dc)
            time_until = (i - state.actor) % n / n
            other_feats.append((dist, (pos.row, pos.col), dr / h, dc / w, time_until))
        other_feats.sort(key=lambda x: (x[0], x[1]))

        # --- assemble ---
        features: list[float] = []
        for _, _, dr, dc in apple_feats:
            features.extend([dr, dc])
        for _, _, dr, dc, tu in other_feats:
            features.extend([dr, dc, tu])

        my_time_until = (agent_idx - state.actor) % n / n
        features.append(my_time_until)
        features.append(my_r / h)
        features.append(my_c / w)
        features.append(1.0 if agent_idx == state.actor else 0.0)

        return EncoderOutput(scalar=torch.tensor(features, dtype=torch.float32))

    def input_dim(self) -> int:
        n_apples = self.env_cfg.n_apples
        n_agents = self.env_cfg.n_agents
        return n_apples * 2 + (n_agents - 1) * 3 + 4

class RelativeKEncoder(BaseEncoder):
    """K-nearest apples with is_present flag, plus relative agent features.

    Feature vector:
      For each of K apple slots (sorted by distance, padded if fewer):
          delta_row/h, delta_col/w, is_present
      For each OTHER agent (sorted by distance):
          delta_row/h, delta_col/w, time_until_act/n_agents
      Scalars: time_until_self/n_agents, own_row/h, own_col/w, is_actor
    """

    def __init__(self, env_cfg, k_nearest: int) -> None:
        super().__init__(env_cfg)
        self.k_nearest = k_nearest

    def encode(self, state, agent_idx):
        my_r, my_c = state.agent_positions[agent_idx]
        h, w = self.env_cfg.height, self.env_cfg.width
        n = self.env_cfg.n_agents

        # --- apples (same sorting as RelativeEncoder) ---
        apple_feats = []
        for ap in state.apple_positions:
            dr = ap.row - my_r
            dc = ap.col - my_c
            dist = abs(dr) + abs(dc)
            apple_feats.append((dist, (ap.row, ap.col), dr / h, dc / w))
        apple_feats.sort(key=lambda x: (x[0], x[1]))

        # --- other agents (identical to RelativeEncoder) ---
        other_feats = []
        for i, pos in enumerate(state.agent_positions):
            if i == agent_idx:
                continue
            dr = pos.row - my_r
            dc = pos.col - my_c
            dist = abs(dr) + abs(dc)
            time_until = (i - state.actor) % n / n
            other_feats.append((dist, (pos.row, pos.col), dr / h, dc / w, time_until))
        other_feats.sort(key=lambda x: (x[0], x[1]))

        # --- assemble ---
        features = []
        for j in range(self.k_nearest):
            if j < len(apple_feats):
                _, _, dr, dc = apple_feats[j]
                features.extend([dr, dc, 1.0])
            else:
                features.extend([0.0, 0.0, 0.0])

        for _, _, dr, dc, tu in other_feats:
            features.extend([dr, dc, tu])

        my_time_until = (agent_idx - state.actor) % n / n
        features.append(my_time_until)
        features.append(my_r / h)
        features.append(my_c / w)
        features.append(1.0 if agent_idx == state.actor else 0.0)

        return EncoderOutput(scalar=torch.tensor(features, dtype=torch.float32))

    def input_dim(self):
        return self.k_nearest * 3 + (self.env_cfg.n_agents - 1) * 3 + 4
