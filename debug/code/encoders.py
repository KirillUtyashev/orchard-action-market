# debug/code/encoders.py
"""Encoder base classes and concrete implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class EncoderOutput:
    """Output of any encoder. Either grid, scalar, or both may be set."""
    grid:   Optional[torch.Tensor] = None   # (C, H, W) float32
    scalar: Optional[torch.Tensor] = None   # (D,)      float32

    def __post_init__(self):
        if self.grid is None and self.scalar is None:
            raise ValueError("EncoderOutput must have at least one of grid or scalar")


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class BaseEncoder(ABC):
    """Base for all encoders. Operates on raw state dicts."""

    def __init__(self, grid_h: int, grid_w: int, n_agents: int):
        self.H = grid_h
        self.W = grid_w
        self.n_agents = n_agents

    @abstractmethod
    def encode(self, state: dict, agent_idx: int) -> EncoderOutput:
        raise NotImplementedError

    @abstractmethod
    def output_dim(self) -> int:
        """Flat dimension consumed by the network (grid channels or scalar dim)."""
        raise NotImplementedError


class GridEncoder(BaseEncoder, ABC):
    """Base for encoders that produce a (C, H, W) grid + optional scalar."""

    @abstractmethod
    def grid_channels(self) -> int:
        raise NotImplementedError

    def scalar_dim(self) -> int:
        return 0

    def output_dim(self) -> int:
        return self.grid_channels()


# ---------------------------------------------------------------------------
# Decentralized encoders
# ---------------------------------------------------------------------------

class DecGridEncoder(GridEncoder):
    """Grid encoder for decentralized critic.

    Channels:
      0 — apples
      1 — self position
      2 — other agents (excludes self and actor)
      3 — actor position
    Scalar: [is_actor]
    """

    def grid_channels(self) -> int:
        return 4

    def scalar_dim(self) -> int:
        return 1

    def encode(self, state: dict, agent_idx: int) -> EncoderOutput:
        apples      = state["apples"]       # (H, W) ndarray
        agent_pos   = state["agent_positions"]  # list of (r, c)
        actor_idx   = int(state["actor_id"])

        grid = torch.zeros(4, self.H, self.W, dtype=torch.float32)

        # Ch 0: apples
        grid[0] = torch.from_numpy((apples >= 1).astype(np.float32))

        # Ch 1: self
        sr, sc = agent_pos[agent_idx]
        grid[1, sr, sc] = 1.0

        # Ch 2: others (not self, not actor)
        for i, (r, c) in enumerate(agent_pos):
            if i != agent_idx and i != actor_idx:
                grid[2, r, c] += 1.0

        # Ch 3: actor
        ar, ac = agent_pos[actor_idx]
        grid[3, ar, ac] = 1.0

        is_actor = 1.0 if agent_idx == actor_idx else 0.0
        return EncoderOutput(grid=grid, scalar=torch.tensor([is_actor]))


class DecEntityEncoder(BaseEncoder):
    """Entity/MLP encoder for decentralized critic (Design A, self-centered).

    Feature vector:
      Scalars:          [actor_is_self, actor_apple_dist]
      Actor block:      [dx, dy, dist]            (relative to self)
      Other agents:     [dx, dy, dist] * (N-1)    (relative to self, by id)
      Top-K apples rel to SELF:  [dx, dy, dist, mask] * K
      Top-K apples rel to ACTOR: [dx, dy, dist, mask] * K
    """

    def __init__(self, grid_h: int, grid_w: int, n_agents: int, k: int):
        super().__init__(grid_h, grid_w, n_agents)
        self.k = k

    def output_dim(self) -> int:
        return 2 + 3 + 3 * (self.n_agents - 1) + 4 * self.k + 4 * self.k

    def encode(self, state: dict, agent_idx: int) -> EncoderOutput:
        apples    = state["apples"]
        agent_pos = state["agent_positions"]
        actor_idx = int(state["actor_id"])

        dmax   = float(np.sqrt((self.W - 1)**2 + (self.H - 1)**2)) or 1.0
        dnx, dny = max(self.W - 1, 1), max(self.H - 1, 1)

        def rel(r_from, c_from, r_to, c_to):
            dx, dy = c_to - c_from, r_to - r_from
            return dx / dnx, dy / dny, np.sqrt(dx*dx + dy*dy) / dmax

        sr, sc = agent_pos[agent_idx]
        ar, ac = agent_pos[actor_idx]

        apple_rc = np.argwhere(apples > 0)  # (M, 2)

        if len(apple_rc) == 0:
            actor_apple_dist = 0.0
            topk_self = topk_actor = np.empty((0, 2), dtype=np.int64)
        else:
            rs, cs = apple_rc[:, 0], apple_rc[:, 1]
            d2_self  = (cs - sc)**2 + (rs - sr)**2
            d2_actor = (cs - ac)**2 + (rs - ar)**2
            actor_apple_dist = float(np.sqrt(d2_actor.min())) / dmax
            topk_self  = apple_rc[np.lexsort((rs - sr, cs - sc, d2_self)) [:self.k]]
            topk_actor = apple_rc[np.lexsort((rs - ar, cs - ac, d2_actor))[:self.k]]

        feats = []
        feats.append([1.0 if agent_idx == actor_idx else 0.0, actor_apple_dist])
        feats.append(list(rel(sr, sc, ar, ac)))

        for i, (r, c) in enumerate(agent_pos):
            if i != agent_idx:
                feats.append(list(rel(sr, sc, r, c)))

        for idx in range(self.k):
            if idx < len(topk_self):
                r, c = int(topk_self[idx, 0]), int(topk_self[idx, 1])
                feats.append([*rel(sr, sc, r, c), 1.0])
            else:
                feats.append([0.0, 0.0, 0.0, 0.0])

        for idx in range(self.k):
            if idx < len(topk_actor):
                r, c = int(topk_actor[idx, 0]), int(topk_actor[idx, 1])
                feats.append([*rel(ar, ac, r, c), 1.0])
            else:
                feats.append([0.0, 0.0, 0.0, 0.0])

        flat = np.concatenate([np.array(f, dtype=np.float32) for f in feats])
        return EncoderOutput(scalar=torch.from_numpy(flat))


# ---------------------------------------------------------------------------
# Centralized encoders
# ---------------------------------------------------------------------------

class CenGridEncoder(GridEncoder):
    """Grid encoder for centralized critic.

    Channels:
      0 — apples
      1 — actor position
      2 — all other agents (excludes actor)
      3 — scalars pinned to pixels:
            [0,0] = actor_id_norm
            [0,1] = apple_under_actor
    agent_idx is ignored.
    """

    def grid_channels(self) -> int:
        return 3

    def encode(self, state: dict, agent_idx: int) -> EncoderOutput:
        apples    = state["apples"]
        agent_pos = state["agent_positions"]
        actor_idx = int(state["actor_id"])

        grid = torch.zeros(3, self.H, self.W, dtype=torch.float32)

        grid[0] = torch.from_numpy((apples >= 1).astype(np.float32))

        ar, ac = agent_pos[actor_idx]
        grid[1, ar, ac] = 1.0

        for i, (r, c) in enumerate(agent_pos):
            grid[2, r, c] += 1.0

        return EncoderOutput(grid=grid)


class CenEntityEncoder(BaseEncoder):
    """Entity/MLP encoder for centralized critic (actor-centric).

    Feature vector:
      Scalars:    [actor_id_norm, apple_under_actor]
      All agents: [dx, dy, dist, is_actor] * N  (relative to actor, by id)
      Top-K apples relative to ACTOR: [dx, dy, dist, mask] * K
    """

    def __init__(self, grid_h: int, grid_w: int, n_agents: int, k: int):
        super().__init__(grid_h, grid_w, n_agents)
        self.k = k

    def output_dim(self) -> int:
        return 2 + 4 * self.n_agents + 4 * self.k

    def encode(self, state: dict, agent_idx: int) -> EncoderOutput:
        apples    = state["apples"]
        agent_pos = state["agent_positions"]
        actor_idx = int(state["actor_id"])

        dmax   = float(np.sqrt((self.W - 1)**2 + (self.H - 1)**2)) or 1.0
        dnx, dny = max(self.W - 1, 1), max(self.H - 1, 1)

        def rel(r_from, c_from, r_to, c_to):
            dx, dy = c_to - c_from, r_to - r_from
            return dx / dnx, dy / dny, np.sqrt(dx*dx + dy*dy) / dmax

        ar, ac = agent_pos[actor_idx]
        apple_rc = np.argwhere(apples > 0)

        feats = []
        feats.append([
            float(actor_idx) / float(max(self.n_agents - 1, 1)),
            1.0 if apples[ar, ac] > 0 else 0.0,
            ])

        for i, (r, c) in enumerate(agent_pos):
            feats.append([*rel(ar, ac, r, c), 1.0 if i == actor_idx else 0.0])

        if len(apple_rc) == 0:
            topk = np.empty((0, 2), dtype=np.int64)
        else:
            rs, cs = apple_rc[:, 0], apple_rc[:, 1]
            d2     = (cs - ac)**2 + (rs - ar)**2
            topk   = apple_rc[np.lexsort((rs - ar, cs - ac, d2))[:self.k]]

        for idx in range(self.k):
            if idx < len(topk):
                r, c = int(topk[idx, 0]), int(topk[idx, 1])
                feats.append([*rel(ar, ac, r, c), 1.0])
            else:
                feats.append([0.0, 0.0, 0.0, 0.0])

        flat = np.concatenate([np.array(f, dtype=np.float32) for f in feats])
        return EncoderOutput(scalar=torch.from_numpy(flat))


class CenConcatEncoder(BaseEncoder):
    """Centralized encoder: concatenates per-agent dec encodings.

    Wraps any scalar BaseEncoder, calls it for each agent, concatenates.
    agent_idx is ignored.
    """

    def __init__(self, dec_encoder: BaseEncoder):
        super().__init__(dec_encoder.H, dec_encoder.W, dec_encoder.n_agents)
        self.dec = dec_encoder

    def output_dim(self) -> int:
        return self.n_agents * self.dec.output_dim()

    def encode(self, state: dict, agent_idx: int) -> EncoderOutput:
        parts = [self.dec.encode(state, i).scalar for i in range(self.n_agents)]
        return EncoderOutput(scalar=torch.cat(parts))
