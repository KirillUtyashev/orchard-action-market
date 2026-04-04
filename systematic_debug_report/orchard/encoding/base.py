"""Encoder base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from orchard.datatypes import EncoderOutput, EnvConfig, State


class BaseEncoder(ABC):
    """Base encoder. Every encoder produces scalar features."""

    def __init__(self, env_cfg: EnvConfig) -> None:
        self.env_cfg = env_cfg

    @abstractmethod
    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        """Encode a single state for a specific agent (used by CPU Trainer).
        
        Returns:
            EncoderOutput containing:
            - grid: shape (C, H, W)
            - scalar: shape (S)
        """
        ...

    @abstractmethod
    def scalar_dim(self) -> int:
        ...

    def grid_channels(self) -> int:
        return 0

    def grid_height(self) -> int:
        return self.env_cfg.height

    def grid_width(self) -> int:
        return self.env_cfg.width

    def encode_batch_for_actions(self, state: State, agent_idx: int, after_states: list[State]) -> EncoderOutput:
        """Batch encode multiple possible actions for ONE agent (used by CPU Trainer). Used for action selection in value policy learning.
        
        Args:
            state: The current state before the action.
            agent_idx: The agent making the decision.
            after_states: A list of B possible states resulting from B different actions.
            
        Returns:
            EncoderOutput containing:
            - grid: shape (B, C, H, W)
            - scalar: shape (B, S)
        """
        raise NotImplementedError(f"{type(self).__name__} does not support encode_batch_for_actions")

    def encode_all_agents(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the current state for ALL N networks simultaneously (used by GPU Trainer).
        
        Returns:
            grids: shape (N, C, H, W)
            scalars: shape (N, S)
        """
        raise NotImplementedError(f"{type(self).__name__} does not support encode_all_agents")

    def encode_all_agents_for_actions(self, state: State, after_states: list[State]) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch encode multiple actions for ALL N networks simultaneously (used by GPU Trainer).
        
        Args:
            state: The current state before the action.
            after_states: A list of B possible states resulting from B different actions.
            
        Returns:
            grids: shape (N, B, C, H, W)
            scalars: shape (N, B, S)
        """
        raise NotImplementedError(f"{type(self).__name__} does not support encode_all_agents_for_actions")


class GridEncoder(BaseEncoder):
    """Base for encoders that produce grid + scalar output."""

    @abstractmethod
    def grid_channels(self) -> int:
        ...
