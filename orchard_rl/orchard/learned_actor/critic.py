"""Team-value critic interface for the learned actor.

The actor depends only on `team_values(state, after_states)`. The decentralized
critic returns sum_i V_i over per-agent networks, the centralized critic returns
V_cen from its single network. Both are trained with backward-view TD(lambda)
on the same underlying objective: sum_i r_i is the team reward, so
sum_i V_i and V_cen estimate the same quantity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

import orchard.encoding as encoding
from orchard.datatypes import EncoderOutput, State
from orchard.encoding.grid import EverythingEncoder
from orchard.model import ValueNetwork


class TeamCritic(ABC):
    """Critic over afterstates, queried as a single team value."""

    def __init__(self, networks: list[ValueNetwork], reward_scale: float = 1.0) -> None:
        self.networks = networks
        # All rewards seen by the critic (TD targets and Q) are multiplied by this
        # constant. A positive constant does not change the optimal policy, and keeps
        # TD updates stable when raw rewards are large (e.g. sigma_b = 8).
        self.reward_scale = float(reward_scale)

    @abstractmethod
    def _encoder_agent_index(self, net_idx: int) -> int:
        """Agent index passed to the encoder for network `net_idx`."""

    @abstractmethod
    def targets(self, rewards: tuple[float, ...]) -> tuple[float, ...]:
        """Per-network TD reward for one transition's per-agent rewards."""

    def encode(self, state: State) -> list[EncoderOutput]:
        return [
            encoding.encode(state, self._encoder_agent_index(i))
            for i in range(len(self.networks))
        ]

    def team_values(self, state: State, after_states: list[State]) -> torch.Tensor:
        """Team value of each afterstate, shape (B,). Never carries gradients."""
        shared = isinstance(encoding.get_encoder(), EverythingEncoder)
        total: torch.Tensor | None = None
        batch_enc: EncoderOutput | None = None
        with torch.no_grad():
            for i, net in enumerate(self.networks):
                if batch_enc is None or not shared:
                    batch_enc = encoding.encode_batch_for_actions(
                        state, self._encoder_agent_index(i), after_states,
                    )
                vals = net(batch_enc).reshape(-1)
                total = vals if total is None else total + vals
        assert total is not None
        return total.detach()

    def team_value_of(self, encs: list[EncoderOutput]) -> float:
        """Team value of one already-encoded afterstate."""
        with torch.no_grad():
            return float(sum(net(enc).item() for net, enc in zip(self.networks, encs)))

    def td_step(
        self,
        prev: list[EncoderOutput],
        rewards: tuple[float, ...],
        discount: float,
        current: list[EncoderOutput],
        alpha: float,
    ) -> float:
        """TD(lambda) update of every network; returns sum of squared deltas."""
        targets = self.targets(rewards)
        total = 0.0
        for idx, net in enumerate(self.networks):
            delta = net.td_step(
                s_enc=prev[idx],
                reward=float(targets[idx]),
                discount=discount,
                s_next_enc=current[idx],
                alpha=alpha,
            )
            total += delta ** 2
        return total

    def reset_traces(self) -> None:
        for net in self.networks:
            net.reset_traces()


class DecentralizedTeamCritic(TeamCritic):
    """One network V_i per agent, trained on the individual reward r_i."""

    def _encoder_agent_index(self, net_idx: int) -> int:
        return net_idx

    def targets(self, rewards: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(self.reward_scale * float(r) for r in rewards)


class CentralizedTeamCritic(TeamCritic):
    """A single network V_cen, trained on the team reward sum_i r_i."""

    def __init__(self, networks: list[ValueNetwork], reward_scale: float = 1.0) -> None:
        if len(networks) != 1:
            raise ValueError("Centralized critic expects exactly one network.")
        super().__init__(networks, reward_scale)

    def _encoder_agent_index(self, net_idx: int) -> int:
        return 0

    def targets(self, rewards: tuple[float, ...]) -> tuple[float, ...]:
        return (self.reward_scale * float(sum(rewards)),)


def build_team_critic(networks: list[ValueNetwork], reward_scale: float = 1.0) -> TeamCritic:
    """Pick the critic implementation from the network count (set by learning_type)."""
    if len(networks) == 1:
        return CentralizedTeamCritic(networks, reward_scale)
    return DecentralizedTeamCritic(networks, reward_scale)


__all__ = [
    "TeamCritic",
    "DecentralizedTeamCritic",
    "CentralizedTeamCritic",
    "build_team_critic",
]
