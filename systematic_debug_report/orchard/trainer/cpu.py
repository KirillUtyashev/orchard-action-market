"""CPU trainer: sequential forward passes for any number of networks."""

from __future__ import annotations

from typing import Any

import torch

import orchard.encoding as encoding
from orchard.datatypes import EncoderOutput, ScheduleConfig, State
from orchard.enums import Heuristic
from orchard.env.base import BaseEnv
from orchard.model import ValueNetwork
from orchard.schedule import compute_schedule_value
from orchard.trainer.timer import Timer
from orchard.trainer.value_base import ValueTrainerBase


class CpuTrainer(ValueTrainerBase):
    """CPU training with sequential forward passes.

    Supports both centralized (1 network) and decentralized (N networks).
    """

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def _encode_all(self, state: State) -> list[EncoderOutput]:
        if self._centralized:
            return [encoding.encode(state, 0)]
        return [encoding.encode(state, i) for i in range(self._n_agents)]

    # ------------------------------------------------------------------
    # TD step
    # ------------------------------------------------------------------
    def _td_step(
        self, prev: Any, rewards: tuple[float, ...],
        discount: float, current: Any, t: int,
        teammate_indices: list[int] | None = None,
    ) -> float:
        s_encs: list[EncoderOutput] = prev
        s_next_encs: list[EncoderOutput] = current

        alpha = compute_schedule_value(self._lr_schedule, t, self._total_steps)
        indices = teammate_indices if teammate_indices is not None else range(self._n_networks)
        total_loss = 0.0
        for i in indices:
            delta = self._networks_list[i].td_step(
                s_enc=s_encs[i], reward=rewards[i],
                discount=discount, s_next_enc=s_next_encs[i],
                alpha=alpha,
            )
            total_loss += delta ** 2
        return total_loss

    # ------------------------------------------------------------------
    # Value computation for greedy action selection
    # ------------------------------------------------------------------
    def _compute_team_values(
        self, state: State, after_states: list[State],
    ) -> list[float]:
        n_actions = len(after_states)
        team_values = [0.0] * n_actions
        with torch.no_grad():
            for i, net in enumerate(self._networks_list):
                agent_idx = 0 if self._centralized else i
                batch_enc = encoding.encode_batch_for_actions(state, agent_idx, after_states)
                vals = net(batch_enc)
                for k in range(n_actions):
                    team_values[k] += vals[k].item()
        return team_values

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------
    def sync_to_cpu(self) -> None:
        pass  # already on CPU
