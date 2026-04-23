"""GPU trainer: vmap-batched training and action selection via BatchedTrainer."""

from __future__ import annotations

from typing import Any

import torch

import orchard.encoding as encoding
from orchard.batched_training import BatchedTrainer
from orchard.datatypes import ScheduleConfig, State
from orchard.enums import Heuristic
from orchard.env.base import BaseEnv
from orchard.model import ValueNetwork
from orchard.schedule import compute_schedule_value
from orchard.trainer.timer import Timer
from orchard.trainer.value_base import ValueTrainerBase


class GpuTrainer(ValueTrainerBase):
    """Batched GPU training for N networks (decentralized or centralized).

    Uses BatchedTrainer (vmap over N networks) for both training and
    action selection.
    """

    def __init__(
        self,
        network_list: list[ValueNetwork],
        bt: BatchedTrainer,
        env: BaseEnv,
        gamma: float,
        epsilon_schedule: ScheduleConfig,
        lr_schedule: ScheduleConfig,
        total_steps: int,
        heuristic: Heuristic,
        timer: Timer | None = None,
    ) -> None:
        super().__init__(
            network_list=network_list, env=env, gamma=gamma,
            epsilon_schedule=epsilon_schedule, lr_schedule=lr_schedule,
            total_steps=total_steps, heuristic=heuristic, timer=timer,
        )
        self._bt = bt

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def _encode_all(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        return encoding.encode_all_agents(state)

    # ------------------------------------------------------------------
    # TD step
    # ------------------------------------------------------------------
    def _td_step(
        self, prev: Any, rewards: tuple[float, ...],
        discount: float, current: Any, t: int,
    ) -> float:
        grids_t, scalars_t = prev
        grids_next, scalars_next = current
        rewards_t = torch.tensor(rewards, dtype=torch.float32)

        return self._bt.td_lambda_step_batched(
            grids_t, scalars_t, rewards_t, discount,
            grids_next, scalars_next,
            alpha=compute_schedule_value(self._lr_schedule, t, self._total_steps),
        )

    # ------------------------------------------------------------------
    # Value computation for greedy action selection
    # ------------------------------------------------------------------
    def _compute_team_values(
        self, state: State, after_states: list[State],
    ) -> list[float]:
        grids, scalars = encoding.encode_all_agents_for_actions(state, after_states)
        values = self._bt.forward_batched(grids, scalars)  # (N, B)
        return values.sum(dim=0).tolist()

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------
    def sync_to_cpu(self) -> None:
        self._bt.sync_to_networks()

    def load_checkpoint(self, path: str | Any) -> int | None:
        step = super().load_checkpoint(path)
        self._bt.sync_from_networks()
        return step
