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
        train_only_teammates: bool = False,
    ) -> None:
        super().__init__(
            network_list=network_list, env=env, gamma=gamma,
            epsilon_schedule=epsilon_schedule, lr_schedule=lr_schedule,
            total_steps=total_steps, heuristic=heuristic, timer=timer,
            train_only_teammates=train_only_teammates,
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
        teammate_indices: list[int] | None = None,
    ) -> float:
        grids_t, scalars_t = prev
        grids_next, scalars_next = current
        rewards_t = torch.tensor(rewards, dtype=torch.float32)

        agent_mask: torch.Tensor | None = None
        if teammate_indices is not None:
            agent_mask = torch.zeros(self._n_agents, dtype=torch.float32)
            agent_mask[teammate_indices] = 1.0

        return self._bt.td_lambda_step_batched(
            grids_t, scalars_t, rewards_t, discount,
            grids_next, scalars_next,
            alpha=compute_schedule_value(self._lr_schedule, t, self._total_steps),
            agent_mask=agent_mask,
        )

    # ------------------------------------------------------------------
    # Value computation for greedy action selection
    # ------------------------------------------------------------------
    def _compute_team_values(
        self, state: State, after_states: list[State],
    ) -> list[float]:
        from orchard.trainer.timer import TimerSection
        self._timer.start(TimerSection.ACTION_ENCODE)
        grids, scalars = encoding.encode_all_agents_for_actions(state, after_states)
        self._timer.stop()
        self._timer.start(TimerSection.ACTION_FORWARD)
        values = self._bt.forward_batched(grids, scalars)  # (N, B)
        self._timer.stop()
        self._last_action_grids = grids      # (N, B, C, H, W) — kept for enc caching
        self._last_action_scalars = scalars  # (N, B, S)
        return values.sum(dim=0).tolist()

    def _cache_selected_enc(self, best_idx: int) -> None:
        """Cache the encoding of the selected after-state for reuse in train_move/train_pick."""
        self._cached_enc = (
            self._last_action_grids[:, best_idx].contiguous(),
            self._last_action_scalars[:, best_idx].contiguous(),
        )

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------
    def sync_to_cpu(self) -> None:
        self._bt.sync_to_networks()

    def load_checkpoint(self, path: str | Any) -> int | None:
        step = super().load_checkpoint(path)
        self._bt.sync_from_networks()
        return step
