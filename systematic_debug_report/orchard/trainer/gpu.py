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
        comm_weight: float = 0.0,
        timer: Timer | None = None,
    ) -> None:
        super().__init__(
            network_list=network_list, env=env, gamma=gamma,
            epsilon_schedule=epsilon_schedule, lr_schedule=lr_schedule,
            total_steps=total_steps, heuristic=heuristic,
            comm_weight=comm_weight, timer=timer,
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

        if self._comm_weight > 0.0 and self._n_networks > 1:
            w = self._comm_weight
            with torch.no_grad():
                v_s = self._bt.forward_single_batched(grids_t, scalars_t)
                v_next = self._bt.forward_single_batched(grids_next, scalars_next)
            
            team_v_s = v_s.sum()
            team_v_next = v_next.sum()
            team_r = rewards_t.sum()
            # below is just r_i' = r_i + w * sum_{j != i} [ r_j + discount * V_j(s') - V_j(s) ] in a different form
            rewards_t = rewards_t.to(self._bt.device)
            rewards_t = (
                rewards_t 
                + w * (team_r - rewards_t)
                + discount * w * (team_v_next - v_next)
                - w * (team_v_s - v_s)
            )

        return self._bt.td_lambda_step_batched(
            grids_t, scalars_t, rewards_t, discount,
            grids_next, scalars_next,
            alpha=compute_schedule_value(self._lr_schedule, t, self._total_steps),
        )

    # ------------------------------------------------------------------
    # Value computation for greedy action selection
    # ------------------------------------------------------------------
    def _compute_team_values(
        self, state: State, after_states: list[State], actor: int,
    ) -> list[float]:
        grids, scalars = encoding.encode_all_agents_for_actions(state, after_states)
        values = self._bt.forward_batched(grids, scalars)  # (N, B)
        n_nets = values.size(0)

        if n_nets > 1 and self._comm_weight < 1.0:
            weights = torch.full((n_nets, 1), self._comm_weight, device=values.device)
            weights[actor] = 1.0
            team_values = (values * weights).sum(dim=0)
        else: # centralized case
            team_values = values.sum(dim=0)

        return team_values.tolist()

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------
    def sync_to_cpu(self) -> None:
        self._bt.sync_to_networks()

    def load_checkpoint(self, path: str | Any) -> int | None:
        step = super().load_checkpoint(path)
        self._bt.sync_from_networks()
        return step
