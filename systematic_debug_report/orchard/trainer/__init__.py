"""Trainer package: create_trainer factory."""

from __future__ import annotations

import torch

from orchard.batched_training import BatchedTrainer
from orchard.datatypes import ExperimentConfig
from orchard.env.base import BaseEnv
from orchard.model import ValueNetwork
from orchard.trainer.base import TrainerBase
from orchard.trainer.timer import Timer


def create_trainer(
    cfg: ExperimentConfig,
    networks: list[ValueNetwork],
    env: BaseEnv,
) -> TrainerBase:
    """Create the appropriate trainer based on config.

    use_gpu=True  → GpuTrainer (BatchedTrainer + vmap, works for any n_networks)
    use_gpu=False → CpuTrainer (sequential forwards, works for any n_networks)
    """
    gpu_sync = cfg.train.use_gpu and torch.cuda.is_available()
    timer = Timer(enabled=False, gpu_sync=gpu_sync)  # enable via code if needed

    if cfg.train.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bt = BatchedTrainer(
            networks, td_lambda=cfg.train.td_lambda, device=device,
        )
        print(f"GpuTrainer: {len(networks)} networks on {device}")
        if device == "cuda":
            alloc = torch.cuda.memory_allocated() / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            print(f"  VRAM: {alloc:.0f}MB / {total:.0f}MB ({alloc/total*100:.1f}%)")

        from orchard.trainer.gpu import GpuTrainer
        return GpuTrainer(
            network_list=networks,
            bt=bt,
            env=env,
            gamma=cfg.env.gamma,
            epsilon_schedule=cfg.train.epsilon,
            lr_schedule=cfg.train.lr,
            total_steps=cfg.train.total_steps,
            heuristic=cfg.train.heuristic,
            comm_weight=cfg.train.comm_weight,
            timer=timer,
        )
    else:
        from orchard.trainer.cpu import CpuTrainer
        print(f"CpuTrainer: {len(networks)} networks on CPU")
        return CpuTrainer(
            network_list=networks,
            env=env,
            gamma=cfg.env.gamma,
            epsilon_schedule=cfg.train.epsilon,
            lr_schedule=cfg.train.lr,
            total_steps=cfg.train.total_steps,
            heuristic=cfg.train.heuristic,
            comm_weight=cfg.train.comm_weight,
            timer=timer,
        )
