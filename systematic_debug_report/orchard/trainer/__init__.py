"""Trainer package: create_trainer factory."""

from __future__ import annotations

import torch

from orchard.batched_training import BatchedTrainer
from orchard.datatypes import ExperimentConfig
from orchard.env.base import BaseEnv
from orchard.enums import AlgorithmName
from orchard.model import create_actor_networks, create_networks
from orchard.trainer.base import TrainerBase
from orchard.trainer.timer import Timer


def create_trainer(
    cfg: ExperimentConfig,
    env: BaseEnv,
) -> TrainerBase:
    """Create the appropriate trainer based on config.

    use_gpu=True  → GpuTrainer (BatchedTrainer + vmap, works for any n_networks)
    use_gpu=False → CpuTrainer (sequential forwards, works for any n_networks)
    """
    gpu_sync = cfg.train.use_gpu and torch.cuda.is_available()
    timer = Timer(enabled=cfg.logging.timing_csv_freq > 0, gpu_sync=gpu_sync)

    if cfg.train.algorithm.name == AlgorithmName.ACTOR_CRITIC:
        critics = create_networks(cfg.model, cfg.env, cfg.train)
        actor_model_cfg = cfg.actor_model or cfg.model
        actors = create_actor_networks(actor_model_cfg, cfg.env, cfg.train)
        if cfg.train.use_gpu:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            for actor in actors:
                actor.to(device)
            bt = BatchedTrainer(critics, td_lambda=cfg.train.td_lambda, device=device)
            print(f"ActorCriticGpuTrainer: {len(critics)} critics on {device}")
            if device == "cuda":
                alloc = torch.cuda.memory_allocated() / 1024**2
                total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                print(f"  VRAM: {alloc:.0f}MB / {total:.0f}MB ({alloc/total*100:.1f}%)")
            from orchard.trainer.actor_critic import ActorCriticGpuTrainer
            return ActorCriticGpuTrainer(
                critic_networks=critics,
                actor_networks=actors,
                bt=bt,
                env=env,
                gamma=cfg.env.gamma,
                critic_lr_schedule=cfg.train.lr,
                actor_lr_schedule=cfg.train.actor_lr or cfg.train.lr,
                total_steps=cfg.train.total_steps,
                heuristic=cfg.train.heuristic,
                freeze_critic=cfg.train.freeze_critic,
                following_rates_cfg=cfg.train.following_rates,
                influencer_cfg=cfg.train.influencer,
                comm_only_teammates=cfg.train.comm_only_teammates,
                timer=timer,
                warmup_steps=cfg.train.warmup_steps,
            )
        from orchard.trainer.actor_critic import ActorCriticCpuTrainer
        print(f"ActorCriticCpuTrainer: {len(critics)} critics on CPU")
        return ActorCriticCpuTrainer(
            critic_networks=critics,
            actor_networks=actors,
            env=env,
            gamma=cfg.env.gamma,
            critic_lr_schedule=cfg.train.lr,
            actor_lr_schedule=cfg.train.actor_lr or cfg.train.lr,
            total_steps=cfg.train.total_steps,
            heuristic=cfg.train.heuristic,
            freeze_critic=cfg.train.freeze_critic,
            following_rates_cfg=cfg.train.following_rates,
            influencer_cfg=cfg.train.influencer,
            comm_only_teammates=cfg.train.comm_only_teammates,
            timer=timer,
            warmup_steps=cfg.train.warmup_steps,
        )

    networks = create_networks(cfg.model, cfg.env, cfg.train)
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
            reward_scale=cfg.train.algorithm.reward_scale,
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
            reward_scale=cfg.train.algorithm.reward_scale,
            timer=timer,
        )
