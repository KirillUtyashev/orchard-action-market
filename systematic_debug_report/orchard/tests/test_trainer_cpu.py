"""Tests for CpuTrainer (sequential forward passes)."""

import pytest
import torch

from orchard.enums import EncoderType, Heuristic, LearningType, DespawnMode
from orchard.datatypes import (
    EnvConfig, ModelConfig, ScheduleConfig, StochasticConfig,
    State, Grid, TrainConfig, StoppingConfig
)
from orchard.env.stochastic import StochasticEnv
from orchard.model import create_networks
import orchard.encoding as encoding
from orchard.trainer.cpu import CpuTrainer
from orchard.seed import set_all_seeds


def _setup_cpu_trainer(n_agents=2):
    set_all_seeds(0)
    env_cfg = EnvConfig(
        height=3, width=3, n_agents=n_agents, n_tasks=1,
        gamma=0.99, n_task_types=1, max_tasks_per_type=1,
        stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=DespawnMode.NONE, despawn_prob=0.0)
    )
    model_cfg = ModelConfig(
        encoder=EncoderType.GENERAL_DEC_CNN_GRID,
        mlp_dims=(16,),
    )
    lr_cfg = ScheduleConfig(start=0.01, end=0.01)
    eps_cfg = ScheduleConfig(start=0.1, end=0.1)

    train_cfg = TrainConfig(
        total_steps=100, seed=42, lr=lr_cfg, epsilon=eps_cfg,
        learning_type=LearningType.DECENTRALIZED, use_gpu=False,
        td_lambda=0.0, heuristic=Heuristic.NEAREST,
        stopping=StoppingConfig()
    )

    env = StochasticEnv(env_cfg)
    encoding.init_encoder(model_cfg.encoder, env)
    networks = create_networks(model_cfg, env_cfg, train_cfg)

    trainer = CpuTrainer(
        network_list=networks, env=env, gamma=0.99,
        epsilon_schedule=eps_cfg, lr_schedule=lr_cfg,
        total_steps=100, heuristic=Heuristic.NEAREST,
    )
    return env, networks, trainer


class TestCpuTrainerTDStep:
    def test_train_move_stores_prev(self):
        env, _, trainer = _setup_cpu_trainer()
        s = State(agent_positions=(Grid(0, 0), Grid(2, 2)), task_positions=(), actor=0, task_types=())

        # Phase 1: Move. First step shouldn't calculate loss, just store _prev
        assert trainer._prev is None
        trainer.train_move(s_moved=s, on_task=False, t=0)
        assert trainer._prev is not None
        assert len(trainer._prev) == 2  # Encoded state for both agents

    def test_train_pick_calculates_loss(self):
        env, _, trainer = _setup_cpu_trainer()
        s = State(agent_positions=(Grid(0, 0), Grid(2, 2)), task_positions=(), actor=0, task_types=())

        # Step 1: establish _prev
        trainer.train_move(s_moved=s, on_task=False, t=0)

        # Step 2: a pick happens. Loss should be tracked.
        assert trainer._td_loss_accum == 0.0
        s_picked = State(agent_positions=(Grid(0, 0), Grid(2, 2)), task_positions=(), actor=0, task_types=())
        trainer.train_pick(s_picked, rewards=(1.0, 0.0), t=1)

        assert trainer._td_loss_accum > 0.0
