"""Tests for CpuTrainer (sequential forward passes)."""

import pytest
import torch

from orchard.enums import Action, EncoderType, Heuristic, PickMode, LearningType
from orchard.datatypes import (
    EnvConfig, ModelConfig, ScheduleConfig, StochasticConfig, 
    State, Grid, TrainConfig, StoppingConfig
)
from orchard.env.stochastic import StochasticEnv
from orchard.model import create_networks
import orchard.encoding as encoding
from orchard.trainer.cpu import CpuTrainer


def _setup_cpu_trainer(
    n_agents=2,
    r_picker=1.0,
    reward_scale=1.0,
    reward_translation=0.0,
):
    env_cfg = EnvConfig(
        height=3, width=3, n_agents=n_agents, n_tasks=1,
        gamma=0.99, r_picker=r_picker, n_task_types=1,
        task_assignments=tuple((0,) for _ in range(n_agents)),
        pick_mode=PickMode.FORCED, max_tasks_per_type=1,
        stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=None, despawn_prob=0.0)
    )
    model_cfg = ModelConfig(
        encoder=EncoderType.BLIND_TASK_CNN_GRID,
        mlp_dims=(16,),
    )
    lr_cfg = ScheduleConfig(start=0.01, end=0.01)
    eps_cfg = ScheduleConfig(start=0.1, end=0.1)

    train_cfg = TrainConfig(
        total_steps=100, seed=42, lr=lr_cfg, epsilon=eps_cfg,
        learning_type=LearningType.DECENTRALIZED, use_gpu=False,
        td_lambda=0.0, heuristic=Heuristic.NEAREST_TASK,
        stopping=StoppingConfig()
    )

    encoding.init_encoder(model_cfg.encoder, env_cfg)
    env = StochasticEnv(env_cfg)
    networks = create_networks(model_cfg, env_cfg, train_cfg)

    trainer = CpuTrainer(
        network_list=networks, env=env, gamma=0.99,
        epsilon_schedule=eps_cfg, lr_schedule=lr_cfg,
        total_steps=100, heuristic=Heuristic.NEAREST_TASK,
        reward_scale=reward_scale,
        reward_translation=reward_translation,
    )
    return env, networks, trainer


class TestCpuTrainerTDStep:
    def test_train_move_stores_prev(self):
        env, _, trainer = _setup_cpu_trainer()
        s = State(agent_positions=(Grid(0, 0), Grid(2, 2)), task_positions=(), actor=0, task_types=())
        
        # Phase 1: Move. Because it's the very first step, it shouldn't calculate loss, just store _prev
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

    def test_train_pick_scales_decentralized_rewards(self):
        _, _, trainer = _setup_cpu_trainer(n_agents=3, reward_scale=0.25)
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1), Grid(2, 2)),
            task_positions=(), actor=0, task_types=(),
        )
        trainer.train_move(s_moved=s, on_task=True, t=0)
        captured = {}

        def capture_td_step(prev, rewards, discount, current, t):
            captured["rewards"] = rewards
            captured["discount"] = discount
            return 0.0

        trainer._td_step = capture_td_step
        trainer.train_pick(s, rewards=(-1.0, 0.0, 2.0 / 3.0), t=1)

        assert captured["rewards"] == pytest.approx((-0.25, 0.0, 1.0 / 6.0))
        assert captured["discount"] == 1.0

    def test_train_pick_default_reward_scale_preserves_rewards(self):
        _, _, trainer = _setup_cpu_trainer(n_agents=3)
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1), Grid(2, 2)),
            task_positions=(), actor=0, task_types=(),
        )
        trainer.train_move(s_moved=s, on_task=True, t=0)
        captured = {}

        def capture_td_step(prev, rewards, discount, current, t):
            captured["rewards"] = rewards
            return 0.0

        trainer._td_step = capture_td_step
        trainer.train_pick(s, rewards=(-1.0, 0.0, 2.0 / 3.0), t=1)

        assert captured["rewards"] == pytest.approx((-1.0, 0.0, 2.0 / 3.0))

    def test_train_pick_translates_decentralized_rewards(self):
        _, _, trainer = _setup_cpu_trainer(
            n_agents=3,
            reward_scale=0.25,
            reward_translation=0.5,
        )
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1), Grid(2, 2)),
            task_positions=(), actor=0, task_types=(),
        )
        trainer.train_move(s_moved=s, on_task=True, t=0)
        captured = {}

        def capture_td_step(prev, rewards, discount, current, t):
            captured["rewards"] = rewards
            return 0.0

        trainer._td_step = capture_td_step
        trainer.train_pick(s, rewards=(-1.0, 0.0, 2.0 / 3.0), t=1)

        assert captured["rewards"] == pytest.approx((0.25, 0.5, 2.0 / 3.0))

    def test_train_move_translates_zero_rewards(self):
        _, _, trainer = _setup_cpu_trainer(n_agents=2, reward_translation=0.5)
        s = State(
            agent_positions=(Grid(0, 0), Grid(1, 1)),
            task_positions=(), actor=0, task_types=(),
        )
        trainer.train_move(s_moved=s, on_task=False, t=0)
        captured = {}

        def capture_td_step(prev, rewards, discount, current, t):
            captured["rewards"] = rewards
            return 0.0

        trainer._td_step = capture_td_step
        trainer.train_move(s_moved=s, on_task=False, t=1)

        assert captured["rewards"] == pytest.approx((0.5, 0.5))

    def test_greedy_action_uses_scaled_immediate_rewards(self):
        _, _, trainer = _setup_cpu_trainer(n_agents=2, r_picker=-1.0, reward_scale=0.25)
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0)),
            task_positions=(Grid(1, 1),), actor=0, task_types=(0,),
            pick_phase=True,
        )

        def compute_team_values(state, after_states):
            assert len(after_states) == 2
            return [0.3, 0.0]

        trainer._compute_team_values = compute_team_values

        assert trainer._greedy_action(s) == Action.STAY
