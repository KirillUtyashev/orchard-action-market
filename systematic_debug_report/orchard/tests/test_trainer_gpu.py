"""Tests for GpuTrainer and vmap batching correctness."""

import pytest
import torch

from orchard.enums import EncoderType, Heuristic, PickMode, LearningType
from orchard.datatypes import (
    EnvConfig, ModelConfig, ScheduleConfig, StochasticConfig, 
    State, Grid, TrainConfig, StoppingConfig
)
from orchard.env.stochastic import StochasticEnv
from orchard.model import create_networks
import orchard.encoding as encoding
from orchard.trainer.cpu import CpuTrainer
from orchard.trainer.gpu import GpuTrainer
from orchard.batched_training import BatchedTrainer


def _clone_networks(networks):
    import copy
    return [copy.deepcopy(net) for net in networks]


def _setup_dual_trainers():
    """Sets up a CPU and GPU trainer with perfectly mirrored network weights."""
    torch.manual_seed(42)
    
    env_cfg = EnvConfig(
        height=3, width=3, n_agents=4, n_tasks=2,
        gamma=0.99, r_picker=1.0, n_task_types=1,
        task_assignments=tuple((0,) for _ in range(4)),
        pick_mode=PickMode.FORCED, max_tasks_per_type=2,
        stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=None, despawn_prob=0.0)
    )
    model_cfg = ModelConfig(encoder=EncoderType.BLIND_TASK_CNN_GRID, mlp_dims=(16,))
    lr_cfg = ScheduleConfig(start=0.1, end=0.1)
    eps_cfg = ScheduleConfig(start=0.1, end=0.1)
    
    train_cfg = TrainConfig(
        total_steps=100, seed=42, lr=lr_cfg, epsilon=eps_cfg,
        learning_type=LearningType.DECENTRALIZED, use_gpu=True,
        td_lambda=0.0, heuristic=Heuristic.NEAREST_TASK,
        stopping=StoppingConfig()
    )
    
    encoding.init_encoder(model_cfg.encoder, env_cfg)
    env = StochasticEnv(env_cfg)
    
    cpu_nets = create_networks(model_cfg, env_cfg, train_cfg)
    gpu_nets = _clone_networks(cpu_nets)
    
    cpu_trainer = CpuTrainer(
        network_list=cpu_nets, env=env, gamma=0.99,
        epsilon_schedule=eps_cfg, lr_schedule=lr_cfg,
        total_steps=100, heuristic=Heuristic.NEAREST_TASK,
    )

    bt = BatchedTrainer(gpu_nets, td_lambda=0.0, device="cpu") # Run vmap on CPU for float match
    gpu_trainer = GpuTrainer(
        network_list=gpu_nets, bt=bt, env=env, gamma=0.99,
        epsilon_schedule=eps_cfg, lr_schedule=lr_cfg,
        total_steps=100, heuristic=Heuristic.NEAREST_TASK,
    )
    
    return cpu_trainer, gpu_trainer


class TestVmapCorrectness:
    def test_single_step_params_match(self):
        """After 1 training step, vmap and sequential paths must have identical weights."""
        cpu_trainer, gpu_trainer = _setup_dual_trainers()
        
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(1, 0)), 
            task_positions=(Grid(2, 2),), actor=0, task_types=(0,)
        )
        
        # Step 1: Establish _prev
        cpu_trainer.train_move(s, on_task=False, t=0)
        gpu_trainer.train_move(s, on_task=False, t=0)
        
        # Step 2: The actual TD update
        s_next = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(1, 0)), 
            task_positions=(Grid(2, 2),), actor=1, task_types=(0,) # Actor advanced
        )
        
        cpu_trainer.train_move(s_next, on_task=False, t=1)
        gpu_trainer.train_move(s_next, on_task=False, t=1)
        
        # Pull parameters from the vmap wrapper back to the individual PyTorch modules
        gpu_trainer.sync_to_cpu()
        
        # Compare every weight in every agent
        for i in range(4):
            for (name1, p1), (name2, p2) in zip(
                cpu_trainer.networks[i].named_parameters(),
                gpu_trainer.networks[i].named_parameters()
            ):
                assert torch.allclose(p1, p2, atol=1e-5), \
                    f"Agent {i}, Param {name1} mismatch after TD update."

    def test_team_values_match(self):
        """Action selection `_compute_team_values` must match exactly."""
        cpu_trainer, gpu_trainer = _setup_dual_trainers()
        
        s = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(1, 0)), 
            task_positions=(Grid(2, 2),), actor=0, task_types=(0,)
        )
        
        # 3 arbitrary candidate after-states
        after_states = [s, s, s]
        
        cpu_vals = cpu_trainer._compute_team_values(s, after_states)
        gpu_vals = gpu_trainer._compute_team_values(s, after_states)
        
        assert len(cpu_vals) == 3
        for c_val, g_val in zip(cpu_vals, gpu_vals):
            assert pytest.approx(c_val, rel=1e-5) == g_val