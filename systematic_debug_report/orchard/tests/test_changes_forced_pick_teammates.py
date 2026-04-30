"""Explicit tests for:
  Change 1 - Forced pick respects task assignment (stays on wrong-type task)
  Change 2 - Spawn default is PER_TYPE_UNIQUE (types can coexist on same cell)
  Change 3 - train_only_teammates only updates actor's group networks
"""

import copy
import torch
import pytest

from orchard.enums import (
    Action, PickMode, DespawnMode, TaskSpawnMode, EncoderType, Heuristic, LearningType
)
from orchard.datatypes import (
    EnvConfig, StochasticConfig, Grid, State,
    ModelConfig, ScheduleConfig, TrainConfig, StoppingConfig
)
from orchard.env.stochastic import StochasticEnv
from orchard.model import create_networks
import orchard.encoding as encoding
from orchard.trainer.cpu import CpuTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_type_env_cfg(pick_mode=PickMode.FORCED, task_spawn_mode=None,
                      spawn_prob=0.0, n_agents=4):
    """4 agents: 0,1 → type 0; 2,3 → type 1."""
    return EnvConfig(
        height=3, width=3, n_agents=n_agents, n_tasks=2, gamma=0.99,
        r_picker=1.0, n_task_types=2, r_low=-1.0,
        task_assignments=((0,), (0,), (1,), (1,)),
        pick_mode=pick_mode, max_tasks_per_type=4,
        stochastic=StochasticConfig(
            spawn_prob=spawn_prob, despawn_mode=DespawnMode.NONE, despawn_prob=0.0,
            task_spawn_mode=task_spawn_mode,
        ),
    )


def _make_cpu_trainer(env_cfg, train_only_teammates=False):
    model_cfg = ModelConfig(encoder=EncoderType.BLIND_TASK_CNN_GRID, mlp_dims=(16,))
    lr_cfg = ScheduleConfig(start=0.1, end=0.1)
    eps_cfg = ScheduleConfig(start=0.0, end=0.0)  # fully greedy
    train_cfg = TrainConfig(
        total_steps=200, seed=0, lr=lr_cfg, epsilon=eps_cfg,
        learning_type=LearningType.DECENTRALIZED, use_gpu=False, td_lambda=0.0,
        heuristic=Heuristic.NEAREST_TASK, stopping=StoppingConfig(),
        train_only_teammates=train_only_teammates,
    )
    encoding.init_encoder(model_cfg.encoder, env_cfg)
    env = StochasticEnv(env_cfg)
    networks = create_networks(model_cfg, env_cfg, train_cfg)
    trainer = CpuTrainer(
        network_list=networks, env=env, gamma=0.99,
        epsilon_schedule=eps_cfg, lr_schedule=lr_cfg,
        total_steps=200, heuristic=Heuristic.NEAREST_TASK,
        train_only_teammates=train_only_teammates,
    )
    return env, networks, trainer


def _params_snapshot(networks):
    """Return a list of flat param tensors (one per network), detached."""
    return [
        torch.cat([p.data.flatten() for p in net.parameters()]).clone()
        for net in networks
    ]


# ---------------------------------------------------------------------------
# Change 1: Forced pick respects assignment
# ---------------------------------------------------------------------------

class TestChange1ForcedPickAssignment:
    def test_forced_wrong_type_no_pick_no_reward(self):
        """Agent on wrong-type task must stay (task remains, reward=0)."""
        env = StochasticEnv(_two_type_env_cfg())
        # Agent 0 (type 0) stands on a type-1 task
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0), Grid(2, 2), Grid(2, 0)),
            task_positions=(Grid(1, 1),), actor=0, task_types=(1,),
        )
        s_after, rewards = env.resolve_pick(s)
        assert len(s_after.task_positions) == 1, "Task must not be picked"
        assert all(r == 0.0 for r in rewards), "No reward when staying"

    def test_forced_correct_type_picks_and_rewards(self):
        """Agent on correct-type task must pick and receive r_picker."""
        env = StochasticEnv(_two_type_env_cfg())
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0), Grid(2, 2), Grid(2, 0)),
            task_positions=(Grid(1, 1),), actor=0, task_types=(0,),
        )
        s_after, rewards = env.resolve_pick(s)
        assert len(s_after.task_positions) == 0, "Task must be picked"
        assert rewards[0] == 1.0, "Picker gets r_picker"

    def test_forced_single_type_identical_to_old(self):
        """With n_task_types=1, forced pick behaves exactly as before."""
        cfg = EnvConfig(
            height=3, width=3, n_agents=2, n_tasks=1, gamma=0.99,
            r_picker=1.0, n_task_types=1, r_low=-1.0,
            task_assignments=((0,), (0,)),
            pick_mode=PickMode.FORCED, max_tasks_per_type=2,
            stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=DespawnMode.NONE, despawn_prob=0.0),
        )
        env = StochasticEnv(cfg)
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0)),
            task_positions=(Grid(1, 1),), actor=0, task_types=(0,),
        )
        s_after, rewards = env.resolve_pick(s)
        assert len(s_after.task_positions) == 0
        assert rewards[0] == 1.0

    def test_forced_wrong_type_task_stays_on_board(self):
        """After a refused pick, the wrong-type task still exists for correct agents."""
        env = StochasticEnv(_two_type_env_cfg())
        # Agent 0 (type 0) on a type-1 task: refused
        s = State(
            agent_positions=(Grid(1, 1), Grid(0, 0), Grid(2, 2), Grid(2, 0)),
            task_positions=(Grid(1, 1),), actor=0, task_types=(1,),
        )
        s_after, _ = env.resolve_pick(s)
        # Agent 2 (type 1) can still pick it
        s2 = State(
            agent_positions=(Grid(0, 0), Grid(0, 1), Grid(1, 1), Grid(2, 0)),
            task_positions=s_after.task_positions, actor=2, task_types=s_after.task_types,
        )
        s_after2, rewards2 = env.resolve_pick(s2)
        assert len(s_after2.task_positions) == 0, "Type-1 agent should pick the task"
        assert rewards2[2] == 1.0


# ---------------------------------------------------------------------------
# Change 2: Spawn default is PER_TYPE_UNIQUE
# ---------------------------------------------------------------------------

class TestChange2SpawnDefault:
    def test_forced_default_allows_types_to_coexist(self):
        """With FORCED pick and default spawn, type-0 and type-1 can share a cell."""
        # Fill the board with type-1 tasks on every cell (except agent cells).
        # Then try to spawn type-0 tasks. PER_TYPE_UNIQUE allows coexistence;
        # GLOBAL_UNIQUE would block all spawning.
        cfg = _two_type_env_cfg(spawn_prob=1.0)  # spawn_prob=1 → always spawn if eligible
        env = StochasticEnv(cfg)

        # Place type-1 tasks on all non-agent cells
        h, w = cfg.height, cfg.width
        agent_positions = (Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(1, 0))
        agent_set = set(agent_positions)
        type1_positions = tuple(
            Grid(r, c)
            for r in range(h) for c in range(w)
            if Grid(r, c) not in agent_set
        )
        type1_types = (1,) * len(type1_positions)

        s = State(
            agent_positions=agent_positions,
            task_positions=type1_positions,
            task_types=type1_types,
            actor=0,
        )

        s_after = env.spawn_and_despawn(s)
        type0_count = sum(1 for t in s_after.task_types if t == 0)
        # PER_TYPE_UNIQUE: type-0 tasks can spawn on cells occupied by type-1 tasks
        assert type0_count > 0, (
            "Type-0 tasks must spawn even when all cells have type-1 tasks "
            "(PER_TYPE_UNIQUE default)"
        )

    def test_global_unique_override_blocks_coexistence(self):
        """Explicitly setting task_spawn_mode=global_unique preserves old behavior."""
        cfg = _two_type_env_cfg(spawn_prob=1.0, task_spawn_mode=TaskSpawnMode.GLOBAL_UNIQUE)
        env = StochasticEnv(cfg)

        h, w = cfg.height, cfg.width
        agent_positions = (Grid(0, 0), Grid(0, 1), Grid(0, 2), Grid(1, 0))
        agent_set = set(agent_positions)
        type1_positions = tuple(
            Grid(r, c)
            for r in range(h) for c in range(w)
            if Grid(r, c) not in agent_set
        )
        type1_types = (1,) * len(type1_positions)

        s = State(
            agent_positions=agent_positions,
            task_positions=type1_positions,
            task_types=type1_types,
            actor=0,
        )

        s_after = env.spawn_and_despawn(s)
        type0_count = sum(1 for t in s_after.task_types if t == 0)
        assert type0_count == 0, (
            "With GLOBAL_UNIQUE, type-0 tasks must NOT spawn when all cells are occupied"
        )


# ---------------------------------------------------------------------------
# Change 3: train_only_teammates updates only actor's group
# ---------------------------------------------------------------------------

class TestChange3TrainOnlyTeammates:
    def _do_step(self, trainer, env, actor_idx):
        """Run one training step with a known actor."""
        s = State(
            agent_positions=(Grid(0, 0), Grid(2, 2), Grid(0, 2), Grid(2, 0)),
            task_positions=(), actor=actor_idx, task_types=(),
        )
        trainer.step(s, t=0)

    def test_group0_actor_only_updates_group0(self):
        """When actor is group-0, only group-0 (agents 0,1) networks change."""
        env_cfg = _two_type_env_cfg()
        _, networks, trainer = _make_cpu_trainer(env_cfg, train_only_teammates=True)

        # Warm up _prev so the next step actually runs a TD update
        s_init = State(
            agent_positions=(Grid(0, 0), Grid(2, 2), Grid(0, 2), Grid(2, 0)),
            task_positions=(), actor=0, task_types=(),
        )
        trainer.step(s_init, t=0)

        before = _params_snapshot(networks)

        # Now step with actor=1 (group 0)
        self._do_step(trainer, None, actor_idx=1)

        after = _params_snapshot(networks)
        group0_changed = any(not torch.equal(before[i], after[i]) for i in [0, 1])
        group1_changed = any(not torch.equal(before[i], after[i]) for i in [2, 3])

        assert group0_changed, "Group-0 networks must be updated when actor is group-0"
        assert not group1_changed, "Group-1 networks must NOT be updated when actor is group-0"

    def test_group1_actor_only_updates_group1(self):
        """When actor is group-1, only group-1 (agents 2,3) networks change."""
        env_cfg = _two_type_env_cfg()
        _, networks, trainer = _make_cpu_trainer(env_cfg, train_only_teammates=True)

        s_init = State(
            agent_positions=(Grid(0, 0), Grid(2, 2), Grid(0, 2), Grid(2, 0)),
            task_positions=(), actor=2, task_types=(),
        )
        trainer.step(s_init, t=0)

        before = _params_snapshot(networks)
        self._do_step(trainer, None, actor_idx=3)  # group-1
        after = _params_snapshot(networks)

        group0_changed = any(not torch.equal(before[i], after[i]) for i in [0, 1])
        group1_changed = any(not torch.equal(before[i], after[i]) for i in [2, 3])

        assert not group0_changed, "Group-0 networks must NOT be updated when actor is group-1"
        assert group1_changed, "Group-1 networks must be updated when actor is group-1"

    def test_disabled_updates_all_networks(self):
        """With train_only_teammates=False, all networks are updated regardless of actor."""
        env_cfg = _two_type_env_cfg()
        _, networks, trainer = _make_cpu_trainer(env_cfg, train_only_teammates=False)

        s_init = State(
            agent_positions=(Grid(0, 0), Grid(2, 2), Grid(0, 2), Grid(2, 0)),
            task_positions=(), actor=0, task_types=(),
        )
        trainer.step(s_init, t=0)
        before = _params_snapshot(networks)

        # Step with a group-0 actor; all 4 networks should still update
        self._do_step(trainer, None, actor_idx=1)
        after = _params_snapshot(networks)

        all_changed = all(not torch.equal(before[i], after[i]) for i in range(4))
        assert all_changed, "All 4 networks must update when train_only_teammates=False"
