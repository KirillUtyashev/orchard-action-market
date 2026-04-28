"""Runtime integration tests for orchard actor-critic training."""

from __future__ import annotations

import copy
import csv
import os
import tempfile
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

import orchard.encoding as encoding
from orchard.actor_critic import PolicyNetwork, build_phase1_legal_mask, build_phase2_legal_mask
from orchard.config import load_config
from orchard.env import create_env
from orchard.enums import (
    Action,
    AlgorithmName,
    DespawnMode,
    EncoderType,
    Heuristic,
    LearningType,
    PickMode,
    make_pick_action,
)
from orchard.datatypes import (
    AlgorithmConfig,
    EnvConfig,
    EvalConfig,
    ExperimentConfig,
    FollowingRatesConfig,
    Grid,
    InfluencerConfig,
    LoggingConfig,
    ModelConfig,
    ScheduleConfig,
    State,
    StochasticConfig,
    StoppingConfig,
    TrainConfig,
)
from orchard.batched_actor_training import BatchedActorTrainer
from orchard.batched_training import BatchedTrainer
from orchard.model import create_actor_networks, create_networks
from orchard.trainer import create_trainer
from orchard.trainer.actor_critic import ActorCriticCpuTrainer, ActorCriticGpuTrainer
from orchard.trainer.timer import Timer, TimerSection
from orchard.train import train


def _write_config(yaml_str: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(yaml_str)
    return path


def _latest_run_dir(output_dir: str) -> Path:
    run_dirs = [
        Path(output_dir) / name
        for name in os.listdir(output_dir)
        if (Path(output_dir) / name).is_dir()
    ]
    assert run_dirs
    return sorted(run_dirs)[-1]


def _read_single_row(csv_path: Path) -> dict[str, str]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    return rows[0]


ACTOR_CRITIC_CONFIG = """
env:
  height: 4
  width: 4
  n_agents: 2
  n_tasks: 2
  n_task_types: 2
  gamma: 0.99
  r_picker: 1.0
  r_low: 0.0
  pick_mode: {pick_mode}
  max_tasks_per_type: 2
  task_assignments: [[0], [1]]
  stochastic:
    spawn_prob: 0.1
    despawn_mode: probability
    despawn_prob: 0.05
model:
  encoder: blind_task_cnn_grid
  mlp_dims: [16]
  conv_specs: [[4, 3]]
train:
  learning_type: decentralized
  use_gpu: {use_gpu}
  td_lambda: 0.3
  total_steps: 5
  seed: 42
  heuristic: nearest_correct_task_stay_wrong
  lr:
    start: 0.01
  epsilon:
    start: 0.3
  algorithm:
    name: actor_critic
{extra_train_blocks}
eval:
  eval_steps: 5
  n_test_states: 2
logging:
  main_csv_freq: 5
  detail_csv_freq: 5
  output_dir: {output_dir}
{extra_logging_blocks}
"""


def _run_actor_critic_case(
    *,
    pick_mode: str,
    output_dir: str,
    extra_train_blocks: str = "",
    extra_logging_blocks: str = "",
    resume_checkpoint: str | None = None,
    use_gpu: str = "false",
) -> Path:
    yaml_str = ACTOR_CRITIC_CONFIG.format(
        pick_mode=pick_mode,
        output_dir=output_dir,
        extra_train_blocks=extra_train_blocks,
        extra_logging_blocks=extra_logging_blocks,
        resume_checkpoint=resume_checkpoint,
        use_gpu=use_gpu,
    )
    path = _write_config(yaml_str)
    cfg = load_config(path)
    train(cfg, resume_checkpoint=resume_checkpoint)
    os.unlink(path)
    return _latest_run_dir(output_dir)


def _make_actor_critic_trainer(
    pick_mode: PickMode,
    *,
    n_task_types: int = 1,
    task_assignments: tuple[tuple[int, ...], ...] | None = None,
    batch_forced_actor_updates: bool = True,
    per_type_seeds: tuple[int, ...] | None = None,
):
    assignments = task_assignments or ((0,), (0,))
    env_cfg = EnvConfig(
        height=3,
        width=3,
        n_agents=len(assignments),
        n_tasks=1,
        gamma=0.99,
        r_picker=1.0,
        n_task_types=n_task_types,
        r_low=0.0,
        task_assignments=assignments,
        pick_mode=pick_mode,
        max_tasks_per_type=1,
        stochastic=StochasticConfig(
            spawn_prob=0.0,
            despawn_mode=None,
            despawn_prob=0.0,
            per_type_seeds=per_type_seeds,
        ),
    )
    model_cfg = ModelConfig(
        encoder=EncoderType.BLIND_TASK_CNN_GRID,
        mlp_dims=(8,),
        conv_specs=((4, 3),),
    )
    train_cfg = TrainConfig(
        total_steps=1,
        seed=7,
        lr=ScheduleConfig(start=0.01, end=0.01),
        epsilon=ScheduleConfig(start=0.0, end=0.0),
        actor_lr=ScheduleConfig(start=0.01, end=0.01),
        algorithm=AlgorithmConfig(name=AlgorithmName.ACTOR_CRITIC),
        freeze_critic=False,
        learning_type=LearningType.DECENTRALIZED,
        use_gpu=False,
        td_lambda=0.0,
        batch_forced_actor_updates=batch_forced_actor_updates,
        heuristic=Heuristic.NEAREST_TASK,
        stopping=StoppingConfig(),
    )
    cfg = ExperimentConfig(
        env=env_cfg,
        model=model_cfg,
        actor_model=None,
        train=train_cfg,
        eval=EvalConfig(),
        logging=LoggingConfig(output_dir="unused"),
    )
    encoding.init_encoder(model_cfg.encoder, env_cfg)
    env = create_env(env_cfg)
    trainer = create_trainer(cfg, env)
    return env, trainer


def _make_probability_count_trainer():
    env_cfg = EnvConfig(
        height=3,
        width=3,
        n_agents=2,
        n_tasks=5,
        gamma=0.99,
        r_picker=1.0,
        n_task_types=1,
        r_low=0.0,
        task_assignments=((0,), (0,)),
        pick_mode=PickMode.CHOICE,
        max_tasks_per_type=5,
        stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=None, despawn_prob=0.0),
    )
    model_cfg = ModelConfig(
        encoder=EncoderType.BLIND_TASK_CNN_GRID,
        mlp_dims=(8,),
        conv_specs=((4, 3),),
    )
    train_cfg = TrainConfig(
        total_steps=1,
        seed=9,
        lr=ScheduleConfig(start=0.01, end=0.01),
        epsilon=ScheduleConfig(start=0.0, end=0.0),
        actor_lr=ScheduleConfig(start=0.01, end=0.01),
        algorithm=AlgorithmConfig(name=AlgorithmName.ACTOR_CRITIC),
        freeze_critic=False,
        learning_type=LearningType.DECENTRALIZED,
        use_gpu=False,
        td_lambda=0.0,
        heuristic=Heuristic.NEAREST_TASK,
        stopping=StoppingConfig(),
    )
    cfg = ExperimentConfig(
        env=env_cfg,
        model=model_cfg,
        actor_model=None,
        train=train_cfg,
        eval=EvalConfig(),
        logging=LoggingConfig(output_dir="unused"),
    )
    encoding.init_encoder(model_cfg.encoder, env_cfg)
    env = create_env(env_cfg)
    trainer = create_trainer(cfg, env)
    return env, trainer


def _make_dual_actor_critic_trainers(
    pick_mode: PickMode,
    *,
    following_rates_cfg: FollowingRatesConfig | None = None,
    task_assignments: tuple[tuple[int, ...], ...] | None = None,
    comm_only_teammates: bool = False,
):
    torch.manual_seed(11)
    assignments = task_assignments or ((0,), (1,))
    n_agents = len(assignments)
    all_task_types = {task_type for group in assignments for task_type in group}
    n_task_types = max(all_task_types) + 1 if all_task_types else 1

    env_cfg = EnvConfig(
        height=3,
        width=3,
        n_agents=n_agents,
        n_tasks=2,
        gamma=0.99,
        r_picker=1.0,
        n_task_types=n_task_types,
        r_low=-0.25,
        task_assignments=assignments,
        pick_mode=pick_mode,
        max_tasks_per_type=2,
        stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=None, despawn_prob=0.0),
    )
    model_cfg = ModelConfig(
        encoder=EncoderType.BLIND_TASK_CNN_GRID,
        mlp_dims=(8,),
        conv_specs=((4, 3),),
    )
    lr_cfg = ScheduleConfig(start=0.01, end=0.01)
    actor_lr_cfg = ScheduleConfig(start=0.01, end=0.01)
    train_cfg = TrainConfig(
        total_steps=20,
        seed=11,
        lr=lr_cfg,
        epsilon=ScheduleConfig(start=0.0, end=0.0),
        actor_lr=actor_lr_cfg,
        algorithm=AlgorithmConfig(name=AlgorithmName.ACTOR_CRITIC),
        freeze_critic=False,
        learning_type=LearningType.DECENTRALIZED,
        use_gpu=True,
        td_lambda=0.0,
        comm_only_teammates=comm_only_teammates,
        heuristic=Heuristic.NEAREST_TASK,
        stopping=StoppingConfig(),
    )

    encoding.init_encoder(model_cfg.encoder, env_cfg)
    env = create_env(env_cfg)
    following_cfg = following_rates_cfg or FollowingRatesConfig()

    cpu_critics = create_networks(model_cfg, env_cfg, train_cfg)
    cpu_actors = create_actor_networks(model_cfg, env_cfg, train_cfg)
    gpu_critics = [copy.deepcopy(net) for net in cpu_critics]
    gpu_actors = [copy.deepcopy(net) for net in cpu_actors]

    cpu_trainer = ActorCriticCpuTrainer(
        critic_networks=cpu_critics,
        actor_networks=cpu_actors,
        env=env,
        gamma=env_cfg.gamma,
        critic_lr_schedule=lr_cfg,
        actor_lr_schedule=actor_lr_cfg,
        total_steps=train_cfg.total_steps,
        heuristic=Heuristic.NEAREST_TASK,
        freeze_critic=False,
        following_rates_cfg=following_cfg,
        influencer_cfg=InfluencerConfig(),
        comm_only_teammates=comm_only_teammates,
    )
    gpu_trainer = ActorCriticGpuTrainer(
        critic_networks=gpu_critics,
        actor_networks=gpu_actors,
        bt=BatchedTrainer(gpu_critics, td_lambda=0.0, device="cpu"),
        env=env,
        gamma=env_cfg.gamma,
        critic_lr_schedule=lr_cfg,
        actor_lr_schedule=actor_lr_cfg,
        total_steps=train_cfg.total_steps,
        heuristic=Heuristic.NEAREST_TASK,
        freeze_critic=False,
        following_rates_cfg=following_cfg,
        influencer_cfg=InfluencerConfig(),
        comm_only_teammates=comm_only_teammates,
    )
    return env, cpu_trainer, gpu_trainer


def _make_single_actor_critic_trainer(
    pick_mode: PickMode,
    *,
    task_assignments: tuple[tuple[int, ...], ...],
    following_rates_cfg: FollowingRatesConfig,
):
    env_cfg = EnvConfig(
        height=3,
        width=3,
        n_agents=len(task_assignments),
        n_tasks=2,
        gamma=0.99,
        r_picker=1.0,
        n_task_types=max({task_type for group in task_assignments for task_type in group}, default=0) + 1,
        r_low=-0.25,
        task_assignments=task_assignments,
        pick_mode=pick_mode,
        max_tasks_per_type=2,
        stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=None, despawn_prob=0.0),
    )
    model_cfg = ModelConfig(
        encoder=EncoderType.BLIND_TASK_CNN_GRID,
        mlp_dims=(8,),
        conv_specs=((4, 3),),
    )
    train_cfg = TrainConfig(
        total_steps=20,
        seed=11,
        lr=ScheduleConfig(start=0.01, end=0.01),
        epsilon=ScheduleConfig(start=0.0, end=0.0),
        actor_lr=ScheduleConfig(start=0.01, end=0.01),
        algorithm=AlgorithmConfig(name=AlgorithmName.ACTOR_CRITIC),
        freeze_critic=False,
        following_rates=following_rates_cfg,
        learning_type=LearningType.DECENTRALIZED,
        use_gpu=False,
        td_lambda=0.0,
        heuristic=Heuristic.NEAREST_TASK,
        stopping=StoppingConfig(),
    )
    cfg = ExperimentConfig(
        env=env_cfg,
        model=model_cfg,
        actor_model=None,
        train=train_cfg,
        eval=EvalConfig(),
        logging=LoggingConfig(output_dir="unused"),
    )
    encoding.init_encoder(model_cfg.encoder, env_cfg)
    env = create_env(env_cfg)
    trainer = create_trainer(cfg, env)
    return env, trainer


def _clone_actor_params(actors: list[PolicyNetwork]) -> list[dict[str, torch.Tensor]]:
    return [
        {name: tensor.detach().clone() for name, tensor in actor.state_dict().items()}
        for actor in actors
    ]


def _assert_actor_params_close(
    left: list[PolicyNetwork],
    right: list[PolicyNetwork],
    *,
    atol: float = 1e-6,
) -> None:
    for left_actor, right_actor in zip(left, right):
        left_state = left_actor.state_dict()
        right_state = right_actor.state_dict()
        assert left_state.keys() == right_state.keys()
        for name in left_state:
            torch.testing.assert_close(left_state[name], right_state[name], atol=atol, rtol=0.0)


def _make_multi_pick_phase_state() -> State:
    return State(
        agent_positions=(Grid(1, 1), Grid(2, 2)),
        task_positions=(Grid(1, 1), Grid(1, 1)),
        actor=0,
        task_types=(0, 1),
        pick_phase=True,
    )


def _make_phase1_decision_state() -> State:
    return State(
        agent_positions=(Grid(0, 0), Grid(2, 2)),
        task_positions=(Grid(0, 2), Grid(1, 0)),
        actor=0,
        task_types=(0, 1),
    )


def _make_choice_cycle_start_state() -> State:
    return State(
        agent_positions=(Grid(0, 0), Grid(2, 2)),
        task_positions=(Grid(0, 1),),
        actor=0,
        task_types=(0,),
    )


def _make_move_only_state() -> State:
    return State(
        agent_positions=(Grid(0, 0), Grid(2, 2)),
        task_positions=(Grid(2, 2),),
        actor=0,
        task_types=(0,),
    )


def _make_always_pick_choice_state() -> State:
    return State(
        agent_positions=(Grid(1, 1), Grid(2, 2)),
        task_positions=(Grid(1, 1), Grid(0, 1), Grid(2, 1), Grid(1, 0), Grid(1, 2)),
        actor=0,
        task_types=(0, 0, 0, 0, 0),
    )


def _install_identity_critic_spy(trainer):
    encode_calls: list[State] = []
    td_calls: list[dict[str, object]] = []

    def _encode_all_critics(self, state: State) -> State:
        encode_calls.append(state)
        return state

    def _critic_values(self, encoded_state: State) -> list[float]:
        return [0.0 for _ in range(self._n_agents)]

    def _critic_td_step(self, prev: State, rewards, discount: float, current: State, t: int) -> float:
        td_calls.append(
            {
                "prev": prev,
                "rewards": rewards,
                "discount": discount,
                "current": current,
                "t": t,
            }
        )
        return 0.0

    trainer._encode_all_critics = MethodType(_encode_all_critics, trainer)
    trainer._critic_values = MethodType(_critic_values, trainer)
    trainer._critic_td_step = MethodType(_critic_td_step, trainer)
    return {
        "encode_calls": encode_calls,
        "td_calls": td_calls,
    }


def _install_scripted_actions(trainer, *, move_action: Action, pick_action: Action | None = None) -> None:
    def _sample_action(self, state: State):
        actor_state = self._encode_actor_state(state, state.actor)
        if state.pick_phase:
            assert pick_action is not None
            mask = build_phase2_legal_mask(state, self._env.cfg)
            return pick_action, actor_state, np.zeros(mask.shape[0], dtype=float), mask
        mask = build_phase1_legal_mask(state, self._env.cfg)
        return move_action, actor_state, np.zeros(mask.shape[0], dtype=float), mask

    trainer._sample_action = MethodType(_sample_action, trainer)


def _install_two_actor_choice_cycle_actions(
    trainer,
    *,
    actor0_move: Action,
    actor0_pick: Action,
    actor1_move: Action,
) -> None:
    def _sample_action(self, state: State):
        actor_state = self._encode_actor_state(state, state.actor)
        if state.pick_phase:
            mask = build_phase2_legal_mask(state, self._env.cfg)
            return actor0_pick, actor_state, np.zeros(mask.shape[0], dtype=float), mask
        mask = build_phase1_legal_mask(state, self._env.cfg)
        action = actor0_move if state.actor == 0 else actor1_move
        return action, actor_state, np.zeros(mask.shape[0], dtype=float), mask

    trainer._sample_action = MethodType(_sample_action, trainer)


class TestActorCriticTrainingLoop:
    def test_actor_critic_action_sampling_uses_per_type_rngs(self):
        _, source_trainer = _make_actor_critic_trainer(
            PickMode.FORCED,
            n_task_types=2,
            task_assignments=((0,), (0,), (1,), (1,)),
            per_type_seeds=(1000, 1001),
        )
        _, isolated_trainer = _make_actor_critic_trainer(
            PickMode.FORCED,
            n_task_types=1,
            task_assignments=((0,), (0,)),
            per_type_seeds=(1001,),
        )

        probs = np.asarray([0.05, 0.15, 0.2, 0.25, 0.35], dtype=float)

        source_agent_2 = [
            source_trainer._sample_action_index_from_probs(2, probs)
            for _ in range(12)
        ]
        isolated_agent_0 = [
            isolated_trainer._sample_action_index_from_probs(0, probs)
            for _ in range(12)
        ]
        source_agent_3 = [
            source_trainer._sample_action_index_from_probs(3, probs)
            for _ in range(12)
        ]
        isolated_agent_1 = [
            isolated_trainer._sample_action_index_from_probs(1, probs)
            for _ in range(12)
        ]

        assert source_agent_2 == isolated_agent_0
        assert source_agent_3 == isolated_agent_1

    def test_batched_actor_trainer_matches_delayed_sequential_update(self):
        torch.manual_seed(17)

        env_cfg = EnvConfig(
            height=3,
            width=3,
            n_agents=2,
            n_tasks=2,
            gamma=0.99,
            r_picker=1.0,
            n_task_types=2,
            r_low=-0.25,
            task_assignments=((0,), (1,)),
            pick_mode=PickMode.CHOICE,
            max_tasks_per_type=2,
            stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=None, despawn_prob=0.0),
        )
        model_cfg = ModelConfig(
            encoder=EncoderType.BLIND_TASK_CNN_GRID,
            mlp_dims=(8,),
            conv_specs=((4, 3),),
        )
        train_cfg = TrainConfig(
            total_steps=4,
            seed=17,
            lr=ScheduleConfig(start=0.01, end=0.01),
            epsilon=ScheduleConfig(start=0.0, end=0.0),
            actor_lr=ScheduleConfig(start=0.01, end=0.01),
            algorithm=AlgorithmConfig(name=AlgorithmName.ACTOR_CRITIC),
            freeze_critic=False,
            learning_type=LearningType.DECENTRALIZED,
            use_gpu=True,
            td_lambda=0.0,
            heuristic=Heuristic.NEAREST_TASK,
            stopping=StoppingConfig(),
        )
        encoding.init_encoder(model_cfg.encoder, env_cfg)

        sequential_actors = create_actor_networks(model_cfg, env_cfg, train_cfg)
        batched_actors = [copy.deepcopy(net) for net in sequential_actors]
        actor_bt = BatchedActorTrainer(batched_actors, device="cpu")

        actor0_state = _make_phase1_decision_state()
        actor1_move_state = State(
            agent_positions=(Grid(0, 0), Grid(1, 1)),
            task_positions=(Grid(2, 2), Grid(1, 1)),
            actor=1,
            task_types=(0, 1),
        )
        actor1_pick_state = actor1_move_state.with_pick_phase()

        experiences = [
            (0, actor0_state, build_phase1_legal_mask(actor0_state, env_cfg), Action.RIGHT, 0.35),
            (1, actor1_move_state, build_phase1_legal_mask(actor1_move_state, env_cfg), Action.STAY, -0.2),
            (
                1,
                actor1_pick_state,
                build_phase2_legal_mask(actor1_pick_state, env_cfg),
                make_pick_action(1),
                0.5,
            ),
        ]

        for actor_id, state, legal_mask, action, advantage in experiences:
            seq_state = encoding.encode(state, actor_id)
            batched_state = encoding.encode(state, actor_id)
            sequential_actors[actor_id].add_experience(seq_state, legal_mask, action, advantage)
            batched_actors[actor_id].add_experience(batched_state, legal_mask, action, advantage)

        for actor in sequential_actors:
            actor.set_lr(0.01)
            actor.train_batch()
        batched_metrics = actor_bt.train_batch_batched(alpha=0.01)

        assert batched_metrics is not None
        assert batched_metrics["sample_count"] == pytest.approx(3.0)
        _assert_actor_params_close(sequential_actors, batched_actors)

    def test_gpu_enumerate_action_objectives_matches_cpu_for_pick_phase(self):
        env, cpu_trainer, gpu_trainer = _make_dual_actor_critic_trainers(PickMode.CHOICE)
        state = _make_multi_pick_phase_state()
        legal_mask = build_phase2_legal_mask(state, env.cfg)

        cpu_q_values, cpu_after_states, cpu_rewards, cpu_after_values = (
            cpu_trainer._enumerate_action_objectives(state, legal_mask, discount=1.0)
        )
        gpu_q_values, gpu_after_states, gpu_rewards, gpu_after_values = (
            gpu_trainer._enumerate_action_objectives(state, legal_mask, discount=1.0)
        )

        np.testing.assert_allclose(cpu_q_values, gpu_q_values, atol=1e-6)
        assert cpu_rewards == gpu_rewards

        legal_indices = [int(idx) for idx in np.flatnonzero(legal_mask)]
        for action_idx in legal_indices:
            assert cpu_after_states[action_idx] == gpu_after_states[action_idx]
            np.testing.assert_allclose(
                cpu_after_values[action_idx],
                gpu_after_values[action_idx],
                atol=1e-6,
            )

    def test_gpu_enumerate_action_objectives_batches_legal_actions_once(self):
        env, _, gpu_trainer = _make_dual_actor_critic_trainers(PickMode.CHOICE)
        state = _make_phase1_decision_state()
        legal_mask = build_phase1_legal_mask(state, env.cfg)

        calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        original_forward_batched = gpu_trainer._bt.forward_batched

        def _spy_forward_batched(grids, scalars):
            calls.append((tuple(grids.shape), tuple(scalars.shape)))
            return original_forward_batched(grids, scalars)

        gpu_trainer._bt.forward_batched = _spy_forward_batched  # type: ignore[method-assign]

        _, after_states, _, _ = gpu_trainer._enumerate_action_objectives(
            state,
            legal_mask,
            discount=env.cfg.gamma,
        )

        assert len(calls) == 1
        assert calls[0][0][1] == int(legal_mask.sum())
        assert len(after_states) == int(legal_mask.sum())

    @pytest.mark.parametrize(
        ("following_rates_cfg", "discount"),
        [
            (FollowingRatesConfig(), 1.0),
            (
                FollowingRatesConfig(
                    enabled=True,
                    budget=1.0,
                    rho=0.5,
                    reallocation_freq=1,
                    fixed=True,
                ),
                1.0,
            ),
        ],
    )
    def test_gpu_tensor_action_objectives_match_cpu_legal_q_values(
        self,
        following_rates_cfg: FollowingRatesConfig,
        discount: float,
    ):
        env, cpu_trainer, gpu_trainer = _make_dual_actor_critic_trainers(
            PickMode.CHOICE,
            following_rates_cfg=following_rates_cfg,
        )
        state = _make_multi_pick_phase_state()
        legal_mask = build_phase2_legal_mask(state, env.cfg)

        cpu_q_values, _, _, _ = cpu_trainer._enumerate_action_objectives(state, legal_mask, discount=discount)
        gpu_q_legal, legal_indices, _, _, _ = gpu_trainer._enumerate_action_objectives_tensor(
            state,
            legal_mask,
            discount=discount,
        )

        np.testing.assert_allclose(
            cpu_q_values[np.asarray(legal_indices, dtype=int)],
            gpu_q_legal.detach().cpu().numpy(),
            atol=1e-6,
        )

    def test_gpu_comm_only_teammates_masks_non_teammates_without_following_rates(self):
        _, _, gpu_trainer = _make_dual_actor_critic_trainers(
            PickMode.CHOICE,
            task_assignments=((0,), (0,), (1,), (1,)),
            comm_only_teammates=True,
        )
        rewards_t = torch.tensor(
            [
                [1.0, 10.0, 100.0, 1000.0],
                [2.0, 20.0, 200.0, 2000.0],
            ],
            dtype=torch.float32,
        )
        after_values_t = torch.tensor(
            [
                [0.5, 5.0, 50.0, 500.0],
                [1.5, 6.0, 60.0, 600.0],
            ],
            dtype=torch.float32,
        )

        q_values = gpu_trainer._action_objectives_tensor(
            actor_id=0,
            rewards_t=rewards_t,
            after_values_t=after_values_t,
            discount=0.5,
        )

        expected = (
            rewards_t[:, 0] + 0.5 * after_values_t[:, 0] +
            rewards_t[:, 1] + 0.5 * after_values_t[:, 1]
        )
        torch.testing.assert_close(q_values, expected, atol=1e-6, rtol=0.0)

    def test_gpu_comm_only_teammates_keeps_default_full_team_objective_when_disabled(self):
        _, _, gpu_trainer = _make_dual_actor_critic_trainers(
            PickMode.CHOICE,
            task_assignments=((0,), (0,), (1,), (1,)),
            comm_only_teammates=False,
        )
        rewards_t = torch.tensor(
            [[1.0, 10.0, 100.0, 1000.0]],
            dtype=torch.float32,
        )
        after_values_t = torch.tensor(
            [[0.5, 5.0, 50.0, 500.0]],
            dtype=torch.float32,
        )

        q_values = gpu_trainer._action_objectives_tensor(
            actor_id=0,
            rewards_t=rewards_t,
            after_values_t=after_values_t,
            discount=0.5,
        )

        expected = (
            rewards_t[:, 0] + 0.5 * after_values_t[:, 0] +
            rewards_t[:, 1] + 0.5 * after_values_t[:, 1] +
            rewards_t[:, 2] + 0.5 * after_values_t[:, 2] +
            rewards_t[:, 3] + 0.5 * after_values_t[:, 3]
        )
        torch.testing.assert_close(q_values, expected, atol=1e-6, rtol=0.0)

    def test_gpu_comm_only_teammates_preserves_teammate_following_weights_only(self):
        following_cfg = FollowingRatesConfig(
            enabled=True,
            budget=3.0,
            rho=0.5,
            reallocation_freq=1,
            fixed=True,
        )
        _, _, gpu_trainer = _make_dual_actor_critic_trainers(
            PickMode.CHOICE,
            following_rates_cfg=following_cfg,
            task_assignments=((0,), (0,), (1,), (1,)),
            comm_only_teammates=True,
        )
        gpu_trainer._following_states[1].set_following_rates([0.7, 0.0, 0.0, 0.0])
        gpu_trainer._following_states[2].set_following_rates([1.5, 0.0, 0.0, 0.0])
        gpu_trainer._following_states[3].set_following_rates([2.0, 0.0, 0.0, 0.0])

        rewards_t = torch.tensor(
            [[1.0, 10.0, 100.0, 1000.0]],
            dtype=torch.float32,
        )
        after_values_t = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0]],
            dtype=torch.float32,
        )

        q_values = gpu_trainer._action_objectives_tensor(
            actor_id=0,
            rewards_t=rewards_t,
            after_values_t=after_values_t,
            discount=1.0,
        )

        teammate_weight = 1.0 - np.exp(-0.7)
        expected = torch.tensor(
            [1.0 + teammate_weight * 11.0],
            dtype=torch.float32,
        )
        torch.testing.assert_close(q_values, expected, atol=1e-6, rtol=0.0)

    def test_fixed_following_rates_dual_budgets_initialize_expected_rates(self):
        _, trainer = _make_single_actor_critic_trainer(
            PickMode.CHOICE,
            task_assignments=((0,), (0,), (1,), (1,)),
            following_rates_cfg=FollowingRatesConfig(
                enabled=True,
                teammate_budget=2.0,
                non_teammate_budget=6.0,
                rho=0.5,
                reallocation_freq=1,
                solver="closed_form",
                fixed=True,
            ),
        )

        np.testing.assert_allclose(
            trainer._following_states[0].following_rates,
            np.array([0.0, 2.0, 3.0, 3.0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            trainer._following_states[2].following_rates,
            np.array([3.0, 3.0, 0.0, 2.0]),
            atol=1e-6,
        )

    def test_gpu_actor_updates_sequentially_per_decision(self):
        _, _, gpu_trainer = _make_dual_actor_critic_trainers(PickMode.CHOICE)
        state = _make_choice_cycle_start_state()
        _install_two_actor_choice_cycle_actions(
            gpu_trainer,
            actor0_move=Action.RIGHT,
            actor0_pick=make_pick_action(0),
            actor1_move=Action.STAY,
        )

        train_calls = [0, 0]
        before_params = _clone_actor_params(gpu_trainer.actor_networks)
        original_train_batches = [actor_net.train_batch for actor_net in gpu_trainer.actor_networks]

        for actor_id, actor_net in enumerate(gpu_trainer.actor_networks):
            original_train_batch = original_train_batches[actor_id]

            def _spy_train_batch(self, _orig=original_train_batch, _actor_id=actor_id):
                train_calls[_actor_id] += 1
                return _orig()

            actor_net.train_batch = MethodType(_spy_train_batch, actor_net)

        next_state = gpu_trainer.step(state, 0)
        assert next_state.actor == 1
        assert train_calls == [2, 0]
        assert all(len(actor_net.batch_states) == 0 for actor_net in gpu_trainer.actor_networks)

        params_changed = False
        for actor_before, actor_after in zip(before_params, gpu_trainer.actor_networks):
            for name, tensor in actor_after.state_dict().items():
                if not torch.allclose(actor_before[name], tensor):
                    params_changed = True
                    break
            if params_changed:
                break
        assert params_changed

        next_state = gpu_trainer.step(next_state, 1)
        assert next_state.actor == 0
        assert train_calls == [2, 1]
        assert all(len(actor_net.batch_states) == 0 for actor_net in gpu_trainer.actor_networks)

    def test_gpu_actor_checkpoints_do_not_store_pending_batches(self):
        _, _, gpu_trainer = _make_dual_actor_critic_trainers(PickMode.CHOICE)
        state = _make_choice_cycle_start_state()
        _install_two_actor_choice_cycle_actions(
            gpu_trainer,
            actor0_move=Action.RIGHT,
            actor0_pick=make_pick_action(0),
            actor1_move=Action.STAY,
        )

        gpu_trainer.step(state, 0)
        assert all(len(actor_net.batch_states) == 0 for actor_net in gpu_trainer.actor_networks)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "final.pt"
            gpu_trainer.sync_to_cpu()
            gpu_trainer.save_checkpoint(ckpt_path, step=1)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            pending_batches = ckpt["pending_actor_batches"]
            assert all(len(payload["states"]) == 0 for payload in pending_batches)

    def test_gpu_actor_flush_pending_updates_is_noop(self):
        _, _, gpu_trainer = _make_dual_actor_critic_trainers(PickMode.CHOICE)
        state = _make_choice_cycle_start_state()
        _install_two_actor_choice_cycle_actions(
            gpu_trainer,
            actor0_move=Action.RIGHT,
            actor0_pick=make_pick_action(0),
            actor1_move=Action.STAY,
        )

        gpu_trainer.step(state, 0)
        before_params = _clone_actor_params(gpu_trainer.actor_networks)
        gpu_trainer.flush_pending_updates()

        for actor_before, actor_after in zip(before_params, gpu_trainer.actor_networks):
            for name, tensor in actor_after.state_dict().items():
                torch.testing.assert_close(actor_before[name], tensor, atol=0.0, rtol=0.0)

    def test_step_reuses_sampled_actor_probabilities(self):
        _, move_only_trainer = _make_probability_count_trainer()
        move_only_calls = 0
        move_only_actor = move_only_trainer.actor_networks[0]
        original_move_only_get_action_probabilities = move_only_actor.get_action_probabilities

        def _spy_move_only_get_action_probabilities(self, enc, legal_mask):
            nonlocal move_only_calls
            move_only_calls += 1
            return original_move_only_get_action_probabilities(enc, legal_mask)

        move_only_actor.get_action_probabilities = MethodType(
            _spy_move_only_get_action_probabilities,
            move_only_actor,
        )

        move_only_trainer.step(_make_move_only_state(), 0)
        assert move_only_calls == 1

        _, choice_trainer = _make_probability_count_trainer()
        choice_calls = 0
        choice_actor = choice_trainer.actor_networks[0]
        original_choice_get_action_probabilities = choice_actor.get_action_probabilities

        def _spy_choice_get_action_probabilities(self, enc, legal_mask):
            nonlocal choice_calls
            choice_calls += 1
            return original_choice_get_action_probabilities(enc, legal_mask)

        choice_actor.get_action_probabilities = MethodType(
            _spy_choice_get_action_probabilities,
            choice_actor,
        )

        choice_trainer.step(_make_always_pick_choice_state(), 0)
        assert choice_calls == 2

    def test_gpu_step_reuses_sampled_actor_probability_tensors(self):
        _, _, gpu_trainer = _make_dual_actor_critic_trainers(PickMode.CHOICE)
        calls = 0
        actor = gpu_trainer.actor_networks[0]
        original_get_action_probabilities_tensor = actor.get_action_probabilities_tensor

        def _spy_get_action_probabilities_tensor(self, enc, legal_mask):
            nonlocal calls
            calls += 1
            return original_get_action_probabilities_tensor(enc, legal_mask)

        actor.get_action_probabilities_tensor = MethodType(
            _spy_get_action_probabilities_tensor,
            actor,
        )

        gpu_trainer.step(_make_always_pick_choice_state(), 0)
        assert calls == 2

    def test_warmup_skips_critic_updates(self):
        _, trainer = _make_actor_critic_trainer(PickMode.CHOICE)
        trainer._warmup_steps = 5
        spy = _install_identity_critic_spy(trainer)
        _install_scripted_actions(
            trainer,
            move_action=Action.RIGHT,
            pick_action=make_pick_action(0),
        )

        sentinel_after = State(
            agent_positions=(Grid(2, 0), Grid(2, 2)),
            task_positions=(),
            actor=1,
            task_types=(),
        )
        trainer._critic_prev_after = sentinel_after

        state = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(Grid(0, 1),),
            actor=0,
            task_types=(0,),
        )

        trainer.step(state, t=0)

        assert spy["td_calls"] == []
        assert trainer._critic_prev_after is None

    def test_warmup_skips_actor_updates(self):
        _, trainer = _make_actor_critic_trainer(PickMode.CHOICE)
        trainer._warmup_steps = 5
        _install_scripted_actions(
            trainer,
            move_action=Action.RIGHT,
            pick_action=make_pick_action(0),
        )

        before_params = _clone_actor_params(trainer.actor_networks)

        state = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(Grid(0, 1),),
            actor=0,
            task_types=(0,),
        )

        trainer.step(state, t=0)

        for actor_before, actor_after in zip(before_params, trainer.actor_networks):
            for name, tensor in actor_after.state_dict().items():
                torch.testing.assert_close(actor_before[name], tensor, atol=0.0, rtol=0.0)

    def test_warmup_resumes_critic_updates_after_threshold(self):
        _, trainer = _make_actor_critic_trainer(PickMode.CHOICE)
        trainer._warmup_steps = 5
        spy = _install_identity_critic_spy(trainer)
        _install_scripted_actions(
            trainer,
            move_action=Action.RIGHT,
            pick_action=make_pick_action(0),
        )

        sentinel_after = State(
            agent_positions=(Grid(2, 0), Grid(2, 2)),
            task_positions=(),
            actor=1,
            task_types=(),
        )
        trainer._critic_prev_after = sentinel_after

        state = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(Grid(0, 1),),
            actor=0,
            task_types=(0,),
        )

        trainer.step(state, t=5)

        assert len(spy["td_calls"]) == 2
        assert trainer._critic_prev_after is not None

    def test_warmup_does_not_skip_following_rate_alpha_updates(self):
        env_cfg = EnvConfig(
            height=3,
            width=3,
            n_agents=2,
            n_tasks=1,
            gamma=0.5,
            r_picker=1.0,
            n_task_types=1,
            r_low=0.0,
            task_assignments=((0,), (0,)),
            pick_mode=PickMode.CHOICE,
            max_tasks_per_type=1,
            stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=None, despawn_prob=0.0),
        )
        model_cfg = ModelConfig(
            encoder=EncoderType.BLIND_TASK_CNN_GRID,
            mlp_dims=(8,),
            conv_specs=((4, 3),),
        )
        train_cfg = TrainConfig(
            total_steps=1,
            seed=7,
            lr=ScheduleConfig(start=0.0, end=0.0),
            epsilon=ScheduleConfig(start=0.0, end=0.0),
            actor_lr=ScheduleConfig(start=0.0, end=0.0),
            algorithm=AlgorithmConfig(name=AlgorithmName.ACTOR_CRITIC),
            freeze_critic=True,
            following_rates=FollowingRatesConfig(
                enabled=True,
                budget=1.0,
                rho=1.0,
                reallocation_freq=1,
                fixed=True,
            ),
            learning_type=LearningType.DECENTRALIZED,
            use_gpu=False,
            td_lambda=0.0,
            heuristic=Heuristic.NEAREST_TASK,
            stopping=StoppingConfig(),
            warmup_steps=100,
        )
        cfg = ExperimentConfig(
            env=env_cfg,
            model=model_cfg,
            actor_model=None,
            train=train_cfg,
            eval=EvalConfig(),
            logging=LoggingConfig(output_dir="unused"),
        )
        encoding.init_encoder(model_cfg.encoder, env_cfg)
        env = create_env(env_cfg)
        trainer = create_trainer(cfg, env)

        def _critic_values_for_after_states(self, state, after_states):
            return [
                [10.0 * len(s.task_positions) for _ in range(self._n_agents)]
                for s in after_states
            ]

        trainer._critic_values_for_after_states = MethodType(
            _critic_values_for_after_states, trainer
        )

        pick_state = State(
            agent_positions=(Grid(1, 1), Grid(2, 2)),
            task_positions=(Grid(1, 1),),
            actor=0,
            task_types=(0,),
            pick_phase=True,
        )
        legal_mask = build_phase2_legal_mask(pick_state, env.cfg)
        actor_state = trainer._encode_actor_state(pick_state, 0)
        probs = trainer._actor_networks_list[0].get_action_probabilities(actor_state, legal_mask)

        prev_decision_count = trainer._decision_count
        trainer._train_decision(
            state=pick_state,
            action=make_pick_action(0),
            legal_mask=legal_mask,
            discount=1.0,
            t=0,
            actor_state=actor_state,
            probs=probs,
        )

        assert trainer._decision_count == prev_decision_count + 1
        assert not np.isclose(trainer._following_states[1].agent_alphas[0], 0.0)

    def test_alpha_update_uses_stay_baseline(self):
        env_cfg = EnvConfig(
            height=3,
            width=3,
            n_agents=2,
            n_tasks=1,
            gamma=0.5,
            r_picker=1.0,
            n_task_types=1,
            r_low=0.0,
            task_assignments=((0,), (0,)),
            pick_mode=PickMode.CHOICE,
            max_tasks_per_type=1,
            stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=None, despawn_prob=0.0),
        )
        model_cfg = ModelConfig(
            encoder=EncoderType.BLIND_TASK_CNN_GRID,
            mlp_dims=(8,),
            conv_specs=((4, 3),),
        )
        train_cfg = TrainConfig(
            total_steps=1,
            seed=7,
            lr=ScheduleConfig(start=0.0, end=0.0),
            epsilon=ScheduleConfig(start=0.0, end=0.0),
            actor_lr=ScheduleConfig(start=0.0, end=0.0),
            algorithm=AlgorithmConfig(name=AlgorithmName.ACTOR_CRITIC),
            freeze_critic=True,
            following_rates=FollowingRatesConfig(
                enabled=True,
                budget=1.0,
                rho=1.0,
                reallocation_freq=1,
                fixed=True,
            ),
            learning_type=LearningType.DECENTRALIZED,
            use_gpu=False,
            td_lambda=0.0,
            heuristic=Heuristic.NEAREST_TASK,
            stopping=StoppingConfig(),
        )
        cfg = ExperimentConfig(
            env=env_cfg,
            model=model_cfg,
            actor_model=None,
            train=train_cfg,
            eval=EvalConfig(),
            logging=LoggingConfig(output_dir="unused"),
        )
        encoding.init_encoder(model_cfg.encoder, env_cfg)
        env = create_env(env_cfg)
        trainer = create_trainer(cfg, env)

        def _critic_values_for_after_states(self, state, after_states):
            return [
                [10.0 * len(s.task_positions) for _ in range(self._n_agents)]
                for s in after_states
            ]

        trainer._critic_values_for_after_states = MethodType(
            _critic_values_for_after_states, trainer
        )

        pick_state = State(
            agent_positions=(Grid(1, 1), Grid(2, 2)),
            task_positions=(Grid(1, 1),),
            actor=0,
            task_types=(0,),
            pick_phase=True,
        )
        legal_mask = build_phase2_legal_mask(pick_state, env.cfg)
        actor_state = trainer._encode_actor_state(pick_state, 0)
        probs = trainer._actor_networks_list[0].get_action_probabilities(actor_state, legal_mask)

        trainer._train_decision(
            state=pick_state,
            action=make_pick_action(0),
            legal_mask=legal_mask,
            discount=1.0,
            t=0,
            actor_state=actor_state,
            probs=probs,
        )

        # Pick removes the only task → after_state has 0 tasks → V(after_pick) = 0
        # Stay keeps the task → after_state has 1 task → V(after_stay) = 10
        # Pick rewards: (1.0, 0.0); Stay rewards: (0.0, 0.0); discount = 1.0
        # Old definition would store Q1(s, pick) = 0 + 1.0*0 = 0
        # New definition stores Q1(s, pick) - Q1(s, Stay) = 0 - 10 = -10
        assert np.isclose(trainer._following_states[1].agent_alphas[0], -10.0)
        # Actor's own following state never updates its self-edge
        assert trainer._following_states[0].agent_alphas[0] == 0.0
        # Other observer dimensions of the non-actor remain at their initial values
        assert trainer._following_states[0].agent_alphas[1] == 0.0

    def test_freeze_critic_skips_critic_updates(self):
        _, trainer = _make_actor_critic_trainer(PickMode.CHOICE)
        trainer._freeze_critic = True
        spy = _install_identity_critic_spy(trainer)
        _install_scripted_actions(
            trainer,
            move_action=Action.RIGHT,
            pick_action=make_pick_action(0),
        )

        sentinel_after = State(
            agent_positions=(Grid(2, 0), Grid(2, 2)),
            task_positions=(),
            actor=1,
            task_types=(),
        )
        trainer._critic_prev_after = sentinel_after

        state = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(Grid(0, 1),),
            actor=0,
            task_types=(0,),
        )

        trainer.step(state, 0)

        assert spy["td_calls"] == []
        move_mask = build_phase1_legal_mask(state, trainer._env.cfg)
        pick_state = trainer._env.apply_action(state, Action.RIGHT).with_pick_phase()
        pick_mask = build_phase2_legal_mask(pick_state, trainer._env.cfg)
        assert len(spy["encode_calls"]) == int(move_mask.sum() + pick_mask.sum())
        assert trainer._critic_prev_after is None

    def test_actor_critic_step_records_action_and_env_timing(self):
        _, trainer = _make_actor_critic_trainer(PickMode.CHOICE)
        trainer._timer = Timer(enabled=True)
        _install_scripted_actions(
            trainer,
            move_action=Action.RIGHT,
            pick_action=make_pick_action(0),
        )

        trainer.step(_make_choice_cycle_start_state(), 0)

        assert trainer._timer._step_count == 1
        report = trainer._timer.report_and_reset()
        assert report[TimerSection.ACTION] > 0.0
        assert report[TimerSection.ENV] > 0.0
        assert report[TimerSection.ENCODE] > 0.0
        assert report[TimerSection.TRAIN] > 0.0

    def test_choice_mode_critic_td_uses_previous_after_state_chain(self):
        env, trainer = _make_actor_critic_trainer(PickMode.CHOICE)
        spy = _install_identity_critic_spy(trainer)
        _install_scripted_actions(
            trainer,
            move_action=Action.RIGHT,
            pick_action=make_pick_action(0),
        )

        sentinel_after = State(
            agent_positions=(Grid(2, 0), Grid(2, 2)),
            task_positions=(),
            actor=1,
            task_types=(),
        )
        trainer._critic_prev_after = sentinel_after

        state = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(Grid(0, 1),),
            actor=0,
            task_types=(0,),
        )

        trainer.step(state, 0)

        calls = spy["td_calls"]
        assert len(calls) == 2
        assert calls[0]["prev"] == sentinel_after
        assert calls[0]["rewards"] == (0.0, 0.0)
        assert calls[0]["discount"] == pytest.approx(env.cfg.gamma)
        assert calls[0]["current"].pick_phase is True
        assert calls[1]["prev"].pick_phase is True
        assert calls[1]["rewards"] == (1.0, 0.0)
        assert calls[1]["discount"] == pytest.approx(1.0)
        assert calls[1]["current"].pick_phase is False
        assert trainer._critic_prev_after.pick_phase is False

    def test_forced_pick_turn_inserts_critic_only_pick_followup(self):
        env, trainer = _make_actor_critic_trainer(PickMode.FORCED)
        spy = _install_identity_critic_spy(trainer)
        _install_scripted_actions(trainer, move_action=Action.RIGHT)

        sentinel_after = State(
            agent_positions=(Grid(2, 0), Grid(2, 2)),
            task_positions=(),
            actor=1,
            task_types=(),
        )
        trainer._critic_prev_after = sentinel_after

        state = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(Grid(0, 1),),
            actor=0,
            task_types=(0,),
        )

        trainer.step(state, 0)

        calls = spy["td_calls"]
        assert len(calls) == 2
        assert calls[0]["prev"] == sentinel_after
        assert calls[0]["rewards"] == (0.0, 0.0)
        assert calls[0]["discount"] == pytest.approx(env.cfg.gamma)
        assert calls[0]["current"].pick_phase is True
        assert calls[1]["prev"].pick_phase is True
        assert calls[1]["rewards"] == (1.0, 0.0)
        assert calls[1]["discount"] == pytest.approx(1.0)
        assert calls[1]["current"].pick_phase is False
        assert trainer._critic_prev_after.pick_phase is False

    def test_forced_mode_defers_actor_updates_until_round_robin_wrap(self):
        _, trainer = _make_actor_critic_trainer(PickMode.FORCED)
        _install_identity_critic_spy(trainer)
        _install_scripted_actions(trainer, move_action=Action.STAY)

        assert trainer._actor_bt is not None
        train_alphas: list[float] = []
        original_train_batch_batched = trainer._actor_bt.train_batch_batched

        def _spy_train_batch_batched(self, alpha: float, _orig=original_train_batch_batched):
            train_alphas.append(float(alpha))
            return _orig(alpha)

        trainer._actor_bt.train_batch_batched = MethodType(
            _spy_train_batch_batched,
            trainer._actor_bt,
        )

        state = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(),
            actor=0,
            task_types=(),
        )

        next_state = trainer.step(state, 0)
        assert next_state.actor == 1
        assert train_alphas == []
        assert [len(actor_net.batch_states) for actor_net in trainer.actor_networks] == [1, 0]

        next_state = trainer.step(next_state, 1)
        assert next_state.actor == 0
        assert len(train_alphas) == 1
        assert all(len(actor_net.batch_states) == 0 for actor_net in trainer.actor_networks)

    def test_forced_mode_flushes_partial_actor_cycle(self):
        _, trainer = _make_actor_critic_trainer(PickMode.FORCED)
        _install_identity_critic_spy(trainer)
        _install_scripted_actions(trainer, move_action=Action.STAY)

        assert trainer._actor_bt is not None
        train_alphas: list[float] = []
        original_train_batch_batched = trainer._actor_bt.train_batch_batched

        def _spy_train_batch_batched(self, alpha: float, _orig=original_train_batch_batched):
            train_alphas.append(float(alpha))
            return _orig(alpha)

        trainer._actor_bt.train_batch_batched = MethodType(
            _spy_train_batch_batched,
            trainer._actor_bt,
        )

        state = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(),
            actor=0,
            task_types=(),
        )

        trainer.step(state, 0)
        trainer.flush_pending_updates()

        assert len(train_alphas) == 1
        assert all(len(actor_net.batch_states) == 0 for actor_net in trainer.actor_networks)

    def test_forced_mode_can_disable_batched_actor_updates_and_train_immediately(self):
        _, trainer = _make_actor_critic_trainer(
            PickMode.FORCED,
            batch_forced_actor_updates=False,
        )
        _install_identity_critic_spy(trainer)
        _install_scripted_actions(trainer, move_action=Action.STAY)

        assert trainer._actor_bt is None
        train_counts = [0 for _ in trainer.actor_networks]
        for actor_id, actor_net in enumerate(trainer.actor_networks):
            original_train_batch = actor_net.train_batch

            def _spy_train_batch(_orig=original_train_batch, _actor_id=actor_id):
                train_counts[_actor_id] += 1
                return _orig()

            actor_net.train_batch = _spy_train_batch

        state = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(),
            actor=0,
            task_types=(),
        )

        next_state = trainer.step(state, 0)
        assert next_state.actor == 1
        assert train_counts == [1, 0]
        assert all(len(actor_net.batch_states) == 0 for actor_net in trainer.actor_networks)

        next_state = trainer.step(next_state, 1)
        assert next_state.actor == 0
        assert train_counts == [1, 1]
        assert all(len(actor_net.batch_states) == 0 for actor_net in trainer.actor_networks)

    @pytest.mark.parametrize("pick_mode", [PickMode.FORCED, PickMode.CHOICE])
    def test_wrong_type_task_does_not_enter_pick_phase(self, pick_mode: PickMode):
        env, trainer = _make_actor_critic_trainer(
            pick_mode,
            n_task_types=2,
            task_assignments=((0,), (1,)),
        )
        spy = _install_identity_critic_spy(trainer)
        _install_scripted_actions(trainer, move_action=Action.RIGHT)

        sentinel_after = State(
            agent_positions=(Grid(2, 0), Grid(2, 2)),
            task_positions=(),
            actor=1,
            task_types=(),
        )
        trainer._critic_prev_after = sentinel_after

        state = State(
            agent_positions=(Grid(0, 0), Grid(2, 2)),
            task_positions=(Grid(0, 1),),
            actor=0,
            task_types=(1,),
        )

        next_state = trainer.step(state, 0)

        calls = spy["td_calls"]
        assert len(calls) == 1
        assert calls[0]["prev"] == sentinel_after
        assert calls[0]["rewards"] == (0.0, 0.0)
        assert calls[0]["discount"] == pytest.approx(env.cfg.gamma)
        assert calls[0]["current"].pick_phase is False
        assert calls[0]["current"].task_positions == (Grid(0, 1),)
        assert trainer._critic_prev_after.pick_phase is False
        assert next_state.task_positions == (Grid(0, 1),)

    @pytest.mark.parametrize("pick_mode", ["forced", "choice"])
    def test_actor_critic_end_to_end_writes_runtime_artifacts(self, pick_mode: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _run_actor_critic_case(pick_mode=pick_mode, output_dir=tmpdir)

            assert (run_dir / "metadata.yaml").exists()
            assert (run_dir / "metrics.csv").exists()
            assert (run_dir / "details.csv").exists()
            assert (run_dir / "checkpoints" / "step_0.pt").exists()
            assert (run_dir / "checkpoints" / "final.pt").exists()
            assert (run_dir / "phase1_policy_probabilities.csv").exists()
            assert (run_dir / "phase2_policy_probabilities.csv").exists()
            final_ckpt = torch.load(run_dir / "checkpoints" / "final.pt", map_location="cpu", weights_only=True)
            assert all(len(payload["states"]) == 0 for payload in final_ckpt["pending_actor_batches"])

            metrics_row = _read_single_row(run_dir / "metrics.csv")
            assert "actor_lr" in metrics_row
            assert "actor_loss_mean" in metrics_row
            assert "advantage_mean" in metrics_row
            assert "policy_entropy_mean" in metrics_row

            details_row = _read_single_row(run_dir / "details.csv")
            assert "current_actor_lr" in details_row

    def test_actor_critic_env_trace_writes_value_style_rows(self, monkeypatch):
        if not hasattr(os, "sched_getaffinity"):
            monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0}, raising=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _run_actor_critic_case(
                pick_mode="forced",
                output_dir=tmpdir,
                extra_logging_blocks="""
  env_trace: true
""",
            )

            trace_path = run_dir / "env_trace.csv"
            assert trace_path.exists()

            with open(trace_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert rows
            assert reader.fieldnames == [
                "step", "actor", "epsilon", "action", "on_task", "pick_happened", "pick_task_type",
                "reward_0", "reward_1",
                "n_tasks_before_spawn", "tasks_despawned", "tasks_spawned",
                "n_tasks_after", "task_positions_after", "task_types_after",
                "agent_positions", "agent_positions_indexed",
                "was_greedy", "best_val", "td_delta_sq",
                "actor_selected_q", "actor_baseline", "actor_advantage",
                "enc_grid_l2", "enc_scalar",
            ]
            first = rows[0]
            assert first["step"] != ""
            assert first["actor"] != ""
            assert first["action"] in {"UP", "DOWN", "LEFT", "RIGHT", "STAY"}
            assert first["n_tasks_after"] != ""
            assert first["agent_positions_indexed"] != ""

    def test_actor_critic_timing_csv_reports_action_and_env_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _run_actor_critic_case(
                pick_mode="choice",
                output_dir=tmpdir,
                extra_logging_blocks="""
  timing_csv_freq: 5
""",
            )

            timing_path = run_dir / "timing.csv"
            assert timing_path.exists()

            with open(timing_path) as f:
                rows = list(csv.DictReader(f))

            assert len(rows) == 1
            assert float(rows[0]["total_step_ms"]) > 0.0
            assert float(rows[0]["env_ms"]) > 0.0

    def test_actor_critic_following_rates_write_snapshots_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _run_actor_critic_case(
                pick_mode="choice",
                output_dir=tmpdir,
                extra_train_blocks="""
  following_rates:
    enabled: true
    budget: 1.0
    rho: 0.5
    reallocation_freq: 1
    solver: closed_form
""",
            )

            assert (run_dir / "following_rates_agent_0.csv").exists()
            assert (run_dir / "following_rates_agent_1.csv").exists()
            metrics_row = _read_single_row(run_dir / "metrics.csv")
            assert "alpha_mean" in metrics_row
            assert "following_weight_mean" in metrics_row
            assert "effective_follow_weight_mean" in metrics_row

    def test_actor_critic_influencer_writes_snapshots_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _run_actor_critic_case(
                pick_mode="choice",
                output_dir=tmpdir,
                extra_train_blocks="""
  following_rates:
    enabled: true
    budget: 1.0
    rho: 0.5
    reallocation_freq: 1
    solver: closed_form
  influencer:
    enabled: true
    budget: 0.5
""",
            )

            assert (run_dir / "external_influencer.csv").exists()
            metrics_row = _read_single_row(run_dir / "metrics.csv")
            assert "beta_mean" in metrics_row
            assert "influencer_weight_mean" in metrics_row

    def test_actor_critic_resume_from_new_checkpoint_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first_run = _run_actor_critic_case(pick_mode="choice", output_dir=tmpdir)
            resume_path = first_run / "checkpoints" / "final.pt"
            second_run = _run_actor_critic_case(
                pick_mode="choice",
                output_dir=tmpdir,
                resume_checkpoint=str(resume_path),
            )
            assert second_run != first_run
            assert (second_run / "checkpoints" / "final.pt").exists()

    def test_actor_critic_can_load_value_checkpoint_and_freeze_critic(self):
        env_cfg = EnvConfig(
            height=4,
            width=4,
            n_agents=2,
            n_tasks=2,
            gamma=0.99,
            r_picker=1.0,
            n_task_types=2,
            r_low=0.0,
            task_assignments=((0,), (1,)),
            pick_mode=PickMode.FORCED,
            max_tasks_per_type=2,
            stochastic=StochasticConfig(
                spawn_prob=0.1,
                despawn_mode=DespawnMode.PROBABILITY,
                despawn_prob=0.05,
            ),
        )
        model_cfg = ModelConfig(
            encoder=EncoderType.BLIND_TASK_CNN_GRID,
            mlp_dims=(16,),
            conv_specs=((4, 3),),
        )
        train_cfg = TrainConfig(
            total_steps=5,
            seed=42,
            lr=ScheduleConfig(start=0.01, end=0.01),
            epsilon=ScheduleConfig(start=0.1, end=0.1),
            learning_type=LearningType.DECENTRALIZED,
            use_gpu=False,
            td_lambda=0.0,
            heuristic=Heuristic.NEAREST_TASK,
            stopping=StoppingConfig(),
        )
        encoding.init_encoder(model_cfg.encoder, env_cfg)
        networks = create_networks(model_cfg, env_cfg, train_cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            value_ckpt = Path(tmpdir) / "value.pt"
            torch.save(
                {"algorithm": "value", "step": 7, "critics": [net.state_dict() for net in networks]},
                value_ckpt,
            )

            run_dir = _run_actor_critic_case(
                pick_mode="forced",
                output_dir=tmpdir,
                resume_checkpoint=str(value_ckpt),
                extra_train_blocks="""
  freeze_critic: true
""",
            )
            assert (run_dir / "checkpoints" / "final.pt").exists()

    def test_value_mode_resume_accepts_legacy_checkpoint_format(self):
        env_cfg = EnvConfig(
            height=3,
            width=3,
            n_agents=2,
            n_tasks=1,
            gamma=0.99,
            r_picker=1.0,
            n_task_types=1,
            task_assignments=((0,), (0,)),
            pick_mode=PickMode.FORCED,
            max_tasks_per_type=1,
            stochastic=StochasticConfig(spawn_prob=0.0, despawn_mode=None, despawn_prob=0.0),
        )
        model_cfg = ModelConfig(encoder=EncoderType.BLIND_TASK_CNN_GRID, mlp_dims=(16,))
        train_cfg = TrainConfig(
            total_steps=5,
            seed=42,
            lr=ScheduleConfig(start=0.01, end=0.01),
            epsilon=ScheduleConfig(start=0.1, end=0.1),
            learning_type=LearningType.DECENTRALIZED,
            use_gpu=False,
            td_lambda=0.0,
            heuristic=Heuristic.NEAREST_TASK,
            stopping=StoppingConfig(),
        )
        encoding.init_encoder(model_cfg.encoder, env_cfg)
        networks = create_networks(model_cfg, env_cfg, train_cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_ckpt = Path(tmpdir) / "legacy.pt"
            torch.save(
                {"step": 7, "networks": [net.state_dict() for net in networks]},
                legacy_ckpt,
            )

            yaml_str = """
env:
  height: 3
  width: 3
  n_agents: 2
  n_tasks: 1
  gamma: 0.99
  r_picker: 1.0
  pick_mode: forced
  task_assignments: [[0], [0]]
  stochastic:
    spawn_prob: 0.0
    despawn_mode: none
    despawn_prob: 0.0
model:
  encoder: blind_task_cnn_grid
  mlp_dims: [16]
train:
  learning_type: decentralized
  use_gpu: false
  total_steps: 5
  seed: 42
  heuristic: nearest_task
  lr:
    start: 0.01
  epsilon:
    start: 0.1
eval:
  eval_steps: 5
  n_test_states: 2
logging:
  main_csv_freq: 5
  detail_csv_freq: 5
  output_dir: {output_dir}
""".format(output_dir=tmpdir)
            path = _write_config(yaml_str)
            cfg = load_config(path)
            train(cfg, resume_checkpoint=str(legacy_ckpt))
            os.unlink(path)

            run_dir = _latest_run_dir(tmpdir)
            assert (run_dir / "checkpoints" / "final.pt").exists()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_actor_critic_gpu_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _run_actor_critic_case(
                pick_mode="forced",
                output_dir=tmpdir,
                use_gpu="true",
            )
            assert (run_dir / "checkpoints" / "final.pt").exists()
