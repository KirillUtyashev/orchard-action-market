"""Runtime integration tests for orchard actor-critic training."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

import orchard.encoding as encoding
from orchard.actor_critic import build_phase1_legal_mask, build_phase2_legal_mask
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
    Grid,
    LoggingConfig,
    ModelConfig,
    ScheduleConfig,
    State,
    StochasticConfig,
    StoppingConfig,
    TrainConfig,
)
from orchard.model import create_networks
from orchard.trainer import create_trainer
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
"""


def _run_actor_critic_case(
    *,
    pick_mode: str,
    output_dir: str,
    extra_train_blocks: str = "",
    resume_checkpoint: str | None = None,
    use_gpu: str = "false",
) -> Path:
    yaml_str = ACTOR_CRITIC_CONFIG.format(
        pick_mode=pick_mode,
        output_dir=output_dir,
        extra_train_blocks=extra_train_blocks,
        resume_checkpoint=resume_checkpoint,
        use_gpu=use_gpu,
    )
    path = _write_config(yaml_str)
    cfg = load_config(path)
    train(cfg, resume_checkpoint=resume_checkpoint)
    os.unlink(path)
    return _latest_run_dir(output_dir)


def _make_actor_critic_trainer(pick_mode: PickMode):
    env_cfg = EnvConfig(
        height=3,
        width=3,
        n_agents=2,
        n_tasks=1,
        gamma=0.99,
        r_picker=1.0,
        n_task_types=1,
        r_low=0.0,
        task_assignments=((0,), (0,)),
        pick_mode=pick_mode,
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


def _install_identity_critic_spy(trainer):
    calls: list[dict[str, object]] = []

    def _encode_all_critics(self, state: State) -> State:
        return state

    def _critic_values(self, encoded_state: State) -> list[float]:
        return [0.0 for _ in range(self._n_agents)]

    def _critic_td_step(self, prev: State, rewards, discount: float, current: State, t: int) -> float:
        calls.append(
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
    return calls


def _install_scripted_actions(trainer, *, move_action: Action, pick_action: Action | None = None) -> None:
    def _sample_action(self, state: State, phase2: bool):
        if phase2:
            assert pick_action is not None
            mask = build_phase2_legal_mask(state, self._env.cfg)
            return pick_action, np.zeros(mask.shape[0], dtype=float), mask
        mask = build_phase1_legal_mask(state, self._env.cfg)
        return move_action, np.zeros(mask.shape[0], dtype=float), mask

    trainer._sample_action = MethodType(_sample_action, trainer)


class TestActorCriticTrainingLoop:
    def test_freeze_critic_skips_critic_updates(self):
        _, trainer = _make_actor_critic_trainer(PickMode.CHOICE)
        trainer._freeze_critic = True
        calls = _install_identity_critic_spy(trainer)
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

        assert calls == []
        assert trainer._critic_prev_after.pick_phase is False

    def test_choice_mode_critic_td_uses_previous_after_state_chain(self):
        env, trainer = _make_actor_critic_trainer(PickMode.CHOICE)
        calls = _install_identity_critic_spy(trainer)
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
        calls = _install_identity_critic_spy(trainer)
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

            metrics_row = _read_single_row(run_dir / "metrics.csv")
            assert "actor_lr" in metrics_row
            assert "actor_loss_mean" in metrics_row
            assert "advantage_mean" in metrics_row
            assert "policy_entropy_mean" in metrics_row

            details_row = _read_single_row(run_dir / "details.csv")
            assert "current_actor_lr" in details_row

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
