"""Tests for choice pick T=1 ≡ T=M equivalence and STAY after-state fix.

1. STAY fix: under blind encoding, Q(STAY) and Q(pick(stranger)) use identical
   after-state encodings (both pick_phase=False, stranger task invisible in Ch0).

2. Gamma accumulation: _gamma_accum_per_team correctly tracks accumulated discount
   across stranger steps, and simulate_stranger_gap reproduces T=M accumulation in T=1.

3. T=1 ≡ T=M weight-by-weight for choice pick (using simulate_stranger_gap).
"""

from __future__ import annotations

import copy
import dataclasses

import torch

import orchard.encoding as encoding
from orchard.datatypes import EnvConfig, ExperimentConfig, Grid, State, StochasticConfig
from orchard.encoding.grid import BlindTaskGridEncoder
from orchard.enums import PickMode
from orchard.config import _parse_env, _parse_model, _parse_train, _parse_eval, _parse_logging
from orchard.env import create_env
from orchard.seed import set_all_seeds
from orchard.trainer import create_trainer


# ---------------------------------------------------------------------------
# Helpers shared with test_per_type_seeds
# ---------------------------------------------------------------------------

def _make_cfg(n_task_types: int, per_type_seeds: list[int] | None,
              pick_mode: str = "forced", base_seed: int = 1234,
              total_steps: int = 50, simulate_stranger_gap: int = 0,
              use_gpu: bool = False):
    n_agents = 2 * n_task_types
    task_assignments = [[k] for k in range(n_task_types) for _ in range(2)]
    raw = {
        "env": {
            "height": 6, "width": 6,
            "n_agents": n_agents,
            "n_tasks": 9,
            "gamma": 0.99,
            "r_picker": -1.0,
            "n_task_types": n_task_types,
            "r_low": -1.0,
            "task_assignments": task_assignments,
            "pick_mode": pick_mode,
            "max_tasks_per_type": 9,
            "stochastic": {
                "spawn_prob": 0.08,
                "despawn_mode": "probability",
                "despawn_prob": 0.1,
                "task_spawn_mode": None,
                "spawn_on_agent_cells": True,
                "spawn_at_round_end": True,
                **({"per_type_seeds": per_type_seeds} if per_type_seeds is not None else {}),
            },
        },
        "model": {
            "encoder": "blind_task_cnn_grid",
            "mlp_dims": [],
            "conv_specs": [[64, 3], [64, 3]],
            "activation": "relu",
            "weight_init": "default",
        },
        "actor_model": None,
        "train": {
            "total_steps": total_steps,
            "seed": base_seed,
            "lr": {"start": 0.001, "end": 0.001, "schedule": "none",
                   "step_size": 0, "step_factor": 1.0, "step_start": 0},
            "epsilon": {"start": 0.5, "end": 0.5, "schedule": "none",
                        "step_size": 0, "step_factor": 1.0, "step_start": 0},
            "actor_lr": None,
            "freeze_critic": False,
            "algorithm": {"name": "value"},
            "following_rates": {
                "enabled": False, "budget": 0.0, "teammate_budget": None,
                "non_teammate_budget": None, "rho": 0.0, "reallocation_freq": 1,
                "solver": "closed_form", "fixed": False,
            },
            "influencer": {"enabled": False, "budget": 0.0},
            "learning_type": "decentralized",
            "use_gpu": use_gpu,
            "td_lambda": 0.3,
            "comm_only_teammates": False,
            "heuristic": "nearest_correct_task_stay_wrong",
            "stopping": {
                "condition": "none", "patience_steps": 10000,
                "improvement_threshold": 0.01, "min_steps_before_stop": 0,
            },
            "warmup_steps": 0,
            "train_only_teammates": True,
            "simulate_stranger_gap": simulate_stranger_gap,
        },
        "eval": {"eval_steps": 100, "n_test_states": 5, "checkpoint_freq": 0},
        "logging": {
            "output_dir": "/tmp/orchard_test_choice_pick",
            "main_csv_freq": 1000000,
            "detail_csv_freq": 1000000,
            "timing_csv_freq": 0,
            "alpha_state_log_freq": 0,
            "env_trace": False,
        },
    }
    env_cfg = _parse_env(raw["env"])
    model_cfg = _parse_model(raw["model"])
    train_cfg = _parse_train(raw["train"], n_task_types=env_cfg.n_task_types)
    return ExperimentConfig(
        env=env_cfg, model=model_cfg, actor_model=None, train=train_cfg,
        eval=_parse_eval(raw.get("eval", {})),
        logging=_parse_logging(raw.get("logging", {})),
    )


def _get_init_weights(cfg) -> list[dict]:
    set_all_seeds(cfg.train.seed)
    env = create_env(cfg.env)
    encoding.init_encoder(cfg.model.encoder, cfg.env)
    trainer = create_trainer(cfg, env)
    return [copy.deepcopy(net.state_dict()) for net in trainer.critic_networks]


def _run(cfg, init_weights: list[dict] | None = None) -> list[dict]:
    set_all_seeds(cfg.train.seed)
    env = create_env(cfg.env)
    encoding.init_encoder(cfg.model.encoder, cfg.env)
    trainer = create_trainer(cfg, env)
    if init_weights is not None:
        for net, sd in zip(trainer.critic_networks, init_weights):
            net.load_state_dict(sd)
    state = env.init_state()
    for t in range(cfg.train.total_steps):
        state = trainer.step(state, t)
    return [copy.deepcopy(net.state_dict()) for net in trainer.critic_networks]


def _tensor_dict_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------------------
# Test 1: STAY after-state fix
# ---------------------------------------------------------------------------

def test_stay_and_pick_stranger_identical_encoding():
    """After the STAY fix, blind encoder gives identical after-states for STAY
    and pick(stranger_type) during choice pick phase.

    Setup: actor is on a cell with one own-type task (type 0) AND one stranger
    task (type 1). With blind encoding, the stranger task is invisible.
    After STAY: pick_phase=False, tasks unchanged → same encoding as...
    After pick(stranger): pick_phase=False, stranger task removed (still invisible).
    Both should produce identical grids and scalars.
    """
    from orchard.enums import make_pick_action

    cfg = _make_cfg(n_task_types=2, per_type_seeds=None, pick_mode="choice")
    encoding.init_encoder(cfg.model.encoder, cfg.env)
    env = create_env(cfg.env)

    enc = encoding.get_encoder()
    assert isinstance(enc, BlindTaskGridEncoder), "Expected BlindTaskGridEncoder"

    # Manually construct a state: actor=0 (type 0) on cell (2,2)
    # Own task (type 0) at (2,2), stranger task (type 1) also at (2,2)
    agent_positions = (
        Grid(2, 2),  # agent 0 (type 0) — the actor
        Grid(0, 0),  # agent 1 (type 0) — teammate
        Grid(5, 5),  # agent 2 (type 1) — stranger
        Grid(5, 4),  # agent 3 (type 1) — stranger
    )
    task_positions = (Grid(2, 2), Grid(2, 2))
    task_types = (0, 1)  # own task type 0 and stranger task type 1, same cell

    pick_phase_state = State(
        agent_positions=agent_positions,
        task_positions=task_positions,
        actor=0,
        task_types=task_types,
        pick_phase=True,
    )

    # STAY after-state: pick_phase=False, tasks unchanged
    stay_after = dataclasses.replace(pick_phase_state, pick_phase=False)

    # pick(stranger=type 1) after-state: stranger task removed, pick_phase=False
    # Task at index 1 is type 1 (stranger). Remove it.
    pick_stranger_after = State(
        agent_positions=agent_positions,
        task_positions=(Grid(2, 2),),   # only own task remains
        actor=0,
        task_types=(0,),
        pick_phase=False,
    )

    # Encode both after-states for agent 0 (actor, type 0)
    stay_enc = enc.encode(stay_after, agent_idx=0)
    stranger_enc = enc.encode(pick_stranger_after, agent_idx=0)

    assert torch.equal(stay_enc.grid, stranger_enc.grid), (
        "STAY and pick(stranger) after-state grids differ under blind encoding.\n"
        f"STAY grid:\n{stay_enc.grid}\nStranger grid:\n{stranger_enc.grid}"
    )
    assert torch.equal(stay_enc.scalar, stranger_enc.scalar), (
        "STAY and pick(stranger) after-state scalars differ under blind encoding.\n"
        f"STAY scalar: {stay_enc.scalar}\nStranger scalar: {stranger_enc.scalar}"
    )


# ---------------------------------------------------------------------------
# Test 2: gamma accumulation correctness
# ---------------------------------------------------------------------------

def test_gamma_accum_accumulates_for_strangers():
    """_gamma_accum_per_team[k] grows by gamma for each stranger step and resets
    to simulate_stranger_gap**gamma on own step.
    """
    cfg = _make_cfg(n_task_types=2, per_type_seeds=[1000, 1001],
                    simulate_stranger_gap=0, total_steps=0)
    env = create_env(cfg.env)
    encoding.init_encoder(cfg.model.encoder, cfg.env)

    # Build trainer with simulate_stranger_gap=0 (T=M mode)
    from orchard.trainer.cpu import CpuTrainer
    from orchard.model import create_networks
    networks = create_networks(cfg.model, cfg.env, cfg.train)
    trainer = CpuTrainer(
        network_list=networks,
        env=env,
        gamma=cfg.env.gamma,
        epsilon_schedule=cfg.train.epsilon,
        lr_schedule=cfg.train.lr,
        total_steps=10,
        heuristic=cfg.train.heuristic,
        train_only_teammates=True,
        per_type_seeds=cfg.env.stochastic.per_type_seeds,
        simulate_stranger_gap=0,
    )

    gamma = cfg.env.gamma
    assert trainer._gamma_accum_per_team is not None

    # Simulate: team 0 acts, accumulates for team 1
    # Before: both accum = 1.0
    assert trainer._gamma_accum_per_team[0] == 1.0
    assert trainer._gamma_accum_per_team[1] == 1.0

    # Manually trigger the accumulation logic as step() would:
    # team 0 agent acts → team 1 accumulates
    team_idx = 0
    eff = trainer._gamma_accum_per_team[team_idx] * gamma
    trainer._gamma_accum_per_team[team_idx] = 1.0  # reset (simulate_stranger_gap=0)
    for k in range(len(trainer._gamma_accum_per_team)):
        if k != team_idx:
            trainer._gamma_accum_per_team[k] *= gamma

    assert abs(eff - gamma) < 1e-9, f"First own step eff_gamma should be gamma, got {eff}"
    assert abs(trainer._gamma_accum_per_team[0] - 1.0) < 1e-9
    assert abs(trainer._gamma_accum_per_team[1] - gamma) < 1e-9, (
        f"After team 0 step, team 1 accum should be gamma, got {trainer._gamma_accum_per_team[1]}"
    )

    # Team 1 acts → team 0 accumulates, team 1 resets
    team_idx = 1
    eff1 = trainer._gamma_accum_per_team[team_idx] * gamma
    trainer._gamma_accum_per_team[team_idx] = 1.0
    for k in range(len(trainer._gamma_accum_per_team)):
        if k != team_idx:
            trainer._gamma_accum_per_team[k] *= gamma

    assert abs(eff1 - gamma * gamma) < 1e-9, (
        f"Team 1 eff_gamma should be gamma^2 (1 stranger step before), got {eff1}"
    )
    assert abs(trainer._gamma_accum_per_team[0] - gamma) < 1e-9
    assert abs(trainer._gamma_accum_per_team[1] - 1.0) < 1e-9


def test_simulate_stranger_gap_matches_TM_accumulation():
    """T=1 with simulate_stranger_gap=n_strangers produces the same eff_gamma
    sequence as T=M after the first round.
    """
    # In T=2 with 2 agents per team: n_strangers=2, so between team k's consecutive
    # turns gamma accumulates gamma^2 from strangers → eff_gamma = gamma^3.
    # T=1 with simulate_stranger_gap=2 should also see eff_gamma = gamma^3.
    cfg = _make_cfg(n_task_types=1, per_type_seeds=[1000],
                    simulate_stranger_gap=2, total_steps=0)
    env = create_env(cfg.env)
    encoding.init_encoder(cfg.model.encoder, cfg.env)

    from orchard.trainer.cpu import CpuTrainer
    from orchard.model import create_networks
    networks = create_networks(cfg.model, cfg.env, cfg.train)
    trainer = CpuTrainer(
        network_list=networks, env=env,
        gamma=cfg.env.gamma,
        epsilon_schedule=cfg.train.epsilon,
        lr_schedule=cfg.train.lr,
        total_steps=10,
        heuristic=cfg.train.heuristic,
        train_only_teammates=True,
        per_type_seeds=cfg.env.stochastic.per_type_seeds,
        simulate_stranger_gap=2,
    )

    gamma = cfg.env.gamma
    assert trainer._gamma_accum_per_team is not None

    # First own step: accum=1.0, eff=gamma, then reset to gamma^2
    eff0 = trainer._gamma_accum_per_team[0] * gamma
    trainer._gamma_accum_per_team[0] = gamma ** trainer._simulate_stranger_gap
    assert abs(eff0 - gamma) < 1e-9

    # Second own step: accum=gamma^2, eff=gamma^3
    eff1 = trainer._gamma_accum_per_team[0] * gamma
    trainer._gamma_accum_per_team[0] = gamma ** trainer._simulate_stranger_gap
    assert abs(eff1 - gamma ** 3) < 1e-9, (
        f"T=1 with simulate_stranger_gap=2 should give eff_gamma=gamma^3, got {eff1}"
    )

    # Third own step: same pattern
    eff2 = trainer._gamma_accum_per_team[0] * gamma
    assert abs(eff2 - gamma ** 3) < 1e-9


# ---------------------------------------------------------------------------
# Test 3: T=1 ≡ T=M weight-by-weight for choice pick
# ---------------------------------------------------------------------------

def _run_choice_equivalence(use_gpu: bool):
    """T=1 with simulate_stranger_gap=2 is bitwise identical to T=2 for choice pick."""
    seeds = [1000, 1001]
    base_steps = 50
    n_strangers = 2  # T=2 has 4 agents, 2 per team

    cfg_t1_run0 = _make_cfg(n_task_types=1, per_type_seeds=[seeds[0]],
                             pick_mode="choice", total_steps=base_steps,
                             simulate_stranger_gap=n_strangers, use_gpu=use_gpu)
    cfg_t1_run1 = _make_cfg(n_task_types=1, per_type_seeds=[seeds[1]],
                             pick_mode="choice", total_steps=base_steps,
                             simulate_stranger_gap=n_strangers, use_gpu=use_gpu)
    w_t1_run0 = _run(cfg_t1_run0)
    w_t1_run1 = _run(cfg_t1_run1)

    cfg_t2 = _make_cfg(n_task_types=2, per_type_seeds=seeds,
                       pick_mode="choice", total_steps=2 * base_steps, use_gpu=use_gpu)
    init_t1 = _get_init_weights(cfg_t1_run0)
    w_t2 = _run(cfg_t2, init_weights=init_t1 + init_t1)

    label = f"gpu={use_gpu}, choice"
    assert _tensor_dict_equal(w_t2[0], w_t1_run0[0]), f"Team 0 agent 0 differs ({label})"
    assert _tensor_dict_equal(w_t2[1], w_t1_run0[1]), f"Team 0 agent 1 differs ({label})"
    assert _tensor_dict_equal(w_t2[2], w_t1_run1[0]), f"Team 1 agent 0 differs ({label})"
    assert _tensor_dict_equal(w_t2[3], w_t1_run1[1]), f"Team 1 agent 1 differs ({label})"


def test_choice_pick_T1_equiv_T2_cpu():
    _run_choice_equivalence(use_gpu=False)


def test_choice_pick_T1_equiv_T2_gpu():
    _run_choice_equivalence(use_gpu=True)
