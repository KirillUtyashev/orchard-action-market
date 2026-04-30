"""Config loading: YAML → ExperimentConfig with validation and enum conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from orchard.enums import (
    Activation,
    AlgorithmName,
    DespawnMode,
    EncoderType,
    Heuristic,
    LearningType,
    PickMode,
    Schedule,
    SpawnZoneMode,
    StoppingCondition,
    TaskSpawnMode,
    WeightInit,
)
from orchard.datatypes import (
    AlgorithmConfig,
    EnvConfig,
    EvalConfig,
    ExperimentConfig,
    FollowingRatesConfig,
    InfluencerConfig,
    LoggingConfig,
    ModelConfig,
    ScheduleConfig,
    StochasticConfig,
    StoppingConfig,
    TrainConfig,
    compute_task_assignments,
)
from orchard.following_rates import get_supported_rate_solver_names, is_scipy_rate_solver_available


# ---------------------------------------------------------------------------
# String → Enum maps
# ---------------------------------------------------------------------------
_ENUM_MAPS: dict[str, dict[str, Any]] = {
    "encoder": {
        "cnn_grid": EncoderType.CNN_GRID,
        "blind_task_cnn_grid": EncoderType.BLIND_TASK_CNN_GRID,
        "filtered_task_cnn_grid": EncoderType.FILTERED_TASK_CNN_GRID,
        "position_aware_task_cnn_grid": EncoderType.POSITION_AWARE_TASK_CNN_GRID,
        "centralized_task_cnn_grid": EncoderType.CENTRALIZED_TASK_CNN_GRID,
    },
    "learning_type": {
        "decentralized": LearningType.DECENTRALIZED,
        "centralized": LearningType.CENTRALIZED,
    },
    "algorithm_name": {
        "value": AlgorithmName.VALUE,
        "actor_critic": AlgorithmName.ACTOR_CRITIC,
    },
    "pick_mode": {
        "forced": PickMode.FORCED,
        "choice": PickMode.CHOICE,
    },
    "heuristic": {
        "nearest_task": Heuristic.NEAREST_TASK,
        "nearest_correct_task": Heuristic.NEAREST_CORRECT_TASK,
        "nearest_correct_task_stay_wrong": Heuristic.NEAREST_CORRECT_TASK_STAY_WRONG,
    },
    "activation": {
        "relu": Activation.RELU,
        "leaky_relu": Activation.LEAKY_RELU,
        "none": Activation.NONE,
    },
    "weight_init": {
        "default": WeightInit.DEFAULT,
        "zero_bias": WeightInit.ZERO_BIAS,
    },
    "schedule": {
        "none": Schedule.NONE,
        "linear": Schedule.LINEAR,
        "step": Schedule.STEP,
    },
    "stopping_condition": {
        "none": StoppingCondition.NONE,
        "running_max_rps": StoppingCondition.RUNNING_MAX_RPS,
    },
    "despawn_mode": {
        "none": DespawnMode.NONE,
        "probability": DespawnMode.PROBABILITY,
    },
    "spawn_zone_mode": {
        "none": SpawnZoneMode.NONE,
        "random": SpawnZoneMode.RANDOM,
        "edge_switch": SpawnZoneMode.EDGE_SWITCH,
        "fixed_spread_agents_center_start": SpawnZoneMode.FIXED_SPREAD_AGENTS_CENTER_START,
    },
    "task_spawn_mode": {
        "global_unique": TaskSpawnMode.GLOBAL_UNIQUE,
        "per_type_unique": TaskSpawnMode.PER_TYPE_UNIQUE,
    },
}


def _enum(value: str, field: str) -> Any:
    mapping = _ENUM_MAPS.get(field)
    if mapping is None:
        raise ValueError(f"No enum map for field: {field}")
    key = value.lower().strip()
    if key not in mapping:
        raise ValueError(f"Invalid {field}: '{value}'. Valid: {list(mapping.keys())}")
    return mapping[key]


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------
def _parse_schedule(d: dict[str, Any], name: str) -> ScheduleConfig:
    return ScheduleConfig(
        start=float(d["start"]),
        end=float(d.get("end", d["start"])),
        schedule=_enum(d.get("schedule", "none"), "schedule"),
        step_size=int(d.get("step_size", 0)),
        step_factor=float(d.get("step_factor", 1.0)),
        step_start=int(d.get("step_start", 0)),
    )


def _parse_env(d: dict[str, Any]) -> EnvConfig:
    n_task_types = int(d.get("n_task_types", 1))
    n_agents = int(d["n_agents"])

    # Pick mode
    pick_mode = _enum(d.get("pick_mode", "forced"), "pick_mode")

    # Task assignments
    if "task_assignments" in d:
        task_assignments = tuple(tuple(int(t) for t in g) for g in d["task_assignments"])
    elif "rho" in d:
        rho = float(d["rho"])
        task_assignments = compute_task_assignments(n_agents, n_task_types, rho)
    elif n_task_types == 1:
        task_assignments = tuple((0,) for _ in range(n_agents))
    else:
        raise ValueError(
            "Must specify 'task_assignments' or 'rho' when n_task_types > 1"
        )

    # Stochastic config
    sd = d.get("stochastic")
    if sd is None:
        raise ValueError("env.stochastic block is required")
    tsm_raw = sd.get("task_spawn_mode")
    stochastic_cfg = StochasticConfig(
        spawn_prob=float(sd["spawn_prob"]),
        despawn_mode=_enum(sd.get("despawn_mode", "probability"), "despawn_mode"),
        despawn_prob=float(sd.get("despawn_prob", 0.0)),
        task_spawn_mode=_enum(tsm_raw, "task_spawn_mode") if tsm_raw else None,
        spawn_on_agent_cells=bool(sd.get("spawn_on_agent_cells", False)),
        spawn_at_round_end=bool(sd.get("spawn_at_round_end", False)),
        per_type_seeds=tuple(int(s) for s in sd["per_type_seeds"]) if sd.get("per_type_seeds") else None,
        spawn_area_size=int(sd["spawn_area_size"]) if sd.get("spawn_area_size") else None,
        spawn_zone_mode=_enum(sd.get("spawn_zone_mode", "none"), "spawn_zone_mode"),
        spawn_zone_interval=int(sd.get("spawn_zone_interval", 0)),
        spawn_flip_interval=int(sd.get("spawn_flip_interval", 0)),
        eval_spawn_zone_mode=_enum(sd.get("eval_spawn_zone_mode", "none"), "spawn_zone_mode"),
        eval_spawn_zone_interval=int(sd.get("eval_spawn_zone_interval", 0)),
        eval_spawn_flip_interval=int(sd.get("eval_spawn_flip_interval", 0)),
        reset_agent_pos_interval=int(sd.get("reset_agent_pos_interval", 0)),
        eval_reset_agent_pos_interval=int(sd.get("eval_reset_agent_pos_interval", 0)),
    )

    if stochastic_cfg.spawn_zone_mode == SpawnZoneMode.FIXED_SPREAD_AGENTS_CENTER_START:
        if stochastic_cfg.eval_spawn_zone_mode != SpawnZoneMode.NONE:
            raise ValueError(
                "spawn_zone_mode=fixed_spread_agents_center_start requires "
                "eval_spawn_zone_mode=none (eval uses the same fixed training zones)"
            )

    return EnvConfig(
        height=int(d["height"]),
        width=int(d["width"]),
        n_agents=n_agents,
        n_tasks=int(d.get("n_tasks", d.get("n_apples", 3))),
        gamma=float(d["gamma"]),
        r_picker=float(d.get("r_picker", 1.0)),
        n_task_types=n_task_types,
        r_low=float(d.get("r_low", 0.0)),
        task_assignments=task_assignments,
        pick_mode=pick_mode,
        max_tasks_per_type=int(d.get("max_tasks_per_type", 3)),
        stochastic=stochastic_cfg,
        allow_cross_type_picks=bool(d.get("allow_cross_type_picks", True)),
    )


def _parse_model(d: dict[str, Any]) -> ModelConfig:
    # Accept both "encoder" and "input_type" for backward compat during transition
    encoder_str = d.get("encoder", d.get("input_type"))
    if encoder_str is None:
        raise ValueError("model.encoder is required")
    encoder = _enum(encoder_str, "encoder")

    mlp_dims = tuple(int(x) for x in d.get("mlp_dims", [64, 64]))
    conv_specs = None
    if "conv_specs" in d and d["conv_specs"] is not None:
        conv_specs = tuple((int(s[0]), int(s[1])) for s in d["conv_specs"])

    return ModelConfig(
        encoder=encoder,
        mlp_dims=mlp_dims,
        conv_specs=conv_specs,
        activation=_enum(d.get("activation", "leaky_relu"), "activation"),
        weight_init=_enum(d.get("weight_init", "zero_bias"), "weight_init"),
    )


def _parse_train(d: dict[str, Any], n_task_types: int = 1) -> TrainConfig:
    lr_cfg = _parse_schedule(d["lr"], "train.lr")

    # Epsilon: accept both flat and nested
    eps_d = d.get("epsilon", d.get("policy_learning", {}).get("epsilon",
                   {"start": 0.1, "end": 0.01, "schedule": "linear"}))
    eps_cfg = _parse_schedule(eps_d, "train.epsilon")

    # Heuristic
    if "heuristic" in d:
        heuristic = _enum(d["heuristic"], "heuristic")
    elif n_task_types > 1:
        heuristic = Heuristic.NEAREST_CORRECT_TASK
    else:
        heuristic = Heuristic.NEAREST_TASK

    # Stopping config: accept both flat and nested
    stop_d = d.get("stopping", {})
    if not stop_d and "stopping_condition" in d:
        # Backward compat: flat fields
        stop_d = {
            "condition": d.get("stopping_condition", "none"),
            "patience_steps": d.get("patience_steps", 10000),
            "improvement_threshold": d.get("improvement_threshold", 0.01),
            "min_steps_before_stop": d.get("min_steps_before_stop", 0),
        }
    stopping = StoppingConfig(
        condition=_enum(stop_d.get("condition", "none"), "stopping_condition"),
        patience_steps=int(stop_d.get("patience_steps", 10000)),
        improvement_threshold=float(stop_d.get("improvement_threshold", 0.01)),
        min_steps_before_stop=int(stop_d.get("min_steps_before_stop", 0)),
    )

    algorithm_d = d.get("algorithm", {})
    algorithm_name = _enum(algorithm_d.get("name", "value"), "algorithm_name")
    actor_lr_d = d.get("actor_lr", algorithm_d.get("actor_lr"))
    actor_lr_cfg = _parse_schedule(actor_lr_d, "train.actor_lr") if actor_lr_d else None
    freeze_critic = bool(d.get("freeze_critic", False))
    comm_only_teammates = bool(d.get("comm_only_teammates", False))
    use_gpu = bool(d.get("use_gpu", d.get("use_gpu_batched", True)))

    following_d = d.get("following_rates", {})
    following_cfg = FollowingRatesConfig(
        enabled=bool(following_d.get("enabled", False)),
        budget=float(following_d.get("budget", 0.0)),
        teammate_budget=(
            float(following_d["teammate_budget"])
            if "teammate_budget" in following_d and following_d.get("teammate_budget") is not None
            else None
        ),
        non_teammate_budget=(
            float(following_d["non_teammate_budget"])
            if "non_teammate_budget" in following_d and following_d.get("non_teammate_budget") is not None
            else None
        ),
        rho=float(following_d.get("rho", 0.0)),
        reallocation_freq=int(following_d.get("reallocation_freq", 1)),
        solver=str(following_d.get("solver", "closed_form")),
        fixed=bool(following_d.get("fixed", False)),
    )

    influencer_d = d.get("influencer", {})
    influencer_cfg = InfluencerConfig(
        enabled=bool(influencer_d.get("enabled", False)),
        budget=float(influencer_d.get("budget", 0.0)),
    )

    if algorithm_name == AlgorithmName.ACTOR_CRITIC:
        if d.get("learning_type", "decentralized").strip().lower() != "decentralized":
            raise ValueError("train.algorithm.name=actor_critic requires train.learning_type=decentralized.")
        if comm_only_teammates and not use_gpu:
            raise ValueError("train.comm_only_teammates=true is only supported for GPU actor-critic.")
    elif freeze_critic:
        raise ValueError("train.freeze_critic is only supported for train.algorithm.name=actor_critic.")
    elif comm_only_teammates:
        raise ValueError("train.comm_only_teammates is only supported for train.algorithm.name=actor_critic.")
    if following_cfg.enabled:
        if algorithm_name != AlgorithmName.ACTOR_CRITIC:
            raise ValueError("train.following_rates.enabled=true requires train.algorithm.name=actor_critic.")
        if following_cfg.fixed:
            if following_cfg.teammate_budget is None or following_cfg.non_teammate_budget is None:
                raise ValueError(
                    "train.following_rates.fixed=true requires both "
                    "train.following_rates.teammate_budget and "
                    "train.following_rates.non_teammate_budget."
                )
            if following_cfg.teammate_budget < 0.0:
                raise ValueError("train.following_rates.teammate_budget must be >= 0.")
            if following_cfg.non_teammate_budget < 0.0:
                raise ValueError("train.following_rates.non_teammate_budget must be >= 0.")
        else:
            if following_cfg.budget < 0.0:
                raise ValueError("train.following_rates.budget must be >= 0.")
            if following_cfg.teammate_budget is not None or following_cfg.non_teammate_budget is not None:
                raise ValueError(
                    "train.following_rates.teammate_budget and "
                    "train.following_rates.non_teammate_budget are only supported when "
                    "train.following_rates.fixed=true."
                )
        if not (0.0 < following_cfg.rho <= 1.0):
            raise ValueError("train.following_rates.rho must be in (0, 1].")
        if following_cfg.reallocation_freq <= 0:
            raise ValueError("train.following_rates.reallocation_freq must be >= 1.")
        if following_cfg.solver not in get_supported_rate_solver_names():
            raise ValueError(
                f"train.following_rates.solver must be one of {get_supported_rate_solver_names()}."
            )
        if following_cfg.solver == "scipy" and not is_scipy_rate_solver_available():
            raise ValueError(
                "train.following_rates.solver=scipy requires scipy to be installed in the active environment."
            )
    if influencer_cfg.enabled and not following_cfg.enabled:
        raise ValueError("train.influencer.enabled=true requires train.following_rates.enabled=true.")

    warmup_steps = int(d.get("warmup_steps", 0))
    if warmup_steps < 0:
        raise ValueError("train.warmup_steps must be >= 0.")
    if warmup_steps > 0 and algorithm_name != AlgorithmName.ACTOR_CRITIC:
        raise ValueError("train.warmup_steps>0 requires train.algorithm.name=actor_critic.")

    train_only_teammates = bool(d.get("train_only_teammates", False))
    simulate_stranger_gap = int(d.get("simulate_stranger_gap", 0))
    greedy_own_type_only = bool(d.get("greedy_own_type_only", False))
    discount_method = str(d.get("discount_method", "team_steps"))
    if discount_method not in ("team_steps", "world_steps", "round_steps"):
        raise ValueError(f"train.discount_method must be 'team_steps', 'world_steps', or 'round_steps', got {discount_method!r}")

    return TrainConfig(
        total_steps=int(d["total_steps"]),
        seed=int(d.get("seed", 42)),
        lr=lr_cfg,
        actor_lr=actor_lr_cfg,
        freeze_critic=freeze_critic,
        epsilon=eps_cfg,
        algorithm=AlgorithmConfig(name=algorithm_name),
        following_rates=following_cfg,
        influencer=influencer_cfg,
        learning_type=_enum(d.get("learning_type", "decentralized"), "learning_type"),
        use_gpu=use_gpu,
        td_lambda=float(d.get("td_lambda", 0.0)),
        comm_only_teammates=comm_only_teammates,
        heuristic=heuristic,
        stopping=stopping,
        warmup_steps=int(d.get("warmup_steps", 0)),
        train_only_teammates=train_only_teammates,
        simulate_stranger_gap=simulate_stranger_gap,
        greedy_own_type_only=greedy_own_type_only,
        discount_method=discount_method,
    )


def _parse_eval(d: dict[str, Any]) -> EvalConfig:
    return EvalConfig(
        eval_steps=int(d.get("eval_steps", 1000)),
        n_test_states=int(d.get("n_test_states", 50)),
        checkpoint_freq=int(d.get("checkpoint_freq", 0)),
    )


def _parse_logging(d: dict[str, Any]) -> LoggingConfig:
    return LoggingConfig(
        output_dir=str(d.get("output_dir", "runs/")),
        main_csv_freq=int(d.get("main_csv_freq", 10000)),
        detail_csv_freq=int(d.get("detail_csv_freq", 50000)),
        timing_csv_freq=int(d.get("timing_csv_freq", 0)),
        alpha_state_log_freq=int(d.get("alpha_state_log_freq", 0)),
        env_trace=bool(d.get("env_trace", False)),
    )


# ---------------------------------------------------------------------------
# Override application
# ---------------------------------------------------------------------------
def _apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: '{override}'")
        key, value_str = override.split("=", 1)
        parts = key.strip().split(".")
        d = raw
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = _parse_override_value(value_str.strip())
    return raw


def _parse_override_value(s: str) -> Any:
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        parts, depth, current = [], 0, []
        for ch in inner:
            if ch == "[":
                depth += 1; current.append(ch)
            elif ch == "]":
                depth -= 1; current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current).strip()); current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current).strip())
        return [_parse_override_value(p) for p in parts]

    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if overrides:
        raw = _apply_overrides(raw, overrides)

    for section in ("env", "model", "train"):
        if section not in raw:
            raise ValueError(f"Missing required config section: '{section}'")

    env_cfg = _parse_env(raw["env"])
    model_cfg = _parse_model(raw["model"])
    actor_model_raw = raw.get("actor_model")
    actor_model_cfg = _parse_model(actor_model_raw) if actor_model_raw is not None else None
    if actor_model_cfg is not None and actor_model_cfg.encoder != model_cfg.encoder:
        raise ValueError("actor_model.encoder must match model.encoder because orchard uses a single encoder.")
    train_cfg = _parse_train(raw["train"], n_task_types=env_cfg.n_task_types)
    if train_cfg.following_rates.enabled and env_cfg.n_agents < 2:
        raise ValueError("train.following_rates.enabled=true requires env.n_agents >= 2.")
    return ExperimentConfig(
        env=env_cfg,
        model=model_cfg,
        actor_model=actor_model_cfg,
        train=train_cfg,
        eval=_parse_eval(raw.get("eval", {})),
        logging=_parse_logging(raw.get("logging", {})),
    )
