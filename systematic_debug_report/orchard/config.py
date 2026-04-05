"""Config loading: YAML → ExperimentConfig with validation and enum conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from orchard.enums import (
    Activation,
    DespawnMode,
    EncoderType,
    Heuristic,
    LearningType,
    PickMode,
    Schedule,
    StoppingCondition,
    TaskSpawnMode,
    WeightInit,
)
from orchard.datatypes import (
    EnvConfig,
    EvalConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    ScheduleConfig,
    StochasticConfig,
    StoppingConfig,
    TrainConfig,
    compute_task_assignments,
)


# ---------------------------------------------------------------------------
# String → Enum maps
# ---------------------------------------------------------------------------
_ENUM_MAPS: dict[str, dict[str, Any]] = {
    "encoder": {
        "blind_task_cnn_grid": EncoderType.BLIND_TASK_CNN_GRID,
        "filtered_task_cnn_grid": EncoderType.FILTERED_TASK_CNN_GRID,
        "position_aware_task_cnn_grid": EncoderType.POSITION_AWARE_TASK_CNN_GRID,
        "centralized_task_cnn_grid": EncoderType.CENTRALIZED_TASK_CNN_GRID,
    },
    "learning_type": {
        "decentralized": LearningType.DECENTRALIZED,
        "centralized": LearningType.CENTRALIZED,
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

    return TrainConfig(
        total_steps=int(d["total_steps"]),
        seed=int(d.get("seed", 42)),
        lr=lr_cfg,
        epsilon=eps_cfg,
        learning_type=_enum(d.get("learning_type", "decentralized"), "learning_type"),
        use_gpu=bool(d.get("use_gpu", d.get("use_gpu_batched", True))),
        td_lambda=float(d.get("td_lambda", 0.0)),
        comm_weight=float(d.get("comm_weight", 0.0)),
        heuristic=heuristic,
        stopping=stopping,
    )


def _parse_eval(d: dict[str, Any]) -> EvalConfig:
    return EvalConfig(
        rollout_len=int(d.get("rollout_len", 2000)),
        eval_steps=int(d.get("eval_steps", 1000)),
        n_test_states=int(d.get("n_test_states", 50)),
        checkpoint_freq=int(d.get("checkpoint_freq", 0)),
    )


def _parse_logging(d: dict[str, Any]) -> LoggingConfig:
    return LoggingConfig(
        output_dir=str(d.get("output_dir", "runs/")),
        main_csv_freq=int(d.get("main_csv_freq", 10000)),
        detail_csv_freq=int(d.get("detail_csv_freq", 50000)),
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
    return ExperimentConfig(
        env=env_cfg,
        model=_parse_model(raw["model"]),
        train=_parse_train(raw["train"], n_task_types=env_cfg.n_task_types),
        eval=_parse_eval(raw.get("eval", {})),
        logging=_parse_logging(raw.get("logging", {})),
    )
