"""Config loading: YAML → ExperimentConfig with full validation and enum conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from orchard.enums import (
    Activation,
    DespawnMode,
    EncoderType,
    EnvType,
    Heuristic,
    ModelType,
    PickMode,
    Schedule,
    StoppingCondition,
    TaskSpawnMode,
    TDTarget,
    TrainMethod,
    TrainMode,
    WeightInit,
)
from orchard.datatypes import (
    EnvConfig,
    EvalConfig,
    ExperimentConfig,
    LearningType,
    LoggingConfig,
    ModelConfig,
    PolicyLearningConfig,
    ScheduleConfig,
    StochasticConfig,
    TrainConfig,
    ValueLearningConfig,
    compute_task_assignments,
)


# ---------------------------------------------------------------------------
# String → Enum maps
# ---------------------------------------------------------------------------
_ENV_TYPE_MAP: dict[str, EnvType] = {
    "deterministic": EnvType.DETERMINISTIC,
    "stochastic": EnvType.STOCHASTIC,
}

_DESPAWN_MODE_MAP: dict[str, DespawnMode] = {
    "none": DespawnMode.NONE,
    "probability": DespawnMode.PROBABILITY,
}

_TASK_SPAWN_MODE_MAP: dict[str, TaskSpawnMode] = {
    "global_unique": TaskSpawnMode.GLOBAL_UNIQUE,
    "per_type_unique": TaskSpawnMode.PER_TYPE_UNIQUE,
}

_ENCODER_TYPE_MAP: dict[str, EncoderType] = {
    "relative": EncoderType.RELATIVE,
    "relative_k": EncoderType.RELATIVE_K,
    "grid_mlp": EncoderType.GRID_MLP,
    "centralized_cnn_grid": EncoderType.CENTRALIZED_CNN_GRID,
    "cnn_grid": EncoderType.CNN_GRID,
    "egocentric_cnn_grid": EncoderType.EGOCENTRIC_CNN_GRID,
    "no_redundant_agent_grid": EncoderType.NO_REDUNDANT_AGENT_GRID,
    "task_cnn_grid": EncoderType.TASK_CNN_GRID,
    "centralized_task_cnn_grid": EncoderType.CENTRALIZED_TASK_CNN_GRID,
    "blind_task_cnn_grid": EncoderType.BLIND_TASK_CNN_GRID,
    "filtered_task_cnn_grid": EncoderType.FILTERED_TASK_CNN_GRID,
}

_LEARNING_TYPE_MAP: dict[str, LearningType] = {
    "decentralized": LearningType.DECENTRALIZED,
    "centralized": LearningType.CENTRALIZED,
}

_MODEL_TYPE_MAP: dict[str, ModelType] = {
    "mlp": ModelType.MLP,
    "cnn": ModelType.CNN,
}

_TRAIN_MODE_MAP: dict[str, TrainMode] = {
    "value_learning": TrainMode.VALUE_LEARNING,
    "policy_learning": TrainMode.POLICY_LEARNING,
    "reward_learning": TrainMode.REWARD_LEARNING,
}

_TD_TARGET_MAP: dict[str, TDTarget] = {
    "pre_action": TDTarget.PRE_ACTION,
    "after_state": TDTarget.AFTER_STATE,
}

_SCHEDULE_MAP: dict[str, Schedule] = {
    "none": Schedule.NONE,
    "linear": Schedule.LINEAR,
    "step": Schedule.STEP,
}

_STOPPING_CONDITION_MAP: dict[str, StoppingCondition] = {
    "none": StoppingCondition.NONE,
    "running_max_pps": StoppingCondition.RUNNING_MAX_PPS,
    "running_min_mae": StoppingCondition.RUNNING_MIN_MAE,
    "running_max_rps": StoppingCondition.RUNNING_MAX_RPS,
}

_TRAIN_METHOD_MAP: dict[str, TrainMethod] = {
    "nstep": TrainMethod.NSTEP,
    "backward_view": TrainMethod.BACKWARD_VIEW,
}

_PICK_MODE_MAP: dict[str, PickMode] = {
    "forced": PickMode.FORCED,
    "choice": PickMode.CHOICE,
}

_HEURISTIC_MAP: dict[str, Heuristic] = {
    "nearest_task": Heuristic.NEAREST_TASK,
    "nearest_correct_task": Heuristic.NEAREST_CORRECT_TASK,
    "nearest_correct_task_stay_wrong": Heuristic.NEAREST_CORRECT_TASK_STAY_WRONG,
}

_ACTIVATION_MAP: dict[str, Activation] = {
    "relu": Activation.RELU,
    "leaky_relu": Activation.LEAKY_RELU,
    "none": Activation.NONE,
}

_WEIGHT_INIT_MAP: dict[str, WeightInit] = {
    "default": WeightInit.DEFAULT,
    "zero_bias": WeightInit.ZERO_BIAS,
}


def _enum_lookup(value: str, mapping: dict[str, Any], field_name: str) -> Any:
    """Convert string to enum, raising ValueError with context on failure."""
    key = value.lower().strip()
    if key not in mapping:
        valid = list(mapping.keys())
        raise ValueError(f"Invalid {field_name}: '{value}'. Valid options: {valid}")
    return mapping[key]


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------
def _parse_schedule(d: dict[str, Any], section_name: str) -> ScheduleConfig:
    """Parse a schedule config block (used for lr and epsilon)."""
    schedule = _enum_lookup(d.get("schedule", "none"), _SCHEDULE_MAP, f"{section_name}.schedule")
    return ScheduleConfig(
        start=float(d["start"]),
        end=float(d.get("end", d["start"])),
        schedule=schedule,
        step_size=int(d.get("step_size", 0)),
        step_factor=float(d.get("step_factor", 1.0)),
        step_start=int(d.get("step_start", 0)),
    )


def _parse_env(d: dict[str, Any]) -> EnvConfig:
    """Parse env config section."""
    env_type = _enum_lookup(d["type"], _ENV_TYPE_MAP, "env.type")

    stochastic_cfg: StochasticConfig | None = None
    if env_type == EnvType.STOCHASTIC:
        sd = d.get("stochastic")
        if sd is None:
            raise ValueError("env.stochastic block required when env.type='stochastic'")
        despawn_mode = _enum_lookup(
            sd.get("despawn_mode", "none"), _DESPAWN_MODE_MAP, "env.stochastic.despawn_mode"
        )
        tsm_raw = sd.get("task_spawn_mode", None)
        task_spawn_mode = (
            _enum_lookup(tsm_raw, _TASK_SPAWN_MODE_MAP, "env.stochastic.task_spawn_mode")
            if tsm_raw is not None else None
        )
        stochastic_cfg = StochasticConfig(
            spawn_prob=float(sd["spawn_prob"]),
            despawn_mode=despawn_mode,
            despawn_prob=float(sd.get("despawn_prob", 0.0)),
            task_spawn_mode=task_spawn_mode,
        )

    # --- Task specialization fields ---
    n_task_types = int(d.get("n_task_types", 1))
    r_low = float(d.get("r_low", 0.0))
    n_agents = int(d["n_agents"])

    # Pick mode: parse from pick_mode string, or fall back to force_pick bool
    if "pick_mode" in d:
        pick_mode = _enum_lookup(d["pick_mode"], _PICK_MODE_MAP, "env.pick_mode")
    elif "force_pick" in d:
        pick_mode = PickMode.FORCED if bool(d["force_pick"]) else PickMode.CHOICE
    else:
        pick_mode = PickMode.FORCED

    max_tasks_per_type = int(d.get("max_tasks_per_type", 3))

    # --- Task assignments (always populated) ---
    has_rho = "rho" in d
    has_assignments = "task_assignments" in d

    if has_rho and has_assignments:
        raise ValueError("Specify either 'rho' or 'task_assignments', not both.")

    if has_assignments:
        raw_assignments = d["task_assignments"]
        task_assignments = tuple(tuple(int(t) for t in g) for g in raw_assignments)
    elif has_rho:
        rho = float(d["rho"])
        # Validate rho bounds
        min_rho = 1.0 / n_task_types  # at least 1 type per agent
        if rho < min_rho - 1e-9 or rho > 1.0 + 1e-9:
            raise ValueError(
                f"rho={rho} out of bounds [{min_rho}, 1.0] for "
                f"n_task_types={n_task_types}"
            )
        # Validate round(rho * T) is near-integer
        g_size_raw = rho * n_task_types
        if abs(g_size_raw - round(g_size_raw)) > 0.01:
            raise ValueError(
                f"rho * n_task_types = {g_size_raw} is not near an integer. "
                f"Choose rho so that rho * {n_task_types} is an integer."
            )
        task_assignments = compute_task_assignments(n_agents, n_task_types, rho)
    elif n_task_types == 1:
        # Single task type: all agents form one big group
        task_assignments = tuple((0,) for _ in range(n_agents))
    else:
        # Default for n_task_types > 1: all agents care about all types (rho=1)
        task_assignments = tuple(
            tuple(range(n_task_types)) for _ in range(n_agents)
        )

    # Validate: all type indices in range
    for i, g in enumerate(task_assignments):
        for t in g:
            if t < 0 or t >= n_task_types:
                raise ValueError(
                    f"task_assignments[{i}] contains type {t}, "
                    f"but n_task_types={n_task_types} (valid: 0..{n_task_types-1})"
                )

    # Validate: every type covered by at least one agent
    covered = set()
    for g in task_assignments:
        covered.update(g)
    if covered != set(range(n_task_types)):
        missing = set(range(n_task_types)) - covered
        raise ValueError(f"task_assignments do not cover all types. Missing: {missing}")

    # Validation
    if n_task_types < 1:
        raise ValueError(f"n_task_types must be >= 1, got {n_task_types}")

    # n_tasks: backward compat with n_apples
    n_tasks = int(d.get("n_tasks", d.get("n_apples", 4)))
    # max_tasks: backward compat with max_apples
    max_tasks = int(d.get("max_tasks", d.get("max_apples", n_tasks)))

    return EnvConfig(
        height=int(d["height"]),
        width=int(d["width"]),
        n_agents=n_agents,
        n_tasks=n_tasks,
        gamma=float(d["gamma"]),
        r_picker=float(d.get("r_picker", 1.0)),
        n_task_types=n_task_types,
        r_low=r_low,
        task_assignments=task_assignments,
        pick_mode=pick_mode,
        max_tasks_per_type=max_tasks_per_type,
        max_tasks=max_tasks,
        env_type=env_type,
        stochastic=stochastic_cfg,
    )


def _parse_model(d: dict[str, Any]) -> ModelConfig:
    """Parse model config section."""
    input_type = _enum_lookup(d["input_type"], _ENCODER_TYPE_MAP, "model.input_type")
    model_type = _enum_lookup(d["model_type"], _MODEL_TYPE_MAP, "model.model_type")
    k = d.get("k", None)
    if k is not None:
        k = int(k)

    mlp_dims = tuple(int(x) for x in d.get("mlp_dims", [64, 64]))

    conv_specs: tuple[tuple[int, int], ...] | None = None
    if "conv_specs" in d and d["conv_specs"] is not None:
        conv_specs = tuple(
            (int(spec[0]), int(spec[1])) for spec in d["conv_specs"]
        )

    activation = _enum_lookup(d.get("activation", "leaky_relu"), _ACTIVATION_MAP, "model.activation")
    weight_init = _enum_lookup(d.get("weight_init", "zero_bias"), _WEIGHT_INIT_MAP, "model.weight_init")

    return ModelConfig(
        input_type=input_type,
        model_type=model_type,
        mlp_dims=mlp_dims,
        conv_specs=conv_specs,
        k_nearest=k,
        activation=activation,
        weight_init=weight_init,
    )


def _parse_train(d: dict[str, Any], n_task_types: int = 1) -> TrainConfig:
    """Parse train config section."""
    mode = _enum_lookup(d["mode"], _TRAIN_MODE_MAP, "train.mode")
    td_target = _enum_lookup(d.get("td_target", "pre_action"), _TD_TARGET_MAP, "train.td_target")
    lr_cfg = _parse_schedule(d["lr"], "train.lr")

    vl_cfg: ValueLearningConfig | None = None
    pl_cfg: PolicyLearningConfig | None = None

    if mode == TrainMode.VALUE_LEARNING:
        vld = d.get("value_learning", {})
        vl_cfg = ValueLearningConfig(reset_freq=int(vld.get("reset_freq", 20)))
    elif mode == TrainMode.POLICY_LEARNING:
        pld = d.get("policy_learning", {})
        eps_d = pld.get("epsilon", {"start": 0.1, "end": 0.01, "schedule": "linear"})
        pl_cfg = PolicyLearningConfig(epsilon=_parse_schedule(eps_d, "train.policy_learning.epsilon"))

    # Heuristic: default depends on n_task_types
    if "heuristic" in d:
        heuristic = _enum_lookup(d["heuristic"], _HEURISTIC_MAP, "train.heuristic")
    elif n_task_types > 1:
        heuristic = Heuristic.NEAREST_CORRECT_TASK
    else:
        heuristic = Heuristic.NEAREST_TASK

    return TrainConfig(
        mode=mode,
        td_target=td_target,
        total_steps=int(d["total_steps"]),
        seed=int(d.get("seed", 42)),
        nstep=int(d.get("nstep", 1)),
        lr=lr_cfg,
        td_lambda=float(d.get("td_lambda", 0.0)),
        comm_weight=float(d.get("comm_weight", 0.0)),
        value_learning=vl_cfg,
        policy_learning=pl_cfg,
        train_method=_enum_lookup(d.get("train_method", "nstep"), _TRAIN_METHOD_MAP, "train.train_method"),
        learning_type=_enum_lookup(d.get("learning_type", "decentralized"), _LEARNING_TYPE_MAP, "train.learning_type"),
        stopping_condition=_enum_lookup(d.get("stopping_condition", "none"), _STOPPING_CONDITION_MAP, "train.stopping_condition"),
        patience_steps=int(d.get("patience_steps", 10000)),
        improvement_threshold=float(d.get("improvement_threshold", 0.01)),
        min_steps_before_stop=int(d.get("min_steps_before_stop", 0)),
        batch_actions=bool(d.get("batch_actions", True)),
        heuristic=heuristic,
        use_vmap=bool(d.get("use_vmap", False)),
        use_vec_encode=bool(d.get("use_vec_encode", True)),
        use_gpu_batched=bool(d.get("use_gpu_batched", False)),
        time_debug=bool(d.get("time_debug", False)),
        time_csv_freq=int(d.get("time_csv_freq", 100)),
    )


def _parse_eval(d: dict[str, Any]) -> EvalConfig:
    """Parse eval config section."""
    return EvalConfig(
        rollout_len=int(d.get("rollout_len", 2000)),
        eval_steps=int(d.get("eval_steps", 1000)),
        n_test_states=int(d.get("n_test_states", 50)),
        checkpoint_freq=int(d.get("checkpoint_freq", 0)),
    )


def _parse_logging(d: dict[str, Any]) -> LoggingConfig:
    """Parse logging config section."""
    return LoggingConfig(
        output_dir=str(d.get("output_dir", "runs/")),
        main_csv_freq=int(d.get("main_csv_freq", 100)),
        detail_csv_freq=int(d.get("detail_csv_freq", 100)),
    )


# ---------------------------------------------------------------------------
# Override application
# ---------------------------------------------------------------------------
def _apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dot-notation overrides to raw YAML dict.

    Example: 'train.lr.start=0.01' sets raw['train']['lr']['start'] = 0.01
    """
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
    """Parse an override value string into a Python object."""
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        # Bracket-aware split
        parts, depth, current = [], 0, []
        for ch in inner:
            if ch == "[":
                depth += 1
                current.append(ch)
            elif ch == "]":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
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
    """Load YAML config, apply overrides, validate, return typed ExperimentConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    if overrides:
        raw = _apply_overrides(raw, overrides)

    for section in ("env", "model", "train"):
        if section not in raw:
            raise ValueError(f"Missing required config section: '{section}'")

    env_cfg = _parse_env(raw["env"])

    cfg = ExperimentConfig(
        env=env_cfg,
        model=_parse_model(raw["model"]),
        train=_parse_train(raw["train"], n_task_types=env_cfg.n_task_types),
        eval=_parse_eval(raw.get("eval", {})),
        logging=_parse_logging(raw.get("logging", {})),
    )

    # Validate encoder-specific requirements
    if cfg.train.td_target == TDTarget.AFTER_STATE and cfg.model.input_type == EncoderType.RELATIVE:
        raise ValueError(
            "td_target='after_state' requires input_type='relative_k' "
            "(the 'relative' encoder has fixed apple count and cannot handle "
            "pre-spawn after-states where apples are missing). "
            "Set model.input_type=relative_k and model.k_nearest=<N>."
        )

    # Warn if using old encoder with n_task_types > 1
    if env_cfg.n_task_types > 1 and cfg.model.input_type in (
        EncoderType.CNN_GRID, EncoderType.CENTRALIZED_CNN_GRID,
    ):
        import warnings
        warnings.warn(
            f"Using old encoder {cfg.model.input_type.name} with n_task_types={env_cfg.n_task_types}. "
            f"Consider using TASK_CNN_GRID or CENTRALIZED_TASK_CNN_GRID.",
            stacklevel=2,
        )

    return cfg
