"""Config loading: YAML → ExperimentConfig with full validation and enum conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from orchard.enums import (
    DespawnMode,
    EncoderType,
    EnvType,
    ModelType,
    Schedule,
    TDTarget,
    TrainMode,
)
from orchard.datatypes import (
    EnvConfig,
    EvalConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    PolicyLearningConfig,
    ScheduleConfig,
    StochasticConfig,
    TrainConfig,
    ValueLearningConfig,
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
    "lifetime": DespawnMode.LIFETIME,
}

_ENCODER_TYPE_MAP: dict[str, EncoderType] = {
    "relative": EncoderType.RELATIVE,
    "relative_k": EncoderType.RELATIVE_K,
    "positional_k": EncoderType.POSITIONAL_K,
    "stable_id": EncoderType.STABLE_ID,
    "grid_mlp": EncoderType.GRID_MLP,
    "cnn_grid": EncoderType.CNN_GRID,
}

_MODEL_TYPE_MAP: dict[str, ModelType] = {
    "mlp": ModelType.MLP,
    "cnn": ModelType.CNN,
}

_TRAIN_MODE_MAP: dict[str, TrainMode] = {
    "value_learning": TrainMode.VALUE_LEARNING,
    "policy_learning": TrainMode.POLICY_LEARNING,
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
        stochastic_cfg = StochasticConfig(
            spawn_prob=float(sd["spawn_prob"]),
            despawn_mode=despawn_mode,
            despawn_prob=float(sd.get("despawn_prob", 0.0)),
            apple_lifetime=int(sd.get("apple_lifetime", 0)),
        )

    return EnvConfig(
        height=int(d["height"]),
        width=int(d["width"]),
        n_agents=int(d["n_agents"]),
        n_apples=int(d["n_apples"]),
        gamma=float(d["gamma"]),
        r_picker=float(d["r_picker"]),
        force_pick=bool(d.get("force_pick", True)),
        max_apples=int(d.get("max_apples", d["n_apples"])),
        env_type=env_type,
        stochastic=stochastic_cfg,
    )


def _parse_model(d: dict[str, Any]) -> ModelConfig:
    """Parse model config section."""
    input_type = _enum_lookup(d["input_type"], _ENCODER_TYPE_MAP, "model.input_type")
    model_type = _enum_lookup(d["model_type"], _MODEL_TYPE_MAP, "model.model_type")
    k = d.get("k", None) # number of apples to consider fo k-nearest input.
    if k is not None:
        k = int(k)

    mlp_dims = tuple(int(x) for x in d.get("mlp_dims", [64, 64]))

    conv_specs: tuple[tuple[int, int], ...] | None = None
    if "conv_specs" in d and d["conv_specs"] is not None:
        conv_specs = tuple(
            (int(spec[0]), int(spec[1])) for spec in d["conv_specs"]
        )

    return ModelConfig(
        input_type=input_type,
        model_type=model_type,
        mlp_dims=mlp_dims,
        conv_specs=conv_specs,
        k_nearest=k,
    )


def _parse_train(d: dict[str, Any]) -> TrainConfig:
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

    return TrainConfig(
        mode=mode,
        td_target=td_target,
        total_steps=int(d["total_steps"]),
        seed=int(d.get("seed", 42)),
        nstep=int(d.get("nstep", 1)),
        lr=lr_cfg,
        value_learning=vl_cfg,
        policy_learning=pl_cfg,
    )


def _parse_eval(d: dict[str, Any]) -> EvalConfig:
    """Parse eval config section."""
    return EvalConfig(
        freq=int(d.get("freq", 1000)),
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

        # Navigate to parent
        d = raw
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]

        # Parse value: try int, float, list, then string
        d[parts[-1]] = _parse_override_value(value_str.strip())

    return raw


def _parse_override_value(s: str) -> Any:
    """Parse an override value string into a Python object."""
    # List notation: [64,64]
    # config.py, inside _parse_override_value, replace lines 238-242
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

    # Bool
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False

    # Int
    try:
        return int(s)
    except ValueError:
        pass

    # Float
    try:
        return float(s)
    except ValueError:
        pass

    # String
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

    # Validate top-level sections
    for section in ("env", "model", "train"):
        if section not in raw:
            raise ValueError(f"Missing required config section: '{section}'")

    cfg = ExperimentConfig(
        env=_parse_env(raw["env"]),
        model=_parse_model(raw["model"]),
        train=_parse_train(raw["train"]),
        eval=_parse_eval(raw.get("eval", {})),
        logging=_parse_logging(raw.get("logging", {})),
    )
    
    # validate encoder-specific requirements
    if cfg.train.td_target == TDTarget.AFTER_STATE and cfg.model.input_type == EncoderType.RELATIVE:
        raise ValueError(
            "td_target='after_state' requires input_type='relative_k' "
            "(the 'relative' encoder has fixed apple count and cannot handle "
            "pre-spawn after-states where apples are missing). "
            "Set model.input_type=relative_k and model.k_nearest=<N>."
        )

    return cfg
    
