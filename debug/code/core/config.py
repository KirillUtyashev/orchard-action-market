from __future__ import annotations

from pathlib import Path
from dataclasses import fields
from typing import Any

import yaml

from debug.code.core.enums import AlgorithmConfig, EnvironmentConfig, EvalConfig, \
    ExperimentConfig, \
    LoggingConfig, NetworkConfig, \
    ProfilingConfig, RewardConfig, SupervisedConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Generic dataclass populator
# ---------------------------------------------------------------------------
def _populate_dataclass(cls, d: dict):
    """Fill a dataclass from a dict, ignoring unknown keys, using defaults for missing."""
    valid_fields = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in d.items() if k in valid_fields}
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Override helpers (generic, works for any config)
# ---------------------------------------------------------------------------
def _parse_override_value(s: str) -> Any:
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
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


def _apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dot-notation overrides, e.g. 'train.alpha=0.001'."""
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load YAML config, apply overrides, return typed ExperimentConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    if overrides:
        raw = _apply_overrides(raw, overrides)

    base_network = raw.get("network", {})

    return ExperimentConfig(
        network=_populate_dataclass(NetworkConfig, base_network),
        critic_network=_populate_dataclass(NetworkConfig, {**base_network, **raw.get("critic_network", {})}),
        actor_network=_populate_dataclass(NetworkConfig, {**base_network, **raw.get("actor_network", {})}),
        train=_populate_dataclass(TrainingConfig, raw.get("train", {})),
        algorithm=_populate_dataclass(AlgorithmConfig, raw.get("algorithm", {})),
        reward=_populate_dataclass(RewardConfig, raw.get("reward", {})),
        supervised=_populate_dataclass(SupervisedConfig, raw.get("supervised", {})),
        eval=_populate_dataclass(EvalConfig, raw.get("eval", {})),
        env=_populate_dataclass(EnvironmentConfig, raw.get("env", {})),
        logging=_populate_dataclass(LoggingConfig, raw.get("logging", {})),
        profiling=_populate_dataclass(ProfilingConfig, raw.get("profiling", {})),
    )
