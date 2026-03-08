"""Logging: CSVLogger, run directory setup, metadata writing."""

from __future__ import annotations
import sys
sys.path.append("../")

import csv
import socket
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from debug.code.core.enums import ExperimentConfig


class CSVLogger:
    """Incremental CSV writer with immediate flush."""

    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def log(self, row: dict[str, float | int | str]) -> None:
        """Write one row and flush."""
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _get_git_hash() -> str | None:
    """Try to get current git hash, return None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Serialize ExperimentConfig to a dict suitable for YAML.

    Converts enums to their names.
    """
    d = asdict(cfg)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        if hasattr(obj, "name"):  # Enum
            return obj.name.lower()
        return obj

    return _convert(d)


def setup_logging(cfg: ExperimentConfig) -> Path:
    """Create run directory and write initial metadata. Returns run_dir."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    run_dir = Path(cfg.logging.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "weights").mkdir(exist_ok=True)

    # Write initial metadata
    metadata: dict[str, Any] = {
        "config": _config_to_dict(cfg),
        "run": {
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_wall_seconds": None,
            "hostname": socket.gethostname(),
            "git_hash": _get_git_hash(),
        },
    }
    with open(run_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    return run_dir


def finalize_logging(run_dir: Path, start_time: float) -> None:
    """Update metadata with end time and total wall seconds."""
    meta_path = run_dir / "metadata.yaml"
    with open(meta_path) as f:
        metadata = yaml.safe_load(f)

    metadata["run"]["end_time"] = datetime.now().isoformat()
    metadata["run"]["total_wall_seconds"] = round(time.time() - start_time, 2)

    with open(meta_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


def build_main_csv_fieldnames(*, reward_learning: bool) -> list[str]:
    """Build column names for metrics.csv based on training mode."""
    fields = ["step", "current_lr"]
    if reward_learning:
        fields.extend([
            "reward_acc_mean",
            "reward_mae_mean",
        ])
    else:
        fields.extend([
            "greedy_pps",
            "greedy_ratio",
            "nearest_pps",
            "nearest_ratio",
            "total_apples",
            "nearest_total_apples",
        ])
    return fields


def build_action_prob_csv_fieldnames() -> list[str]:
    """Build column names for action_probabilities.csv."""
    return ["step", "wall_time", "left", "right", "up", "down", "stay"]


def build_value_track_csv_fieldnames(num_states: int) -> list[str]:
    """Build column names for tracked state-value trajectories."""
    fields = ["step", "wall_time"]
    fields.extend([f"state_{i}" for i in range(max(0, int(num_states)))])
    return fields


def build_weight_sample_csv_fieldnames() -> list[str]:
    """Build column names for sampled weight trajectories."""
    return ["step", "wall_time", "tensor_name", "sample_id", "flat_index", "value"]


def build_detail_csv_fieldnames(n_agents: int, networks: list[Any]) -> list[str]:
    """Build column names for details.csv."""
    fields = ["step", "wall_time", "ram_mb", "current_lr", "current_epsilon"]

    # Weight and grad norm columns per agent per layer
    for agent_idx in range(n_agents):
        for name, _ in networks[agent_idx].named_parameters():
            if "weight" in name:
                fields.append(f"weight_norm_agent_{agent_idx}_{name}")
                fields.append(f"grad_norm_agent_{agent_idx}_{name}")

    fields.extend(["td_loss_step", "value_pred_mean", "value_pred_std"])
    return fields
