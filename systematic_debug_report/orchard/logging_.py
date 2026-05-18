"""Logging: CSVLogger, run directory setup, metadata writing."""

from __future__ import annotations

import csv
import multiprocessing
import os
import socket
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from orchard.datatypes import ExperimentConfig


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
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _get_git_hash() -> str | None:
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
    d = asdict(cfg)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        if hasattr(obj, "name"):
            return obj.name.lower()
        return obj

    return _convert(d)


def setup_logging(cfg: ExperimentConfig) -> Path:
    """Create run directory and write initial metadata. Returns run_dir."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    run_dir = Path(cfg.logging.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    metadata: dict[str, Any] = {
        "config": _config_to_dict(cfg),
        "run": {
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_wall_seconds": None,
            "hostname": socket.gethostname(),
            "git_hash": _get_git_hash(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", None),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", None),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", None),
        },
    }
    cpu_count = multiprocessing.cpu_count()
    get_affinity = getattr(os, "sched_getaffinity", None)
    hw: dict[str, Any] = {
        "cpu_logical_cores": cpu_count,
        "cpu_affinity_cores": len(get_affinity(0)) if get_affinity is not None else cpu_count,
    }
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        hw["gpu_name"] = pynvml.nvmlDeviceGetName(handle)
        hw["gpu_vram_mb"] = pynvml.nvmlDeviceGetMemoryInfo(handle).total // 1024**2
        try:
            import torch
            hw["gpu_sm_count"] = torch.cuda.get_device_properties(0).multi_processor_count
        except Exception:
            hw["gpu_sm_count"] = None
        pynvml.nvmlShutdown()
    except Exception as e:
        hw["gpu_name"] = None
        hw["gpu_sm_count"] = None
        hw["gpu_vram_mb"] = None
        hw["gpu_pynvml_error"] = str(e)
    metadata["run"].update(hw)

    with open(run_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    return run_dir


def write_reward_vectors_csv(run_dir: Path, env: Any) -> Path | None:
    """Write generated category reward vectors and their components for this run."""
    rewards = getattr(env, "category_rewards", None)
    if rewards is None:
        return None

    n_task_types, n_agents = rewards.shape
    baselines = getattr(env, "category_reward_baselines", rewards.mean(axis=1))
    baseline_raw = getattr(env, "category_reward_baseline_raw", ["" for _ in range(n_task_types)])
    baseline_standardized = getattr(
        env,
        "category_reward_baseline_standardized",
        ["" for _ in range(n_task_types)],
    )
    agent_offsets = getattr(env, "category_reward_agent_offsets", rewards - baselines[:, None])
    stoch = getattr(getattr(env, "cfg", None), "stochastic", None)

    fieldnames = [
        "reward_seed",
        "sigma_a",
        "sigma_b",
        "reward_generation",
        "reward_seed_attempts",
        "task_type",
        "b_raw",
        "b_standardized",
        "b_normalized",
        "reward_sum",
        "reward_mean",
        "reward_variance",
    ]
    fieldnames += [f"reward_agent_{i}" for i in range(n_agents)]
    fieldnames += [f"a_offset_agent_{i}" for i in range(n_agents)]

    path = run_dir / "reward_vectors.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for task_type in range(n_task_types):
            reward_vec = rewards[task_type]
            offset_vec = agent_offsets[task_type]
            row: dict[str, float | int | str] = {
                "reward_seed": getattr(env, "category_reward_seed", ""),
                "sigma_a": getattr(stoch, "sigma_a", ""),
                "sigma_b": getattr(stoch, "sigma_b", ""),
                "reward_generation": (
                    getattr(getattr(stoch, "reward_generation", ""), "name", "")
                    .lower()
                    .strip()
                ),
                "reward_seed_attempts": getattr(env, "category_reward_seed_attempts", ""),
                "task_type": task_type,
                "b_raw": float(baseline_raw[task_type]) if baseline_raw[task_type] != "" else "",
                "b_standardized": (
                    float(baseline_standardized[task_type])
                    if baseline_standardized[task_type] != "" else ""
                ),
                "b_normalized": float(baselines[task_type]),
                "reward_sum": float(reward_vec.sum()),
                "reward_mean": float(reward_vec.mean()),
                "reward_variance": float(reward_vec.var()),
            }
            for agent_idx in range(n_agents):
                row[f"reward_agent_{agent_idx}"] = float(reward_vec[agent_idx])
                row[f"a_offset_agent_{agent_idx}"] = float(offset_vec[agent_idx])
            writer.writerow(row)
    return path


def finalize_logging(run_dir: Path, start_time: float) -> None:
    meta_path = run_dir / "metadata.yaml"
    with open(meta_path) as f:
        metadata = yaml.safe_load(f)
    metadata["run"]["end_time"] = datetime.now().isoformat()
    metadata["run"]["total_wall_seconds"] = round(time.time() - start_time, 2)
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


def build_main_csv_fieldnames(
    heuristic_name: str,
    *,
    actor_critic: bool = False,
    following_rates: bool = False,
    influencer: bool = False,
) -> list[str]:
    fields = [
        "step", "wall_time",
        "greedy_rps", "greedy_team_rps",
        "greedy_tasks_picked_per_step",
        f"{heuristic_name}_rps", f"{heuristic_name}_team_rps",
        f"{heuristic_name}_tasks_picked_per_step",
        "td_loss_avg",
    ]
    if actor_critic:
        fields.extend([
            "actor_lr",
            "actor_loss_mean",
            "advantage_mean",
            "policy_entropy_mean",
        ])
    if following_rates:
        fields.extend([
            "alpha_mean",
            "alpha_positive_frac",
            "following_weight_mean",
            "active_follow_edges_mean",
            "follower_to_influencer_weight_mean",
            "effective_follow_weight_mean",
        ])
    if influencer:
        fields.extend([
            "beta_mean",
            "influencer_weight_mean",
        ])
    return fields


def build_action_prob_csv_fieldnames() -> list[str]:
    return ["step", "wall_time", "left", "right", "up", "down", "stay"]


def _build_full_policy_prob_fieldnames(n_task_types: int) -> list[str]:
    fields = [
        "prob_up",
        "prob_down",
        "prob_left",
        "prob_right",
        "prob_stay",
    ]
    for tau in range(int(n_task_types)):
        fields.append(f"prob_pick_{tau}")
    return fields


def build_phase1_policy_prob_csv_fieldnames(n_task_types: int) -> list[str]:
    fields = [
        "step",
        "wall_time",
        "state_id",
        "actor_id",
        "state_json",
    ]
    fields.extend(_build_full_policy_prob_fieldnames(n_task_types))
    return fields


def build_phase2_policy_prob_csv_fieldnames(n_task_types: int) -> list[str]:
    fields = [
        "step",
        "wall_time",
        "state_id",
        "state_label",
        "actor_id",
        "state_json",
    ]
    for tau in range(int(n_task_types)):
        fields.append(f"present_type_{tau}")
    for tau in range(int(n_task_types)):
        fields.append(f"assigned_type_{tau}")
    fields.extend(_build_full_policy_prob_fieldnames(n_task_types))
    return fields


def build_following_rate_csv_fieldnames(num_agents: int) -> list[str]:
    fields = ["step", "wall_time"]
    for target_id in range(int(num_agents)):
        fields.append(f"alpha_to_{target_id}")
    for target_id in range(int(num_agents)):
        fields.append(f"lambda_to_{target_id}")
    for target_id in range(int(num_agents)):
        fields.append(f"weight_to_{target_id}")
    fields.extend([
        "lambda_to_influencer",
        "weight_to_influencer",
        "influencer_value",
    ])
    return fields


def build_influencer_csv_fieldnames(num_agents: int) -> list[str]:
    fields = ["step", "wall_time"]
    for target_id in range(int(num_agents)):
        fields.append(f"beta_to_actor_{target_id}")
    for target_id in range(int(num_agents)):
        fields.append(f"lambda_to_actor_{target_id}")
    for target_id in range(int(num_agents)):
        fields.append(f"weight_to_actor_{target_id}")
    return fields


def build_detail_csv_fieldnames(
    critic_networks: list[Any],
    actor_networks: list[Any] | None = None,
) -> list[str]:
    n_networks = len(critic_networks)
    fields = ["step", "wall_time", "ram_mb", "current_lr", "current_epsilon"]
    for agent_idx in range(n_networks):
        for name, _ in critic_networks[agent_idx].named_parameters():
            if "weight" in name:
                fields.append(f"critic_weight_norm_agent_{agent_idx}_{name}")
                fields.append(f"critic_grad_norm_agent_{agent_idx}_{name}")
    if actor_networks:
        fields.append("current_actor_lr")
        for agent_idx in range(len(actor_networks)):
            for name, _ in actor_networks[agent_idx].named_parameters():
                if "weight" in name:
                    fields.append(f"actor_weight_norm_agent_{agent_idx}_{name}")
                    fields.append(f"actor_grad_norm_agent_{agent_idx}_{name}")
    fields.extend(["td_loss_step", "value_pred_mean", "value_pred_std"])
    fields.extend(["vram_allocated_mb", "vram_peak_mb", "vram_total_mb"])
    return fields
