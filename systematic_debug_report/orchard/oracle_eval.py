"""Optimistic oracle evaluation for orchard runs.

The oracle uses the real stochastic environment mechanics, but replaces the
learned policy with an immediate best-available-task picker. It ignores
movement, so it is an optimistic inventory upper bound rather than an
achievable movement-constrained policy.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml

from orchard.config import load_config
from orchard.datatypes import ExperimentConfig, State
from orchard.env import create_env
from orchard.env.base import BaseEnv
from orchard.seed import set_all_seeds


ORACLE_CSV_FIELDS = [
    "source_path",
    "run_dir",
    "oracle_rps_mean",
    "oracle_rps_std",
    "oracle_team_rps_mean",
    "oracle_team_rps_std",
    "oracle_tasks_picked_per_step_mean",
    "oracle_tasks_picked_per_step_std",
    "oracle_eval_steps",
    "oracle_n_test_states",
    "oracle_eval_seed",
    "oracle_train_seed",
    "oracle_reward_seed",
]


def _load_config_or_metadata(path: str | Path, overrides: Sequence[str] | None = None) -> ExperimentConfig:
    """Load a raw experiment config, run metadata.yaml, or run directory."""
    path = Path(path)
    if path.is_dir():
        metadata_path = path / "metadata.yaml"
        config_path = path / "config.yaml"
        if metadata_path.exists():
            path = metadata_path
        elif config_path.exists():
            path = config_path
        else:
            raise FileNotFoundError(f"No metadata.yaml or config.yaml found in {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if "config" in raw and "env" not in raw:
        raw = raw["config"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(raw, tmp)
        tmp_path = tmp.name
    try:
        return load_config(tmp_path, overrides=list(overrides or ()))
    finally:
        os.unlink(tmp_path)


def _remove_task(state: State, task_idx: int) -> State:
    assert state.task_types is not None, "task_types must be set"
    return State(
        agent_positions=state.agent_positions,
        task_positions=state.task_positions[:task_idx] + state.task_positions[task_idx + 1:],
        actor=state.actor,
        task_types=state.task_types[:task_idx] + state.task_types[task_idx + 1:],
    )


def oracle_step(state: State, env: BaseEnv) -> tuple[State, tuple[float, ...], int]:
    """Advance one actor turn using the best positive available team reward."""
    assert state.task_types is not None, "task_types must be set"
    actor = state.actor
    best_idx: int | None = None
    best_tau: int | None = None
    best_team_reward = float("-inf")

    for idx, tau in enumerate(state.task_types):
        team_reward = float(np.sum(env._pick_rewards[actor, tau]))
        if team_reward > best_team_reward:
            best_team_reward = team_reward
            best_idx = idx
            best_tau = tau

    if best_idx is not None and best_tau is not None and best_team_reward > 0.0:
        rewards = tuple(float(x) for x in env._pick_rewards[actor, best_tau].tolist())
        after_choice = _remove_task(state, best_idx)
        tasks_picked = 1
    else:
        rewards = tuple(0.0 for _ in range(env.cfg.n_agents))
        after_choice = state
        tasks_picked = 0

    next_state = env.advance_actor(env.spawn_and_despawn(after_choice))
    return next_state, rewards, tasks_picked


def evaluate_oracle_metrics(start_state: State, env: BaseEnv, n_steps: int) -> dict[str, float]:
    """Compute oracle reward and pick rates over ``n_steps`` actor turns."""
    total_reward = 0.0
    total_team_reward = 0.0
    tasks_picked = 0
    state = start_state

    for _ in range(n_steps):
        actor = state.actor
        state, rewards, picked = oracle_step(state, env)
        total_reward += rewards[actor]
        total_team_reward += sum(rewards)
        tasks_picked += picked

    denom = float(n_steps) if n_steps > 0 else 1.0
    return {
        "rps": total_reward / denom if n_steps > 0 else 0.0,
        "team_rps": total_team_reward / denom if n_steps > 0 else 0.0,
        "tasks_picked_per_step": tasks_picked / denom if n_steps > 0 else 0.0,
    }


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _apply_reward_vectors(env: BaseEnv, reward_vectors_path: str | Path | None) -> None:
    if reward_vectors_path is None:
        return
    path = Path(reward_vectors_path)
    if not path.exists():
        return

    rows_by_task_type: dict[int, dict[str, str]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_task_type[int(row["task_type"])] = row

    if len(rows_by_task_type) != env.cfg.n_task_types:
        raise ValueError(
            f"Expected {env.cfg.n_task_types} reward rows in {path}, "
            f"found {len(rows_by_task_type)}."
        )

    rewards = np.zeros((env.cfg.n_task_types, env.cfg.n_agents), dtype=np.float32)
    for tau in range(env.cfg.n_task_types):
        row = rows_by_task_type[tau]
        for agent_idx in range(env.cfg.n_agents):
            rewards[tau, agent_idx] = float(row[f"reward_agent_{agent_idx}"])

    env.category_rewards = rewards
    first_row = rows_by_task_type[0]
    if first_row.get("reward_seed") not in (None, ""):
        env.category_reward_seed = int(first_row["reward_seed"])
    env._precompute_pick_rewards()


def evaluate_oracle(
    cfg: ExperimentConfig,
    *,
    eval_steps: int | None = None,
    n_test_states: int | None = None,
    eval_seed: int | None = None,
    reward_vectors_path: str | Path | None = None,
) -> dict[str, float | int | None]:
    """Evaluate the oracle across one or more eval start states.

    Reward generation is reproduced by reseeding with ``cfg.train.seed`` before
    environment construction. Eval state/spawn randomness then follows
    ``eval_seed`` if provided, otherwise ``cfg.eval.eval_seed``.
    """
    steps = int(eval_steps if eval_steps is not None else cfg.eval.eval_steps)
    n_states = int(n_test_states if n_test_states is not None else cfg.eval.n_test_states)
    effective_eval_seed = eval_seed if eval_seed is not None else cfg.eval.eval_seed

    set_all_seeds(cfg.train.seed)
    env = create_env(cfg.env)
    _apply_reward_vectors(env, reward_vectors_path)

    per_state = []
    env.set_eval_mode(True, seed=effective_eval_seed)
    try:
        for _ in range(max(n_states, 0)):
            start = env.init_state()
            per_state.append(evaluate_oracle_metrics(start, env, steps))
    finally:
        env.set_eval_mode(False)

    rps_mean, rps_std = _mean_std([m["rps"] for m in per_state])
    team_mean, team_std = _mean_std([m["team_rps"] for m in per_state])
    picked_mean, picked_std = _mean_std([m["tasks_picked_per_step"] for m in per_state])

    return {
        "oracle_rps_mean": rps_mean,
        "oracle_rps_std": rps_std,
        "oracle_team_rps_mean": team_mean,
        "oracle_team_rps_std": team_std,
        "oracle_tasks_picked_per_step_mean": picked_mean,
        "oracle_tasks_picked_per_step_std": picked_std,
        "oracle_eval_steps": steps,
        "oracle_n_test_states": n_states,
        "oracle_eval_seed": effective_eval_seed,
        "oracle_train_seed": cfg.train.seed,
        "oracle_reward_seed": getattr(env, "category_reward_seed", None),
    }


def _target_run_dir(path: Path) -> Path | None:
    if path.is_dir() and (path / "metadata.yaml").exists():
        return path
    if path.name == "metadata.yaml":
        return path.parent
    return None


def evaluate_path(
    path: str | Path,
    *,
    eval_steps: int | None = None,
    n_test_states: int | None = None,
    eval_seed: int | None = None,
    overrides: Sequence[str] | None = None,
) -> dict[str, float | int | str | None]:
    path = Path(path)
    cfg = _load_config_or_metadata(path, overrides=overrides)
    run_dir = _target_run_dir(path)
    reward_vectors_path = run_dir / "reward_vectors.csv" if run_dir is not None else None
    metrics = evaluate_oracle(
        cfg,
        eval_steps=eval_steps,
        n_test_states=n_test_states,
        eval_seed=eval_seed,
        reward_vectors_path=reward_vectors_path,
    )
    return {
        "source_path": str(path),
        "run_dir": str(run_dir) if run_dir is not None else "",
        **metrics,
    }


def write_oracle_csv(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ORACLE_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in ORACLE_CSV_FIELDS})


def find_oracle_targets(paths: Sequence[str | Path], *, recursive: bool = False) -> list[Path]:
    targets: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if recursive and path.is_dir() and not (path / "metadata.yaml").exists() and not (path / "config.yaml").exists():
            targets.extend(sorted(p.parent for p in path.rglob("metadata.yaml")))
            targets.extend(sorted(path.rglob("config.yaml")))
        else:
            targets.append(path)
    return targets


def _default_output_path(targets: Sequence[Path], *, recursive: bool) -> Path:
    if len(targets) == 1:
        target = targets[0]
        run_dir = _target_run_dir(target)
        if run_dir is not None:
            return run_dir / "oracle_metrics.csv"
        if target.is_dir():
            return target / "oracle_metrics.csv"
        return target.parent / "oracle_metrics.csv"
    if recursive and targets:
        common = Path(os.path.commonpath([str(t if t.is_dir() else t.parent) for t in targets]))
        return common / "oracle_metrics.csv"
    return Path("oracle_metrics.csv")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optimistic oracle eval for orchard configs or runs.")
    parser.add_argument("paths", nargs="+", type=Path, help="Config YAML, metadata.yaml, run dir, or sweep root.")
    parser.add_argument("--recursive", action="store_true", help="Recurse through sweep roots looking for metadata.yaml/config.yaml.")
    parser.add_argument("--eval-steps", type=int, default=None, help="Override eval.eval_steps.")
    parser.add_argument("--n-test-states", type=int, default=None, help="Override eval.n_test_states.")
    parser.add_argument("--eval-seed", type=int, default=None, help="Override eval.eval_seed.")
    parser.add_argument("--output", type=Path, default=None, help="Combined CSV output path.")
    parser.add_argument("--no-per-run", action="store_true", help="Do not write oracle_metrics.csv into individual run dirs.")
    parser.add_argument("--override", action="append", default=[], help="Config override key=value. Can be repeated.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    targets = find_oracle_targets(args.paths, recursive=args.recursive)
    if not targets:
        raise FileNotFoundError("No oracle eval targets found.")

    rows = []
    for target in targets:
        row = evaluate_path(
            target,
            eval_steps=args.eval_steps,
            n_test_states=args.n_test_states,
            eval_seed=args.eval_seed,
            overrides=args.override,
        )
        rows.append(row)
        run_dir = Path(row["run_dir"]) if row.get("run_dir") else None
        if run_dir is not None and not args.no_per_run:
            write_oracle_csv(run_dir / "oracle_metrics.csv", [row])
        print(
            f"{target}: oracle_team_rps={row['oracle_team_rps_mean']:.6f} "
            f"+/- {row['oracle_team_rps_std']:.6f} over {row['oracle_n_test_states']} states"
        )

    output_path = args.output or _default_output_path(targets, recursive=args.recursive)
    write_oracle_csv(output_path, rows)
    print(f"Wrote {len(rows)} row(s) to {output_path}")


if __name__ == "__main__":
    main()
