"""Diagnostics for pickups outside an actor's relatedness/interests.

A successful pickup is marked "unrelated" when actor i picks task type tau and
actor i is outside task tau's interested set, i.e. circular_dist(i, tau) is
larger than env.relatedness_width. Proficiency still controls whether the actor
can pick the task; this diagnostic asks whether learned policies use that
proficiency to pick tasks they do not personally care about.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

import orchard.encoding as encoding
from orchard.config import load_config
from orchard.datatypes import ExperimentConfig, State
from orchard.env import create_env
from orchard.env.base import BaseEnv
from orchard.enums import Action, LearningType
from orchard.policy import heuristic_action
from orchard.seed import set_all_seeds
from orchard.trainer import create_trainer


INTEREST_CSV_FIELDS = [
    "run_dir",
    "checkpoint_path",
    "checkpoint_step",
    "policy",
    "rel",
    "prof",
    "eval_steps",
    "n_test_states",
    "eval_seed",
    "agent",
    "successful_picks",
    "unrelated_picks",
    "unrelated_pick_rate",
    "picks_per_step",
    "unrelated_picks_per_step",
]


def _load_config_or_metadata(path: str | Path, overrides: Sequence[str] | None = None) -> ExperimentConfig:
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


def find_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints/ directory in {run_dir}")

    if checkpoint == "latest":
        steps = sorted(
            ckpt_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_", 1)[1]),
        )
        if steps:
            return steps[-1]
        final = ckpt_dir / "final.pt"
        if final.exists():
            return final
        raise FileNotFoundError(f"No step_*.pt or final.pt checkpoint found in {ckpt_dir}")

    if checkpoint == "final":
        final = ckpt_dir / "final.pt"
        if final.exists():
            return final
        return find_checkpoint(run_dir, "latest")

    explicit = Path(checkpoint)
    if explicit.is_absolute() and explicit.exists():
        return explicit
    candidate = ckpt_dir / checkpoint
    if not candidate.suffix:
        candidate = candidate.with_suffix(".pt")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Checkpoint not found: {candidate}")


def actor_interested_in_task(env: BaseEnv, actor: int, task_type: int) -> bool:
    """Whether actor is in task_type's interested set."""
    if getattr(env, "relatedness", None) is not None:
        rel = env.relatedness
        if task_type < rel.shape[0] and actor < rel.shape[1]:
            return bool(rel[task_type, actor] > 0)

    n = env.cfg.n_agents
    dist = min(abs(actor - task_type), n - abs(actor - task_type))
    return dist <= env.cfg.relatedness_width


def rollout_unrelated_pick_counts(
    start: State,
    policy_fn: Callable[[State], Action],
    env: BaseEnv,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (successful_picks_by_actor, unrelated_picks_by_actor)."""
    successful = np.zeros(env.cfg.n_agents, dtype=np.int64)
    unrelated = np.zeros(env.cfg.n_agents, dtype=np.int64)
    state = start

    for _ in range(n_steps):
        move_action = policy_fn(state)
        if not move_action.is_move():
            raise AssertionError(f"Phase 1 policy returned non-move action: {move_action}")
        moved = env.apply_action(state, move_action)

        actor = moved.actor
        eligible_types = env.proficiency_positive_types[actor]
        on_task = moved.is_agent_on_task(actor, eligible_types)

        if on_task:
            pick_action = policy_fn(moved.with_pick_phase())
            pick_type = pick_action.pick_type() if pick_action.is_pick() else None
            n_tasks_before = len(moved.task_positions)
            picked, _ = env.resolve_pick(moved, pick_type=pick_type)
            if pick_type is not None and len(picked.task_positions) < n_tasks_before:
                successful[actor] += 1
                if not actor_interested_in_task(env, actor, pick_type):
                    unrelated[actor] += 1
            state = env.advance_actor(env.spawn_and_despawn(picked))
        else:
            state = env.advance_actor(env.spawn_and_despawn(moved))

    return successful, unrelated


def build_loaded_policy(
    run_dir: Path,
    *,
    checkpoint: str = "latest",
    policy: str = "greedy",
    device: str = "cpu",
    overrides: Sequence[str] | None = None,
):
    cfg = _load_config_or_metadata(run_dir, overrides=overrides)
    cfg = dataclasses.replace(
        cfg,
        train=dataclasses.replace(cfg.train, use_gpu=(device == "cuda")),
    )
    set_all_seeds(cfg.train.seed)
    env = create_env(cfg.env)
    n_networks = 1 if cfg.train.learning_type == LearningType.CENTRALIZED else cfg.env.n_agents
    encoding.init_encoder(cfg.model.encoder, env, n_networks=n_networks)
    trainer = create_trainer(cfg, env)
    ckpt_path = find_checkpoint(run_dir, checkpoint)
    loaded_step = trainer.load_checkpoint(ckpt_path)
    trainer.sync_to_cpu()

    if policy == "greedy":
        policy_fn = lambda s: trainer._greedy_action(s)
    elif policy == "nearest":
        policy_fn = lambda s: heuristic_action(s, env, cfg.train.heuristic)
    else:
        raise ValueError(f"Unknown policy {policy!r}; expected 'greedy' or 'nearest'.")

    return cfg, env, policy_fn, ckpt_path, loaded_step


def evaluate_unrelated_pick_rates(
    run_dir: str | Path,
    *,
    checkpoint: str = "latest",
    policy: str = "greedy",
    eval_steps: int | None = None,
    n_test_states: int | None = None,
    eval_seed: int | None = None,
    device: str = "cpu",
    overrides: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Evaluate one run and return one row per actor."""
    run_dir = Path(run_dir)
    cfg, env, policy_fn, ckpt_path, loaded_step = build_loaded_policy(
        run_dir,
        checkpoint=checkpoint,
        policy=policy,
        device=device,
        overrides=overrides,
    )
    steps = int(eval_steps if eval_steps is not None else cfg.eval.eval_steps)
    states = int(n_test_states if n_test_states is not None else cfg.eval.n_test_states)
    seed = eval_seed if eval_seed is not None else cfg.eval.eval_seed

    successful = np.zeros(cfg.env.n_agents, dtype=np.int64)
    unrelated = np.zeros(cfg.env.n_agents, dtype=np.int64)
    env.set_eval_mode(True, seed=seed)
    try:
        for _ in range(max(states, 0)):
            start = env.init_state()
            s_counts, u_counts = rollout_unrelated_pick_counts(start, policy_fn, env, steps)
            successful += s_counts
            unrelated += u_counts
    finally:
        env.set_eval_mode(False)

    denom_steps = float(steps * max(states, 0)) if steps > 0 and states > 0 else np.nan
    rows = []
    for agent in range(cfg.env.n_agents):
        rate = unrelated[agent] / successful[agent] if successful[agent] else np.nan
        rows.append({
            "run_dir": str(run_dir),
            "checkpoint_path": str(ckpt_path),
            "checkpoint_step": loaded_step,
            "policy": policy,
            "rel": cfg.env.relatedness_width,
            "prof": cfg.env.proficiency_width,
            "eval_steps": steps,
            "n_test_states": states,
            "eval_seed": seed,
            "agent": agent,
            "successful_picks": int(successful[agent]),
            "unrelated_picks": int(unrelated[agent]),
            "unrelated_pick_rate": float(rate) if np.isfinite(rate) else np.nan,
            "picks_per_step": float(successful[agent] / denom_steps) if np.isfinite(denom_steps) else np.nan,
            "unrelated_picks_per_step": float(unrelated[agent] / denom_steps) if np.isfinite(denom_steps) else np.nan,
        })
    return pd.DataFrame(rows)


def write_interest_csv(path: str | Path, rows: pd.DataFrame | Iterable[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(output, index=False)
        return
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INTEREST_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in INTEREST_CSV_FIELDS})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate unrelated/interests-outside pickups for orchard runs.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--policy", choices=["greedy", "nearest"], default="greedy")
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--n-test-states", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("unrelated_pick_rates.csv"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    frames = []
    for run_dir in args.run_dirs:
        df = evaluate_unrelated_pick_rates(
            run_dir,
            checkpoint=args.checkpoint,
            policy=args.policy,
            eval_steps=args.eval_steps,
            n_test_states=args.n_test_states,
            eval_seed=args.eval_seed,
            device=args.device,
        )
        frames.append(df)
        mean_rate = df["unrelated_pick_rate"].mean(skipna=True)
        print(f"{run_dir}: mean unrelated pickup rate = {mean_rate:.4f}")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=INTEREST_CSV_FIELDS)
    write_interest_csv(args.output, out)
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
