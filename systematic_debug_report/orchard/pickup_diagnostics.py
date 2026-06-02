"""Evaluate trained policies and summarize successful pickups by task type.

Example:
    python -m orchard.pickup_diagnostics \
      /path/to/slurm_experiments/value_learning_sampled_mean/runs \
      --checkpoint latest \
      --eval-steps 5000 \
      --eval-seeds 42,43,44 \
      --task-sum-min 1 \
      --task-sum-max 2 \
      --out /path/to/pickup_diagnostics
"""

from __future__ import annotations

import argparse
import dataclasses
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

import orchard.encoding as encoding
from orchard.config import _parse_env, _parse_eval, _parse_model, _parse_train
from orchard.datatypes import EvalConfig, ExperimentConfig, LoggingConfig, State
from orchard.env import create_env
from orchard.env.base import BaseEnv
from orchard.enums import Action, LearningType
from orchard.policy import heuristic_action
from orchard.trainer import create_trainer


def find_run_dirs(paths: list[Path]) -> list[Path]:
    run_dirs: set[Path] = set()
    for path in paths:
        path = path.expanduser().resolve()
        if (path / "metadata.yaml").exists():
            run_dirs.add(path)
        elif path.is_dir():
            for meta in path.rglob("metadata.yaml"):
                if (meta.parent / "checkpoints").exists():
                    run_dirs.add(meta.parent)
    return sorted(run_dirs)


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


def load_config_from_run(run_dir: Path, *, device: str, eval_steps: int, eval_seed: int) -> ExperimentConfig:
    with open(run_dir / "metadata.yaml") as f:
        meta = yaml.safe_load(f)
    raw = meta["config"]

    env_cfg = _parse_env(raw["env"])
    model_cfg = _parse_model(raw["model"])
    actor_model_raw = raw.get("actor_model")
    actor_model_cfg = _parse_model(actor_model_raw) if actor_model_raw else None
    train_cfg = _parse_train(raw["train"])
    train_cfg = dataclasses.replace(train_cfg, use_gpu=(device == "cuda"))
    eval_raw = raw.get("eval", {})
    eval_cfg = _parse_eval(eval_raw) if eval_raw else EvalConfig()
    eval_cfg = dataclasses.replace(eval_cfg, eval_steps=eval_steps, eval_seed=eval_seed)

    return ExperimentConfig(
        env=env_cfg,
        model=model_cfg,
        actor_model=actor_model_cfg,
        train=train_cfg,
        eval=eval_cfg,
        logging=LoggingConfig(timing_csv_freq=0),
    )


def build_loaded_trainer(run_dir: Path, checkpoint: str, *, device: str, eval_steps: int, eval_seed: int):
    cfg = load_config_from_run(run_dir, device=device, eval_steps=eval_steps, eval_seed=eval_seed)
    env = create_env(cfg.env)
    n_networks = 1 if cfg.train.learning_type == LearningType.CENTRALIZED else cfg.env.n_agents
    encoding.init_encoder(cfg.model.encoder, env, n_networks=n_networks)
    trainer = create_trainer(cfg, env)
    ckpt_path = find_checkpoint(run_dir, checkpoint)
    loaded_step = trainer.load_checkpoint(ckpt_path)
    trainer.sync_to_cpu()
    return cfg, env, trainer, ckpt_path, loaded_step


def reward_rows_for_env(run_dir: Path, env: BaseEnv) -> pd.DataFrame:
    reward_path = run_dir / "reward_vectors.csv"
    if reward_path.exists():
        rows = pd.read_csv(reward_path)
    else:
        rows = pd.DataFrame({"task_type": np.arange(env.cfg.n_task_types)})
        for i in range(env.cfg.n_agents):
            rows[f"reward_agent_{i}"] = env.category_rewards[:, i]

    agent_cols = sorted(
        [c for c in rows.columns if c.startswith("reward_agent_")],
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    for col in agent_cols:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")

    matrix = rows[agent_cols].to_numpy(dtype=float)
    rows["reward_sum_calc"] = matrix.sum(axis=1)
    rows["reward_mean_calc"] = matrix.mean(axis=1)
    rows["reward_std_calc"] = matrix.std(axis=1)
    rows["reward_min"] = matrix.min(axis=1)
    rows["reward_max"] = matrix.max(axis=1)
    rows["entry_frac_negative"] = (matrix < 0.0).mean(axis=1)
    rows["negative_mass"] = -np.clip(matrix, None, 0.0).sum(axis=1)
    rows["positive_mass"] = np.clip(matrix, 0.0, None).sum(axis=1)
    rows["gross_mass"] = rows["negative_mass"] + rows["positive_mass"]
    safe_sum = rows["reward_sum_calc"].where(rows["reward_sum_calc"].abs() > 1e-12, np.nan)
    rows["negative_mass_over_sum"] = rows["negative_mass"] / safe_sum
    rows["gross_mass_over_sum"] = rows["gross_mass"] / safe_sum
    rows["task_type"] = pd.to_numeric(rows["task_type"], errors="coerce").astype(int)
    return rows


def rollout_pick_counts(
    start: State,
    policy_fn: Callable[[State], Action],
    env: BaseEnv,
    n_steps: int,
) -> dict[str, Counter]:
    s = start
    successful = Counter()
    attempts = Counter()
    opportunities = Counter()
    stays_on_task = Counter()
    team_reward = Counter()
    actor_reward = Counter()

    for _ in range(n_steps):
        move_action = policy_fn(s)
        if not move_action.is_move():
            raise AssertionError(f"Phase 1 policy returned non-move action: {move_action}")
        s_moved = env.apply_action(s, move_action)

        actor = s_moved.actor
        eligible_types = env.phi_positive_types[actor]
        on_task = s_moved.is_agent_on_task(actor, eligible_types)

        if on_task:
            actor_pos = s_moved.agent_positions[actor]
            eligible_here = sorted({tau for _, tau in s_moved.tasks_at(actor_pos) if tau in eligible_types})
            for tau in eligible_here:
                opportunities[tau] += 1

            pick_action = policy_fn(s_moved.with_pick_phase())
            pick_type = pick_action.pick_type() if pick_action.is_pick() else None
            if pick_type is None:
                for tau in eligible_here:
                    stays_on_task[tau] += 1
            else:
                attempts[pick_type] += 1

            n_tasks_before = len(s_moved.task_positions)
            s_picked, rewards = env.resolve_pick(s_moved, pick_type=pick_type)
            if pick_type is not None and len(s_picked.task_positions) < n_tasks_before:
                successful[pick_type] += 1
                team_reward[pick_type] += float(sum(rewards))
                actor_reward[pick_type] += float(rewards[actor])

            s = env.advance_actor(env.spawn_and_despawn(s_picked))
        else:
            s = env.advance_actor(env.spawn_and_despawn(s_moved))

    return {
        "successful": successful,
        "attempts": attempts,
        "opportunities": opportunities,
        "stays_on_task": stays_on_task,
        "team_reward": team_reward,
        "actor_reward": actor_reward,
    }


def run_policy_pick_diagnostics(
    run_dir: Path,
    *,
    checkpoint: str,
    eval_steps: int,
    eval_seeds: list[int],
    policies: list[str],
    device: str,
    task_sum_min: float | None,
    task_sum_max: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg, env, trainer, ckpt_path, loaded_step = build_loaded_trainer(
        run_dir,
        checkpoint,
        device=device,
        eval_steps=eval_steps,
        eval_seed=eval_seeds[0],
    )
    rewards = reward_rows_for_env(run_dir, env)

    policy_fns: dict[str, Callable[[State], Action]] = {}
    if "greedy" in policies:
        policy_fns["greedy"] = lambda s: trainer._greedy_action(s)
    if "nearest" in policies:
        policy_fns["nearest"] = lambda s: heuristic_action(s, env, cfg.train.heuristic)

    all_rows: list[dict[str, object]] = []
    for policy_name, policy_fn in policy_fns.items():
        aggregate = {
            "successful": Counter(),
            "attempts": Counter(),
            "opportunities": Counter(),
            "stays_on_task": Counter(),
            "team_reward": Counter(),
            "actor_reward": Counter(),
        }

        for seed in eval_seeds:
            env.set_eval_mode(True, seed=seed)
            try:
                start = env.init_state()
                counts = rollout_pick_counts(start, policy_fn, env, eval_steps)
            finally:
                env.set_eval_mode(False)
            for key, counter in counts.items():
                aggregate[key].update(counter)

        total_successful = sum(aggregate["successful"].values())
        total_attempts = sum(aggregate["attempts"].values())
        total_opportunities = sum(aggregate["opportunities"].values())

        for tau in range(cfg.env.n_task_types):
            row = {
                "run_dir": str(run_dir),
                "checkpoint_path": str(ckpt_path),
                "checkpoint_step": loaded_step,
                "policy": policy_name,
                "eval_steps_per_seed": eval_steps,
                "n_eval_seeds": len(eval_seeds),
                "task_type": tau,
                "successful_picks": aggregate["successful"][tau],
                "pick_attempts": aggregate["attempts"][tau],
                "task_opportunities": aggregate["opportunities"][tau],
                "stay_on_task_events": aggregate["stays_on_task"][tau],
                "team_reward_from_picks": aggregate["team_reward"][tau],
                "actor_reward_from_picks": aggregate["actor_reward"][tau],
                "mean_team_reward_per_successful_pick": (
                    aggregate["team_reward"][tau] / aggregate["successful"][tau]
                    if aggregate["successful"][tau]
                    else np.nan
                ),
                "mean_actor_reward_per_successful_pick": (
                    aggregate["actor_reward"][tau] / aggregate["successful"][tau]
                    if aggregate["successful"][tau]
                    else np.nan
                ),
                "successful_pick_share": (
                    aggregate["successful"][tau] / total_successful if total_successful else 0.0
                ),
                "attempt_share": aggregate["attempts"][tau] / total_attempts if total_attempts else 0.0,
                "opportunity_share": (
                    aggregate["opportunities"][tau] / total_opportunities if total_opportunities else 0.0
                ),
                "success_per_attempt": (
                    aggregate["successful"][tau] / aggregate["attempts"][tau]
                    if aggregate["attempts"][tau]
                    else np.nan
                ),
                "success_per_opportunity": (
                    aggregate["successful"][tau] / aggregate["opportunities"][tau]
                    if aggregate["opportunities"][tau]
                    else np.nan
                ),
            }
            all_rows.append(row)

    by_task = pd.DataFrame(all_rows).merge(rewards, on="task_type", how="left")
    in_window = pd.Series(True, index=by_task.index)
    if task_sum_min is not None:
        in_window &= by_task["reward_sum_calc"] >= task_sum_min
    if task_sum_max is not None:
        in_window &= by_task["reward_sum_calc"] <= task_sum_max
    by_task["in_task_sum_window"] = in_window
    by_task["task_sum_window_min"] = task_sum_min if task_sum_min is not None else np.nan
    by_task["task_sum_window_max"] = task_sum_max if task_sum_max is not None else np.nan
    add_run_metadata(by_task, cfg)
    by_run = summarize_correlations(by_task)
    return by_task, by_run


def add_run_metadata(df: pd.DataFrame, cfg: ExperimentConfig) -> None:
    stoch = cfg.env.stochastic
    df["learning_type"] = cfg.train.learning_type.name.lower()
    df["decentralized_reward_target"] = cfg.train.decentralized_reward_target.name.lower()
    df["sigma_a"] = stoch.sigma_a if stoch is not None else np.nan
    df["sigma_b"] = stoch.sigma_b if stoch is not None else np.nan
    df["baseline_team_sum_mean"] = stoch.baseline_team_sum_mean if stoch is not None else np.nan
    df["deterministic_baseline_offsets"] = (
        stoch.deterministic_baseline_offsets if stoch is not None else False
    )
    df["reward_generation"] = stoch.reward_generation.name.lower() if stoch is not None else ""
    df["positive_rewards_only"] = stoch.positive_rewards_only if stoch is not None else False
    df["require_no_negative_dominates_positive"] = (
        stoch.require_no_negative_dominates_positive if stoch is not None else False
    )
    df["lr"] = cfg.train.lr.start
    df["width"] = cfg.model.conv_specs[0][0] if cfg.model.conv_specs else np.nan


def summarize_correlations(by_task: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["run_dir", "policy"]
    reward_cols = [
        "reward_sum_calc",
        "reward_mean_calc",
        "reward_min",
        "reward_max",
        "entry_frac_negative",
        "negative_mass",
        "negative_mass_over_sum",
        "gross_mass_over_sum",
    ]
    count_cols = ["successful_picks", "pick_attempts", "task_opportunities", "successful_pick_share"]
    for (run_dir, policy), group in by_task.groupby(group_cols, dropna=False):
        row: dict[str, object] = {
            "run_dir": run_dir,
            "policy": policy,
            "total_successful_picks": int(group["successful_picks"].sum()),
            "total_pick_attempts": int(group["pick_attempts"].sum()),
            "total_task_opportunities": int(group["task_opportunities"].sum()),
        }
        if "in_task_sum_window" in group.columns:
            focus = group["in_task_sum_window"].astype(bool)
            focus_success = int(group.loc[focus, "successful_picks"].sum())
            focus_attempts = int(group.loc[focus, "pick_attempts"].sum())
            focus_opportunities = int(group.loc[focus, "task_opportunities"].sum())
            row["task_sum_window_min"] = group["task_sum_window_min"].iloc[0]
            row["task_sum_window_max"] = group["task_sum_window_max"].iloc[0]
            row["task_sum_window_n_task_types"] = int(focus.sum())
            row["task_sum_window_successful_picks"] = focus_success
            row["task_sum_window_pick_attempts"] = focus_attempts
            row["task_sum_window_task_opportunities"] = focus_opportunities
            row["task_sum_window_successful_pick_share"] = (
                focus_success / row["total_successful_picks"] if row["total_successful_picks"] else 0.0
            )
            row["task_sum_window_opportunity_share"] = (
                focus_opportunities / row["total_task_opportunities"] if row["total_task_opportunities"] else 0.0
            )
        for meta_col in [
            "checkpoint_path",
            "checkpoint_step",
            "learning_type",
            "decentralized_reward_target",
            "sigma_a",
            "sigma_b",
            "baseline_team_sum_mean",
            "deterministic_baseline_offsets",
            "reward_generation",
            "positive_rewards_only",
            "require_no_negative_dominates_positive",
            "lr",
            "width",
        ]:
            if meta_col in group.columns:
                row[meta_col] = group[meta_col].iloc[0]
        for reward_col in reward_cols:
            if reward_col not in group.columns:
                continue
            for count_col in count_cols:
                x = pd.to_numeric(group[reward_col], errors="coerce")
                y = pd.to_numeric(group[count_col], errors="coerce")
                mask = x.notna() & y.notna()
                if mask.sum() >= 2 and x[mask].nunique() > 1 and y[mask].nunique() > 1:
                    row[f"{reward_col}_vs_{count_col}_pearson"] = float(x[mask].corr(y[mask], method="pearson"))
                    row[f"{reward_col}_vs_{count_col}_spearman"] = float(x[mask].corr(y[mask], method="spearman"))
                else:
                    row[f"{reward_col}_vs_{count_col}_pearson"] = np.nan
                    row[f"{reward_col}_vs_{count_col}_spearman"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def parse_eval_seeds(raw: str) -> list[int]:
    if "," in raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    if ":" in raw:
        start, stop = [int(x.strip()) for x in raw.split(":", 1)]
        return list(range(start, stop))
    return [int(raw)]


def parse_policies(raw: str) -> list[str]:
    policies = [x.strip() for x in raw.split(",") if x.strip()]
    allowed = {"greedy", "nearest"}
    unknown = sorted(set(policies) - allowed)
    if unknown:
        raise ValueError(f"Unknown policies {unknown}; allowed: {sorted(allowed)}")
    return policies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Run dir(s), or roots containing metadata.yaml files.")
    parser.add_argument("--checkpoint", default="latest", help="'latest', 'final', step name, or checkpoint path.")
    parser.add_argument("--eval-steps", type=int, default=5000)
    parser.add_argument("--eval-seeds", default="42", help="One seed, comma-list, or half-open range like 42:52.")
    parser.add_argument("--policies", default="greedy,nearest", help="Comma-list: greedy,nearest.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-runs", type=int, default=0, help="Optional cap for quick checks.")
    parser.add_argument("--task-sum-min", type=float, default=None, help="Optional lower bound for focused task-sum window.")
    parser.add_argument("--task-sum-max", type=float, default=None, help="Optional upper bound for focused task-sum window.")
    parser.add_argument("--out", type=Path, default=None, help="Output prefix or directory. Defaults to first root/plots/pickup_diagnostics.")
    args = parser.parse_args()

    run_dirs = find_run_dirs(args.paths)
    if args.max_runs > 0:
        run_dirs = run_dirs[: args.max_runs]
    if not run_dirs:
        raise FileNotFoundError("No run directories with metadata.yaml/checkpoints found.")

    eval_seeds = parse_eval_seeds(args.eval_seeds)
    policies = parse_policies(args.policies)

    all_by_task = []
    all_by_run = []
    for idx, run_dir in enumerate(run_dirs, 1):
        print(f"[{idx}/{len(run_dirs)}] {run_dir}")
        try:
            by_task, by_run = run_policy_pick_diagnostics(
                run_dir,
                checkpoint=args.checkpoint,
                eval_steps=args.eval_steps,
                eval_seeds=eval_seeds,
                policies=policies,
                device=args.device,
                task_sum_min=args.task_sum_min,
                task_sum_max=args.task_sum_max,
            )
        except Exception as exc:
            print(f"  SKIP: {exc!r}")
            continue
        all_by_task.append(by_task)
        all_by_run.append(by_run)

    if not all_by_task:
        raise RuntimeError("No runs evaluated successfully.")

    by_task_df = pd.concat(all_by_task, ignore_index=True)
    by_run_df = pd.concat(all_by_run, ignore_index=True)

    if args.out is None:
        out_prefix = args.paths[0].expanduser().resolve() / "plots" / "pickup_diagnostics"
    else:
        out_prefix = args.out.expanduser().resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if out_prefix.suffix:
        by_task_path = out_prefix.with_name(out_prefix.stem + "_by_task.csv")
        by_run_path = out_prefix.with_name(out_prefix.stem + "_by_run.csv")
    else:
        out_prefix.mkdir(parents=True, exist_ok=True)
        by_task_path = out_prefix / "pickup_diagnostics_by_task.csv"
        by_run_path = out_prefix / "pickup_diagnostics_by_run.csv"

    by_task_df.to_csv(by_task_path, index=False)
    by_run_df.to_csv(by_run_path, index=False)
    print(f"Saved {by_task_path}")
    print(f"Saved {by_run_path}")
    preview_cols = [
        "run_dir",
        "policy",
        "total_successful_picks",
        "task_sum_window_successful_pick_share",
        "reward_sum_calc_vs_successful_picks_spearman",
    ]
    preview_cols = [col for col in preview_cols if col in by_run_df.columns]
    print(by_run_df[preview_cols].head(20))


if __name__ == "__main__":
    main()
