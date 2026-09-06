"""Export: write trajectory.csv and summary.json from a list of Frames."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from orchard.viz.frame import Frame


def write_trajectory_csv(frames: list[Frame], path: Path) -> None:
    """Write one row per transition with all info."""
    if not frames:
        return

    n_agents = len(frames[0].rewards)
    reward_cols = [f"reward_{i}" for i in range(n_agents)]
    agent_pick_cols = [f"agent_picks_{i}" for i in range(n_agents)]

    fieldnames = [
        "transition",
        "state_index",
        "step",
        "actor",
        "action",
        "picked",
        "picked_task_type",
        "picked_correct",
        *reward_cols,
        "discount",
        "n_tasks",
        "n_tasks_after",
        "cum_picks",
        "cum_correct_picks",
        "cum_wrong_picks",
        "team_rps",
        "picks_per_step",
        *agent_pick_cols,
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for frame in frames:
            row = {
                "transition": frame.transition_index,
                "state_index": frame.state_index,
                "step": frame.step,
                "actor": frame.actor,
                "action": frame.action.name,
                "picked": frame.picked,
                "picked_task_type": frame.picked_task_type if frame.picked_task_type is not None else "",
                "picked_correct": frame.picked_correct if frame.picked_correct is not None else "",
                "discount": frame.discount,
                "n_tasks": frame.tasks_on_grid,
                "n_tasks_after": frame.tasks_after,
                "cum_picks": frame.total_picks,
                "cum_correct_picks": frame.total_correct_picks,
                "cum_wrong_picks": frame.total_wrong_picks,
                "team_rps": f"{frame.team_reward_per_step:.6f}",
                "picks_per_step": f"{frame.picks_per_step:.6f}",
            }
            for i, rcol in enumerate(reward_cols):
                row[rcol] = f"{frame.rewards[i]:.4f}"
            for i, pcol in enumerate(agent_pick_cols):
                row[pcol] = frame.agent_picks.get(i, 0) if frame.agent_picks else 0
            writer.writerow(row)


def write_summary_json(
    frames: list[Frame],
    path: Path,
    config_path: str = "",
    checkpoint_path: str = "",
    seed: int = 0,
) -> None:
    """Write summary statistics."""
    if not frames:
        return

    n_agents = len(frames[0].rewards)
    total_decisions = frames[-1].total_decisions
    total_transitions = len(frames)
    total_picks = frames[-1].total_picks
    total_correct = frames[-1].total_correct_picks
    total_wrong = frames[-1].total_wrong_picks
    total_reward = frames[-1].total_reward
    total_team_reward = frames[-1].total_team_reward

    # Per-agent pick counts
    agent_picks = [0] * n_agents
    for frame in frames:
        if frame.picked:
            agent_picks[frame.actor] += 1

    # Task count stats
    task_counts = [f.tasks_on_grid for f in frames]
    avg_tasks_all = sum(task_counts) / len(task_counts)

    last_n = min(100, len(task_counts))
    avg_tasks_last = sum(task_counts[-last_n:]) / last_n

    summary = {
        "policy": frames[0].policy_name,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "seed": seed,
        "gamma": frames[0].discount if frames[0].discount != 1.0 else frames[-1].discount,
        "total_decisions": total_decisions,
        "total_transitions": total_transitions,
        "total_picks": total_picks,
        "total_correct_picks": total_correct,
        "total_wrong_picks": total_wrong,
        "picks_per_step": total_picks / total_decisions if total_decisions > 0 else 0.0,
        "total_reward": total_reward,
        "reward_per_step": total_reward / total_decisions if total_decisions > 0 else 0.0,
        "total_team_reward": total_team_reward,
        "team_reward_per_step": total_team_reward / total_decisions if total_decisions > 0 else 0.0,
        "correct_picks_per_step": total_correct / total_decisions if total_decisions > 0 else 0.0,
        "wrong_picks_per_step": total_wrong / total_decisions if total_decisions > 0 else 0.0,
        "avg_tasks_all": avg_tasks_all,
        f"avg_tasks_last_{last_n}": avg_tasks_last,
        "agent_pick_counts": agent_picks,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def write_reward_variances_json(
    category_rewards,
    sigma_a: float,
    sigma_b: float,
    reward_generation: str,
    path: Path,
    mean_team_reward: float = 1.0,
) -> None:
    """Write per-agent / per-task / team-total variance diagnostics of the reward matrix.

    category_rewards is (n_task_types, n_agents): entry [k, j] is the reward agent j
    receives for picking task type k. All stats are POPULATION (ddof=0), matching the
    generator's internal asserts, so the circulant guarantees can be checked exactly.
    """
    R = np.asarray(category_rewards, dtype=np.float64)  # (T, N) = (task_types, agents)
    T, N = R.shape

    # per agent (column j): distribution of that agent's rewards ACROSS TASKS (over rows)
    agent_mean = R.mean(axis=0)             # (N,)
    agent_var = R.var(axis=0, ddof=0)       # (N,)  <- one variance per agent, no averaging
    agent_std = R.std(axis=0, ddof=0)       # (N,)

    # per task (row k): distribution of a task's rewards ACROSS AGENTS (over cols)
    task_mean = R.mean(axis=1)              # (T,)
    task_var = R.var(axis=1, ddof=0)        # (T,)  <- one variance per task
    task_std = R.std(axis=1, ddof=0)        # (T,)

    # team total per task = sum of reward over agents; its spread ACROSS TASKS
    team_total = R.sum(axis=1)              # (T,)
    team_total_var = float(team_total.var(ddof=0))
    team_total_std = float(team_total.std(ddof=0))

    def _f(a):
        return [float(v) for v in a]

    out = {
        "reward_generation": reward_generation,
        "note": ("category_rewards[task, agent]. Population stats (ddof=0). "
                 "circulant_all_to_all is deterministic (seed-independent)."),
        "shape": {"n_task_types": T, "n_agents": N},
        "config_sigma_a": float(sigma_a),
        "config_sigma_b": float(sigma_b),

        # (1) agent variance across tasks — ONE PER AGENT, no averaging
        "per_agent_across_tasks": {
            "mean": _f(agent_mean),
            "var": _f(agent_var),
            "std": _f(agent_std),
        },
        # (2) variance of each task across agents — ONE PER TASK
        "per_task_across_agents": {
            "mean": _f(task_mean),
            "var": _f(task_var),
            "std": _f(task_std),
            "team_total": _f(team_total),
        },
        # (3) sum of reward across agents (team total), its variance across tasks
        "team_total_across_tasks": {
            "mean": float(team_total.mean()),
            "var": team_total_var,
            "std": team_total_std,
        },
        # extra roll-ups so the whole picture is visible at a glance
        "aggregates": {
            "grand_mean": float(R.mean()),
            "min_reward": float(R.min()),
            "max_reward": float(R.max()),
            "agent_std_across_tasks": {
                "min": float(agent_std.min()), "max": float(agent_std.max()),
                "mean": float(agent_std.mean()),
            },
            "task_std_across_agents": {
                "min": float(task_std.min()), "max": float(task_std.max()),
                "mean": float(task_std.mean()),
            },
            "sum_of_per_agent_vars": float(agent_var.sum()),
        },
        # what the circulant_all_to_all construction is supposed to yield
        "expected_if_circulant_all_to_all": {
            "agent_std_across_tasks": float((sigma_a ** 2 + sigma_b ** 2 / N ** 2) ** 0.5),
            "task_std_across_agents": float(sigma_a),
            "team_total_std_across_tasks": float(sigma_b),
            "agent_mean_across_tasks": float(mean_team_reward / N),
        },
        # full matrix for manual inspection (rounded)
        "reward_matrix": [[round(float(v), 6) for v in row] for row in R],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
