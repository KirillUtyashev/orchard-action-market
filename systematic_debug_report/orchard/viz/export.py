"""Export: write trajectory.csv and summary.json from a list of Frames."""

from __future__ import annotations

import csv
import json
from pathlib import Path

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
        *reward_cols,
        "discount",
        "n_apples",
        "n_apples_after",
        "cum_picks",
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
                "discount": frame.discount,
                "n_apples": frame.apples_on_grid,
                "n_apples_after": frame.apples_after,
                "cum_picks": frame.total_picks,
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

    # Per-agent pick counts
    agent_picks = [0] * n_agents
    for frame in frames:
        if frame.picked:
            agent_picks[frame.actor] += 1

    # Apple count stats (at s_t of each transition)
    apple_counts = [f.apples_on_grid for f in frames]
    avg_apples_all = sum(apple_counts) / len(apple_counts)

    last_n = min(100, len(apple_counts))
    avg_apples_last = sum(apple_counts[-last_n:]) / last_n

    summary = {
        "policy": frames[0].policy_name,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "seed": seed,
        "gamma": frames[0].discount if frames[0].discount != 1.0 else frames[-1].discount,
        "total_decisions": total_decisions,
        "total_transitions": total_transitions,
        "total_picks": total_picks,
        "picks_per_step": total_picks / total_decisions if total_decisions > 0 else 0.0,
        "avg_apples_all": avg_apples_all,
        f"avg_apples_last_{last_n}": avg_apples_last,
        "agent_pick_counts": agent_picks,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
