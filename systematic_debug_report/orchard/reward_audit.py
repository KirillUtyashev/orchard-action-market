"""Reward-vector audit helpers for experiment sanity checks."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiagonalRewardIssue:
    reward_vectors_path: Path
    task_type: int
    agent: int
    reward: float


def _resolve_reward_vector_paths(path: Path) -> list[Path]:
    if path.is_file():
        if path.name != "reward_vectors.csv":
            raise ValueError(f"Expected reward_vectors.csv file, got: {path}")
        return [path]

    direct = path / "reward_vectors.csv"
    if direct.exists():
        return [direct]

    return sorted(path.rglob("reward_vectors.csv"))


def audit_positive_diagonal_rewards(path: str | Path) -> list[DiagonalRewardIssue]:
    """Return assigned-task reward issues from reward_vectors.csv files.

    For S=0 and n_task_types=n_agents experiments, task_type i is meant to be
    picked by agent i. This audit flags rows where reward_agent_i <= 0.
    """
    root = Path(path)
    reward_paths = _resolve_reward_vector_paths(root)
    if not reward_paths:
        raise FileNotFoundError(f"No reward_vectors.csv found under {root}")

    issues: list[DiagonalRewardIssue] = []
    for reward_path in reward_paths:
        with open(reward_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_type = int(row["task_type"])
                reward_key = f"reward_agent_{task_type}"
                if reward_key not in row:
                    continue
                reward = float(row[reward_key])
                if reward <= 0.0:
                    issues.append(
                        DiagonalRewardIssue(
                            reward_vectors_path=reward_path,
                            task_type=task_type,
                            agent=task_type,
                            reward=reward,
                        )
                    )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit S=0 diagonal reward entries in reward_vectors.csv files."
    )
    parser.add_argument(
        "path",
        help="Path to reward_vectors.csv, a run directory, or a directory tree of runs.",
    )
    args = parser.parse_args()

    issues = audit_positive_diagonal_rewards(args.path)
    if issues:
        print(f"Found {len(issues)} non-positive assigned rewards:")
        for issue in issues:
            print(
                f"{issue.reward_vectors_path}: "
                f"task_type={issue.task_type} agent={issue.agent} reward={issue.reward:.8g}"
            )
        raise SystemExit(1)

    print("All assigned diagonal rewards are positive.")


if __name__ == "__main__":
    main()
