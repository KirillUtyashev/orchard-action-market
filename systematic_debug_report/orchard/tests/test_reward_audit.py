"""Tests for reward-vector audit helpers."""

from __future__ import annotations

import csv
from pathlib import Path

from orchard.reward_audit import audit_positive_diagonal_rewards


def _write_reward_vectors(path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames = ["task_type", "reward_agent_0", "reward_agent_1", "reward_agent_2"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_positive_diagonal_rewards_pass(tmp_path):
    reward_path = tmp_path / "reward_vectors.csv"
    _write_reward_vectors(
        reward_path,
        [
            {"task_type": 0, "reward_agent_0": 0.2, "reward_agent_1": -1.0, "reward_agent_2": -1.0},
            {"task_type": 1, "reward_agent_0": -1.0, "reward_agent_1": 0.3, "reward_agent_2": -1.0},
            {"task_type": 2, "reward_agent_0": -1.0, "reward_agent_1": -1.0, "reward_agent_2": 0.4},
        ],
    )

    assert audit_positive_diagonal_rewards(reward_path) == []


def test_non_positive_diagonal_rewards_are_reported(tmp_path):
    reward_path = tmp_path / "reward_vectors.csv"
    _write_reward_vectors(
        reward_path,
        [
            {"task_type": 0, "reward_agent_0": 0.2, "reward_agent_1": 3.0, "reward_agent_2": 3.0},
            {"task_type": 1, "reward_agent_0": 3.0, "reward_agent_1": 0.0, "reward_agent_2": 3.0},
            {"task_type": 2, "reward_agent_0": 3.0, "reward_agent_1": 3.0, "reward_agent_2": -0.1},
        ],
    )

    issues = audit_positive_diagonal_rewards(tmp_path)

    assert len(issues) == 2
    assert issues[0].task_type == 1
    assert issues[0].agent == 1
    assert issues[0].reward == 0.0
    assert issues[1].task_type == 2
    assert issues[1].agent == 2
    assert issues[1].reward == -0.1
