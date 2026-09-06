"""MC validation-set evaluation for TD value networks."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import orchard.encoding as encoding
from orchard.datatypes import Grid, State
from orchard.model import ValueNetwork


@dataclass(frozen=True)
class MCValidationExample:
    state_id: int
    state: State
    source: str
    source_turn: int
    team_target: float
    agent_targets: tuple[float, ...]


@dataclass(frozen=True)
class MCValidationResult:
    summary: dict[str, float | int]
    rows: list[dict[str, float | int | str]]


def _parse_grids(value: str) -> tuple[Grid, ...]:
    value = (value or "").strip()
    if not value:
        return ()
    grids: list[Grid] = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        row_s, col_s = item.split(",", 1)
        grids.append(Grid(int(row_s), int(col_s)))
    return tuple(grids)


def _parse_task_types(value: str) -> tuple[int, ...] | None:
    value = (value or "").strip()
    if not value:
        return ()
    return tuple(int(item) for item in value.split(";") if item.strip())


def _mc_validation_csv_paths(path: Path) -> list[Path]:
    if path.is_dir():
        paths = sorted(p for p in path.glob("*.csv") if p.is_file())
        if not paths:
            raise ValueError(f"MC validation directory contains no CSV files: {path}")
        return paths
    if not path.exists():
        raise FileNotFoundError(f"MC validation CSV not found: {path}")
    return [path]


def load_mc_validation_csv(path: str | Path, n_agents: int) -> list[MCValidationExample]:
    path = Path(path)
    csv_paths = _mc_validation_csv_paths(path)

    examples: list[MCValidationExample] = []
    seen_state_ids: set[int] = set()
    for csv_path in csv_paths:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"MC validation CSV has no header: {csv_path}")
            required = {
                "state_id", "source", "source_turn", "actor", "pick_phase",
                "agent_positions", "task_positions", "task_types", "mc_team_mean",
            }
            missing = sorted(required - set(reader.fieldnames))
            missing.extend(
                f"mc_agent_mean_{idx}"
                for idx in range(n_agents)
                if f"mc_agent_mean_{idx}" not in reader.fieldnames
            )
            if missing:
                raise ValueError(f"MC validation CSV missing columns in {csv_path}: {missing}")

            for row in reader:
                state_id = int(row["state_id"])
                if state_id in seen_state_ids:
                    raise ValueError(f"Duplicate MC validation state_id={state_id} across {path}")
                seen_state_ids.add(state_id)

                agent_positions = _parse_grids(row["agent_positions"])
                if len(agent_positions) != n_agents:
                    raise ValueError(
                        f"state_id={row.get('state_id')} has {len(agent_positions)} agents, expected {n_agents}"
                    )
                task_positions = _parse_grids(row["task_positions"])
                task_types = _parse_task_types(row["task_types"])
                if task_types is not None and len(task_types) != len(task_positions):
                    raise ValueError(
                        f"state_id={row.get('state_id')} has {len(task_positions)} tasks "
                        f"but {len(task_types)} task types"
                    )
                state = State(
                    agent_positions=agent_positions,
                    task_positions=task_positions,
                    actor=int(row["actor"]),
                    task_types=task_types,
                    pick_phase=bool(int(row["pick_phase"])),
                )
                examples.append(
                    MCValidationExample(
                        state_id=state_id,
                        state=state,
                        source=row.get("source", ""),
                        source_turn=int(row.get("source_turn", 0)),
                        team_target=float(row["mc_team_mean"]),
                        agent_targets=tuple(float(row[f"mc_agent_mean_{idx}"]) for idx in range(n_agents)),
                    )
                )

    if not examples:
        raise ValueError(f"MC validation CSV contains no rows: {path}")
    return sorted(examples, key=lambda example: example.state_id)


def _summary_stats(prefix: str, pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    return {
        f"{prefix}_rmse": float(math.sqrt(np.mean(err ** 2))),
        f"{prefix}_mae": float(np.mean(np.abs(err))),
        f"{prefix}_bias": float(np.mean(err)),
        f"{prefix}_pred_mean": float(np.mean(pred)),
        f"{prefix}_target_mean": float(np.mean(target)),
    }


def evaluate_mc_validation(
    examples: Sequence[MCValidationExample],
    networks: Sequence[ValueNetwork],
    *,
    n_agents: int,
) -> MCValidationResult:
    if not examples:
        raise ValueError("MC validation set is empty")
    if not networks:
        raise ValueError("No value networks available for MC validation")

    centralized = len(networks) == 1
    was_training = [net.training for net in networks]
    for net in networks:
        net.eval()

    rows: list[dict[str, float | int | str]] = []
    team_preds: list[float] = []
    team_targets: list[float] = []
    agent_preds_all: list[float] = []
    agent_targets_all: list[float] = []

    try:
        with torch.no_grad():
            for example in examples:
                row: dict[str, float | int | str] = {
                    "state_id": example.state_id,
                    "source": example.source,
                    "source_turn": example.source_turn,
                    "actor": example.state.actor,
                    "pick_phase": int(example.state.pick_phase),
                }

                if centralized:
                    enc = encoding.encode(example.state, 0)
                    team_pred = float(networks[0](enc).item())
                    agent_preds: list[float] = []
                else:
                    agent_preds = []
                    for agent_idx, net in enumerate(networks):
                        enc = encoding.encode(example.state, agent_idx)
                        agent_preds.append(float(net(enc).item()))
                    team_pred = float(sum(agent_preds))
                    agent_preds_all.extend(agent_preds)
                    agent_targets_all.extend(example.agent_targets)
                    for agent_idx in range(n_agents):
                        pred = agent_preds[agent_idx]
                        target = example.agent_targets[agent_idx]
                        row[f"agent_pred_{agent_idx}"] = pred
                        row[f"agent_target_{agent_idx}"] = target
                        row[f"agent_error_{agent_idx}"] = pred - target

                team_target = float(example.team_target)
                team_error = team_pred - team_target
                team_preds.append(team_pred)
                team_targets.append(team_target)
                row.update({
                    "team_pred": team_pred,
                    "team_target": team_target,
                    "team_error": team_error,
                    "team_abs_error": abs(team_error),
                    "team_sq_error": team_error ** 2,
                })
                rows.append(row)
    finally:
        for net, training in zip(networks, was_training):
            net.train(training)

    team_pred_arr = np.asarray(team_preds, dtype=np.float64)
    team_target_arr = np.asarray(team_targets, dtype=np.float64)
    summary: dict[str, float | int] = {"mc_value_n_states": len(examples)}
    summary.update(_summary_stats("mc_value_team", team_pred_arr, team_target_arr))

    if not centralized:
        agent_pred_arr = np.asarray(agent_preds_all, dtype=np.float64)
        agent_target_arr = np.asarray(agent_targets_all, dtype=np.float64)
        summary.update(_summary_stats("mc_value_agent", agent_pred_arr, agent_target_arr))

    return MCValidationResult(summary=summary, rows=rows)


def build_mc_validation_csv_fieldnames(n_agents: int) -> list[str]:
    fields = [
        "step", "wall_time", "state_id", "source", "source_turn", "actor", "pick_phase",
        "team_pred", "team_target", "team_error", "team_abs_error", "team_sq_error",
    ]
    for agent_idx in range(n_agents):
        fields.extend([
            f"agent_pred_{agent_idx}",
            f"agent_target_{agent_idx}",
            f"agent_error_{agent_idx}",
        ])
    return fields


def build_mc_validation_summary_fieldnames() -> list[str]:
    return [
        "mc_value_n_states",
        "mc_value_team_rmse",
        "mc_value_team_mae",
        "mc_value_team_bias",
        "mc_value_team_pred_mean",
        "mc_value_team_target_mean",
        "mc_value_agent_rmse",
        "mc_value_agent_mae",
        "mc_value_agent_bias",
        "mc_value_agent_pred_mean",
        "mc_value_agent_target_mean",
    ]
