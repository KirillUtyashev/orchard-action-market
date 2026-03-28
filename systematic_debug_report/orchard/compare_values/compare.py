"""Generate states and compare N models' value predictions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from orchard.compare_values.loader import LoadedRun
from orchard.datatypes import EnvConfig, State
from orchard.encoding.base import BaseEncoder
from orchard.enums import Heuristic, TDTarget
from orchard.env import create_env
from orchard.eval import collect_after_state_test_states, collect_on_policy_test_states
from orchard.model import ValueNetwork
from orchard.seed import set_all_seeds


@dataclass
class StateComparison:
    """Value predictions from N runs for a single state."""
    state_index: int
    state: State

    # label -> team value
    team_values: dict[str, float]

    # label -> {agent_idx: value}
    agent_values: dict[str, dict[int, float]]


def validate_env_compatibility(runs: list[LoadedRun]) -> None:
    """Check that all runs have compatible env configs.

    Raises ValueError listing all mismatched fields.
    """
    if len(runs) < 2:
        return

    ref = runs[0].cfg.env
    fields_to_check = [
        "height", "width", "n_agents", "n_tasks", "gamma", "pick_mode",
        "max_tasks", "env_type", "n_task_types",
    ]

    for run in runs[1:]:
        mismatches: list[str] = []
        other = run.cfg.env
        for field in fields_to_check:
            va = getattr(ref, field)
            vb = getattr(other, field)
            if va != vb:
                mismatches.append(f"  {field}: {va} vs {vb}")

        if ref.stochastic is not None and other.stochastic is not None:
            for field in ["spawn_prob", "despawn_mode", "despawn_prob"]:
                va = getattr(ref.stochastic, field)
                vb = getattr(other.stochastic, field)
                if va != vb:
                    mismatches.append(f"  stochastic.{field}: {va} vs {vb}")
        elif (ref.stochastic is None) != (other.stochastic is None):
            mismatches.append(f"  stochastic: {ref.stochastic} vs {other.stochastic}")

        if mismatches:
            raise ValueError(
                f"Env config mismatch between '{runs[0].label}' and '{run.label}':\n"
                + "\n".join(mismatches)
            )


def validate_td_target_compatibility(runs: list[LoadedRun]) -> None:
    """Require all runs to use the same td_target."""
    if len(runs) < 2:
        return
    ref_tt = runs[0].cfg.train.td_target
    for run in runs[1:]:
        if run.cfg.train.td_target != ref_tt:
            raise ValueError(
                f"td_target mismatch: '{runs[0].label}' uses {ref_tt.name}, "
                f"'{run.label}' uses {run.cfg.train.td_target.name}. "
                f"Cannot compare models trained on different state types."
            )


def generate_states(
    env_cfg: EnvConfig,
    td_target: TDTarget,
    n_states: int,
    seed: int,
    heuristic: Heuristic | None = None,
) -> list[State]:
    """Generate comparison states using heuristic policy."""
    if heuristic is None:
        heuristic = (Heuristic.NEAREST_CORRECT_TASK
                     if env_cfg.n_task_types > 1
                     else Heuristic.NEAREST_TASK)

    set_all_seeds(seed)
    env = create_env(env_cfg)

    if td_target == TDTarget.AFTER_STATE:
        return collect_after_state_test_states(env, n_states, heuristic=heuristic)
    else:
        return collect_on_policy_test_states(env, n_states, heuristic=heuristic)


def compute_team_value(
    state: State,
    networks: list[ValueNetwork],
    encoder: BaseEncoder,
    is_centralized: bool,
    n_agents: int,
) -> tuple[float, dict[int, float]]:
    """Compute team value and per-agent/per-network values for a state."""
    agent_values: dict[int, float] = {}

    with torch.no_grad():
        for i, net in enumerate(networks):
            v = net(encoder.encode(state, i)).item()
            agent_values[i] = v

    team_value = sum(agent_values.values())
    return team_value, agent_values


def run_comparison(
    runs: list[LoadedRun],
    states: list[State],
) -> list[StateComparison]:
    """Compare N models on a list of states."""
    results: list[StateComparison] = []
    n_agents = runs[0].cfg.env.n_agents

    for idx, state in enumerate(states):
        team_values: dict[str, float] = {}
        agent_values: dict[str, dict[int, float]] = {}

        for run in runs:
            tv, av = compute_team_value(
                state, run.networks, run.encoder, run.is_centralized, n_agents
            )
            team_values[run.label] = tv
            agent_values[run.label] = av

        results.append(StateComparison(
            state_index=idx,
            state=state,
            team_values=team_values,
            agent_values=agent_values,
        ))

    return results
