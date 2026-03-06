"""Generate states and compare two models' value predictions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from orchard.compare_values.loader import LoadedRun
from orchard.datatypes import EnvConfig, State
from orchard.encoding.base import BaseEncoder
from orchard.env import create_env
from orchard.enums import TDTarget
from orchard.eval import collect_after_state_test_states, collect_on_policy_test_states
from orchard.model import ValueNetwork
from orchard.seed import set_all_seeds


@dataclass
class StateComparison:
    state_index: int
    state: State

    # Model A
    team_value_a: float
    agent_values_a: dict[int, float]

    # Model B
    team_value_b: float
    agent_values_b: dict[int, float]

    # Derived
    team_diff: float        # A - B
    team_abs_diff: float    # |A - B|


def validate_env_compatibility(env_a: EnvConfig, env_b: EnvConfig) -> None:
    """Check that two env configs are compatible for comparison.

    Raises ValueError listing all mismatched fields.
    """
    fields_to_check = [
        "height", "width", "n_agents", "n_apples", "gamma",
        "r_picker", "force_pick", "max_apples", "env_type",
    ]
    mismatches: list[str] = []
    for field in fields_to_check:
        va = getattr(env_a, field)
        vb = getattr(env_b, field)
        if va != vb:
            mismatches.append(f"  {field}: {va} vs {vb}")

    # Check stochastic config if both stochastic
    if env_a.stochastic is not None and env_b.stochastic is not None:
        for field in ["spawn_prob", "despawn_mode", "despawn_prob", "apple_lifetime"]:
            va = getattr(env_a.stochastic, field)
            vb = getattr(env_b.stochastic, field)
            if va != vb:
                mismatches.append(f"  stochastic.{field}: {va} vs {vb}")
    elif (env_a.stochastic is None) != (env_b.stochastic is None):
        mismatches.append(f"  stochastic: {env_a.stochastic} vs {env_b.stochastic}")

    if mismatches:
        raise ValueError(
            "Env configs are not compatible for comparison:\n" + "\n".join(mismatches)
        )


def validate_td_target_compatibility(run_a: LoadedRun, run_b: LoadedRun) -> None:
    """Require both runs to use the same td_target."""
    tt_a = run_a.cfg.train.td_target
    tt_b = run_b.cfg.train.td_target
    if tt_a != tt_b:
        raise ValueError(
            f"td_target mismatch: run A uses {tt_a.name}, run B uses {tt_b.name}. "
            f"Cannot compare models trained on different state types."
        )


def generate_states(
    env_cfg: EnvConfig,
    td_target: TDTarget,
    n_states: int,
    seed: int,
) -> list[State]:
    """Generate comparison states using nearest-apple policy."""
    set_all_seeds(seed)
    env = create_env(env_cfg)

    if td_target == TDTarget.AFTER_STATE:
        return collect_after_state_test_states(env, n_states)
    else:
        return collect_on_policy_test_states(env, n_states)


def compute_team_value(
    state: State,
    networks: list[ValueNetwork],
    encoder: BaseEncoder,
    is_centralized: bool,
    n_agents: int,
) -> tuple[float, dict[int, float]]:
    """Compute team value and per-agent/per-network values for a state.

    Returns:
        (team_value, agent_values_dict)

    For decentralized: agent_values_dict has one entry per agent (network).
        team_value = sum of all agent values.
    For centralized: agent_values_dict has one entry (key 0).
        team_value = that single value (it already predicts team value).
    """
    agent_values: dict[int, float] = {}

    with torch.no_grad():
        for i, net in enumerate(networks):
            v = net(encoder.encode(state, i)).item()
            agent_values[i] = v

    team_value = sum(agent_values.values())
    return team_value, agent_values


def run_comparison(
    run_a: LoadedRun,
    run_b: LoadedRun,
    states: list[State],
) -> list[StateComparison]:
    """Compare two models on a list of states."""
    results: list[StateComparison] = []
    n_agents = run_a.cfg.env.n_agents

    for idx, state in enumerate(states):
        tv_a, av_a = compute_team_value(
            state, run_a.networks, run_a.encoder, run_a.is_centralized, n_agents
        )
        tv_b, av_b = compute_team_value(
            state, run_b.networks, run_b.encoder, run_b.is_centralized, n_agents
        )

        diff = tv_a - tv_b
        results.append(StateComparison(
            state_index=idx,
            state=state,
            team_value_a=tv_a,
            agent_values_a=av_a,
            team_value_b=tv_b,
            agent_values_b=av_b,
            team_diff=diff,
            team_abs_diff=abs(diff),
        ))

    return results
