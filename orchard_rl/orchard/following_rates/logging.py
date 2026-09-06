"""Diagnostics helpers for following-rate experiments."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from orchard.following_rates.state import FollowingRateAgentState, InfluencerLike


def compute_following_rate_stats(
    agent_states: Iterable[FollowingRateAgentState],
    influencer: InfluencerLike | None = None,
) -> dict[str, float | None]:
    states = list(agent_states)
    empty = {
        "alpha_mean": None,
        "alpha_positive_frac": None,
        "following_weight_mean": None,
        "active_follow_edges_mean": None,
        "beta_mean": None,
        "influencer_weight_mean": None,
        "follower_to_influencer_weight_mean": None,
        "effective_follow_weight_mean": None,
    }
    if not states:
        return empty

    num_agents = int(states[0].agent_alphas.size)
    if any(int(state.agent_alphas.size) != num_agents for state in states):
        raise ValueError("All following-rate states must have the same number of agents.")

    alpha_rows = np.stack([np.asarray(state.agent_alphas, dtype=float) for state in states], axis=0)
    weight_rows = np.stack(
        [np.asarray(state.agent_observing_probabilities, dtype=float) for state in states],
        axis=0,
    )

    flat_alphas = []
    flat_weights = []
    per_observer_active = []
    flat_effective_weights = []

    for state_idx, state in enumerate(states):
        mask = np.ones(num_agents, dtype=bool)
        mask[int(state.agent_id)] = False
        flat_alphas.append(alpha_rows[state_idx, mask])
        flat_weights.append(weight_rows[state_idx, mask])
        per_observer_active.append(float(np.count_nonzero(weight_rows[state_idx, mask] > 1e-12)))
        for target_id in range(num_agents):
            if target_id == int(state.agent_id):
                continue
            flat_effective_weights.append(
                state.get_effective_observing_probability(target_id, influencer)
            )

    flat_alphas_arr = np.concatenate(flat_alphas) if flat_alphas else np.array([], dtype=float)
    flat_weights_arr = np.concatenate(flat_weights) if flat_weights else np.array([], dtype=float)
    flat_effective_weights_arr = np.asarray(flat_effective_weights, dtype=float)
    follower_to_influencer_weights = np.asarray(
        [state.influencer_observing_probability for state in states],
        dtype=float,
    )

    influencer_beta = np.array([], dtype=float)
    influencer_weights = np.array([], dtype=float)
    if influencer is not None:
        if hasattr(influencer, "beta"):
            influencer_beta = np.asarray(getattr(influencer, "beta"), dtype=float)
        influencer_weights = np.asarray(influencer.outgoing_weights, dtype=float)

    return {
        "alpha_mean": float(np.mean(flat_alphas_arr)) if flat_alphas_arr.size else None,
        "alpha_positive_frac": (
            float(np.mean(flat_alphas_arr > 0.0))
            if flat_alphas_arr.size
            else None
        ),
        "following_weight_mean": (
            float(np.mean(flat_weights_arr))
            if flat_weights_arr.size
            else None
        ),
        "active_follow_edges_mean": (
            float(np.mean(per_observer_active))
            if per_observer_active
            else None
        ),
        "beta_mean": float(np.mean(influencer_beta)) if influencer_beta.size else None,
        "influencer_weight_mean": (
            float(np.mean(influencer_weights))
            if influencer_weights.size
            else None
        ),
        "follower_to_influencer_weight_mean": (
            float(np.mean(follower_to_influencer_weights))
            if follower_to_influencer_weights.size
            else None
        ),
        "effective_follow_weight_mean": (
            float(np.mean(flat_effective_weights_arr))
            if flat_effective_weights_arr.size
            else None
        ),
    }


def build_following_rate_snapshot_row(
    step: int,
    wall_time: float,
    agent_state: FollowingRateAgentState,
) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "step": int(step),
        "wall_time": float(wall_time),
    }
    alphas = np.asarray(agent_state.agent_alphas, dtype=float)
    rates = np.asarray(agent_state.following_rates, dtype=float)
    weights = np.asarray(agent_state.agent_observing_probabilities, dtype=float)
    for target_id in range(alphas.size):
        row[f"alpha_to_{target_id}"] = float(alphas[target_id])
    for target_id in range(rates.size):
        row[f"lambda_to_{target_id}"] = float(rates[target_id])
    for target_id in range(weights.size):
        row[f"weight_to_{target_id}"] = float(weights[target_id])
    row["lambda_to_influencer"] = float(agent_state.following_rate_to_influencer)
    row["weight_to_influencer"] = float(agent_state.influencer_observing_probability)
    row["influencer_value"] = float(agent_state.influencer_value)
    return row


__all__ = [
    "compute_following_rate_stats",
    "build_following_rate_snapshot_row",
]
