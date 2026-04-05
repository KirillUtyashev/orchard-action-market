"""Diagnostics helpers for influencer experiments."""

from __future__ import annotations

import numpy as np

from orchard.influencer.state import ExternalInfluencer


def build_influencer_snapshot_row(
    step: int,
    wall_time: float,
    influencer: ExternalInfluencer,
) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "step": int(step),
        "wall_time": float(wall_time),
    }
    beta = np.asarray(influencer.beta, dtype=float)
    rates = np.asarray(influencer.outgoing_rates, dtype=float)
    weights = np.asarray(influencer.outgoing_weights, dtype=float)
    for target_id in range(influencer.num_agents):
        row[f"beta_to_actor_{target_id}"] = float(beta[target_id])
    for target_id in range(influencer.num_agents):
        row[f"lambda_to_actor_{target_id}"] = float(rates[target_id])
    for target_id in range(influencer.num_agents):
        row[f"weight_to_actor_{target_id}"] = float(weights[target_id])
    return row


__all__ = ["build_influencer_snapshot_row"]
