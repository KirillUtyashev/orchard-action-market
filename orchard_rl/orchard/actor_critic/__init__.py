"""Canonical home for orchard actor-critic scaffolding."""

from orchard.actor_critic.action_space import (
    action_to_policy_index,
    build_phase1_legal_mask,
    build_phase2_legal_mask,
    full_action_head_dim,
    policy_index_to_action,
)
from orchard.actor_critic.policy_eval_states import (
    generate_phase2_policy_eval_states,
    sample_phase1_policy_eval_states,
    serialize_state,
)
from orchard.actor_critic.policy_logging import (
    build_phase1_policy_prob_row,
    build_phase2_policy_prob_row,
)
from orchard.actor_critic.policy_network import PolicyNetwork

__all__ = [
    "PolicyNetwork",
    "full_action_head_dim",
    "action_to_policy_index",
    "policy_index_to_action",
    "build_phase1_legal_mask",
    "build_phase2_legal_mask",
    "serialize_state",
    "sample_phase1_policy_eval_states",
    "generate_phase2_policy_eval_states",
    "build_phase1_policy_prob_row",
    "build_phase2_policy_prob_row",
]
