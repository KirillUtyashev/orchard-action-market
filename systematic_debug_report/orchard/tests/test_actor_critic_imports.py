"""Import-structure tests for the canonical package split."""

import importlib

import pytest

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
from orchard.following_rates import (
    CommunicationBudgetAllocator,
    FollowingRateAgentState,
    build_following_rate_snapshot_row,
    compute_following_rate_stats,
    initial_following_rate_to_influencer,
    initial_following_rate_vector,
)
from orchard.influencer import (
    ExternalInfluencer,
    build_influencer_snapshot_row,
    initial_influencer_outgoing_rates,
)


class TestCanonicalImports:
    def test_actor_critic_package_re_exports_expected_symbols(self):
        from orchard import actor_critic

        assert actor_critic.PolicyNetwork is PolicyNetwork
        assert actor_critic.full_action_head_dim is full_action_head_dim
        assert actor_critic.action_to_policy_index is action_to_policy_index
        assert actor_critic.policy_index_to_action is policy_index_to_action
        assert actor_critic.build_phase1_legal_mask is build_phase1_legal_mask
        assert actor_critic.build_phase2_legal_mask is build_phase2_legal_mask
        assert actor_critic.serialize_state is serialize_state
        assert actor_critic.sample_phase1_policy_eval_states is sample_phase1_policy_eval_states
        assert actor_critic.generate_phase2_policy_eval_states is generate_phase2_policy_eval_states
        assert actor_critic.build_phase1_policy_prob_row is build_phase1_policy_prob_row
        assert actor_critic.build_phase2_policy_prob_row is build_phase2_policy_prob_row

    def test_following_rates_package_re_exports_expected_symbols(self):
        from orchard import following_rates

        assert following_rates.CommunicationBudgetAllocator is CommunicationBudgetAllocator
        assert following_rates.FollowingRateAgentState is FollowingRateAgentState
        assert following_rates.initial_following_rate_vector is initial_following_rate_vector
        assert (
            following_rates.initial_following_rate_to_influencer
            is initial_following_rate_to_influencer
        )
        assert following_rates.compute_following_rate_stats is compute_following_rate_stats
        assert following_rates.build_following_rate_snapshot_row is build_following_rate_snapshot_row

    def test_influencer_package_re_exports_expected_symbols(self):
        from orchard import influencer

        assert influencer.ExternalInfluencer is ExternalInfluencer
        assert influencer.initial_influencer_outgoing_rates is initial_influencer_outgoing_rates
        assert influencer.build_influencer_snapshot_row is build_influencer_snapshot_row

    @pytest.mark.parametrize(
        "module_name",
        [
            "orchard.actor_policy",
            "orchard.actor_diagnostics",
            "orchard.actor_critic.following_rates",
            "orchard.actor_critic.following_rate_logging",
            "orchard.actor_critic.rate_allocators",
        ],
    )
    def test_legacy_modules_are_removed(self, module_name: str):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)
