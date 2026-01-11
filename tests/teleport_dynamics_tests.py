from tadd_helpers.setting_seed import set_all_seeds
from teleport_dynamic.experiment_utils import \
    generate_decentralized_value_test_set
from teleport_dynamic.orchard_generator import init_fixed_apples
from teleport_dynamic.rewards_decentralized import (
    get_reward_minus1_2, make_picker_penalty_reward)

import pytest


WIDTH = 9
HEIGHT = 9
NUM_AGENTS = 4
NUM_APPLES = 40
SEED = 42
NUM_TEST_SAMPLES = 1000
DISCOUNT = 0.99


SEEDS = [13, 42, 87, 123, 256, 511, 777, 1024, 2048, 4096]


class TestState:
    def test_init(self):
        for seed in SEEDS:
            set_all_seeds(seed)
            state = init_fixed_apples(WIDTH, HEIGHT, NUM_AGENTS, NUM_APPLES)
            assert (state.agents.sum() == 4)
            assert (state.apples.sum() == 40)

            for agent in state._agents:
                assert state.agents[state._agents[agent][0], state._agents[agent][1]] >= 1

    def test_generate_decentralized_value_test_set_minus_one(self):
        reward_func = get_reward_minus1_2
        for seed in SEEDS:
            state = init_fixed_apples(WIDTH, HEIGHT, NUM_AGENTS, NUM_APPLES)
            test_sets = generate_decentralized_value_test_set(
                HEIGHT, WIDTH, NUM_AGENTS, NUM_APPLES, NUM_TEST_SAMPLES,
                state.apples, reward_func, DISCOUNT, seed + 1000
            )

            assert (len(test_sets) == 4)
            assert(sum(len(v) for v in test_sets.values())) == NUM_TEST_SAMPLES * 4

            # check self_picker
            for case in test_sets["self_picker"]:
                assert (case.acting_agent_idx == case.self_agent_idx)
                assert (case.state.agents[case.state._agents[case.acting_agent_idx]] >= 1)
                assert (case.state.apples[case.state._agents[case.acting_agent_idx]] >= 1)
                assert(case.true_reward == -1.0)

            # check other_picker
            for case in test_sets["other_picker"]:
                assert (case.acting_agent_idx != case.self_agent_idx)
                assert (case.state.agents[case.state._agents[case.acting_agent_idx]] >= 1)
                assert (case.state.agents[case.state._agents[case.self_agent_idx]] >= 1)
                assert (case.state.apples[case.state._agents[case.acting_agent_idx]] >= 1)
                assert(case.true_reward != 0.0 and case.true_reward != -1.0)

            # check_zero_actor_miss
            for case in test_sets["zero_actor_miss"]:
                assert (case.state.agents[case.state._agents[case.acting_agent_idx]] >= 1)
                assert (case.state.agents[case.state._agents[case.self_agent_idx]] >= 1)
                assert (case.state.apples[case.state._agents[case.acting_agent_idx]] == 0)
                assert(case.true_reward == 0.0)

            # check_zero_self_on_apple
            for case in test_sets["zero_self_on_apple"]:
                assert (case.acting_agent_idx != case.self_agent_idx)
                assert (case.state.agents[case.state._agents[case.acting_agent_idx]] >= 1)
                assert (case.state.agents[case.state._agents[case.self_agent_idx]] >= 1)
                assert (case.state.apples[case.state._agents[case.acting_agent_idx]] == 0)
                assert (case.state.apples[case.state._agents[case.self_agent_idx]] >= 1)
                assert (case.true_reward == 0.0)

    def _helper(self, reward_func, reward_picker, reward_other):
        for seed in SEEDS:
            state = init_fixed_apples(WIDTH, HEIGHT, NUM_AGENTS, NUM_APPLES)
            test_sets = generate_decentralized_value_test_set(
                HEIGHT, WIDTH, NUM_AGENTS, NUM_APPLES, NUM_TEST_SAMPLES,
                state.apples, reward_func, DISCOUNT, seed + 1000
            )

            for set_ in test_sets:
                for case in test_sets[set_]:
                    rewards = reward_func(case.state, case.acting_agent_idx)
                    if case.state.apples[case.state._agents[case.acting_agent_idx]] == 0:
                        picker, other = 0.0, 0.0
                    else:
                        picker, other = reward_picker, reward_other
                    assert (rewards[case.acting_agent_idx] == picker)
                    for idx in range(NUM_AGENTS):
                        if idx != case.acting_agent_idx:
                            assert (rewards[idx] == pytest.approx(other, rel=1e-8, abs=0.0))

    def test_get_reward_minus1_2(self):
        reward_func = get_reward_minus1_2
        self._helper(reward_func, -1.0, 2/3)

    def test_make_picker_penalty_reward(self):
        rewards = [-10.0, -5.0, -1.0, 5.0, 10.0]
        for reward in rewards:
            reward_func = make_picker_penalty_reward(reward)
            self._helper(reward_func, reward, (1 - reward) / (NUM_AGENTS - 1))
