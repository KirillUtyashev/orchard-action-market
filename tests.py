import itertools
import random
from typing import Optional

import torch

from agents.agent import AgentInfo
from agents.communicating_agent import CommAgent
from helpers.helpers import create_env
from models.actor_network import ActorNetwork
from models.value_function import VNetwork
from policies.nearest import nearest_policy
from helpers.controllers import AgentControllerCentralized, \
    AgentControllerDecentralized, ViewController, ViewControllerOrchardSelfless
from orchard.environment import Orchard, OrchardBasic, \
    OrchardBasicNewDynamic, OrchardEuclideanNegativeRewards, \
    OrchardEuclideanNegativeRewardsNewDynamic, OrchardEuclideanRewards, \
    OrchardEuclideanRewardsNewDynamic, OrchardIDs, \
    OrchardMineAllRewards, OrchardMineNoReward, OrchardSelfless
from agents.simple_agent import SimpleAgent
from orchard.algorithms import despawn_apple_selfless_orchard, spawn_apple, \
    despawn_apple, \
    spawn_apple_selfless_orchard
from policies.random_policy import random_policy
import numpy as np
from configs.config import EnvironmentConfig
from configs.config import TrainingConfig
from configs.config import ExperimentConfig
import pytest

from reward_learning.reward_learning import RewardLearningDecentralized
from value_function_learning.train_value_function import \
    CentralizedValueFunction, DecentralizedValueFunction


@pytest.mark.parametrize("env_cls", [OrchardBasic, OrchardEuclideanRewards, OrchardEuclideanNegativeRewards])
class TestOrchard:
    def setup_orchard(self, length, width, env_cls, policy=random_policy):
        self.agents_list = []
        for i in range(2):
            self.agents_list.append(SimpleAgent(AgentInfo(
                policy=policy,
                agent_id=i + 1
            )))
        self.orchard = env_cls(length, width, 2, self.agents_list)

    def test_spawn(self, env_cls):
        self.setup_orchard(10, 1, env_cls)
        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([0, 8])])
        self.orchard.spawn_algorithm(self.orchard, 1)
        self.orchard.spawn_algorithm(self.orchard, 1)
        assert (self.orchard.apples <= 2).all(), "Not all cells contain apples"

    def test_despawn(self, env_cls):
        self.setup_orchard(10, 1, env_cls)
        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([0, 8])])
        self.orchard.spawn_algorithm(self.orchard, 1)
        self.orchard.spawn_algorithm(self.orchard, 1)
        self.orchard.despawn_algorithm(self.orchard, 1)
        assert (self.orchard.apples <= 1).all(), "Not all cells contain apples"

    def test_get_state(self, env_cls):
        self.setup_orchard(10, 1, env_cls)
        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([0, 8])])
        state = self.orchard.get_state()
        assert ((len(state["agents"][0]) == self.orchard.width * self.orchard.length) &
                (len(state["apples"][0]) == self.orchard.width * self.orchard.length))
        assert (self.orchard.apples == 0).all()
        assert self.orchard.agents[0][3] == 1
        assert self.orchard.agents[0][8] == 1

    def test_process_action_no_apple(self, env_cls):
        self.setup_orchard(10, 1, env_cls)
        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([0, 8])])
        action = 0
        action_result = self.orchard.process_action(0, self.agents_list[0].position, action)
        assert int(self.agents_list[0].position[1]) == 2
        assert self.orchard.total_apples == 0

    def test_process_action_apple(self, env_cls):
        self.setup_orchard(10, 1, env_cls)
        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([0, 8])])
        self.orchard.apples[0][2] += 1
        action = 0
        action_result = self.orchard.process_action(0, self.agents_list[0].position, action)
        assert int(self.agents_list[0].position[1]) == 2
        assert (action_result.picked is True)
        if not isinstance(self.orchard, OrchardBasicNewDynamic):
            assert self.orchard.total_apples == 0
        else:
            assert self.orchard.total_apples == 1
            self.orchard.remove_apple(self.agents_list[0].position)
            assert self.orchard.total_apples == 0
        if env_cls is OrchardBasic:
            assert action_result.reward_vector[0] == 1

    def test_actions_2d(self, env_cls):
        self.setup_orchard(10, 2, env_cls)
        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([1, 8])])
        action = 2
        action_result = self.orchard.process_action(0, self.agents_list[1].position, action)
        assert (int(self.agents_list[1].position[0]) == 1) & (int(self.agents_list[1].position[1]) == 8)
        assert self.orchard.total_apples == 0
        assert np.sum(action_result.reward_vector) == 0
        assert action_result.picked is False

    def test_nearest_2d(self, env_cls):
        self.setup_orchard(10, 2, env_cls, nearest_policy)

        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([1, 8])])
        self.orchard.apples[0][3] += 1
        assert self.agents_list[0].policy(self.orchard.get_state(), np.array([0, 3])) == 2
        self.orchard.apples[0][3] -= 1

        self.orchard.apples[1][5] += 1
        assert self.agents_list[1].policy(self.orchard.get_state(), np.array([1, 8])) == 0
        self.orchard.apples[1][5] -= 1

        self.orchard.apples[0][8] += 1
        assert self.agents_list[1].policy(self.orchard.get_state(), np.array([1, 8])) == 3
        self.orchard.apples[0][8] -= 1

        self.orchard.apples[1][3] += 1
        assert self.agents_list[1].policy(self.orchard.get_state(), np.array([0, 3])) == 4
        self.orchard.apples[1][3] -= 1

        self.orchard.apples[0][7] += 1
        assert self.agents_list[1].policy(self.orchard.get_state(), np.array([1, 8])) == 0
        self.orchard.apples[0][7] -= 1


@pytest.mark.parametrize("env_cls", [OrchardBasic])
class TestViewController:
    def setup_tests(self, length, width, agent_cls, positions, env_cls):
        agents_list = [agent_cls(AgentInfo(
            policy=random_policy,
            agent_id=i
        )) for i in range(2)]
        orchard = env_cls(length, width, 2, agents_list, spawn_algo=spawn_apple, despawn_algo=despawn_apple)
        orchard.initialize(agents_list, positions)
        return agents_list, orchard

    def test_perfect_info_simple(self, env_cls):
        agents_list, orchard = self.setup_tests(10, 1, SimpleAgent, positions=[np.array([0, 3]), np.array([0, 8])], env_cls=env_cls)
        view_controller = ViewController(0)
        result = view_controller.process_state(orchard.get_state(), agents_list[0].position)
        expected = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]).reshape(-1, 1)
        assert np.array_equal(result, expected)

    def test_perfect_info_comm(self, env_cls):
        agents_list, orchard = self.setup_tests(10, 1, CommAgent, positions=[np.array([0, 3]), np.array([0, 8])], env_cls=env_cls)
        view_controller = ViewController(0)
        result = view_controller.process_state(orchard.get_state(), agents_list[0].position)
        expected = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]).reshape(-1, 1)
        assert np.array_equal(result, expected)

    def test_local_info_comm(self, env_cls):
        agents_list, orchard = self.setup_tests(10, 1, CommAgent, positions=[np.array([0, 3]), np.array([0, 8])], env_cls=env_cls)
        view_controller = ViewController(vision=5)
        result = view_controller.process_state(orchard.get_state(), agents_list[0].position)
        expected = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3]).reshape(-1, 1)
        assert np.array_equal(result, expected)

    def test_perfrect_simple_2d(self, env_cls):
        agents_list, orchard = self.setup_tests(5, 2, CommAgent, positions=[np.array([0, 0]), np.array([1, 1])], env_cls=env_cls)
        view_controller = ViewController(0)
        result = view_controller.process_state(orchard.get_state(), agents_list[1].position)
        expected = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]).reshape(-1, 1)
        assert np.array_equal(result, expected)

    def test_new_input(self, env_cls):
        agents_list, orchard = self.setup_tests(5, 2, CommAgent, positions=[np.array([0, 0]), np.array([1, 1])], env_cls=env_cls)
        view_controller = ViewController(0, True)
        result = view_controller.process_state(orchard.get_state(), agents_list[1].position)
        expected = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).reshape(-1, 1)
        assert np.array_equal(result, expected)


@pytest.mark.parametrize("env_cls", [OrchardBasic, OrchardEuclideanRewards, OrchardEuclideanNegativeRewards])
@pytest.mark.parametrize("new_dynamic", [True, False])
class TestCentralizedLearning:
    algo: Optional[CentralizedValueFunction] = None

    @pytest.fixture(autouse=True)
    def _build_algo(self, request, env_cls, new_dynamic):
        # build env_config with the current env_cls
        env_config = EnvironmentConfig(
            length=6,
            width=6,
            apple_mean_lifetime=5,
            s_target=0.16,
            env_cls=env_cls.__name__,   # or pass the class itself if your code supports it
        )

        train_config = TrainingConfig(
            num_agents=2,
            hidden_dimensions=16,
            num_layers=4,
            batch_size=2,
            alpha=0.000275,
            critic_vision=0,
            timesteps=10000,
            eval_interval=1,
            test=True,
            new_dynamic=new_dynamic,
        )

        experiment_config = ExperimentConfig(
            env_config=env_config,
            train_config=train_config,
        )

        # attach to the test instance so tests can use self.algo
        request.instance.algo = CentralizedValueFunction(experiment_config)

    def test_build_experiment(self, env_cls):
        self.algo.build_experiment()

        # Test controllers
        assert self.algo.critic_view_controller is not None
        assert type(self.algo.agent_controller) is AgentControllerCentralized
        assert self.algo.agent_controller.agents_list is self.algo.agents_list
        assert self.algo.agent_controller.critic_view_controller is self.algo.critic_view_controller

        # Test agents
        assert len(self.algo.agents_list) == 2
        assert self.algo.agents_list[0].policy_value == self.algo.agents_list[1].policy_value
        assert not hasattr(self.algo.agents_list[0], "policy_network")
        assert self.algo.agents_list[0].policy_value.get_input_dim() == 72
        assert self.algo.agents_list[0].position is not None

        # Test environment
        assert self.algo.env is not None
        assert self.algo.env.spawn_algorithm is spawn_apple
        assert self.algo.env.despawn_algorithm is despawn_apple

    def test_create_env(self, env_cls):
        self.algo._init_agents_for_training(SimpleAgent, self.algo._init_critic_networks(VNetwork), self.algo._init_actor_networks(ActorNetwork), None)

        assert self.algo.agents_list[0].policy_value == self.algo.agents_list[1].policy_value
        self.algo.env = create_env(self.algo.env_config, 2, [np.array([0, 0]), np.array([1, 1])], None, self.algo.agents_list, self.algo.env_cls)

        assert self.algo.env.spawn_algorithm is spawn_apple
        assert self.algo.env.despawn_algorithm is despawn_apple

        assert self.algo.agents_list[0].position[0] == 0
        assert self.algo.agents_list[1].position[0] == 1
        assert (self.algo.env.apples == 0).all()

    def test_collect_observation(self, env_cls):
        self.algo.build_experiment()
        self.algo.collect_observation(0)
        assert len(self.algo.agents_list[0].policy_value.batch_states) == len(self.algo.agents_list[1].policy_value.batch_states) == 2

    def test_init_agents_for_eval(self, env_cls):
        self.algo.build_experiment()
        agents, controller = self.algo.init_agents_for_eval()
        assert len(agents) == 2
        assert controller.agents_list is agents
        assert controller.critic_view_controller is self.algo.critic_view_controller
        assert agents[0] is not self.algo.agents_list[0]
        assert agents[1] is not self.algo.agents_list[1]

    def test_controller(self):
        self.algo.build_experiment()
        self.algo.env.apples[self.algo.agents_list[0].position[0]][self.algo.agents_list[0].position[1]] = 1

        apples_before = self.algo.env.apples.copy()
        ir, agents, apples, new_pos = self.algo.env.calculate_ir(self.algo.agents_list[0].position, [0, 0])
        if isinstance(self.algo.env, OrchardBasicNewDynamic) or isinstance(self.algo.env, OrchardEuclideanRewardsNewDynamic) or isinstance(self.algo.env, OrchardEuclideanNegativeRewardsNewDynamic):
            assert new_pos[0] == self.algo.agents_list[0].position[0]
            assert new_pos[1] == self.algo.agents_list[0].position[1]
            assert ir == 1
            assert np.sum(apples) == 1
            assert np.equal(apples_before, apples).all()
        else:
            assert new_pos[0] == self.algo.agents_list[0].position[0]
            assert new_pos[1] == self.algo.agents_list[0].position[1]
            assert ir == 1
            assert np.sum(apples) == 0

    def test_env_step(self, env_cls):
        self.algo.build_experiment()
        old_state = self.algo.env.get_state()
        old_positions = [self.algo.agents_list[0].position, self.algo.agents_list[1].position]
        env_step_result = self.algo.env_step(0)
        assert (old_state["agents"] == env_step_result.old_state["agents"]).all()
        assert env_step_result.new_state is not None
        assert (env_step_result.picked is False)

        assert old_positions[0][0] == env_step_result.old_positions[0][0]
        assert old_positions[0][1] == env_step_result.old_positions[0][1]
        assert old_positions[1][0] == env_step_result.old_positions[1][0]
        assert old_positions[1][1] == env_step_result.old_positions[1][1]

        assert type(env_step_result.action) is int

        non_acting_agent = 0 if env_step_result.acting_agent_id == 1 else 1
        assert self.algo.agents_list[non_acting_agent].position[0] == old_positions[non_acting_agent][0]

    def test_env_step_pick_apple(self, env_cls):
        np.random.seed(1234)
        torch.manual_seed(1234)
        random.seed(1234)

        self.algo.build_experiment()
        self.algo.env = create_env(self.algo.env_config, self.algo.train_config.num_agents, [np.array([1, 1]), np.array([4, 4])], None, self.algo.agents_list, self.algo.env_cls, debug=self.algo.debug)

        old_state = self.algo.env.get_state()
        old_positions = [self.algo.agents_list[0].position, self.algo.agents_list[1].position]
        self.algo.env.apples[self.algo.agents_list[0].position[0]][self.algo.agents_list[0].position[1]] = 1
        if isinstance(self.algo.env, OrchardBasic):
            print(1)
        env_step_result = self.algo.env_step(0, agent_id=0)
        assert (old_state["agents"] == env_step_result.old_state["agents"]).all()
        assert env_step_result.new_state is not None

        assert (env_step_result.action == 2)
        assert (env_step_result.picked is True)

        assert old_positions[0][0] == env_step_result.old_positions[0][0]
        assert old_positions[0][1] == env_step_result.old_positions[0][1]
        assert old_positions[1][0] == env_step_result.old_positions[1][0]
        assert old_positions[1][1] == env_step_result.old_positions[1][1]

        assert self.algo.agents_list[1].position[0] == old_positions[1][0]
        assert np.sum(env_step_result.reward_vector) == 1
        if isinstance(self.algo.env, OrchardBasicNewDynamic) or isinstance(self.algo.env, OrchardEuclideanRewardsNewDynamic) or isinstance(self.algo.env, OrchardEuclideanNegativeRewardsNewDynamic):
            assert self.algo.env.apples[self.algo.agents_list[0].position[0]][self.algo.agents_list[0].position[1]] == 1
        else:
            assert self.algo.env.apples[self.algo.agents_list[0].position[0]][self.algo.agents_list[0].position[1]] == 0

        if isinstance(self.algo.env, OrchardEuclideanNegativeRewards):
            assert env_step_result.reward_vector[0] == -1
            assert env_step_result.reward_vector[1] == 2
        elif isinstance(self.algo.env, OrchardEuclideanRewards):
            assert env_step_result.reward_vector[0] == 0
            assert env_step_result.reward_vector[1] == 1
        else:
            assert env_step_result.reward_vector[0] == 1
            assert env_step_result.reward_vector[1] == 0

    def test_training_step(self):
        self.algo.build_experiment()
        self.algo.training_step(0)
        assert self.algo.agents_list[0].policy_value.batch_states == []
        assert self.algo.agents_list[1].policy_value.batch_states == []

        assert self.algo.agents_list[0].policy_value.batch_rewards == []

    def test_env_step_2(self, env_cls):
        self.algo.build_experiment()
        self.algo.env.apples[self.algo.agents_list[0].position[0]][self.algo.agents_list[0].position[1]] = 1
        action = 2  # Stay
        res = self.algo.env.process_action(0, self.algo.agents_list[0].position, action)
        assert np.sum(res.reward_vector) == 1
        if env_cls is OrchardEuclideanRewards:
            assert res.reward_vector[0] == 0
            assert res.reward_vector[1] == 1
        elif env_cls is OrchardEuclideanNegativeRewards:
            assert res.reward_vector[0] == -1
            assert res.reward_vector[1] == 2
        elif env_cls is OrchardBasic:
            assert res.reward_vector[0] == 1
            assert res.reward_vector[1] == 0

    def test_env_step_3(self, env_cls):
        self.algo.build_experiment()
        self.algo.env.apples = np.zeros((self.algo.env.width, self.algo.env.length), dtype=int)
        prev = self.algo.agents_list[0].position
        self.algo.env.apples[self.algo.agents_list[0].position[0]][self.algo.agents_list[0].position[1]] = 1
        action = 2  # Stay
        res = self.algo.env.process_action(0, self.algo.agents_list[0].position, action)
        assert np.sum(res.reward_vector) == 1

        if isinstance(self.algo.env, OrchardEuclideanNegativeRewards):
            assert res.reward_vector[0] == -1
            assert res.reward_vector[1] == 2
        elif isinstance(self.algo.env, OrchardEuclideanRewards):
            assert res.reward_vector[0] == 0
            assert res.reward_vector[1] == 1
        else:
            assert res.reward_vector[0] == 1
            assert res.reward_vector[1] == 0
        assert self.algo.agents_list[0].position[0] == prev[0]
        assert self.algo.agents_list[0].position[1] == prev[1]

    def test_remove_apple(self, env_cls):
        self.algo.build_experiment()
        self.algo.env.apples[self.algo.agents_list[0].position[0]][self.algo.agents_list[0].position[1]] = 1
        self.algo.env._consume_apple(self.algo.agents_list[0].position)
        self.algo.env.remove_apple(self.algo.agents_list[0].position)
        assert np.sum(self.algo.env.apples) == 0


@pytest.mark.parametrize("env_cls", [OrchardBasic, OrchardEuclideanRewards, OrchardEuclideanNegativeRewards])
@pytest.mark.parametrize("new_dynamic", [True, False])
@pytest.mark.parametrize("new_input", [True, False])
class TestDecentralizedLearning:
    algo: Optional[DecentralizedValueFunction] = None

    @pytest.fixture(autouse=True)
    def _build_algo(self, request, env_cls, new_dynamic, new_input):
        # build env_config with the current env_cls
        env_config = EnvironmentConfig(
            length=6,
            width=6,
            apple_mean_lifetime=5,
            s_target=0.16,
            env_cls=env_cls.__name__,   # or pass the class itself if your code supports it
        )

        train_config = TrainingConfig(
            num_agents=2,
            hidden_dimensions=16,
            num_layers=4,
            batch_size=2,
            alpha=0.000275,
            critic_vision=0,
            timesteps=10000,
            eval_interval=1,
            test=True,
            new_dynamic=new_dynamic,
            new_input=new_input
        )

        experiment_config = ExperimentConfig(
            env_config=env_config,
            train_config=train_config,
        )

        # attach to the test instance so tests can use self.algo
        request.instance.algo = DecentralizedValueFunction(experiment_config)

    def test_build_experiment(self, env_cls):
        self.algo.build_experiment()

        # Test controllers
        assert self.algo.critic_view_controller is not None
        assert type(self.algo.agent_controller) is AgentControllerDecentralized
        assert self.algo.agent_controller.agents_list is self.algo.agents_list
        assert self.algo.agent_controller.critic_view_controller is self.algo.critic_view_controller

        # Test agents
        assert len(self.algo.agents_list) == 2
        assert self.algo.agents_list[0].policy_value != self.algo.agents_list[1].policy_value
        assert not hasattr(self.algo.agents_list[0], "policy_network")
        if not self.algo.train_config.new_input:
            assert self.algo.agents_list[0].policy_value.get_input_dim() == 74
        else:
            assert self.algo.agents_list[0].policy_value.get_input_dim() == 108
        assert self.algo.agents_list[0].position is not None

        # Test environment
        assert self.algo.env is not None
        assert self.algo.env.spawn_algorithm is spawn_apple or self.algo.env.spawn_algorithm is spawn_apple_selfless_orchard
        assert self.algo.env.despawn_algorithm is despawn_apple or self.algo.env.despawn_algorithm is despawn_apple_selfless_orchard

    def test_create_env(self, env_cls):
        self.algo._init_agents_for_training(CommAgent, self.algo._init_critic_networks(VNetwork), self.algo._init_actor_networks(ActorNetwork), None)

        assert self.algo.agents_list[0].policy_value != self.algo.agents_list[1].policy_value

        self.algo.env = create_env(self.algo.env_config, 2,[np.array([0, 0]), np.array([1, 1])], None, self.algo.agents_list, self.algo.env_cls)

        assert self.algo.env.spawn_algorithm is spawn_apple
        assert self.algo.env.despawn_algorithm is despawn_apple

        assert self.algo.agents_list[0].position[0] == 0
        assert self.algo.agents_list[1].position[0] == 1
        assert (self.algo.env.apples == 0).all()

    def test_collect_observation(self, env_cls):
        self.algo.build_experiment()
        self.algo.collect_observation(0)
        assert len(self.algo.agents_list[0].policy_value.batch_states) == len(self.algo.agents_list[1].policy_value.batch_states) == 2

    def test_init_agents_for_eval(self, env_cls):
        self.algo.build_experiment()
        agents, controller = self.algo.init_agents_for_eval()
        assert len(agents) == 2
        assert controller.agents_list is agents
        assert controller.critic_view_controller is self.algo.critic_view_controller
        assert agents[0] is not self.algo.agents_list[0]
        assert agents[1] is not self.algo.agents_list[1]

    def test_env_step(self, env_cls):
        self.algo.build_experiment()
        old_state = self.algo.env.get_state()
        old_positions = [self.algo.agents_list[0].position, self.algo.agents_list[1].position]
        env_step_result = self.algo.env_step(0)
        assert (old_state["agents"] == env_step_result.old_state["agents"]).all()
        assert env_step_result.new_state is not None
        assert (env_step_result.picked is False)

        assert old_positions[0][0] == env_step_result.old_positions[0][0]
        assert old_positions[0][1] == env_step_result.old_positions[0][1]
        assert old_positions[1][0] == env_step_result.old_positions[1][0]
        assert old_positions[1][1] == env_step_result.old_positions[1][1]

        assert type(env_step_result.action) is int

        non_acting_agent = 0 if env_step_result.acting_agent_id == 1 else 1
        assert self.algo.agents_list[non_acting_agent].position[0] == old_positions[non_acting_agent][0]

    def test_training_step(self):
        self.algo.build_experiment()
        self.algo.training_step(0)
        assert self.algo.agents_list[0].policy_value.batch_states == []
        assert self.algo.agents_list[1].policy_value.batch_states == []

        assert self.algo.agents_list[0].policy_value.batch_rewards == []

    def test_remove_apple(self, env_cls):
        self.algo.build_experiment()
        self.algo.env.apples[self.algo.agents_list[0].position[0]][self.algo.agents_list[0].position[1]] = 1
        self.algo.env._consume_apple(self.algo.agents_list[0].position)
        self.algo.env.remove_apple(self.algo.agents_list[0].position)
        assert np.sum(self.algo.env.apples) == 0

    def test_env_step_pick_apple(self, env_cls):
        np.random.seed(1234)
        torch.manual_seed(1234)
        random.seed(1234)

        self.algo.build_experiment()
        self.algo.env = create_env(self.algo.env_config, self.algo.train_config.num_agents, [np.array([1, 1]), np.array([4, 4])], None, self.algo.agents_list, self.algo.env_cls, debug=self.algo.debug)

        old_state = self.algo.env.get_state()
        old_positions = [self.algo.agents_list[0].position, self.algo.agents_list[1].position]
        self.algo.env.apples[self.algo.agents_list[0].position[0]][self.algo.agents_list[0].position[1]] = 1
        if isinstance(self.algo.env, OrchardBasic):
            print(1)
        env_step_result = self.algo.env_step(0, agent_id=0)
        assert (old_state["agents"] == env_step_result.old_state["agents"]).all()
        assert env_step_result.new_state is not None

        assert (env_step_result.action == 2)
        assert (env_step_result.picked is True)

        assert old_positions[0][0] == env_step_result.old_positions[0][0]
        assert old_positions[0][1] == env_step_result.old_positions[0][1]
        assert old_positions[1][0] == env_step_result.old_positions[1][0]
        assert old_positions[1][1] == env_step_result.old_positions[1][1]

        assert self.algo.agents_list[1].position[0] == old_positions[1][0]
        assert np.sum(env_step_result.reward_vector) == 1
        if isinstance(self.algo.env, OrchardBasicNewDynamic) or isinstance(self.algo.env, OrchardEuclideanRewardsNewDynamic) or isinstance(self.algo.env, OrchardEuclideanNegativeRewardsNewDynamic):
            assert self.algo.env.apples[self.algo.agents_list[0].position[0]][self.algo.agents_list[0].position[1]] == 1
        else:
            assert self.algo.env.apples[self.algo.agents_list[0].position[0]][self.algo.agents_list[0].position[1]] == 0

        if isinstance(self.algo.env, OrchardEuclideanNegativeRewards):
            assert env_step_result.reward_vector[0] == -1
            assert env_step_result.reward_vector[1] == 2
        elif isinstance(self.algo.env, OrchardEuclideanRewards):
            assert env_step_result.reward_vector[0] == 0
            assert env_step_result.reward_vector[1] == 1
        else:
            assert env_step_result.reward_vector[0] == 1
            assert env_step_result.reward_vector[1] == 0


class TestDecentralizedRewardLearning:
    algo: Optional[DecentralizedValueFunction] = None

    def _build_algo(self, env_cls, new_dynamic):
        # build env_config with the current env_cls
        env_config = EnvironmentConfig(
            length=6,
            width=6,
            apple_mean_lifetime=50,
            s_target=0.16,
            env_cls=env_cls.__name__,   # or pass the class itself if your code supports it
        )

        train_config = TrainingConfig(
            num_agents=2,
            hidden_dimensions=64,
            num_layers=4,
            batch_size=2,
            alpha=0.0005,
            critic_vision=0,
            timesteps=10000,
            eval_interval=0,
            new_dynamic=new_dynamic,
        )

        experiment_config = ExperimentConfig(
            env_config=env_config,
            train_config=train_config,
            debug=True
        )

        # attach to the test instance so tests can use self.algo
        return RewardLearningDecentralized(experiment_config)

    def test_identical_orchard(self):

        algos_dict = {}

        for orchard in [OrchardEuclideanRewards, OrchardBasic]:
            for new_dynamic in [True, False]:
                np.random.seed(1234)
                torch.manual_seed(1234)
                random.seed(1234)
                algo = self._build_algo(orchard, new_dynamic)
                algo.train()
                algos_dict[(orchard, new_dynamic)] = algo

        for (orchard1, algo1), (orchard2, algo2) in itertools.combinations(algos_dict.items(), 2):
            assert np.array_equal(algo1.env.state_history, algo2.env.state_history)

            if isinstance(algo1.env, OrchardBasicNewDynamic) and isinstance(algo2.env, OrchardBasicNewDynamic):
                assert algo1.env.total_picked == algo2.env.total_picked


if __name__ == '__main__':
    import pytest

    pytest.main(['tests.py'])
