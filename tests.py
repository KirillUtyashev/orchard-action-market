from typing import Optional

from agents.agent import AgentInfo
from agents.communicating_agent import CommAgent
from models.actor_network import ActorNetwork
from models.value_function import VNetwork
from policies.nearest import nearest_policy
from helpers.controllers import AgentControllerCentralized, \
    AgentControllerDecentralized, ViewController, ViewControllerOrchardSelfless
from orchard.environment import Orchard, OrchardBasic, OrchardSelfless
from agents.simple_agent import SimpleAgent
from orchard.algorithms import despawn_apple_selfless_orchard, spawn_apple, \
    despawn_apple, \
    spawn_apple_selfless_orchard
from policies.random_policy import random_policy
import numpy as np
from plots import graph_plots
from configs.config import EnvironmentConfig
from configs.config import TrainingConfig
from configs.config import ExperimentConfig
import pytest
from value_function_learning.train_value_function import \
    CentralizedValueFunction, DecentralizedValueFunction


@pytest.mark.parametrize("env_cls", [OrchardBasic, OrchardSelfless])
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

        if env_cls is OrchardSelfless:
            max_tries = 100
            for _ in range(max_tries):
                self.orchard.despawn_algorithm(self.orchard, 1)
                self.orchard.spawn_algorithm(self.orchard, 1)
                if np.any(self.orchard.apples == 2):
                    break
            else:
                raise AssertionError("No apple with id=2 after spawning attempts")

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
        assert int(action_result.new_position[1]) == 2
        assert self.orchard.total_apples == 0

    def test_process_action_apple(self, env_cls):
        self.setup_orchard(10, 1, env_cls)
        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([0, 8])])
        self.orchard.apples[0][2] += 2
        action = 0
        action_result = self.orchard.process_action(0, self.agents_list[0].position, action)
        assert int(action_result.new_position[1]) == 2
        assert self.orchard.total_apples == 0
        if env_cls is OrchardBasic:
            assert action_result.picker_reward == 1
        else:
            assert action_result.owner_reward == 1
            assert action_result.owner_id == 2

    def test_actions_2d(self, env_cls):
        self.setup_orchard(10, 2, env_cls)
        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([1, 8])])
        action = 2
        action_result = self.orchard.process_action(0, self.agents_list[1].position, action)
        assert (int(action_result.new_position[0]) == 1) & (int(action_result.new_position[1]) == 8)
        assert self.orchard.total_apples == 0
        assert action_result.picker_reward == 0

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

    def test_process_action(self, env_cls):
        self.setup_orchard(6, 6, env_cls)
        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([1, 1])])
        action = 2

        # Apple in the same field
        self.orchard.apples[0][3] = 2
        result = self.orchard.process_action(0, self.agents_list[0].position, action)

        assert result.new_position[0] == 0 and result.new_position[1] == 3

        if env_cls is OrchardBasic:
            assert result.picker_reward == 1
            assert result.owner_reward is None
            assert result.owner_id is None
            assert int(self.orchard.apples[0][3]) == 1
        else:
            assert result.picker_reward == 0
            assert result.owner_reward == 1
            assert result.owner_id == 2
            assert int(self.orchard.apples[0][3]) == 0

        # Agent picks up its own apple
        self.orchard.apples[0][3] = 1
        result = self.orchard.process_action(0, self.agents_list[0].position, action)

        if env_cls is OrchardBasic:
            assert result.picker_reward == 1
            assert result.owner_reward is None
            assert result.owner_id is None
        else:
            assert result.picker_reward == 1
            assert result.owner_reward == 0
            assert result.owner_id == 1
        assert int(self.orchard.apples[0][3]) == 0


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


@pytest.mark.parametrize("env_cls", [OrchardBasic])
class TestViewControllerOrchardSelfless:
    def setup_tests(self, length, width, agent_cls, positions, env_cls):
        agents_list = [agent_cls(AgentInfo(
            policy=random_policy,
            agent_id=i
        )) for i in range(2)]
        orchard = env_cls(length, width, 2, agents_list, spawn_algo=spawn_apple, despawn_algo=despawn_apple)
        orchard.initialize(agents_list, positions)
        return agents_list, orchard

    def test_no_apples(self, env_cls):
        agents_list, orchard = self.setup_tests(10, 1, CommAgent, positions=[np.array([0, 3]), np.array([0, 8])], env_cls=env_cls)
        view_controller = ViewControllerOrchardSelfless(0)
        result = view_controller.process_state(orchard.get_state(), agents_list[0].position, 1)
        expected = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]).reshape(-1, 1)
        assert np.array_equal(result, expected)

    def test_one_agent_apple(self, env_cls):
        agents_list, orchard = self.setup_tests(10, 1, CommAgent, positions=[np.array([0, 3]), np.array([0, 8])], env_cls=env_cls)
        view_controller = ViewControllerOrchardSelfless(0)
        orchard.apples[0][1] = 1
        result_1 = view_controller.process_state(orchard.get_state(), agents_list[0].position, 1)
        result_2 = view_controller.process_state(orchard.get_state(), agents_list[1].position, 2)
        expected_1 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]).reshape(-1, 1)
        expected_2 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8]).reshape(-1, 1)
        assert np.array_equal(result_1, expected_1)
        assert np.array_equal(result_2, expected_2)

    def test_two_agent_apples(self, env_cls):
        agents_list, orchard = self.setup_tests(10, 1, CommAgent, positions=[np.array([0, 3]), np.array([0, 8])], env_cls=env_cls)
        view_controller = ViewControllerOrchardSelfless(0)
        orchard.apples[0][1] = 1
        orchard.apples[0][2] = 2
        result_1 = view_controller.process_state(orchard.get_state(), agents_list[0].position, 1)
        result_2 = view_controller.process_state(orchard.get_state(), agents_list[1].position, 2)
        expected_1 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]).reshape(-1, 1)
        expected_2 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 8]).reshape(-1, 1)
        assert np.array_equal(result_1, expected_1)
        assert np.array_equal(result_2, expected_2)

    def test_simple_agent(self, env_cls):
        agents_list, orchard = self.setup_tests(10, 1, SimpleAgent, positions=[np.array([0, 3]), np.array([0, 8])], env_cls=env_cls)
        view_controller = ViewControllerOrchardSelfless(0)
        orchard.apples[0][1] = 1
        orchard.apples[0][2] = 2
        result_1 = view_controller.process_state(orchard.get_state(), agents_list[0].position, None)
        result_2 = view_controller.process_state(orchard.get_state(), agents_list[1].position, None)
        expected_1 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3]).reshape(-1, 1)
        expected_2 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 8]).reshape(-1, 1)
        assert np.array_equal(result_1, expected_1)
        assert np.array_equal(result_2, expected_2)


class TestCentralizedController:
    pass


class TestDecentralizedController:
    pass


@pytest.mark.parametrize("env_cls", [OrchardBasic, OrchardSelfless])
class TestCentralizedLearning:
    algo: Optional[CentralizedValueFunction] = None

    @pytest.fixture(autouse=True)
    def _build_algo(self, request, env_cls):
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
        assert self.algo.env.spawn_algorithm is spawn_apple or self.algo.env.spawn_algorithm is spawn_apple_selfless_orchard
        assert self.algo.env.despawn_algorithm is despawn_apple or self.algo.env.despawn_algorithm is despawn_apple_selfless_orchard

    def test_create_env(self, env_cls):
        self.algo._init_agents_for_training(SimpleAgent, self.algo._init_critic_networks(VNetwork), self.algo._init_actor_networks(ActorNetwork))

        assert self.algo.agents_list[0].policy_value == self.algo.agents_list[1].policy_value

        self.algo.env = self.algo.create_env([np.array([0, 0]), np.array([1, 1])], None, self.algo.agents_list, self.algo.env_cls)

        if type(self.algo.env) is OrchardBasic:
            assert self.algo.env.spawn_algorithm is spawn_apple
            assert self.algo.env.despawn_algorithm is despawn_apple
        else:
            assert self.algo.env.spawn_algorithm is spawn_apple_selfless_orchard
            assert self.algo.env.despawn_algorithm is despawn_apple_selfless_orchard

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
        if env_cls is OrchardBasic:
            assert (env_step_result.apple_owner_id is None) and (env_step_result.apple_owner_reward is None)
        else:
            assert env_step_result.apple_owner_reward is not None

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


@pytest.mark.parametrize("env_cls", [OrchardBasic, OrchardSelfless])
class TestDecentralizedLearning:
    algo: Optional[DecentralizedValueFunction] = None

    @pytest.fixture(autouse=True)
    def _build_algo(self, request, env_cls):
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
        assert self.algo.agents_list[0].policy_value.get_input_dim() == 74
        assert self.algo.agents_list[0].position is not None

        # Test environment
        assert self.algo.env is not None
        assert self.algo.env.spawn_algorithm is spawn_apple or self.algo.env.spawn_algorithm is spawn_apple_selfless_orchard
        assert self.algo.env.despawn_algorithm is despawn_apple or self.algo.env.despawn_algorithm is despawn_apple_selfless_orchard

    def test_create_env(self, env_cls):
        self.algo._init_agents_for_training(CommAgent, self.algo._init_critic_networks(VNetwork), self.algo._init_actor_networks(ActorNetwork))

        assert self.algo.agents_list[0].policy_value != self.algo.agents_list[1].policy_value

        self.algo.env = self.algo.create_env([np.array([0, 0]), np.array([1, 1])], None, self.algo.agents_list, self.algo.env_cls)

        if type(self.algo.env) is OrchardBasic:
            assert self.algo.env.spawn_algorithm is spawn_apple
            assert self.algo.env.despawn_algorithm is despawn_apple
        else:
            assert self.algo.env.spawn_algorithm is spawn_apple_selfless_orchard
            assert self.algo.env.despawn_algorithm is despawn_apple_selfless_orchard

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
        if env_cls is OrchardBasic:
            assert (env_step_result.apple_owner_id is None) and (env_step_result.apple_owner_reward is None)
        else:
            assert env_step_result.apple_owner_reward is not None

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


# class TestVLearning:
#     def test_setup(self):
#         env_config = EnvironmentConfig(
#             length=6,
#             width=6,
#             apple_mean_lifetime=5,
#             s_target=0.16,
#         )
#         train_config = TrainingConfig(
#             num_agents=2,
#             hidden_dimensions=16,
#             num_layers=4,
#             batch_size=256,
#             alpha=0.0005,
#             critic_vision=0,
#             timesteps=10000,
#             eval_interval=1,
#             test=True
#         )
#
#         experiment_config = ExperimentConfig(
#             env_config=env_config,
#             train_config=train_config
#         )
#
#         test_algo = DecentralizedValueFunction(experiment_config)
#         test_algo.run()
#
#         assert (test_algo.count_random_actions / (10000 * 2)) <= 0.11
#         assert 0.9 <= (test_algo.count_random_actions / (10000 * 2))

# class TestGraphPlots:
#     def test_no_erros(self):
#         plot1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#         plot2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#         plot3 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#
#         eval_x = [0, 10000, 100000, 500000, 1000000]
#         eval_y = [0, 100, 200, 300, 310]
#
#         graph_plots("Test", plot1, plot2, plot3, eval_x, eval_y)


if __name__ == '__main__':
    import pytest

    pytest.main(['tests.py'])
