from typing import Optional

from agents.agent import AgentInfo
from agents.communicating_agent import CommAgent
from models.actor_network import ActorNetwork
from models.value_function import VNetwork
from policies.nearest import nearest_policy
from helpers.controllers import AgentControllerCentralized, \
    AgentControllerDecentralized, ViewController, ViewControllerOrchardSelfless
from orchard.environment import Orchard, OrchardBasic, \
    OrchardEuclideanNegativeRewards, OrchardEuclideanRewards, \
    OrchardIDs, \
    OrchardMineAllRewards, OrchardMineNoReward, OrchardSelfless
from agents.simple_agent import SimpleAgent
from orchard.algorithms import despawn_apple_selfless_orchard, spawn_apple, \
    despawn_apple, \
    spawn_apple_selfless_orchard, spawn_dirt
from policies.random_policy import random_policy
import numpy as np
from configs.config import EnvironmentConfig
from configs.config import TrainingConfig
from configs.config import ExperimentConfig
import pytest
from value_function_learning.train_value_function import \
    CentralizedValueFunction, DecentralizedValueFunction


@pytest.mark.parametrize("env_cls", [OrchardBasic, OrchardSelfless, OrchardIDs, OrchardMineAllRewards, OrchardMineNoReward])
class TestOrchard:
    def setup_orchard(self, length, width, env_cls, policy=random_policy):
        self.agents_list = []
        for i in range(2):
            self.agents_list.append(SimpleAgent(AgentInfo(
                policy=policy,
                agent_id=i + 1
            )))
        self.orchard = env_cls(length, width, 2, self.agents_list)

    # def test_spawn(self, env_cls):
    #     self.setup_orchard(10, 1, env_cls)
    #     self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([0, 8])])
    #     self.orchard.spawn_algorithm(self.orchard, 1, 2)
    #     self.orchard.spawn_algorithm(self.orchard, 1, 2)
    #     assert (self.orchard.apples <= 2).all(), "Not all cells contain apples"

    #     if env_cls is OrchardSelfless:
    #         max_tries = 100
    #         for _ in range(max_tries):
    #             self.orchard.despawn_algorithm(self.orchard, 1)
    #             self.orchard.spawn_algorithm(self.orchard, 1, 2)
    #             if np.any(self.orchard.apples == 2):
    #                 break
    #         else:
    #             raise AssertionError("No apple with id=2 after spawning attempts")

    def test_spawn_dirt(self, env_cls):
        self.setup_orchard(10, 1, OrchardBasic)
        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([0, 8])])
        print(f'{type(self.orchard)}')
        total_dirt = self.orchard.spawn_dirt_algorithm(self.orchard, 1)
        assert(total_dirt > 0)


    def test_spawn_apple_reduced(self, env_cls):
        self.setup_orchard(10, 1, env_cls)
        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([0, 8])])
        apple_total_clean = self.orchard.spawn_algorithm(self.orchard, 1, 0)

        self.setup_orchard(10, 1, env_cls)
        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([0, 8])])
        apple_total_dirt = self.orchard.spawn_algorithm(self.orchard, 1, 10)

        print(f'Total apple spawned without dirt: {apple_total_clean}; Total apple spawned with dirt {apple_total_dirt}.')
        assert(apple_total_clean > apple_total_dirt)

    # def test_dirt(self, env_cls):
    #     self.setup_orchard(10, 1, env_cls)
    #     self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([0, 8])])
    #     self.orchard.spawn_dirt_algorithm(self.orchard, 1)
    #     self.orchard.spawn_dirt_algorithm(self.orchard, 1)
    #     assert (self.orchard.dirt <= 2).all(), "Not all cells contain dirt"

    #     if env_cls is OrchardSelfless:
    #         max_tries = 100
    #         for _ in range(max_tries):
    #             self.orchard.spawn_dirt_algorithm(self.orchard, 1)
    #             if np.any(self.orchard.dirt == 2):
    #                 break
    #         else:
    #             raise AssertionError("No apple with id=2 after spawning attempts")


if __name__ == '__main__':
    import pytest

    pytest.main(['test_dirt.py'])