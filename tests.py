from agents.communicating_agent import CommAgent
from policies.nearest import nearest_policy
from value_function_learning.controllers import ViewController
from orchard.environment import Orchard
from agents.simple_agent import SimpleAgent
from orchard.algorithms import spawn_apple, despawn_apple
from policies.random_policy import random_policy
import numpy as np
from plots import graph_plots

class TestOrchard:
    def setup_orchard(self, width):
        self.agents_list = [SimpleAgent(policy=random_policy) for _ in range(2)]
        self.orchard = Orchard(10, width, 2, self.agents_list, spawn_algo=spawn_apple, despawn_algo=despawn_apple)

    def test_spawn(self):
        self.setup_orchard(1)
        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([0, 8])])
        self.orchard.spawn_algorithm(self.orchard, 1)
        self.orchard.spawn_algorithm(self.orchard, 1)
        assert (self.orchard.apples == 2).all(), "Not all cells contain apples"

    def test_despawn(self):
        self.setup_orchard(1)
        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([0, 8])])
        self.orchard.spawn_algorithm(self.orchard, 1)
        self.orchard.spawn_algorithm(self.orchard, 1)
        self.orchard.despawn_algorithm(self.orchard, 1)
        assert (self.orchard.apples == 1).all(), "Not all cells contain apples"

    def test_get_state(self):
        self.setup_orchard(1)
        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([0, 8])])
        state = self.orchard.get_state()
        assert ((len(state["agents"][0]) == self.orchard.width * self.orchard.length) &
                (len(state["apples"][0]) == self.orchard.width * self.orchard.length))
        assert (self.orchard.apples == 0).all()
        assert self.orchard.agents[0][3] == 1
        assert self.orchard.agents[0][8] == 1

    def test_process_action_no_apple(self):
        self.setup_orchard(1)
        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([0, 8])])
        action = 0
        reward, new_pos = self.orchard.main_step(self.agents_list[0].position, action)
        assert int(new_pos[1]) == 2
        assert self.orchard.total_apples == 0

    def test_process_action_apple(self):
        self.setup_orchard(1)
        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([0, 8])])
        self.orchard.apples[0][2] += 1
        action = 0
        reward, new_pos = self.orchard.main_step(self.agents_list[0].position, action)
        assert int(new_pos[1]) == 2
        assert self.orchard.total_apples == 0
        assert reward == 1

    def test_actions_2d(self):
        self.setup_orchard(2)
        self.orchard.initialize(self.agents_list, [np.array([0, 3]), np.array([1, 8])])
        action = 2
        reward, new_pos = self.orchard.main_step(self.agents_list[1].position, action)
        assert (int(new_pos[0]) == 0) & (int(new_pos[1]) == 8)
        assert self.orchard.total_apples == 0
        assert reward == 0

    def test_nearest_2d(self):
        self.agents_list = [SimpleAgent(policy=nearest_policy) for _ in range(2)]
        self.orchard = Orchard(10, 2, 2, self.agents_list, spawn_algo=spawn_apple, despawn_algo=despawn_apple)
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


class TestViewController:
    def test_perfect_info_simple(self):
        agents_list = [SimpleAgent(policy=random_policy) for _ in range(2)]
        orchard = Orchard(10, 1, 2, agents_list, spawn_algo=spawn_apple, despawn_algo=despawn_apple)
        orchard.initialize(agents_list, [np.array([0, 3]), np.array([0, 8])])
        view_controller = ViewController()
        result = view_controller.process_state(orchard.get_state(), agents_list[0].position)
        expected = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]).reshape(-1, 1)
        assert np.array_equal(result, expected)

    def test_perfect_info_comm(self):
        agents_list = [CommAgent(policy=random_policy) for _ in range(2)]
        orchard = Orchard(10, 1, 2, agents_list, spawn_algo=spawn_apple, despawn_algo=despawn_apple)
        orchard.initialize(agents_list, [np.array([0, 3]), np.array([0, 8])])
        view_controller = ViewController()
        result = view_controller.process_state(orchard.get_state(), agents_list[0].position)
        expected = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]).reshape(-1, 1)
        assert np.array_equal(result, expected)

    def test_local_info_comm(self):
        agents_list = [CommAgent(policy=random_policy) for _ in range(2)]
        orchard = Orchard(10, 1, 2, agents_list, spawn_algo=spawn_apple, despawn_algo=despawn_apple)
        orchard.initialize(agents_list, [np.array([0, 3]), np.array([0, 8])])
        view_controller = ViewController(vision=5)
        result = view_controller.process_state(orchard.get_state(), agents_list[0].position)
        expected = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3]).reshape(-1, 1)
        assert np.array_equal(result, expected)

    def test_perfrect_simple_2d(self):
        agents_list = [SimpleAgent(policy=random_policy) for _ in range(2)]
        orchard = Orchard(5, 2, 2, agents_list, spawn_algo=spawn_apple, despawn_algo=despawn_apple)
        orchard.initialize(agents_list, [np.array([0, 0]), np.array([1, 1])])
        view_controller = ViewController()
        result = view_controller.process_state(orchard.get_state(), agents_list[1].position)
        expected = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]).reshape(-1, 1)
        assert np.array_equal(result, expected)


class TestGraphPlots:
    def test_no_erros(self):
        plot1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        plot2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        plot3 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        eval_x = [0, 10000, 100000, 500000, 1000000]
        eval_y = [0, 100, 200, 300, 310]

        graph_plots("Test", plot1, plot2, plot3, eval_x, eval_y)


if __name__ == '__main__':
    import pytest

    pytest.main(['tests.py'])
