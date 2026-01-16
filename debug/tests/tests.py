from debug.code.environment import Orchard
from debug.code.reward import Reward
from debug.code.helpers import set_all_seeds, teleport
import numpy as np
from pathlib import Path
data_dir = Path(__file__).parent.parent / "data"
NUM_AGENTS = 4
W, L = 9, 9

REWARD = -1

NUM_STEPS = 1000
SEEDS = [13, 42, 87, 123, 256, 511, 777, 1024, 2048, 4096]


class TestStateGeneration:
    def test_generation(self):
        for seed in SEEDS:
            set_all_seeds(seed)
            reward_module = Reward(REWARD, NUM_AGENTS)
            orchard = Orchard(W, L, NUM_AGENTS, reward_module, 10/(W * L))
            orchard.set_positions()
            np.savez(data_dir / f"init_state_reward_{reward_module.picker_r}", dict=orchard.get_state(), extra=orchard.agent_positions)
            true_state, true_positions = orchard.get_state(), orchard.agent_positions

            npz = np.load(data_dir / f"init_state_reward_{reward_module.picker_r}.npz", allow_pickle=True)
            d_loaded = npz["dict"].item()  # back to a Python dict
            extra_loaded = npz["extra"]  # regular NumPy array
            assert (true_positions == extra_loaded).all()
            assert (true_state["agents"] == d_loaded["agents"]).all()
            assert (true_state["apples"] == d_loaded["apples"]).all()

    def test_orchard_init_initial_state(self):
        reward_module = Reward(REWARD, NUM_AGENTS)
        npz = np.load(data_dir / f"init_state_reward_{reward_module.picker_r}.npz", allow_pickle=True)
        initial_state = npz["dict"].item()   # back to a Python dict
        agent_positions = npz["extra"]     # regular NumPy array

        initial_orchard = Orchard(W, L, NUM_AGENTS, reward_module, 1/(W * L), start_agents_map=initial_state["agents"],
                                  start_apples_map=initial_state["apples"], start_agent_positions=agent_positions)

        assert (initial_orchard.agent_positions == agent_positions).all()
        assert (initial_orchard.get_state()["agents"] == initial_state["agents"]).all()
        assert (initial_orchard.get_state()["apples"] == initial_state["apples"]).all()


class TestReward:
    def test_correct_reward_on_apple(self):
        reward_module = Reward(REWARD, 2)
        sample_state = {
            "agents": np.zeros((6, 6), dtype=int),
            "apples": np.zeros((6, 6), dtype=int)
        }
        # agent on apple
        sample_state["agents"][0][0] = 1

        sample_state["agents"][3][2] = 1

        sample_state["apples"][0][0] = 1

        res = reward_module.get_reward(sample_state, 0, np.array((0, 0)))

        assert res[0] == -1
        assert res[1] == 2

    def test_correct_reward_no_reward(self):
        reward_module = Reward(REWARD, 2)
        sample_state = {
            "agents": np.zeros((6, 6), dtype=int),
            "apples": np.zeros((6, 6), dtype=int)
        }
        # agent on apple
        sample_state["agents"][0][0] = 1

        sample_state["agents"][3][2] = 1

        res = reward_module.get_reward(sample_state, 0, np.array((0, 0)))

        assert res[0] == 0
        assert res[1] == 0


class TestEnvironment:
    def test_correct_sequence(self):
        reward_module = Reward(REWARD, 2)
        sample_state = {
            "agents": np.zeros((6, 6), dtype=int),
            "apples": np.zeros((6, 6), dtype=int)
        }
        sample_state["agents"][0][0] = 1

        sample_state["agents"][3][2] = 1
        orchard = Orchard(6, 6, 2, reward_module, 1, start_agents_map=sample_state["agents"],
                          start_apples_map=sample_state["apples"], start_agent_positions=np.array([[0, 0], [3, 2]]))

        res = orchard.process_action(0, np.array([5, 5]))
        assert (orchard.agent_positions[0] == np.array([5, 5])).all()
        assert res.reward_vector.sum() == 0
        assert (sample_state["apples"] != orchard.get_state()["apples"]).all()

    def test_random_teleportation(self):
        for _ in range(200):
            res = teleport(W)
            assert 0 <= res[0] <= W - 1
            assert 0 <= res[1] <= W - 1

    def test_random_apple_spawn(self):
        orchard = Orchard(6, 6, 2, None, 1/3)
        for _ in range(1000):
            prev = orchard.get_state()["apples"]
            orchard.spawn_apples()
            assert (orchard.get_state()["apples"] != prev).any()
