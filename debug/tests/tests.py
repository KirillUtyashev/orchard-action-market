import random

from matplotlib import pyplot as plt

from debug.code.environment import Orchard
from debug.code.reward import Reward
from debug.code.helpers import random_policy, set_all_seeds, teleport
import numpy as np
from pathlib import Path
data_dir = Path(__file__).parent.parent / "data"
NUM_AGENTS = 4
W, L = 9, 9

REWARD = -1

NUM_STEPS = 1000
SEEDS = [13, 42, 87, 123, 256, 511, 777, 1024, 2048, 4096]


class TimeToAppleTracker:
    def __init__(self, grid_shape):
        self.first_nonempty_tick = -np.ones(grid_shape, dtype=np.int64)
        self.prev_nonempty = np.zeros(grid_shape, dtype=bool)
        self.ticks = 0
        self.ages = []  # collected times-to-apple in ticks

    def observe_grid(self, apples_grid):
        nonempty = apples_grid > 0
        became_nonempty = nonempty & (~self.prev_nonempty)
        self.first_nonempty_tick[became_nonempty] = self.ticks
        # when a cell becomes empty again, clear timestamp
        became_empty = (~nonempty) & self.prev_nonempty
        self.first_nonempty_tick[became_empty] = -1
        self.prev_nonempty = nonempty

    def on_pick(self, x, y):
        t0 = self.first_nonempty_tick[x, y]
        if t0 >= 0:
            self.ages.append(self.ticks - t0)


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

        res = orchard.process_action(0, np.array([5, 5]), mode=0)
        assert (orchard.agent_positions[0] == np.array([5, 5])).all()
        assert res.reward_vector.sum() == 0

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

    def test_spawn_despawn(self):
        q_agent = 0.5
        apple_mean = 5
        reward_module = Reward(REWARD, NUM_AGENTS)
        p_apple = (q_agent * NUM_AGENTS) / (W ** 2)
        d_apple = 1 / apple_mean
        self.env = Orchard(
            W,
            L,
            NUM_AGENTS,
            reward_module,
            p_apple,
            d_apple
        )
        steps = 10_000
        num_apples = np.zeros(steps + 1, dtype=int)

        average = q_agent * apple_mean * NUM_AGENTS

        # initialize
        self.env.spawn_apples()
        num_apples[0] = self.env.get_sum_apples()

        for t in range(1, steps + 1):
            self.env.despawn_apples()
            self.env.spawn_apples()
            num_apples[t] = self.env.get_sum_apples()

        mean = num_apples.mean()
        print(num_apples.std())

        assert abs(average - mean) < 1

        plt.plot(np.arange(steps + 1), num_apples)
        plt.xlabel("Time steps")
        plt.ylabel("Number of apples")
        plt.title("Apple count over time")
        plt.show()

    def test_how_much_time_to_apple(self):
        q_agent = 0.5
        apple_mean = 8
        reward_module = Reward(REWARD, NUM_AGENTS)
        p_apple = (q_agent * NUM_AGENTS) / (W ** 2)
        d_apple = 1 / apple_mean
        self.env = Orchard(
            W,
            L,
            NUM_AGENTS,
            reward_module,
            p_apple,
            d_apple
        )
        self.env.set_positions()
        self.env.spawn_apples()

        tracker = TimeToAppleTracker(self.env.apples.shape)

        # NEW: counters
        total_picked = 0
        total_spawned = int(self.env.apples.sum())  # count initial apples as "spawned"

        seconds = 10_000
        tracker.observe_grid(self.env.apples)
        num_apples = np.zeros(seconds + 1, dtype=int)
        num_apples[0] = self.env.get_sum_apples()

        curr_state = None
        actor_idx = None
        for sec in range(seconds):
            for step in range(-1, NUM_AGENTS):
                if step == -1:
                    if curr_state is None:
                        curr_state = self.env.get_state()
                        actor_idx = random.randint(0, NUM_AGENTS - 1)
                        curr_state["actor_id"] = actor_idx
                        curr_state["mode"] = 0
                    continue

                # Mode 0: move
                self.env.process_action(
                    actor_idx,
                    random_policy(curr_state["agent_positions"][actor_idx]),
                    mode=0,
                )

                semi_state = dict(self.env.get_state())
                semi_state["actor_id"] = actor_idx
                semi_state["mode"] = 1

                sx, sy = semi_state["agent_positions"][actor_idx]
                cx, cy = curr_state["agent_positions"][actor_idx]
                assert abs(sx - cx) + abs(sy - cy) <= 1

                # Mode 1: pick
                ax, ay = semi_state["agent_positions"][actor_idx]

                # NEW: robust pick detection via apple grid (not reward sign/magnitude)
                had_apple_before = bool(self.env.apples[ax, ay])

                res = self.env.process_action(actor_idx, None, mode=1)

                has_apple_after = bool(self.env.apples[ax, ay])
                picked = had_apple_before and (not has_apple_after)

                if picked:
                    total_picked += 1
                    tracker.on_pick(ax, ay)

                tracker.observe_grid(self.env.apples)

                if step == NUM_AGENTS - 1:
                    # End-of-second: despawn then spawn
                    self.env.despawn_apples()

                    # NEW: count how many *new* apples appear at spawn time
                    apples_before_spawn = self.env.apples.copy()
                    self.env.spawn_apples()
                    new_spawns = np.logical_and(self.env.apples == 1, apples_before_spawn == 0).sum()
                    total_spawned += int(new_spawns)

                    tracker.observe_grid(self.env.apples)

                final_state = self.env.get_state()
                actor_idx = random.randint(0, NUM_AGENTS - 1)
                final_state["actor_id"] = actor_idx
                final_state["mode"] = 0

                curr_state = final_state
                tracker.ticks += 1
            num_apples[sec] = self.env.get_sum_apples()

        # NEW: print pickup fraction
        if total_spawned > 0:
            print(f"Picked apples: {total_picked} / {total_spawned} ({total_picked / total_spawned:.4%})")
        else:
            print("No apples ever spawned? total_spawned=0")

        ages_ticks = np.array(tracker.ages, dtype=np.int64)
        if ages_ticks.size == 0:
            print("No apples were picked; nothing to plot.")
            return

        ages_sec = ages_ticks / NUM_AGENTS

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.hist(ages_sec, bins=50, density=True, alpha=0.8)
        ax1.set_xlabel("Time to apple (seconds)")
        ax1.set_ylabel("Density")
        ax1.set_title("Pickup time distribution")

        running_mean = np.cumsum(ages_sec) / (np.arange(len(ages_sec)) + 1)
        ax2.plot(running_mean)
        ax2.set_xlabel("Pickup event index")
        ax2.set_ylabel("Mean time to apple (seconds)")
        ax2.set_title("Running mean time-to-apple")

        plt.tight_layout()
        plt.show()

        plt.plot(np.arange(seconds + 1), num_apples)
        plt.xlabel("Time steps")
        plt.ylabel("Number of apples")
        plt.title("Apple count over time")
        plt.show()
