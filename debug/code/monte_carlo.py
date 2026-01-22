import argparse
import itertools
import os
import time
import logging
import random
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from typing import Literal

from config import (
    NUM_AGENTS,
    W,
    L,
    PROBABILITY_APPLE,
    NUM_WORKERS,
    SEEDS,
    data_dir,
)

import numpy as np

from debug.code.environment import Orchard
from debug.code.helpers import set_all_seeds, teleport
from debug.code.reward import Reward


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("monte_carlo.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def save_results(kind: str, seed: int, rewards_by_agent: np.ndarray, trajectory_length: int, reward: int, state) -> None:
    """Save rewards and metadata to an .npz file."""
    # folder name: function_name-{PROBABILITY_APPLE}-{NUM_AGENTS}-{W}
    folder_name = f"{PROBABILITY_APPLE:.2f}-{NUM_AGENTS}-{W}-{reward}"
    out_dir = (data_dir / state / kind)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = out_dir / f"results_seed{seed}.npz"
    metadata = np.array(
        {
            "seed": seed,
            "num_agents": NUM_AGENTS,
            "width": W,
            "length": L,
            "trajectory_length": trajectory_length,
            "timestamp": datetime.now().isoformat(),
            "kind": kind,
            "probability_apple": PROBABILITY_APPLE,
        },
        dtype=object,
    )

    np.savez_compressed(filename, rewards_by_agent=rewards_by_agent, metadata=metadata)
    logger.info(f"[{kind}] Results saved to {filename}")


StateType = Literal["none_on_apples", "agent_on_apple"]


# ---------------------------------------------------------------------
# Environment init
# ---------------------------------------------------------------------
def init_state(reward_module: Reward, state_type: StateType, trajectory_length) -> tuple[Orchard, np.ndarray]:
    npz = np.load(
        data_dir / f"states/{state_type}/init_state_reward_{reward_module.picker_r}_{state_type}.npz",
        allow_pickle=True,  # needed if "dict" entry is an object array [web:109]
    )
    initial_state = npz["dict"].item()
    agent_positions = npz["extra"]
    init_reward = npz["reward_vector"]

    reward_vector = np.zeros((NUM_AGENTS, trajectory_length), dtype=float)

    reward_vector[:, 0] = init_reward

    orchard = Orchard(
        W,
        L,
        NUM_AGENTS,
        reward_module,
        PROBABILITY_APPLE,
        start_agents_map=initial_state["agents"],
        start_apples_map=initial_state["apples"],
        start_agent_positions=agent_positions,
    )
    return orchard, reward_vector


def generate_initial_state(
        reward_module: Reward,
        state_type: StateType = "none_on_apples",
        *,
        max_attempts: int = 1000,
        actor_id=0
) -> None:
    orchard = Orchard(W, L, NUM_AGENTS, reward_module, PROBABILITY_APPLE)

    def satisfies(state: dict, agent_positions: np.ndarray) -> bool:
        apples = state["apples"]
        rows = agent_positions[:, 0]
        cols = agent_positions[:, 1]
        on_apples = apples[rows, cols].astype(bool)

        left_any = on_apples[:actor_id].any()
        right_any = on_apples[actor_id + 1:].any()
        others_any = left_any or right_any

        match state_type:
            case "none_on_apples":
                return not on_apples.any()
            case "agent_on_apple":
                return bool(on_apples[actor_id]) and (not others_any)
            case _:
                raise ValueError(f"Unknown state_type: {state_type}")

    for attempt in range(1, max_attempts + 1):
        orchard.set_positions()
        state = orchard.get_state()
        agent_positions = np.asarray(orchard.agent_positions)

        if satisfies(state, agent_positions):
            logger.info(f"[generate_state] Generated '{state_type}' after {attempt} attempts")
            break

        orchard.clear_positions()
    else:
        raise RuntimeError(
            f"[generate_state] Could not generate '{state_type}' after {max_attempts} attempts"
        )

    # Compute and store reward vector for this *initial* state
    reward_vector = np.asarray(reward_module.get_reward(state, actor_id, orchard.agent_positions[actor_id]))

    out_dir = data_dir / f"states/{state_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"init_state_reward_{reward_module.picker_r}_{state_type}.npz"

    # Keyword args become names inside the .npz (e.g., "reward_vector") [web:80]
    np.savez_compressed(
        out_path,
        dict=np.array(state, dtype=object),
        extra=agent_positions,
        reward_vector=reward_vector,
    )
    logger.info(f"[generate_state] Initial state '{state_type}' saved to {out_path}")


# ---------------------------------------------------------------------
# Monte Carlo env-based simulation
# ---------------------------------------------------------------------
def monte_carlo(seed: int = 42069, trajectory_length=100000, reward=-1, state: StateType = "none_on_apples") -> None:
    set_all_seeds(seed)
    logger.info(
        f"[mc] Starting Monte Carlo simulation with seed={seed} | "
        f"NUM_AGENTS={NUM_AGENTS}, W={W}, L={L}, T={trajectory_length}"
    )

    reward_module = Reward(reward, NUM_AGENTS)
    orchard, rewards_by_agent = init_state(reward_module, state, trajectory_length)

    for step in range(1, trajectory_length):
        actor_idx = random.randint(0, NUM_AGENTS - 1)
        res = orchard.process_action(actor_idx, teleport(W))
        rewards_by_agent[:, step] = res.reward_vector

    total_reward = rewards_by_agent.sum()
    logger.info(f"[mc] Simulation complete. Total reward: {total_reward:.2f}")
    logger.info(f"[mc] Reward by agent: {rewards_by_agent.sum(axis=1)}")

    save_results("monte-carlo", seed, rewards_by_agent, trajectory_length, reward, state)


# ---------------------------------------------------------------------
# IID baseline simulation
# ---------------------------------------------------------------------
def iid(seed: int, trajectory_length, reward=-1, state: StateType = "agent_on_apple") -> None:
    set_all_seeds(seed)
    logger.info(
        f"[iid] Starting IID simulation with seed={seed} | "
        f"NUM_AGENTS={NUM_AGENTS}, W={W}, L={L}, T={trajectory_length}"
    )

    _, rewards_by_agent = init_state(Reward(reward, NUM_AGENTS), state, trajectory_length)

    reward_other = (1 - reward) / (NUM_AGENTS - 1)

    for step in range(1, trajectory_length):
        actor_id = random.randint(0, NUM_AGENTS - 1)

        picker_reward = reward if random.random() < PROBABILITY_APPLE else 0.0
        if picker_reward != 0.0:
            # everyone gets updated this step
            rewards = np.full(NUM_AGENTS, reward_other, dtype=float)
            rewards[actor_id] = reward
            rewards_by_agent[:, step] = rewards

    total_reward = rewards_by_agent.sum()
    logger.info(f"[iid] Simulation complete. Total reward: {total_reward:.2f}")
    logger.info(f"[iid] Reward by agent: {rewards_by_agent.sum(axis=1)}")

    save_results("iid", seed, rewards_by_agent, trajectory_length, reward, state)


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
def run(sim_fn=monte_carlo, kind="monte-carlo", trajectory_length=100000, reward=-1, state_type: StateType = "agent_on_apple") -> None:
    seeds = list(range(SEEDS))
    trajectory_lengths = itertools.repeat(trajectory_length)
    rewards = itertools.repeat(reward)
    states = itertools.repeat(state_type)
    start = time.time()
    logger.info(
        f"run({kind}) starting with {len(seeds)} seeds and {NUM_WORKERS} workers"
    )

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        ex.map(sim_fn, seeds, trajectory_lengths, rewards, states)

    elapsed = time.time() - start
    logger.info(f"run({kind}) finished in {elapsed:.2f} seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fn",
        choices=["monte_carlo", "iid"],
        default="monte_carlo",
        help="Which simulation function to run.",
    )
    parser.add_argument(
        "--trajectories",
        required=True,
        help="Path (or identifier) for trajectories input.",
    )
    parser.add_argument(
        "--reward",
        required=True,
        help="Reward.",
    )

    parser.add_argument(
        "--state",
        required=True,
        help="Reward.",
    )

    parser.add_argument(
        "--task",
        required=True,
        help="State.",
    )
    args = parser.parse_args()  # parsed values land on `args.<name>` [web:1]
    if args.task == "generate":
        generate_initial_state(Reward(int(args.reward), NUM_AGENTS), args.state)
    else:
        if args.fn == "monte_carlo":
            sim_fn = monte_carlo
            kind = "monte-carlo"
        else:
            sim_fn = iid
            kind = "iid"

        run(sim_fn=sim_fn, kind=kind, trajectory_length=int(args.trajectories), reward=int(args.reward), state_type=args.state)


if __name__ == "__main__":
    main()
