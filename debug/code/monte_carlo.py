import argparse
import time
import logging
import random
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from config import (
    NUM_AGENTS,
    W,
    L,
    REWARD,
    PROBABILITY_APPLE,
    TRAJECTORY_LENGTH,
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
def save_results(kind: str, seed: int, rewards_by_agent: np.ndarray) -> None:
    """Save rewards and metadata to an .npz file."""
    # folder name: function_name-{PROBABILITY_APPLE}-{NUM_AGENTS}-{W}
    folder_name = f"{kind}-{PROBABILITY_APPLE:.2f}-{NUM_AGENTS}-{W}-{REWARD}"
    out_dir = data_dir / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = out_dir / f"results_seed{seed}.npz"
    metadata = np.array(
        {
            "seed": seed,
            "num_agents": NUM_AGENTS,
            "width": W,
            "length": L,
            "trajectory_length": TRAJECTORY_LENGTH,
            "timestamp": datetime.now().isoformat(),
            "kind": kind,
            "probability_apple": PROBABILITY_APPLE,
        },
        dtype=object,
    )

    np.savez_compressed(filename, rewards_by_agent=rewards_by_agent, metadata=metadata)
    logger.info(f"[{kind}] Results saved to {filename}")


def log_progress(step: int, rewards_by_agent: np.ndarray) -> None:
    """Log aggregate stats every N steps."""
    if step % 1000 != 0 or step == 0:
        return
    total_reward = rewards_by_agent[:, :step].sum()
    avg_reward_per_agent = rewards_by_agent[:, :step].mean(axis=1)
    logger.info(
        f"Step {step}/{TRAJECTORY_LENGTH}: "
        f"Total reward={total_reward:.2f}, "
        f"Avg per agent={avg_reward_per_agent.mean():.2f}"
    )


# ---------------------------------------------------------------------
# Environment init
# ---------------------------------------------------------------------
def init_state(reward_module: Reward) -> Orchard:
    npz = np.load(
        data_dir / f"init_state_reward_{reward_module.picker_r}.npz",
        allow_pickle=True,
        )
    initial_state = npz["dict"].item()
    agent_positions = npz["extra"]

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
    return orchard


def generate_initial_state(reward_module: Reward) -> None:
    orchard = Orchard(W, L, NUM_AGENTS, reward_module, PROBABILITY_APPLE)
    orchard.set_positions()
    np.savez(
        data_dir / f"init_state_reward_{reward_module.picker_r}",
        dict=orchard.get_state(),
        extra=orchard.agent_positions,
        )


# ---------------------------------------------------------------------
# Monte Carlo env-based simulation
# ---------------------------------------------------------------------
def monte_carlo(seed: int = 42069) -> None:
    set_all_seeds(seed)
    logger.info(
        f"[mc] Starting Monte Carlo simulation with seed={seed} | "
        f"NUM_AGENTS={NUM_AGENTS}, W={W}, L={L}, T={TRAJECTORY_LENGTH}"
    )

    reward_module = Reward(REWARD, NUM_AGENTS)
    orchard = init_state(reward_module)

    rewards_by_agent = np.zeros((NUM_AGENTS, TRAJECTORY_LENGTH), dtype=float)

    for step in range(TRAJECTORY_LENGTH):
        actor_idx = random.randint(0, NUM_AGENTS - 1)
        res = orchard.process_action(actor_idx, teleport(W))
        rewards_by_agent[:, step] = res.reward_vector

        log_progress(step, rewards_by_agent)

    total_reward = rewards_by_agent.sum()
    logger.info(f"[mc] Simulation complete. Total reward: {total_reward:.2f}")
    logger.info(f"[mc] Reward by agent: {rewards_by_agent.sum(axis=1)}")

    save_results("monte-carlo", seed, rewards_by_agent)


# ---------------------------------------------------------------------
# IID baseline simulation
# ---------------------------------------------------------------------
def iid(seed: int) -> None:
    set_all_seeds(seed)
    logger.info(
        f"[iid] Starting IID simulation with seed={seed} | "
        f"NUM_AGENTS={NUM_AGENTS}, W={W}, L={L}, T={TRAJECTORY_LENGTH}"
    )

    rewards_by_agent = np.zeros((NUM_AGENTS, TRAJECTORY_LENGTH), dtype=float)
    reward_other = (1 - REWARD) / (NUM_AGENTS - 1)

    for step in range(TRAJECTORY_LENGTH):
        actor_id = random.randint(0, NUM_AGENTS - 1)

        picker_reward = REWARD if random.random() < PROBABILITY_APPLE else 0.0
        if picker_reward != 0.0:
            # everyone gets updated this step
            rewards = np.full(NUM_AGENTS, reward_other, dtype=float)
            rewards[actor_id] = REWARD
            rewards_by_agent[:, step] = rewards

        log_progress(step, rewards_by_agent)

    total_reward = rewards_by_agent.sum()
    logger.info(f"[iid] Simulation complete. Total reward: {total_reward:.2f}")
    logger.info(f"[iid] Reward by agent: {rewards_by_agent.sum(axis=1)}")

    save_results("iid", seed, rewards_by_agent)


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
def run(sim_fn=monte_carlo, kind="monte-carlo") -> None:
    seeds = list(range(SEEDS))
    start = time.time()
    logger.info(
        f"run({kind}) starting with {len(seeds)} seeds and {NUM_WORKERS} workers"
    )

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        ex.map(sim_fn, seeds)

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
    args = parser.parse_args()

    if args.fn == "monte_carlo":
        sim_fn = monte_carlo
        kind = "monte-carlo"
    else:
        sim_fn = iid
        kind = "iid"

    run(sim_fn=sim_fn, kind=kind)


if __name__ == "__main__":
    main()
