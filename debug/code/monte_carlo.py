from typing import Dict, Tuple

from debug.code.environment import Orchard
from debug.code.helpers import set_all_seeds, teleport
from debug.code.reward import Reward
import logging
import numpy as np
from datetime import datetime
import random
from concurrent.futures import ProcessPoolExecutor

from pathlib import Path
data_dir = Path(__file__).parent.parent / "data"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monte_carlo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

NUM_AGENTS = 4
W, L = 9, 9
REWARD = -1
PROBABILITY_APPLE = 10 / (W * L)
TRAJECTORY_LENGTH = 10000


def init_state(reward_module):
    npz = np.load(data_dir / f"init_state_reward_{reward_module.picker_r}.npz", allow_pickle=True)
    initial_state = npz["dict"].item()   # back to a Python dict
    agent_positions = npz["extra"]     # regular NumPy array

    orchard = Orchard(W, L, NUM_AGENTS, reward_module, PROBABILITY_APPLE, start_agents_map=initial_state["agents"],
                      start_apples_map=initial_state["apples"], start_agent_positions=agent_positions)

    return orchard


def monte_carlo(seed=42069):
    logger.info(f"Starting Monte Carlo simulation with seed={seed}")
    logger.info(f"Config: NUM_AGENTS={NUM_AGENTS}, W={W}, L={L}, TRAJECTORY_LENGTH={TRAJECTORY_LENGTH}")

    reward_module = Reward(REWARD, NUM_AGENTS)
    orchard = init_state(reward_module)

    # Initialize DS to store results
    rewards_by_agent = np.zeros((NUM_AGENTS, TRAJECTORY_LENGTH), dtype=float)

    set_all_seeds(seed)

    for step in range(TRAJECTORY_LENGTH):
        # act
        actor_idx = random.randint(0, NUM_AGENTS - 1)

        # get reward
        res = orchard.process_action(actor_idx, teleport(W))

        # store data
        rewards_by_agent[:, step] = res.reward_vector

        # Log progress every 1000 steps
        if step % 1000 == 0 and step > 0:
            total_reward = rewards_by_agent[:, :step].sum()
            avg_reward_per_agent = rewards_by_agent[:, :step].mean(axis=1)
            logger.info(
                f"Step {step}/{TRAJECTORY_LENGTH}: "
                f"Total reward={total_reward:.2f}, "
                f"Avg per agent={avg_reward_per_agent.mean():.2f}"
            )

    # Final statistics
    total_reward = rewards_by_agent.sum()
    logger.info(f"Simulation complete. Total reward: {total_reward:.2f}")
    logger.info(f"Reward by agent: {rewards_by_agent.sum(axis=1)}")

    # Save results with descriptive filename
    filename = data_dir / f"monte-carlo/results_seed{seed}_agents{NUM_AGENTS}_w{W}_l{L}.npz"
    np.savez_compressed(
        filename,
        rewards_by_agent=rewards_by_agent,
        metadata=np.array({
            'seed': seed,
            'num_agents': NUM_AGENTS,
            'width': W,
            'length': L,
            'trajectory_length': TRAJECTORY_LENGTH,
            'timestamp': datetime.now().isoformat()
        }, dtype=object)
    )
    logger.info(f"Results saved to {filename}")


def generate_initial_state(reward_module):
    orchard = Orchard(W, L, NUM_AGENTS, reward_module, PROBABILITY_APPLE)
    orchard.set_positions()
    np.savez(data_dir / f"init_state_reward_{reward_module.picker_r}", dict=orchard.get_state(), extra=orchard.agent_positions)


def load_monte_carlo_results(
        seed: int,
        num_agents: int,
        width: int,
        length: int,
) -> Tuple[np.ndarray, Dict]:
    """
    Load Monte Carlo simulation results from .npz file.

    Args:
        seed: Seed value used in the simulation
        num_agents: Number of agents in the simulation
        width: Width of the orchard
        length: Length of the orchard
        results_dir: Directory containing results files

    Returns:
        Tuple of (rewards_by_agent array, metadata dict)
    """
    filepath = data_dir / f"monte-carlo/results_seed{seed}_agents{num_agents}_w{width}_l{length}.npz"

    with np.load(filepath, allow_pickle=True) as data:
        rewards_by_agent = data['rewards_by_agent'].copy()
        metadata = data['metadata'].item() if 'metadata' in data else {}

    return rewards_by_agent, metadata


def run():
    seeds = list(range(10))

    with ProcessPoolExecutor(max_workers=8) as ex:
        ex.map(monte_carlo, seeds)


if __name__ == "__main__":
    run()
