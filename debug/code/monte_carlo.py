import argparse
import itertools
import time
import logging
import random
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from typing import Literal
from typing import Optional, Union

from config import (
    DISCOUNT_FACTOR, NUM_AGENTS,
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
    # logger.info(f"[{kind}] Results saved to {filename}")


StateType = Literal["none_on_apples", "agent_on_apple"]


# ---------------------------------------------------------------------
# Environment init
# ---------------------------------------------------------------------
def init_state(reward_module: Reward, state_type: StateType, trajectory_length):
    npz = np.load(
        data_dir / f"states/{state_type}/init_state_reward_{reward_module.picker_r}_{state_type}.npz",
        allow_pickle=True,  # needed if "dict" entry is an object array [web:109]
    )
    initial_state = npz["dict"].item()
    agent_positions = npz["extra"]
    # init_reward = npz["reward_vector"]
    #
    # reward_vector = np.zeros((NUM_AGENTS, trajectory_length), dtype=float)
    #
    # reward_vector[:, 0] = init_reward

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
    return orchard, initial_state["actor_id"], initial_state["mode"]


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


def generate_initial_state_supervised(
        reward_module: Reward,
        state_type: str,
        *,
        max_attempts: int = 10000,
        actor_id: int = 0,
        save=True
):
    """
    Generates and saves one of the 6 abstract states used in the supervised-learning setup.

    State types:
        Z1  : mode=0, actor=self
        Z0  : mode=0, actor=other
        Y11 : mode=1, actor=self,  apple under actor
        Y10 : mode=1, actor=self,  no apple under actor
        Y01 : mode=1, actor=other, apple under actor
        Y00 : mode=1, actor=other, no apple under actor
    """

    orchard = Orchard(W, L, NUM_AGENTS, reward_module, PROBABILITY_APPLE)

    VALID_STATES = {"Z1", "Z0", "Y11", "Y10", "Y01", "Y00"}
    if state_type not in VALID_STATES:
        raise ValueError(f"Unknown state_type {state_type}. Must be one of {VALID_STATES}")

    def satisfies(state: dict, agent_positions: np.ndarray) -> bool:
        apples = state["apples"]
        mode = state["mode"]

        rows = agent_positions[:, 0]
        cols = agent_positions[:, 1]
        on_apples = apples[rows, cols].astype(bool)

        actor_on_apple = bool(on_apples[actor_id])

        left_any = on_apples[:actor_id].any()
        right_any = on_apples[actor_id + 1:].any()
        other_on_apple = left_any or right_any

        actor_is_self = True
        actor_is_other = False

        # we assume orchard tracks current actor in state["actor"]
        if "actor" in state:
            actor_is_self = state["actor"] == actor_id
            actor_is_other = not actor_is_self

        match state_type:
            # -------- mode 0 --------
            case "Z1":   # m=0, a=1
                return (mode == 0) and actor_is_self

            case "Z0":   # m=0, a=0
                return (mode == 0) and actor_is_other

            # -------- mode 1 --------
            case "Y11":  # m=1, a=1, b=1
                return (mode == 1) and actor_is_self and actor_on_apple

            case "Y10":  # m=1, a=1, b=0
                return (mode == 1) and actor_is_self and (not actor_on_apple)

            case "Y01":  # m=1, a=0, b=1
                return (mode == 1) and actor_is_other and other_on_apple

            case "Y00":  # m=1, a=0, b=0
                return (mode == 1) and actor_is_other and (not other_on_apple)

            case _:
                return False

    # ------------------------------------------------------------

    for attempt in range(1, max_attempts + 1):
        orchard.set_positions()
        state = orchard.get_state()

        # force mode explicitly to avoid accidental matches
        if state_type.startswith("Z"):
            state["mode"] = 0
        else:
            state["mode"] = 1

        # force actor if your env allows it
        if state_type in {"Z1", "Y11", "Y10"}:
            state["actor_id"] = actor_id
        else:
            # pick some other agent
            other = (actor_id + 1) % NUM_AGENTS
            state["actor_id"] = other

        agent_positions = np.asarray(orchard.agent_positions)

        if satisfies(state, agent_positions):
            logger.info(f"[generate_state] Generated '{state_type}' after {attempt} attempts")
            break

        orchard.clear_positions()

    else:
        raise RuntimeError(
            f"[generate_state] Could not generate '{state_type}' after {max_attempts} attempts"
        )

    # ------------------------------------------------------------
    # compute reward vector for this initial state

    # reward_vector = np.asarray(
    #     reward_module.get_reward(state, actor_id, orchard.agent_positions[actor_id], state["mode"])
    # )
    if save:
        out_dir = data_dir / "states" / state_type
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"init_state_reward_{reward_module.picker_r}_{state_type}.npz"

        np.savez_compressed(
            out_path,
            dict=np.array(state, dtype=object),
            extra=agent_positions,
            # reward_vector=reward_vector,
        )
        logger.info(f"[generate_state] Initial state '{state_type}' saved to {out_path}")
        return None
    else:
        return state, agent_positions


# ---------------------------------------------------------------------
# Monte Carlo env-based simulation
# ---------------------------------------------------------------------
def monte_carlo(seed: int = 42069, trajectory_length=100000, reward=-1, state: StateType = "none_on_apples", run_id=0) -> None:
    set_all_seeds(seed)
    # logger.info(
    #     f"[mc] Starting Monte Carlo simulation with seed={seed} | "
    #     f"NUM_AGENTS={NUM_AGENTS}, W={W}, L={L}, T={trajectory_length}"
    # )

    reward_module = Reward(reward, NUM_AGENTS)
    orchard, rewards_by_agent = init_state(reward_module, state, trajectory_length)

    for step in range(1, trajectory_length):
        actor_idx = random.randint(0, NUM_AGENTS - 1)
        res = orchard.process_action(actor_idx, teleport(W), mode=1)  # always act
        rewards_by_agent[:, step] = res.reward_vector

    total_reward = rewards_by_agent.sum()
    logger.info(f"[mc] Simulation complete. Total reward: {total_reward:.2f}")
    logger.info(f"[mc] Reward by agent: {rewards_by_agent.sum(axis=1)}")

    save_results("monte-carlo", run_id, rewards_by_agent, trajectory_length, reward, state)


# (state_dict, agent_positions) OR (state_dict, agent_positions, init_actor, init_mode)
InitPayload = Union[
    tuple[dict, np.ndarray],
    tuple[dict, np.ndarray, int, int],
]


def _orchard_from_payload(reward_module: Reward, payload: InitPayload):
    state_dict = payload[0]
    agent_positions = np.asarray(payload[1])

    # Prefer explicit values in the payload tuple if provided
    if len(payload) >= 4:
        init_actor = int(payload[2])
        init_mode = int(payload[3])
    else:
        init_actor = int(state_dict.get("actor_id", state_dict.get("actor", 0)))
        init_mode = int(state_dict.get("mode", 1))

    orchard = Orchard(
        W,
        L,
        NUM_AGENTS,
        reward_module,
        PROBABILITY_APPLE,
        start_agents_map=state_dict["agents"],
        start_apples_map=state_dict["apples"],
        start_agent_positions=agent_positions,
    )
    return orchard, init_actor, init_mode


def monte_carlo_supervised(
        seed: int = 42069,
        trajectory_length=100000,
        reward=-1,
        state: StateType = "none_on_apples",
        run_id=0,
        init_payload: Optional[InitPayload] = None,
) -> float:
    set_all_seeds(seed)

    reward_module = Reward(reward, NUM_AGENTS)
    rewards_by_agent = np.zeros((NUM_AGENTS, trajectory_length), dtype=float)

    if init_payload is None:
        orchard, init_actor, init_mode = init_state(reward_module, state, trajectory_length)
    else:
        orchard, init_actor, init_mode = _orchard_from_payload(reward_module, init_payload)

    if init_mode == 0:
        res = orchard.process_action(init_actor, teleport(W), mode=0)
        rewards_by_agent[:, 0] = res.reward_vector
        res = orchard.process_action(init_actor, None, mode=1)
        rewards_by_agent[:, 1] = res.reward_vector
        start, end = 2, trajectory_length
    else:
        res = orchard.process_action(init_actor, None, mode=1)
        rewards_by_agent[:, 0] = res.reward_vector
        start, end = 1, trajectory_length

    for step in range(start, end, 2):
        actor_idx = random.randint(0, NUM_AGENTS - 1)

        res = orchard.process_action(actor_idx, teleport(W), mode=0)
        rewards_by_agent[:, step] = res.reward_vector

        if step + 1 < end:
            res = orchard.process_action(actor_idx, None, mode=1)
            rewards_by_agent[:, step + 1] = res.reward_vector

    save_results("monte-carlo", run_id, rewards_by_agent, trajectory_length, reward, state)
    discounts = np.power(DISCOUNT_FACTOR, np.arange(trajectory_length, dtype=np.float64))
    values_by_agent = (rewards_by_agent * discounts).sum(axis=1)

    return values_by_agent


# ---------------------------------------------------------------------
# IID baseline simulation
# ---------------------------------------------------------------------
def iid(seed: int, trajectory_length, reward=-1, state: StateType = "agent_on_apple", run_id=0) -> None:
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

    save_results("iid", run_id, rewards_by_agent, trajectory_length, reward, state)


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
def run(
        sim_fn=monte_carlo_supervised,
        kind="monte-carlo-supervised",
        trajectory_length=100000,
        reward=-1,
        state_type: StateType = "agent_on_apple",
        seedgen_seed: int | None = None,
        init_payload: Optional[InitPayload] = None,
) -> float:
    if seedgen_seed is not None:
        random.seed(seedgen_seed)

    seeds = random.sample(range(1, 10_000_000), SEEDS)
    run_ids = list(range(SEEDS))

    trajectory_lengths = itertools.repeat(trajectory_length)
    rewards = itertools.repeat(reward)
    states = itertools.repeat(state_type)
    init_payloads = itertools.repeat(init_payload)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        results = list(ex.map(sim_fn, seeds, trajectory_lengths, rewards, states, run_ids, init_payloads))

    return np.mean(results, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fn",
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
    elif args.task == "generate_supervised":
        generate_initial_state_supervised(Reward(int(args.reward), NUM_AGENTS), args.state)
    else:
        if args.fn == "monte_carlo":
            sim_fn = monte_carlo
            kind = "monte-carlo"
        elif args.fn == "iid":
            sim_fn = iid
            kind = "iid"
        else:
            sim_fn = monte_carlo_supervised
            kind = "monte-carlo-supervised"

        # run(sim_fn, kind, int(args.trajectories), int(args.reward), args.state)
        run(sim_fn=sim_fn, kind=kind, trajectory_length=int(args.trajectories), reward=int(args.reward), state_type=args.state)


if __name__ == "__main__":
    main()
