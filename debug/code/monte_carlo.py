import argparse
import itertools
import time
import logging
import random
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal
from typing import Optional, Union
import copy
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
from debug.code.helpers import make_env, random_policy, set_all_seeds, teleport, \
    transition
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
        save=True,
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
        start_agents_map=state_dict["agents"].copy(),
        start_apples_map=state_dict["apples"].copy(),
        start_agent_positions=agent_positions.copy(),
    )
    return orchard, init_actor, init_mode


def deep_copy_state(s: dict) -> dict:
    # safer than s.copy() for nested structures [web:394]
    return copy.deepcopy(s)


def monte_carlo_full(
        seed: int = 42069,
        trajectory_length: int = 1,
        init_env=None,
        init_state: dict = None,
        discount_factor: float = 0.99,
        num_trajectories: int = 1,
        num_rollouts: int = 1,
):
    assert init_env is not None, "Pass init_env (or refactor to pass make_env=...)"
    assert init_state is not None

    T = trajectory_length * NUM_AGENTS * 2
    gamma = float(discount_factor)

    d = np.tile(np.array([gamma, 1.0], dtype=float), T // 2 + 1)[:T]

    weights = np.empty(T, dtype=float)
    weights[0] = 1.0
    if T > 1:
        weights[1:] = np.cumprod(d[:-1])

    base_state = deep_copy_state(init_state)

    # Collect ALL rollout returns as samples for mean/std/SE
    n_total = int(num_trajectories * num_rollouts)
    all_returns = np.zeros((NUM_AGENTS, n_total), dtype=float)

    k = 0
    for i in range(num_trajectories):
        for j in range(num_rollouts):
            set_all_seeds(seed + i * num_rollouts + j)

            curr_state = deep_copy_state(base_state)

            env = make_env(
                init_env.reward_module,
                init_env.p_apple,
                init_env.d_apple,
                curr_state["apples"],
                curr_state["agents"].copy(),
                curr_state["agent_positions"].copy(),
            )

            actor_idx = curr_state["actor_id"]

            rewards_by_agent = np.zeros((NUM_AGENTS, T), dtype=float)
            t = 0

            for _ in range(trajectory_length):
                for step in range(NUM_AGENTS):
                    curr_state, _, res, actor_idx = transition(step, curr_state, env, actor_idx, random_policy(curr_state["agent_positions"][actor_idx]))
                    rewards_by_agent[:, t] = 0.0
                    t += 1
                    rewards_by_agent[:, t] = res.reward_vector
                    t += 1

            assert t == T, f"Filled {t} reward slots, expected {T}"

            # One return sample per agent
            all_returns[:, k] = rewards_by_agent @ weights
            k += 1

    # Mean return per agent
    mc_mean = all_returns.mean(axis=1)

    # Std of returns across rollouts (sample std with ddof=1)
    mc_std = all_returns.std(axis=1, ddof=1)

    # Standard error of the MC mean: SE = std / sqrt(n) [web:49]
    mc_se = mc_std / np.sqrt(n_total)

    # 95% CI for the MC mean using normal approx: mean ± 1.96*SE [web:52]
    ci_low = mc_mean - 1.96 * mc_se
    ci_high = mc_mean + 1.96 * mc_se

    return {
        "mc_mean": mc_mean,     # shape (NUM_AGENTS,)
        "mc_se": mc_se,         # shape (NUM_AGENTS,)
        "ci95_low": ci_low,     # shape (NUM_AGENTS,)
        "ci95_high": ci_high,   # shape (NUM_AGENTS,)
        "n": n_total,
    }


def generate_initial_state_full(
        reward_module: Reward,
        run_id: int,
        seed: int,
        discount_factor: float,
        p_apple: float,
        d_apple: float,
        q_agent: float,
        tau: float,
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

    orchard = Orchard(W, L, NUM_AGENTS, reward_module, p_apple=(q_agent * NUM_AGENTS * tau) / (W ** 2), d_apple=d_apple)
    orchard.p_apple = p_apple
    orchard.set_positions()
    state = dict(orchard.get_state())
    state["actor_id"] = random.randint(0, NUM_AGENTS - 1)
    state["mode"] = 0

    res = monte_carlo_full(seed, init_env=orchard, init_state=state, discount_factor=discount_factor)

    state["mc"] = res["mc_mean"]
    state["std"] = res["mc_se"]
    state["ci95_low"] = res["ci95_low"]
    state["ci95_high"] = res["ci95_high"]
    if save:
        out_dir = data_dir / "states" / "full"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"init_state_reward_{reward_module.picker_r}_{run_id}.npz"

        np.savez_compressed(
            out_path,
            dict=np.array(state, dtype=object),
        )
    return state


def _sample_other_agents_positions(rng, width, length, num_agents, reserved):
    """
    Returns positions array (num_agents, 2) with unique cells.
    reserved: set of (x,y) that cannot be used.
    """
    positions = np.full((num_agents, 2), -1, dtype=int)

    used = set(reserved)
    for a in range(num_agents):
        while True:
            x = int(rng.integers(0, width))
            y = int(rng.integers(0, length))
            if (x, y) not in used:
                used.add((x, y))
                positions[a] = (x, y)
                break
    return positions


def _agents_map_from_positions(width, length, agent_positions):
    agents = np.zeros((width, length), dtype=int)
    for (x, y) in agent_positions:
        agents[x, y] += 1
    return agents


def _make_distance_positions(center, distances, width, length):
    """
    Produce self positions at given Manhattan distances from center along a line,
    clamped to valid cells. Keeps it simple/deterministic.
    """
    cx, cy = center
    out = []
    for d in distances:
        x = cx - d
        y = cy
        if x < 0:
            # fallback: go upward instead
            x = cx
            y = cy - d
        if not (0 <= x < width and 0 <= y < length):
            raise ValueError(f"Distance {d} doesn't fit on grid from center {center}.")
        out.append((x, y))
    return out


def generate_careful_distance_series(
        reward_module,
        seed: int,
        discount_factor: float,
        p_apple: float,
        d_apple: float,
        distances=(4, 3, 2, 1, 0),
        self_id: int = 0,
):
    width, length = W, L
    center = (width // 2, length // 2)

    # Fixed diamond apples around center
    start_apples = np.zeros((width, length), dtype=int)
    start_apples[center[0],     center[1]    ] = 1  # center
    start_apples[center[0] - 2, center[1]    ] = 1  # top
    start_apples[center[0] - 1, center[1] + 1] = 1  # top-right
    start_apples[center[0] - 1, center[1] - 1] = 1  # top-left

    # 6 hardcoded outskirt apples (fixed across all states)
    outskirt_apples = [
        (1,          1          ),  # top-left corner area
        (1,          length - 2 ),  # top-right corner area
        (width - 2,  1          ),  # bottom-left corner area
        (width - 2,  length - 2 ),  # bottom-right corner area
        (width // 2, 1          ),  # left edge midpoint
        (width // 2, length - 2 ),  # right edge midpoint
    ]
    for (r, c) in outskirt_apples:
        start_apples[r, c] = 1

    # Choose self positions at different Manhattan distances
    self_positions = _make_distance_positions(center, distances, width, length)

    # Fix other agents' positions ONCE (reused across all distances)
    rng = np.random.default_rng(seed)

    # Sample a base placement for other agents, avoiding the apple and any potential self cells
    reserved = {center} | set(self_positions)
    other_positions = _sample_other_agents_positions(
        rng, width, length, NUM_AGENTS - 1, reserved=reserved
    )

    for d, self_pos in zip(distances, self_positions):
        # Build full agent_positions (NUM_AGENTS,2)
        agent_positions = np.zeros((NUM_AGENTS, 2), dtype=int)
        agent_positions[self_id] = np.array(self_pos, dtype=int)

        k = 0
        for a in range(NUM_AGENTS):
            if a == self_id:
                continue
            agent_positions[a] = other_positions[k]
            k += 1

        start_agents = _agents_map_from_positions(width, length, agent_positions)

        # Build init_state explicitly (don’t rely on get_state())
        init_state = {
            "apples": start_apples.copy(),
            "agents": start_agents.copy(),
            "agent_positions": agent_positions.copy(),
            "actor_id": self_id,
            "mode": 0,  # start in move mode so distance can matter [file:1]
        }

        # Build an env factory that recreates the exact same env per trajectory
        def init_env_factory():
            return Orchard(
                length=length,
                width=width,
                num_agents=NUM_AGENTS,
                reward=reward_module,
                p_apple=p_apple,
                d_apple=d_apple,
                start_agents_map=start_agents,
                start_apples_map=start_apples,
                start_agent_positions=agent_positions,
            )

        res = monte_carlo_full(
            seed=seed,
            init_env=init_env_factory(),
            init_state=init_state,
            discount_factor=discount_factor,
        )

        init_state["distance"] = d
        init_state["mc"] = res["mc_mean"]
        init_state["std"] = res["mc_se"]
        init_state["ci95_low"] = res["ci95_low"]
        init_state["ci95_high"] = res["ci95_high"]

        out_dir = data_dir / "states" / "careful"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"careful_agent{self_id}_seed{seed}_d{d}.npz"

        np.savez_compressed(
            out_path,
            dict=np.array(init_state, dtype=object),
        )
    return init_state


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


def iid_supervised(
        seed: int = 42069,
        trajectory_length: int = 100000,
        reward: float = -1,
        state: StateType = "none_on_apples",
        run_id: int = 0,
        init_payload: Optional[InitPayload] = None,
) -> np.ndarray:
    set_all_seeds(seed)

    reward_other = (1 - reward) / (NUM_AGENTS - 1)

    # Init (kept for parity with your setup; orchard isn't otherwise used in IID)
    if init_payload is None:
        orchard, actor_id, init_mode = init_state(Reward(reward, NUM_AGENTS), state, trajectory_length)
    else:
        orchard, actor_id, init_mode = _orchard_from_payload(Reward(reward, NUM_AGENTS), init_payload)

    values_by_agent = np.zeros(NUM_AGENTS, dtype=np.float64)

    # Preallocate once; reuse to avoid per-step allocations
    rewards_vec = np.empty(NUM_AGENTS, dtype=np.float64)

    # Running discount for time t
    discount = 1.0

    def maybe_apply_reward(curr_actor: int, curr_discount: float) -> None:
        # reward happens only with probability PROBABILITY_APPLE
        if random.random() < PROBABILITY_APPLE:
            rewards_vec.fill(reward_other)
            rewards_vec[curr_actor] = reward
            values_by_agent[:] += curr_discount * rewards_vec  # broadcasted vector add [web:27]

    # t = 0 is always zero reward in your original code (explicitly set in mode 0)
    # So we just advance the discount once.
    discount *= DISCOUNT_FACTOR

    if init_mode == 0:
        # t = 1: possible reward using initial actor_id
        maybe_apply_reward(actor_id, discount)
        discount *= DISCOUNT_FACTOR
        start, end = 2, trajectory_length
    else:
        # Your original code writes to rewards_by_agent[:, 1] in this branch.
        # Keeping behavior consistent: treat this as reward at t = 1, then start loop at 1.
        maybe_apply_reward(actor_id, discount)
        discount *= DISCOUNT_FACTOR
        start, end = 1, trajectory_length

    # Loop steps: only odd indices (step+1) can have rewards in your original logic
    for step in range(start, end, 2):
        actor_id = random.randint(0, NUM_AGENTS - 1)
        # step itself has zero reward; just advance discount by one timestep
        discount *= DISCOUNT_FACTOR

        if step + 1 < end:
            maybe_apply_reward(actor_id, discount)
            discount *= DISCOUNT_FACTOR

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

    orchard, actor_id, init_mode = init_state(Reward(reward, NUM_AGENTS), state, trajectory_length)

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
        trajectory_length=100000,
        reward=-1,
        state_type: StateType = "agent_on_apple",
        seedgen_seed: int | None = None,
        init_payload: Optional[InitPayload] = None,
        num_seeds: int = 10
) -> float:
    if seedgen_seed is not None:
        random.seed(seedgen_seed)

    seeds = random.sample(range(1, 10_000_000), num_seeds)
    run_ids = list(range(num_seeds))

    trajectory_lengths = itertools.repeat(trajectory_length)
    rewards = itertools.repeat(reward)
    states = itertools.repeat(state_type)
    init_payloads = itertools.repeat(init_payload)

    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(sim_fn, seeds, trajectory_lengths, rewards, states, run_ids, init_payloads, chunksize=10))
    # sim_fn(0, 1000, -1, run_id=0, init_payload=init_payload)

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
        elif args.fn == "iid":
            sim_fn = iid
        elif args.fn == "iid_supervised":
            sim_fn = iid_supervised
        else:
            sim_fn = monte_carlo_supervised

        # run(sim_fn, kind, int(args.trajectories), int(args.reward), args.state)

        start = time.time()
        run(sim_fn=sim_fn, trajectory_length=int(args.trajectories), reward=int(args.reward), state_type=args.state)
        end = time.time()
        print(end - start)


if __name__ == "__main__":
    main()
