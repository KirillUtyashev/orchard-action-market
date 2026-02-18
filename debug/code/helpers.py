import random
import torch
import os
import numpy as np

from debug.code.config import NUM_AGENTS, W
from debug.code.environment import Orchard

same_cell_no_reward = 0
count = 0
UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4


def set_all_seeds(seed: int = 42, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch (CPU & CUDA) for reproducibility.

    Args:
        seed: The seed value to use.
        deterministic: If True, enable extra settings for deterministic
                       behavior in cuDNN at the possible cost of speed.
    """
    # print(f"[seed] Using seed={seed}")

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU & CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Hash-based things (e.g. dict iteration order)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # cuDNN settings for (more) deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def teleport(n):
    new_i = np.random.randint(0, n - 1)
    new_j = np.random.randint(0, n - 1)
    return np.array([new_i, new_j])


def random_policy(agent_pos):
    """
    Returns the *new (r, c) position* after taking a uniformly random action
    from {UP, DOWN, LEFT, RIGHT, STAY}. If the sampled move would go off-grid,
    the agent stays in place. [web:198]
    """
    r, c = agent_pos
    a = random.randrange(5)  # 0..4

    nr, nc = r, c
    if a == UP:
        nr = r - 1
    elif a == DOWN:
        nr = r + 1
    elif a == LEFT:
        nc = c - 1
    elif a == RIGHT:
        nc = c + 1
    elif a == STAY:
        pass

    # If illegal, "pick stay" (i.e., revert)
    if not (0 <= nr < W and 0 <= nc < W):
        nr, nc = r, c

    return np.array([nr, nc])


def transition(step, curr_state, env, actor_idx, new_pos):
    if step == -1:
        # init-only: do NOT mutate env
        if curr_state is None:
            curr_state = dict(env.get_state())
        if actor_idx is None:
            actor_idx = random.randint(0, NUM_AGENTS - 1)
        curr_state["actor_id"] = actor_idx
        curr_state["mode"] = 0
        return curr_state, None, None, actor_idx

    env.process_action(
        actor_idx,
        new_pos,
        mode=0,
    )

    semi_state = dict(env.get_state())
    semi_state["actor_id"] = actor_idx
    semi_state["mode"] = 1

    res = env.process_action(actor_idx, None, mode=1)

    if step == NUM_AGENTS - 1:
        env.despawn_apples()
        env.spawn_apples()

    final_state = dict(env.get_state())
    actor_idx = random.randint(0, NUM_AGENTS - 1)
    final_state["actor_id"] = actor_idx
    final_state["mode"] = 0

    # return final_state as the next curr_state
    return final_state, semi_state, res, actor_idx


def make_env(reward_module, p_apple, d_apple, apples, agents, agent_positions):
    return Orchard(W, W, NUM_AGENTS, reward_module, p_apple=p_apple, d_apple=d_apple,
                   start_apples_map=apples, start_agents_map=agents, start_agent_positions=agent_positions)
