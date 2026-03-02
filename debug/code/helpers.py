import random
import torch
import os
import numpy as np

from debug.code.enums import NUM_AGENTS, W
from debug.code.environment import Orchard

same_cell_no_reward = 0
count = 0
UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4


def ten(c: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy array to torch tensor on specified device.

    Args:
        c: Input numpy array.
        device: Target device for the tensor.

    Returns:
        A torch tensor on the specified device.
    """
    return torch.from_numpy(c).to(device).float()


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


def nearest_apple_policy(agent_pos, apples_matrix):
    """
    Greedy 4-neighborhood step toward the nearest apple (by Manhattan distance).
    If no apples exist, STAY (or you can fall back to random_policy).
    Returns the new (r, c) position after one step.
    """
    r, c = int(agent_pos[0]), int(agent_pos[1])
    H, W = apples_matrix.shape

    apple_rc = np.argwhere(apples_matrix > 0)
    if apple_rc.size == 0:
        return np.array([r, c])  # or: return random_policy(agent_pos)

    rs = apple_rc[:, 0]
    cs = apple_rc[:, 1]
    d = np.abs(rs - r) + np.abs(cs - c)
    min_d = d.min()

    # Tie-break deterministically but stably: pick the first minimum
    idx = int(np.flatnonzero(d == min_d)[0])
    tr, tc = int(rs[idx]), int(cs[idx])

    nr, nc = r, c

    # Move to reduce Manhattan distance by 1 (one axis at a time)
    if tr < r:
        nr = r - 1
    elif tr > r:
        nr = r + 1
    elif tc < c:
        nc = c - 1
    elif tc > c:
        nc = c + 1
    else:
        # Already on an apple
        nr, nc = r, c  # STAY

    # Boundary check (same style as your random_policy)
    if not (0 <= nr < H and 0 <= nc < W):
        nr, nc = r, c

    return np.array([nr, nc])


def env_step(env, actor_idx, new_pos, num_agents):
    s_moved = env.apply_action(actor_idx, new_pos)
    on_apple = env.is_on_apple(s_moved, actor_idx)

    if on_apple:
        s_picked, pick_rewards = env.resolve_pick(actor_idx)
    else:
        pick_rewards = [0.0] * num_agents

    s_next, next_actor_idx = env.advance_actor(actor_idx, num_agents)

    return s_moved, s_next, pick_rewards, on_apple, next_actor_idx


def make_env(reward_module, p_apple, d_apple, apples=None, agents=None, agent_positions=None):
    return Orchard(W, W, NUM_AGENTS, reward_module, p_apple=p_apple, d_apple=d_apple,
                   start_apples_map=apples, start_agents_map=agents, start_agent_positions=agent_positions)


def eval_performance(
        agent_controller,
        env,
        timesteps=5000,
        num_agents=NUM_AGENTS,
):
    reward = 0

    # Init state
    curr_state = dict(env.get_state())
    curr_state["actor_id"] = 0
    actor_idx = 0

    for sec in range(timesteps):
        for _ in range(NUM_AGENTS):
            new_pos = agent_controller.agent_get_action(env, actor_idx)
            s_moved, s_next, pick_rewards, on_apple, next_actor_idx = env_step(
                env, actor_idx, new_pos, num_agents
            )

            if on_apple:
                reward += 1

            actor_idx = next_actor_idx

        if sec % 1000 == 0:
            print(sec)
    assert env.total_picked == reward
    print("Results")
    print("Reward: ", reward)
    print("Apples per agent:", reward / NUM_AGENTS)
    print("Average Reward: ", reward / env.apples_spawned)
    print("Total apples: ", env.apples_spawned)
    # print(
    #     "Picked vs Spawned per agent",
    #     (reward / NUM_AGENTS) / (env.total_apples / NUM_AGENTS),
    #     )
    return {
        "greedy_pps": reward,
        "total_apples": env.apples_spawned
    }
