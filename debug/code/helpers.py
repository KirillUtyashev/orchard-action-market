import random
import torch
import os
import numpy as np

from debug.code.config import NUM_AGENTS, W
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


def transition(step, curr_state, env, actor_idx, new_pos):
    if step == -1:
        # init-only: do NOT mutate env
        if curr_state is None:
            curr_state = dict(env.get_state())
        if actor_idx is None:
            # actor_idx = random.randint(0, NUM_AGENTS - 1)
            actor_idx = 0
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
    # actor_idx = random.randint(0, NUM_AGENTS - 1)
    actor_idx = step % 2
    final_state["actor_id"] = actor_idx
    final_state["mode"] = 0

    # return final_state as the next curr_state
    return final_state, semi_state, res, actor_idx


def make_env(reward_module, p_apple, d_apple, apples=None, agents=None, agent_positions=None):
    return Orchard(W, W, NUM_AGENTS, reward_module, p_apple=p_apple, d_apple=d_apple,
                   start_apples_map=apples, start_agents_map=agents, start_agent_positions=agent_positions)


def eval_performance(
        agent_controller,
        env,
        timesteps=10000,
        agents_list=None,
):
    reward = 0

    # Calculate mean distance between agents and their nearest neighbors at each timestep
    nearest_neighbour_mean_distance = []

    num_of_apples_per_second = []

    # Function to compute distance between two points
    def distance(pos1, pos2):
        return np.sqrt(np.sum((pos1 - pos2) ** 2))

    # Function to find nearest neighbor distance for one agent
    def get_nearest_neighbor_distance(agent_idx, agents):
        current_pos = agents[agent_idx].position
        distances = []
        for i, other_agent in enumerate(agents):
            if i != agent_idx:
                distances.append(distance(current_pos, other_agent.position))
        return min(distances) if distances else float("inf")

    os.makedirs("positions", exist_ok=True)
    curr_state = None
    actor_idx = None
    new_pos = None
    for sec in range(timesteps):
        for step in range(-1, NUM_AGENTS):
            if step != -1:
                new_pos = agent_controller.agent_get_action(env, actor_idx)
            final_state, semi_state, res, actor_idx = transition(step, curr_state, env, actor_idx, new_pos)
            if res is not None and sum(res.reward_vector) != 0.0:
                reward += 1
            curr_state = final_state
        # IGNORE END
        if sec % 1000 == 0:
            print(sec)
    print("Results")
    print("Reward: ", reward)
    print("Apples per agent:", reward / NUM_AGENTS)
    print("Average Reward: ", reward / env.apples_spawned)
    print("Total apples: ", env.apples_spawned)
    # print(
    #     "Picked vs Spawned per agent",
    #     (reward / NUM_AGENTS) / (env.total_apples / NUM_AGENTS),
    #     )
    return (
        reward
    )
