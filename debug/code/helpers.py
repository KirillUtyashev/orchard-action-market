import random
import torch
import os
import numpy as np
from orchard.environment import (
    Orchard,
    OrchardBasic,
    OrchardBasicNewDynamic,
    OrchardEuclideanNegativeRewardsNewDynamic,
    OrchardEuclideanRewardsNewDynamic,
)


same_cell_no_reward = 0
count = 0


def create_env(
    env_config,
    num_agents,
    agent_pos,
    apples,
    agents_list,
    env_cls=OrchardBasic,
    debug=False,
):
    env = env_cls(
        env_config.length,
        env_config.width,
        num_agents,
        agents_list,
        s_target=env_config.s_target,
        apple_mean_lifetime=env_config.apple_mean_lifetime,
        debug=debug,
    )
    env.initialize(agents_list, agent_pos=agent_pos, apples=apples)
    return env


def set_all_seeds(seed: int = 42, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch (CPU & CUDA) for reproducibility.

    Args:
        seed: The seed value to use.
        deterministic: If True, enable extra settings for deterministic
                       behavior in cuDNN at the possible cost of speed.
    """
    print(f"[seed] Using seed={seed}")

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

def convert_position(pos):
    if pos is not None:
        temp = np.zeros((len(pos), 1), dtype=int)
        for i in range(len(pos)):
            temp[i] = np.array(pos[i])
        return temp
    return None


def get_empty_fields(env_length, env_width):
    return {
        "agents": np.zeros((env_width, env_length), dtype=int),
        "apples": np.zeros((env_width, env_length), dtype=int),
        "poses": np.zeros((env_width, env_length), dtype=int),
    }


def generate_sample_states(env_length, env_width, num_agents, alt_vision=False):
    """
    Create three sample states. In state i (i=0,1,2), all agents are placed at (row=0, col=i+1),
    and a single apple is placed at (0, 0). Returns a tuple of three state dicts.

    Assumes get_empty_fields(L, W) -> dict with keys:
      - "agents": 2D array-like [L x W] (agent count per cell)
      - "apples": 2D array-like [L x W]
      - "poses":  array of shape [num_agents, 2] (row, col) for each agent
    """
    if num_agents < 0:
        raise ValueError("num_agents must be non-negative")
    # Need columns 1,2,3 => require width >= 4 (since we index i+1)
    if env_width < 4:
        raise ValueError(
            "env_width must be at least 4 (to place agents at columns 1..3)"
        )
    if env_length < 1:
        raise ValueError("env_length must be >= 1")

    states = []
    for i in range(3):
        s = get_empty_fields(env_length, env_width)

        # Place all agents at (0, i+1)
        col = i + 1
        # Use numpy-style indexing if arrays, but remain compatible with lists
        s["agents"][0][col] = num_agents
        s["poses"] = (
            np.repeat([[0, col]], repeats=num_agents, axis=0)
            if num_agents > 0
            else np.empty((0, 2), dtype=int)
        )

        # One apple at (0, 0)
        s["apples"][0][0] = 1

        states.append(s)

    return tuple(states)


def generate_alt_states(env_length, num_agents):
    res = [
        get_empty_fields(env_length),
        get_empty_fields(env_length),
        get_empty_fields(env_length),
    ]
    for i in range(len(res)):
        res[i]["agents"][i + 1] = np.array(num_agents)


def get_discounted_value(old, new, discount_factor=0.05):
    return old * (1 - discount_factor) + new * discount_factor


def step(agents_list, environment: Orchard, agent_controller, epsilon, inference=False):
    agent_idx = random.randint(0, len(agents_list) - 1)
    agent = agents_list[agent_idx]
    action = None
    # if agents_list[agent].policy is not random_policy:
    #     action = agent_controller.agent_get_action(environment, agent, epsilon)
    # else:
    #     action = random_policy(environment.available_actions)
    if agent_controller is not None:
        action = agent_controller.agent_get_action(environment, agent_idx, epsilon)
    else:
        state = environment.get_state()
        action = agent.policy(state, agent.position, environment.available_actions)
        # action = random_policy(environment.available_actions)
    environment.process_action_eval(agent_idx, agent.position.copy(), action)

    if (
        isinstance(environment, OrchardBasicNewDynamic)
        or isinstance(environment, OrchardEuclideanRewardsNewDynamic)
        or isinstance(environment, OrchardEuclideanNegativeRewardsNewDynamic)
    ):
        environment.remove_apple(agents_list[agent_idx].position.copy())

