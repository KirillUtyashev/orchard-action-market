import numpy as np
from tqdm import tqdm
from notebooks.Compare_to_True_value.env_functions import (
    State,
    despawn_apples,
    init_empty_state,
    place_agents_randomly,
    spawn_apples,
)
from orchard.environment import Action2D


def get_true_value_random_policy(
    states_to_evaluate: list[State],
    num_agents: int,
    spawn_prob_per_cell: float,
    despawn_prob_per_cell: float,
    discount_factor: float,
    simulate_step
):
    NUM_TO_AVERAGE_OVER = 100
    for state in states_to_evaluate:
        total_values = []
        initial_agent_positions = np.argwhere(state.agents == 1)
        for n in tqdm(range(NUM_TO_AVERAGE_OVER), desc=f"iterating over {state.name}"):
            agent_positions = initial_agent_positions.copy()
            s_t = state
            total_value = 0
            discount_factor = 1.0
            for t in range(1000):
                c = np.random.randint(0, num_agents)
                action = Action2D.get_random_action()

                r_t, s_t_plus_1, agent_positions = simulate_step(
                    s_t, c, agent_positions, action.vector
                )
                spawn_apples(s_t_plus_1, spawn_prob_per_cell)
                despawn_apples(s_t_plus_1, despawn_prob_per_cell)

                total_value += discount_factor * r_t
                discount_factor *= discount_factor
                s_t = s_t_plus_1
            total_values.append(total_value)
        avg_value = np.mean(total_values)
        print(
            f"Estimated true value of state:\n{state.name}\n is approximately: {avg_value}\n"
        )
