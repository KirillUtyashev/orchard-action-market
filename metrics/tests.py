from metrics import avg_apples_picked, avg_apples_on_field, distance_to_nearest_apple, average_agent_distance
import numpy as np

state = {
    "agents": np.array([[0, 0, 1, 0, 1, 0, 1]]),
    "apples": np.array([[0, 0, 0, 0, 1, 0, 0]])
}
total_reward = 0
timestep = 5

assert distance_to_nearest_apple(state, total_reward, timestep) == 4/3
assert average_agent_distance(state, total_reward, timestep) == 2

state = {
    "agents": np.array([[1, 0, 0, 0, 0, 0, 3]]),
    "apples": np.array([[1, 0, 0, 0, 0, 0, 0]])
}
total_reward = 0
timestep = 5

assert distance_to_nearest_apple(state, total_reward, timestep) == 18/4
assert average_agent_distance(state, total_reward, timestep) == 6/4
