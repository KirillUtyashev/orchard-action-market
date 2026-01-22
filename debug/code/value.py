from config import DISCOUNT_FACTOR, PROBABILITY_APPLE
import numpy as np


class Value:
    def __init__(self, picker_r, num_agents):
        self.picker_r = picker_r
        self.other_r = (1 - picker_r) / (num_agents - 1)
        self.num_agents = num_agents

        self.future_value = (DISCOUNT_FACTOR * (
                picker_r * (1.0 / num_agents) * PROBABILITY_APPLE
                + self.other_r * (1.0 - 1.0 / num_agents) * PROBABILITY_APPLE
        )) / (1.0 - DISCOUNT_FACTOR)

    def theoretical_value(self, state: dict, actor_id: int, agent_positions):
        res = np.zeros(self.num_agents)

        r, c = agent_positions[actor_id]
        on_apple = state["apples"][r, c] >= 1

        if on_apple:
            res[:] = self.other_r + self.future_value
            res[actor_id] = self.picker_r + self.future_value
        else:
            res[:] = self.future_value

        return res

