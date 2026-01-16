import numpy as np


class Reward:
    def __init__(self, picker_r, num_agents):
        self.picker_r = picker_r
        self.other_r = (1 - picker_r) / (num_agents - 1)
        self.num_agents = num_agents

    def get_reward(self, state: dict, actor_id, actor_pos):
        res = np.zeros(self.num_agents)
        if state["apples"][tuple(actor_pos)] >= 1:
            res[actor_id] = self.picker_r
            for a in range(self.num_agents):
                if a != actor_id:
                    res[a] = self.other_r
        return res
