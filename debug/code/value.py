from config import DISCOUNT_FACTOR, PROBABILITY_APPLE
import numpy as np


# class Value:
#     def __init__(self, picker_r, num_agents):
#         self.picker_r = picker_r
#         self.other_r = (1 - picker_r) / (num_agents - 1)
#         self.num_agents = num_agents
#
#         self.future_value = (DISCOUNT_FACTOR * (
#                 picker_r * (1.0 / num_agents) * PROBABILITY_APPLE
#                 + self.other_r * (1.0 - 1.0 / num_agents) * PROBABILITY_APPLE
#         )) / (1.0 - DISCOUNT_FACTOR)
#
#     def theoretical_value(self, state: dict, actor_id: int, agent_positions):
#         res = np.zeros(self.num_agents)
#
#         r, c = agent_positions[actor_id]
#         on_apple = state["apples"][r, c] >= 1
#
#         if on_apple:
#             res[:] = self.other_r + self.future_value
#             res[actor_id] = self.picker_r + self.future_value
#         else:
#             res[:] = self.future_value
#
#         return res

class Value:
    def __init__(self, picker_r, num_agents, discount, prob_apple):
        self.picker_r = float(picker_r)          # r_pick
        self.num_agents = int(num_agents)        # N
        self.gamma = float(discount)             # γ
        self.P = float(prob_apple)               # P

        if self.num_agents <= 1:
            raise ValueError("num_agents must be >= 2")

        # Your reward scheme: distribute total reward 1 across agents when an apple event happens.
        self.other_r = (1.0 - self.picker_r) / (self.num_agents - 1)

        # Closed-form constants for the 2-mode supervised environment (from your write-up)
        exp_reward_given_apple = (
                (1.0 / self.num_agents) * self.picker_r
                + (1.0 - 1.0 / self.num_agents) * self.other_r
        )

        # M = E[value of next mode-0 state after a mode-1 step]
        self.M = (self.gamma * self.P * exp_reward_given_apple) / (1.0 - self.gamma ** 2)

        # Cache the two mode-0 state-type values (actor vs non-actor)
        self.V_Z1 = (self.gamma ** 2) * self.M + self.gamma * self.P * self.picker_r
        self.V_Z0 = (self.gamma ** 2) * self.M + self.gamma * self.P * self.other_r

        # Cache the mode-1 base term
        self.mode1_base = self.gamma * self.M

    def theoretical_value(self, state: dict, actor_id: int, agent_positions):
        """
        Returns a vector res[j] = V_j(state) for all agents j, under the
        2-mode supervised dynamics in the PDF.

        Assumptions (as in the write-up):
        - Mode alternates deterministically: 0 -> 1 -> 0 -> ...
        - Only mode 1 can give nonzero reward.
        - Apples relevant for reward are checked in mode 1 under the acting agent.
        - From mode 0, apples are (re)sampled before the subsequent mode-1 reward check,
          so the *current* apple grid in mode 0 should NOT affect the value.
        """
        res = np.zeros(self.num_agents, dtype=np.float32)

        mode = int(state["mode"])  # adjust key if yours differs

        if mode == 0:
            # Mode 0: reward is always 0, actor persists to mode 1, apple under actor will be Bernoulli(P)
            res[:] = self.V_Z0
            res[actor_id] = self.V_Z1
            return res

        if mode != 1:
            raise ValueError(f"Unknown mode={mode}")

        # Mode 1: reward depends on whether there is an apple under the acting agent now
        r, c = agent_positions[actor_id]
        on_apple = state["apples"][r, c] >= 1

        if on_apple:
            res[:] = self.other_r + self.mode1_base
            res[actor_id] = self.picker_r + self.mode1_base
        else:
            res[:] = self.mode1_base

        return res
