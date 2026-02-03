import numpy as np


class Value:
    def __init__(self, picker_r, num_agents, discount, prob_apple, variance=0.0):
        self.picker_r = float(picker_r)          # r_pick
        self.num_agents = int(num_agents)        # N
        self.gamma = float(discount)             # γ
        self.P = float(prob_apple)               # P
        self.variance = float(variance)

        if self.num_agents <= 1:
            raise ValueError("num_agents must be >= 2")
        if self.variance < 0:
            raise ValueError("variance must be >= 0")

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

    def _maybe_add_noise(self, v: np.ndarray) -> np.ndarray:
        """If variance > 0, return Normal(mean=v, var=variance); else return v."""
        if self.variance == 0.0:
            return v
        std = np.sqrt(self.variance)
        return np.random.normal(loc=v, scale=std, size=v.shape).astype(v.dtype, copy=False)

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
            res[:] = self.V_Z0
            res[actor_id] = self.V_Z1
            return self._maybe_add_noise(res)

        if mode != 1:
            raise ValueError(f"Unknown mode={mode}")

        r, c = agent_positions[actor_id]
        on_apple = state["apples"][r, c] >= 1

        if on_apple:
            res[:] = self.other_r + self.mode1_base
            res[actor_id] = self.picker_r + self.mode1_base
        else:
            res[:] = self.mode1_base

        return self._maybe_add_noise(res)
