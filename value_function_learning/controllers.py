import math
import random
from abc import abstractmethod

from agents.agent import calculate_ir
from config import get_config
from helpers import get_discounted_value, unwrap_state, convert_position
import numpy as np


class AgentController:
    def __init__(self, agents, view_controller):
        self.agents_list = agents
        self.view_controller = view_controller

    def get_best_action(self, state, agent_id, available_actions):
        agents, apples = unwrap_state(state)
        action = available_actions.STAY
        best_val = 0

        for act in available_actions:
            val, new_a, new_b, new_pos = calculate_ir(agents, apples, self.agents_list[agent_id].position, act.vector)
            positions = []
            for agent in range(len(self.agents_list)):
                if agent != agent_id:
                    positions.append(self.agents_list[agent].position)
                else:
                    positions.append(new_pos)
            observations = self.get_all_agent_obs({"agents": new_a, "apples": new_b}, positions)
            val += get_config()["discount"] * self.get_collective_value(observations, agent_id)
            if val > best_val:
                action = act
                best_val = val
        return action.idx

    def get_agent_obs(self, state, agent_pos):
        return self.view_controller.process_state(state, agent_pos)

    def get_all_agent_obs(self, state, positions):
        obs = []
        for agent in range(len(self.agents_list)):
            obs.append(self.get_agent_obs(state, positions[agent]))
        return obs

    @abstractmethod
    def get_collective_value(self, states, agent_id):
        raise NotImplementedError


class AgentControllerDecentralized(AgentController):
    def __init__(self, agents, view_controller):
        super().__init__(agents, view_controller)

    def get_collective_value(self, states, agent_id):
        # sum_ = self.agents_list[agent_id].get_value_function(states[agent_id])
        sum_ = 0
        for num, agent in enumerate(self.agents_list):
            value = agent.get_value_function(states[num])
            sum_ += value
        return sum_


class AgentControllerCentralized(AgentController):
    def __init__(self, agents, view_controller):
        super().__init__(agents, view_controller)

    def get_collective_value(self, states, agent_id):
        return self.agents_list[0].get_value_function(states[agent_id])


class AgentControllerActorCritic(AgentControllerDecentralized):
    def __init__(self, agents, view_controller):
        super().__init__(agents, view_controller)

    def get_best_action(self, state, agent_id, available_actions):
        probs = self.agents_list[agent_id].policy_network.get_function_output(self.view_controller.process_state(state, self.agents_list[agent_id].position))
        action = np.random.choice(len(probs), p=probs)
        return action

    def collective_value_from_state(self, state, positions, agent_id=None):
        observations = self.get_all_agent_obs(state, positions)
        return self.get_collective_value(observations, agent_id)

    def get_collective_value(self, states, agent_id,):
        sum_ = 0
        for num, agent in enumerate(self.agents_list):
            value = agent.get_value_function(states[num])
            sum_ += value
        return sum_


class AgentControllerActorCriticRatesFixed(AgentControllerActorCritic):
    def __init__(self, agents, view_controller):
        super().__init__(agents, view_controller)

    def get_collective_value(self, states, agent_id):
        sum_ = 0
        for num, agent in enumerate(self.agents_list):
            if num != agent_id:
                sum_ += agent.get_value_function(states[num]) * agent.agent_observing_probabilities[agent_id]
            else:
                sum_ += agent.get_value_function(states[num])
        return sum_


class AgentControllerActorCriticRates(AgentControllerActorCritic):
    def __init__(self, agents, view_controller):
        super().__init__(agents, view_controller)

    def get_collective_advantage(self, state, positions, new_state, new_positions, agent_id=None):
        new_observations = self.get_all_agent_obs(new_state, new_positions)
        old_observations = self.get_all_agent_obs(state, positions)
        sum_ = 0
        for num, agent in enumerate(self.agents_list):
            if num != agent_id:
                q_value = get_config()["discount"] * agent.get_value_function(new_observations[num])
                v_value = agent.get_value_function(old_observations[num])
                agent.agent_alphas[agent_id] = get_discounted_value(agent.agent_alphas[agent_id], q_value - v_value, agent.rate)
                sum_ += (q_value - v_value) * agent.agent_observing_probabilities[agent_id]
            else:
                sum_ += agent.get_value_function(new_observations[num]) - agent.get_value_function(old_observations[num])
        return sum_

    def collective_value_from_state(self, state, positions, agent_id=None, discount=None):
        observations = self.get_all_agent_obs(state, positions)
        return self.get_collective_value(observations, agent_id, discount)

    def get_collective_value(self, states, agent_id, discount=None):
        sum_ = 0
        for num, agent in enumerate(self.agents_list):
            if num != agent_id:
                value = agent.get_value_function(states[num])
                agent.agent_alphas[agent_id] = get_discounted_value(agent.agent_alphas[agent_id], get_config()["discount"] * value.item(), agent.rate)
                sum_ += value * agent.agent_observing_probabilities[agent_id]
            else:
                sum_ += agent.get_value_function(states[num])
        return sum_


class ViewController:
    def __init__(self, vision=None):
        if vision == 0:
            self.perfect_info = True
            self.vision = vision
        else:
            self.perfect_info = False
            self.vision = vision

    def process_state(self, state, agent_pos):
        agents, apples = unwrap_state(state)
        H, W = agents.shape

        # Helper to append agent_pos if provided
        def append_pos(vec):
            if agent_pos is None:
                return vec
            pos = convert_position(agent_pos)
            return np.concatenate((vec, pos), axis=0)

        if self.perfect_info:
            # Flatten once; no per-row concatenation
            res = np.concatenate(
                (agents.ravel()[:, None],
                 apples.ravel()[:, None]),
                axis=0
            )
            return append_pos(res)

        # Partial info path
        half = self.vision // 2

        # Pad horizontally always; pad vertically only if H != 1 (to match your original logic)
        pad_y = (half, half) if H != 1 else (0, 0)
        pad_x = (half, half)

        ap = np.pad(agents, (pad_y, pad_x), mode='constant', constant_values=-1)
        bp = np.pad(apples, (pad_y, pad_x), mode='constant', constant_values=-1)

        r, c = agent_pos
        r_true = r + pad_y[0]
        c_true = c + half

        # Extract local vision window
        rs = slice(r_true - half, r_true + half + 1)
        cs = slice(c_true - half, c_true + half + 1)
        true_a = ap[rs, cs]
        true_b = bp[rs, cs]

        # Vectorize without loops; preserve your 1D handling
        if H != 1:
            av = true_a.ravel()[:, None]
            bv = true_b.ravel()[:, None]
        else:
            # match your original: use transposed window so shape is (2*half+1, 1)
            av = true_a.T.ravel()[:, None]
            bv = true_b.T.ravel()[:, None]

        res = np.concatenate((av, bv), axis=0)
        return append_pos(res)

