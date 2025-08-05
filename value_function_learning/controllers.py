from abc import abstractmethod

from agents.agent import calculate_ir
from config import get_config
from helpers import unwrap_state, convert_position
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
        sum_ = 0
        for num, agent in enumerate(self.agents_list):
            sum_ += agent.get_value_function(states[num])
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

    def collective_value_from_state(self, state, positions):
        observations = self.get_all_agent_obs(state, positions)
        return self.get_collective_value(observations, None)


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

        height, length = agents.shape

        if self.perfect_info:
            agents_vector = agents[0].reshape(-1, 1)
            apples_vector = apples[0].reshape(-1, 1)
            for i in range(1, height):
                agents_vector = np.concatenate([agents_vector, agents[i].reshape(-1, 1)], axis=0)
                apples_vector = np.concatenate([apples_vector, apples[i].reshape(-1, 1)], axis=0)

            res = np.concatenate([agents_vector, apples_vector], axis=0)
            if agent_pos is not None:
                pos = convert_position(agent_pos)
                res = np.concatenate([res, pos], axis=0)
            return res
        else:
            half = self.vision // 2

            fill = np.full((half, len(agents[0])), -1, dtype=int)

            if height != 1:
                ap = np.concatenate((fill, agents, fill))
                bp = np.concatenate((fill, apples, fill))
            else:
                ap = agents
                bp = apples

            # build a column‐fill of shape (ap.shape[0], pad)
            col_fill = np.full((ap.shape[0], half), -1, dtype=int)

            # now pad left and right:
            ap = np.concatenate((col_fill, ap, col_fill), axis=1)
            bp = np.concatenate((col_fill, bp, col_fill), axis=1)
            r, c = agent_pos

            # get initial coordinate
            r_true, c_true = r + half, c + half

            true_a = ap[r_true - half:r_true + half + 1, c_true - half:c_true + half + 1]
            true_b = bp[r_true - half:r_true + half + 1, c_true - half:c_true + half + 1]

            if height != 1:
                agents_vector = true_a[0].reshape(-1, 1)
                apples_vector = true_b[0].reshape(-1, 1)
                for i in range(1, self.vision):
                    agents_vector = np.concatenate([agents_vector, true_a[i].reshape(-1, 1)], axis=0)
                    apples_vector = np.concatenate([apples_vector, true_b[i].reshape(-1, 1)], axis=0)
                res = np.concatenate([agents_vector, apples_vector], axis=0)
            else:
                res = np.concatenate([true_a.T, true_b.T], axis=0)
            if agent_pos is not None:
                pos = convert_position(agent_pos)
                res = np.concatenate([res, pos], axis=0)
            return res
