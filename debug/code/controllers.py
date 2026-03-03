import copy
from abc import abstractmethod

import numpy as np
import random
import torch

from debug.code.encoders import BaseEncoder, EncoderOutput
from debug.code.enums import W
from debug.code.environment import MoveAction
from debug.code.helpers import random_policy


class AgentController:
    def __init__(self, agents, encoder: BaseEncoder, discount: float, epsilon: float):
        self.agents_list = agents
        self.encoder = encoder
        self.discount = discount
        self.epsilon = epsilon

    @abstractmethod
    def get_best_action(self, env, agent_id):
        raise NotImplementedError

    def get_agent_obs(self, state: dict, agent_id: int) -> EncoderOutput:
        return self.encoder.encode(state, agent_id)

    def get_all_agent_obs(self, state: dict) -> list[EncoderOutput]:
        return [self.encoder.encode(state, i) for i in range(len(self.agents_list))]

    @abstractmethod
    def get_collective_value(self, encoded_states: list[EncoderOutput], agent_id: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def agent_get_action(self, env, agent_id, epsilon=None):
        raise NotImplementedError


class AgentControllerValue(AgentController):
    def __init__(self, agents, encoder: BaseEncoder, discount: float, epsilon: float):
        super().__init__(agents, encoder, discount, epsilon)

    @abstractmethod
    def get_collective_value(self, encoded_states: list[EncoderOutput], agent_id: int) -> float:
        pass

    def get_best_action(self, env, actor_id):
        best_action = env.agent_positions[actor_id]
        best_val = -1_000_000
        position = env.agent_positions[actor_id]

        for act in MoveAction:
            curr_state = copy.deepcopy(env.get_state())
            r, c = curr_state["agent_positions"][actor_id]
            nr, nc = r + act.vector[0], c + act.vector[1]
            if not (0 <= nr < W and 0 <= nc < W):
                continue

            new_pos = np.array([nr, nc])
            curr_state["agents"][tuple(new_pos)] += 1
            curr_state["agents"][tuple(position)] -= 1
            curr_state["agent_positions"][actor_id] = new_pos
            curr_state["actor_id"] = actor_id

            encoded_states = self.get_all_agent_obs(curr_state)
            val = self.discount * self.get_collective_value(encoded_states, actor_id)
            if val > best_val:
                best_action = new_pos
                best_val = val

        return best_action

    def agent_get_action(self, env, agent_id, epsilon=None):
        eps = epsilon if epsilon is not None else self.epsilon
        if random.random() < eps:
            return random_policy(env.agent_positions[agent_id])
        with torch.no_grad():
            return self.get_best_action(env, agent_id)


class AgentControllerDecentralized(AgentControllerValue):
    def get_collective_value(self, encoded_states: list[EncoderOutput], agent_id: int) -> float:
        total = 0.0
        for agent, enc in zip(self.agents_list, encoded_states):
            total += agent.policy_value.get_value_function(enc)
        return total


class AgentControllerCentralized(AgentControllerValue):
    def get_collective_value(self, states, agent_id):
        return self.agents_list[0].get_value_function(states[0])

    def get_all_agent_obs(self, state: dict) -> list[EncoderOutput]:
        return [self.encoder.encode(state, 0)]
