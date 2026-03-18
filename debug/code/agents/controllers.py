from abc import abstractmethod

import numpy as np
import random
import torch

from debug.code.nn.encoders import BaseEncoder, EncoderOutput, stack_encoder_outputs
from debug.code.env.environment import MoveAction
from debug.code.training.helpers import random_policy


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

    @abstractmethod
    def _score_candidate_positions(
        self,
        base_state: dict,
        actor_id: int,
        candidate_positions: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def _candidate_positions(self, env, actor_id: int) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(env.agent_positions[actor_id], dtype=np.int64)
        candidate_positions: list[np.ndarray] = []

        for act in MoveAction:
            nr = int(position[0] + act.vector[0])
            nc = int(position[1] + act.vector[1])
            if 0 <= nr < env.width and 0 <= nc < env.length:
                candidate_positions.append(np.array([nr, nc], dtype=np.int64))

        return position.copy(), np.asarray(candidate_positions, dtype=np.int64)

    def _build_candidate_states(
        self,
        base_state: dict,
        actor_id: int,
        position: np.ndarray,
        candidate_positions: np.ndarray,
    ) -> list[dict]:
        base_agents = base_state["agents"]
        base_apples = base_state["apples"]
        base_positions = base_state["agent_positions"]
        pr, pc = map(int, position)

        states: list[dict] = []
        for new_pos in candidate_positions:
            nr, nc = map(int, new_pos)
            agents = base_agents.copy()
            agent_positions = base_positions.copy()
            agents[nr, nc] += 1
            agents[pr, pc] -= 1
            agent_positions[actor_id] = new_pos
            states.append(
                {
                    "agents": agents,
                    "apples": base_apples,
                    "agent_positions": agent_positions,
                    "actor_id": actor_id,
                }
            )
        return states

    def _encode_candidate_batch(
        self,
        base_state: dict,
        actor_id: int,
        position: np.ndarray,
        candidate_positions: np.ndarray,
        observer_id: int,
        candidate_states: list[dict] | None,
    ) -> tuple[EncoderOutput, list[dict] | None]:
        fast_batch = getattr(self.encoder, "encode_candidate_positions", None)
        if callable(fast_batch):
            return fast_batch(base_state, observer_id, candidate_positions), candidate_states

        if candidate_states is None:
            candidate_states = self._build_candidate_states(base_state, actor_id, position, candidate_positions)

        outputs = [self.encoder.encode(state, observer_id) for state in candidate_states]
        return stack_encoder_outputs(outputs), candidate_states

    def get_best_action(self, env, actor_id):
        position, candidate_positions = self._candidate_positions(env, actor_id)
        if candidate_positions.size == 0:
            return position

        base_state = env.get_state()
        base_state["actor_id"] = actor_id

        values = self.discount * self._score_candidate_positions(base_state, actor_id, candidate_positions)
        best_idx = int(np.argmax(values))
        return candidate_positions[best_idx]

    def agent_get_action(self, env, agent_id, epsilon=None):
        eps = epsilon if epsilon is not None else self.epsilon
        if random.random() < eps:
            return random_policy(env.agent_positions[agent_id], width=env.width, length=env.length)
        with torch.no_grad():
            return self.get_best_action(env, agent_id)


class AgentControllerDecentralized(AgentControllerValue):
    def get_collective_value(self, encoded_states: list[EncoderOutput], agent_id: int) -> float:
        total = 0.0
        for agent, enc in zip(self.agents_list, encoded_states):
            total += agent.policy_value.get_value_function(enc)
        return total

    def _score_candidate_positions(
        self,
        base_state: dict,
        actor_id: int,
        candidate_positions: np.ndarray,
    ) -> np.ndarray:
        position = np.asarray(base_state["agent_positions"][actor_id], dtype=np.int64)
        values = np.zeros(int(candidate_positions.shape[0]), dtype=np.float64)
        candidate_states = None

        for observer_id, agent in enumerate(self.agents_list):
            batched_enc, candidate_states = self._encode_candidate_batch(
                base_state,
                actor_id,
                position,
                candidate_positions,
                observer_id,
                candidate_states,
            )
            values += np.asarray(agent.policy_value.get_value_function_batch(batched_enc), dtype=np.float64)

        return values


class AgentControllerCentralized(AgentControllerValue):
    def get_collective_value(self, states, agent_id):
        return self.agents_list[0].get_value_function(states[0])

    def get_all_agent_obs(self, state: dict) -> list[EncoderOutput]:
        return [self.encoder.encode(state, 0)]

    def _score_candidate_positions(
        self,
        base_state: dict,
        actor_id: int,
        candidate_positions: np.ndarray,
    ) -> np.ndarray:
        position = np.asarray(base_state["agent_positions"][actor_id], dtype=np.int64)
        batched_enc, _ = self._encode_candidate_batch(
            base_state,
            actor_id,
            position,
            candidate_positions,
            observer_id=0,
            candidate_states=None,
        )
        return np.asarray(self.agents_list[0].policy_value.get_value_function_batch(batched_enc), dtype=np.float64)
