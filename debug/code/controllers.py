import copy
from abc import abstractmethod

from debug.code.config import W
from debug.code.environment import MoveAction
import numpy as np
import random

import torch

from debug.code.helpers import random_policy


class AgentController:
    def __init__(self, agents, critic_view_controller, discount):
        self.agents_list = agents
        self.critic_view_controller = critic_view_controller
        self.discount = discount

    @abstractmethod
    def get_best_action(self, env, agent_id):
        raise NotImplementedError

    def get_agent_obs(self, state, agent_id=None):
        return self.critic_view_controller.state_to_nn_input(state, agent_id)

    def get_all_agent_obs(self, state):
        obs = []
        for agent in range(len(self.agents_list)):
            obs.append(self.get_agent_obs(state, agent))
        return obs

    @abstractmethod
    def get_collective_value(self, states, agent_id):
        raise NotImplementedError

    @abstractmethod
    def agent_get_action(self, env, agent_id, epsilon=None):
        raise NotImplementedError


class AgentControllerValue(AgentController):
    def __init__(self, agents, critic_view_controller, discount):
        super().__init__(agents, critic_view_controller, discount)

    @abstractmethod
    def get_collective_value(self, processed_states, agent_id) -> float:
        pass

    def get_best_action(self, env, actor_id):
        action = env.agent_positions[actor_id]
        best_val = -1000000
        position = env.agent_positions[actor_id]

        for act in MoveAction:

            curr_state = copy.deepcopy(env.get_state())
            r, c = curr_state["agent_positions"][actor_id]
            nr = r + act.vector[0]
            nc = c + act.vector[1]
            if not (0 <= nr < W and 0 <= nc < W):
                continue

            new_pos = np.array([nr, nc])
            curr_state["agents"][tuple(new_pos)] += 1
            curr_state["agents"][tuple(position)] -= 1

            curr_state["agent_positions"][actor_id] = new_pos

            curr_state["mode"] = 0
            curr_state["actor_id"] = actor_id

            observations = self.get_all_agent_obs(curr_state)
            val = self.discount * self.get_collective_value(
                observations, actor_id
            )
            if val > best_val:
                action = new_pos
                best_val = val
        return action

    def agent_get_action(self, env, agent_id, epsilon=0.1):
        action = None
        if random.random() < epsilon:
            action = random_policy(env.agent_positions[agent_id])
        else:
            with torch.no_grad():
                action = self.get_best_action(env, agent_id)
        return action


class AgentControllerDecentralized(AgentControllerValue):
    def get_collective_value(self, states, agent_id):
        sum_ = 0
        for num, agent in enumerate(self.agents_list):
            value = agent.get_value_function(states[num])
            sum_ += value
        return sum_


class ViewControllerDec:
    def __init__(self, input_dim, k):
        self.input_dim = input_dim
        self.k = k

    def state_to_nn_input(self, state, agent_id=None) -> np.ndarray:
        """
        Design A (self-centered apples) entity encoding (global observability, fixed-size, egocentric):

          Scalars: [actor_is_self, mode]
          Actor block (relative to self): [dx_actor_norm, dy_actor_norm, dist_actor_norm]
          Other agents (relative to self, deterministic order by id): for each j != self
              [dx_norm, dy_norm, dist_norm]
          Apples (top-K nearest to SELF, relative to self): K slots
              [dx_norm, dy_norm, dist_norm, mask]

        Extra rule to keep reward-relevant info recoverable with top-K:
          - If there is an apple at the actor cell, we force-include (actor_r, actor_c)
            among the K apple slots (even if it is far from self), so the network can infer
            "apple under actor" by checking whether any apple-slot (dx,dy) matches
            (dx_actor,dy_actor). [file:1]
        """
        if agent_id is None:
            raise ValueError("agent_id must be provided")

        actor_id = state["actor_id"]
        agent_positions = state["agent_positions"]  # index -> (r,c)
        self_r, self_c = agent_positions[agent_id]
        actor_r, actor_c = agent_positions[actor_id]

        # ---- keep your existing encodings unchanged ----
        if self.input_dim == 326:
            apples_matrix = state["apples"].copy()
            agents_matrix = state["agents"].copy()

            other_agents = agents_matrix.copy()
            other_agents[self_r, self_c] = max(0, other_agents[self_r, self_c] - 1)
            if actor_id != agent_id:
                other_agents[actor_r, actor_c] = max(0, other_agents[actor_r, actor_c] - 1)

            self_pos = np.zeros_like(agents_matrix)
            self_pos[self_r, self_c] = 1

            actor_pos = np.zeros_like(agents_matrix)
            actor_pos[actor_r, actor_c] = 1

            actor_is_self = 1 if actor_id == agent_id else 0
            mode = int(state["mode"])

            features = [
                apples_matrix.flatten(),
                other_agents.flatten(),
                self_pos.flatten(),
                actor_pos.flatten(),
                np.array([actor_is_self, mode], dtype=np.int64),
            ]
            return np.concatenate(features)

        elif self.input_dim == 3:
            actor_is_self = float(actor_id == agent_id)
            mode = float(state["mode"])
            apple_under_actor = float(state["apples"][actor_r, actor_c] >= 1)
            return np.array([actor_is_self, mode, apple_under_actor], dtype=np.float32)

        # ---- Design A entity encoding (self-centered apples) ----
        if not hasattr(self, "k") or self.k is None:
            raise ValueError("For entity encoding, set self.k (top-K apples).")

        apples_matrix = state["apples"]
        H, W = apples_matrix.shape
        denom_x = max(W - 1, 1)
        denom_y = max(H - 1, 1)
        dmax = float(np.sqrt((W - 1) ** 2 + (H - 1) ** 2))
        if dmax <= 0:
            dmax = 1.0

        def rel_norm(r_from, c_from, r_to, c_to):
            """Return (dx_norm, dy_norm, dist_norm) from (from)->(to), using fixed map bounds."""
            dx = c_to - c_from
            dy = r_to - r_from
            dxn = dx / denom_x
            dyn = dy / denom_y
            distn = float(np.sqrt(dx * dx + dy * dy)) / dmax
            return float(dxn), float(dyn), float(distn)

        actor_is_self = 1.0 if actor_id == agent_id else 0.0
        mode = float(int(state["mode"]))
        apple_under_actor = 1.0 if apples_matrix[actor_r, actor_c] > 0 else 0.0  # extra feature [file:1]

        feats = []
        # Scalars: add apple_under_actor
        feats.append(np.array([actor_is_self, mode, apple_under_actor], dtype=np.float32))

        # Actor block: actor position relative to self
        dxn_a, dyn_a, distn_a = rel_norm(self_r, self_c, actor_r, actor_c)
        feats.append(np.array([dxn_a, dyn_a, distn_a], dtype=np.float32))

        # Other agents: relative to self (includes actor too; redundancy is OK)
        for j, (rj, cj) in enumerate(agent_positions):
            if j == agent_id:
                continue
            dxn, dyn, distn = rel_norm(self_r, self_c, rj, cj)
            feats.append(np.array([dxn, dyn, distn], dtype=np.float32))

        # Apples: top-K nearest to SELF, encoded relative to self with mask padding
        apple_rc = np.argwhere(apples_matrix > 0)  # rows are [r, c]

        if apple_rc.size == 0:
            topk = np.empty((0, 2), dtype=np.int64)
        else:
            rs = apple_rc[:, 0]
            cs = apple_rc[:, 1]
            dx = cs - self_c
            dy = rs - self_r
            d2 = dx * dx + dy * dy
            # Deterministic: sort by distance^2, then dx, then dy
            order = np.lexsort((dy, dx, d2))
            topk = apple_rc[order[: self.k]]

        for idx in range(self.k):
            if idx < len(topk):
                r, c = int(topk[idx, 0]), int(topk[idx, 1])
                dxn, dyn, distn = rel_norm(self_r, self_c, r, c)  # relative to SELF
                feats.append(np.array([dxn, dyn, distn, 1.0], dtype=np.float32))
            else:
                feats.append(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

        out = np.concatenate(feats).astype(np.float32)
        return out

    def __call__(self, state, agent_id):
            """Make the controller callable for compatibility with existing code."""
            return self.state_to_nn_input(state, agent_id)


class ViewControllerCen:
    """
    Centralized critic view controller with two modes:

    1) concat=True:
       - Calls ViewControllerDec(state, agent_id=i) for each agent i
       - Concatenates all per-agent NN inputs into a single vector

    2) concat=False (actor_view):
       - Builds ONE global, actor-centric (egocentric) encoding:
         everything is expressed relative to the current actor position.
    """

    def __init__(self, k, concat: bool, dec_controller=None):
        self.k = k
        self.concat = bool(concat)
        self.dec = dec_controller  # expected to be callable: dec(state, agent_id) -> np.ndarray

        if self.concat and self.dec is None:
            # Lazily require ViewControllerDec only in concat mode
            try:
                self.dec = ViewControllerDec(input_dim=0, k=self.k)  # noqa: F821 (ViewControllerDec expected in your codebase)
            except NameError as e:
                raise ValueError(
                    "concat=True requires a dec_controller or a ViewControllerDec in scope."
                ) from e

    def state_to_nn_input(self, state) -> np.ndarray:
        if self.concat:
            return self._concat_dec_views(state)

        # actor-centric centralized input (agent_id not needed)
        return self._actor_view(state)

    def _concat_dec_views(self, state) -> np.ndarray:
        agent_positions = state["agent_positions"]
        n_agents = len(agent_positions)

        parts = []
        for i in range(n_agents):
            parts.append(np.asarray(self.dec(state, i), dtype=np.float32))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _actor_view(self, state) -> np.ndarray:
        if not hasattr(self, "k") or self.k is None:
            raise ValueError("For entity encoding, set self.k (top-K apples).")

        actor_id = int(state["actor_id"])
        agent_positions = state["agent_positions"]  # index -> (r,c)
        apples_matrix = state["apples"]
        H, W = apples_matrix.shape

        actor_r, actor_c = agent_positions[actor_id]

        denom_x = max(W - 1, 1)
        denom_y = max(H - 1, 1)
        dmax = float(np.sqrt((W - 1) ** 2 + (H - 1) ** 2)) or 1.0

        def rel_norm(r_from, c_from, r_to, c_to):
            dx = c_to - c_from
            dy = r_to - r_from
            dxn = dx / denom_x
            dyn = dy / denom_y
            distn = float(np.sqrt(dx * dx + dy * dy)) / dmax
            return float(dxn), float(dyn), float(distn)

        # Scalars
        mode = float(int(state["mode"]))
        n_agents = len(agent_positions)
        actor_id_norm = float(actor_id) / float(max(n_agents - 1, 1))
        apple_under_actor = 1.0 if apples_matrix[actor_r, actor_c] > 0 else 0.0

        feats = []
        feats.append(np.array([mode, actor_id_norm, apple_under_actor], dtype=np.float32))

        # Agents: relative to ACTOR, deterministic order by id
        for j, (rj, cj) in enumerate(agent_positions):
            dxn, dyn, distn = rel_norm(actor_r, actor_c, rj, cj)
            is_actor = 1.0 if j == actor_id else 0.0
            feats.append(np.array([dxn, dyn, distn, is_actor], dtype=np.float32))

        # Apples: top-K nearest to ACTOR, relative to actor, with mask padding
        apple_rc = np.argwhere(apples_matrix > 0)  # rows: [r, c]

        if apple_rc.size == 0:
            topk = np.empty((0, 2), dtype=np.int64)
        else:
            rs = apple_rc[:, 0]
            cs = apple_rc[:, 1]
            dx = cs - actor_c
            dy = rs - actor_r
            d2 = dx * dx + dy * dy
            # Deterministic: sort by distance^2, then dx, then dy
            order = np.lexsort((dy, dx, d2))
            topk = apple_rc[order[: self.k]]

        for idx in range(self.k):
            if idx < len(topk):
                r, c = int(topk[idx, 0]), int(topk[idx, 1])
                dxn, dyn, distn = rel_norm(actor_r, actor_c, r, c)
                feats.append(np.array([dxn, dyn, distn, 1.0], dtype=np.float32))
            else:
                feats.append(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

        return np.concatenate(feats).astype(np.float32)

    def __call__(self, state, agent_id=None):
        return self.state_to_nn_input(state)
