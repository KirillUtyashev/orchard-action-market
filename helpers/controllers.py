import random
from abc import abstractmethod

import torch

from policies.nearest import nearest_policy
from policies.random_policy import random_policy
from config import get_config
from helpers.helpers import get_discounted_value, unwrap_state, convert_position
import numpy as np

from policies.random_policy import random_policy


class AgentController:
    def __init__(self, agents, critic_view_controller, actor_view_controller=None):
        self.agents_list = agents
        self.critic_view_controller = critic_view_controller
        self.actor_view_controller = actor_view_controller

    @abstractmethod
    def get_best_action(self, env, agent_id):
        raise NotImplementedError

    def get_agent_obs(self, state, agent_pos, agent_id=None):
        return self.critic_view_controller.process_state(state, agent_pos, agent_id)

    def get_all_agent_obs(self, state, positions):
        obs = []
        for agent in range(len(self.agents_list)):
            obs.append(self.get_agent_obs(state, positions[agent], agent + 1))
        return obs

    @abstractmethod
    def get_collective_value(self, states, agent_id):
        raise NotImplementedError

    @abstractmethod
    def agent_get_action(self, env, agent_id, epsilon=None):
        raise NotImplementedError


class AgentControllerRandom(AgentController):
    def get_best_action(self, env, agent_id):
        return nearest_policy(env.get_state(), self.agents_list[agent_id].position)
        # return random_policy(env.available_actions)

    def get_collective_value(self, states, agent_id):
        pass

    def agent_get_action(self, env, agent_id, epsilon=None):
        return self.get_best_action(env, agent_id)


class AgentControllerValue(AgentController):
    def __init__(self, agents, critic_view_controller, agent_view_controller=None, test=False):
        super().__init__(agents, critic_view_controller, agent_view_controller)
        self.test = test
        if test:
            self.count_random_actions = 0

    @abstractmethod
    def get_collective_value(self, states, agent_id):
        pass

    def get_best_action(self, env, agent_id, communal=True):
        action = env.available_actions.STAY
        best_val = -1000000

        for act in env.available_actions:
            val, new_a, new_b, new_pos = env.calculate_ir(self.agents_list[agent_id].position, act.vector, communal, agent_id)
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

    def agent_get_action(self, env, agent_id, epsilon=0.1):
        action = None
        if random.random() < epsilon:
            action = random_policy(env.available_actions)
            if self.test:
                self.count_random_actions += 1
        else:
            with torch.no_grad():
                action = self.get_best_action(env, agent_id)
        return action


class AgentControllerCentralized(AgentControllerValue):
    def get_collective_value(self, states, agent_id):
        return self.agents_list[0].get_value_function(states[agent_id])

    def get_all_agent_obs(self, state, positions):
        obs = []
        for agent in range(len(self.agents_list)):
            obs.append(self.get_agent_obs(state, None, None))
        return obs


class AgentControllerDecentralized(AgentControllerValue):
    def get_collective_value(self, states, agent_id):
        sum_ = 0
        for num, agent in enumerate(self.agents_list):
            value = agent.get_value_function(states[num])
            sum_ += value
        return sum_


class AgentControllerDecentralizedPersonal(AgentControllerValue):
    def get_collective_value(self, states, agent_id):
        value = self.agents_list[agent_id].get_value_function(states[agent_id])
        return value

    def agent_get_action(self, env, agent_id, epsilon=0.1):
        action = None
        if random.random() < epsilon:
            action = random_policy(env.available_actions)
            if self.test:
                self.count_random_actions += 1
        else:
            with torch.no_grad():
                action = self.get_best_action(env, agent_id, False)
        return action


class AgentControllerActorCritic(AgentControllerDecentralized):
    def __init__(self, agents, critic_view_controller, actor_view_controller):
        super().__init__(agents, critic_view_controller)
        self.actor_view_controller = actor_view_controller

    def get_best_action(self, env, agent_id, communal=True):
        probs = self.agents_list[agent_id].policy_network.get_function_output(self.actor_view_controller.process_state(env.get_state(), self.agents_list[agent_id].position))
        action = np.random.choice(len(probs), p=probs)
        return action

    def collective_value_from_state(self, state, positions, agent_id=None):
        observations = self.get_all_agent_obs(state, positions)
        return self.get_collective_value(observations, agent_id)

    def agent_get_action(self, env, agent_id, epsilon=None) -> int:
        with torch.no_grad():
            action = self.get_best_action(env, agent_id, env.available_actions)
        return action


class AgentControllerActorCriticIndividual(AgentControllerActorCritic):
    def __init__(self, agents, critic_view_controller, actor_view_controller):
        super().__init__(agents, critic_view_controller, actor_view_controller)

    def get_collective_value(self, states, agent_id):
        value = self.agents_list[agent_id].get_value_function(states[agent_id])
        self.agents_list[agent_id].personal_q_value = get_config()["discount"] * value.item()
        return value


class AgentControllerActorCriticRatesFixed(AgentControllerActorCritic):
    def __init__(self, agents, critic_view_controller, actor_view_controller):
        super().__init__(agents, critic_view_controller, actor_view_controller)

    def get_collective_value(self, states, agent_id):
        sum_ = 0
        for num, agent in enumerate(self.agents_list):
            value = agent.get_value_function(states[num])
            if num != agent_id:
                sum_ += value * agent.agent_observing_probabilities[agent_id]
            else:
                sum_ += value
            agent.personal_q_value = get_config()["discount"] * value.item()
        return sum_


class AgentControllerActorCriticRates(AgentControllerActorCritic):
    def __init__(self, agents, critic_view_controller, actor_view_controller):
        super().__init__(agents, critic_view_controller, actor_view_controller)

    def get_collective_advantage(self, state, positions, new_state, new_positions, agent_id=None):
        new_observations = self.get_all_agent_obs(new_state, new_positions)
        old_observations = self.get_all_agent_obs(state, positions)
        sum_ = 0
        for num, agent in enumerate(self.agents_list):
            if num != agent_id:
                q_value = get_config()["discount"] * agent.get_value_function(new_observations[num])
                v_value = agent.get_value_function(old_observations[num])
                agent.agent_alphas[agent_id] = get_discounted_value(agent.agent_alphas[agent_id], q_value, agent.rate)
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


class AgentControllerActorCriticRatesAdvantage(AgentControllerActorCriticRates):
    def __init__(self, agents, critic_view_controller, actor_view_controller):
        super().__init__(agents, critic_view_controller, actor_view_controller)

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


class ViewController:
    def __init__(self, vision=None, new_input=False):
        if vision == 0:
            self.perfect_info = True
            self.vision = vision
        else:
            self.perfect_info = False
            self.vision = vision
        self.new_input = new_input

    def process_state(self, state, agent_pos, agent_id=None):
        agents, apples = unwrap_state(state)
        H, W = agents.shape

        # Helper to append agent_pos if provided
        def append_pos(vec):
            if agent_pos is None:
                return vec
            pos = convert_position(agent_pos)
            return np.concatenate((vec, pos), axis=0)

        if self.perfect_info:
            if not self.new_input or agent_pos is None:
                # Flatten once; no per-row concatenation
                res = np.concatenate(
                    (agents.ravel()[:, None],
                     apples.ravel()[:, None]),
                    axis=0
                )
                return append_pos(res)
            else:
                # new_input = True
                # 1) agents grid without the acting agent
                agents_wo = agents.astype(np.float32).copy()
                if agent_pos is not None:
                    r, c = agent_pos
                    agents_wo[r, c] = max(0.0, agents_wo[r, c] - 1.0)

                agents_col = agents_wo.ravel()[:, None]          # (H*W, 1)

                # 2) full apples vector (same as non-new_input branch)
                apples_col = apples.ravel().astype(np.float32)[:, None]

                # 3) one-hot vector for the acting agent's position (same length as agents vector)
                pos_onehot = np.zeros((H, W), dtype=np.float32)
                if agent_pos is not None:
                    r, c = agent_pos
                    pos_onehot[r, c] = 1.0
                pos_col = pos_onehot.ravel()[:, None]

                # Concatenate: [agents_wo_self, apples_full, onehot_pos]
                res = np.concatenate((agents_col, apples_col, pos_col), axis=0)

                return res

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


class ViewControllerOrchardSelfless(ViewController):
    def process_state(self, state, agent_pos, agent_id=None):
        if self.perfect_info:
            if not self.new_input:
                agents, apples = unwrap_state(state)

                # Helper to append agent_pos if provided
                def append_pos(vec):
                    if agent_pos is None:
                        return vec
                    pos = convert_position(agent_pos)
                    pos = np.asarray(pos, dtype=np.float32).ravel()[:, None]  # (k,1)
                    return np.concatenate((vec, pos), axis=0)

                if agent_id is None:
                    # Centralized value function - call parent class
                    apple_grid = (apples != 0).astype(np.float32)           # collapse all IDs to 1
                    apples_col = apple_grid.ravel()[:, None]
                else:
                    # Decentralized value function - return only agent's apples
                    # Only keep apples belonging to this agent
                    my_apples = (apples == agent_id).astype(np.float32)
                    # Flatten into column vectors (N,1)
                    apples_col = my_apples.ravel()[:, None]

                agents_col = agents.ravel().astype(np.float32)[:, None]
                # Concatenate along the first axis
                res = np.concatenate((agents_col, apples_col), axis=0)
                return append_pos(res)
        return None
