import numpy as np
import torch
from models.main_net import MainNet
import torch.nn.functional as F
import torch.optim as optim
from helpers import convert_position, ten
from config import DEVICE
from abc import abstractmethod
torch.set_default_dtype(torch.float64)


action_vectors = [
            np.array([-1, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([0, -1]),
            np.array([0, 0])
        ]


class ActorNetworkBase:
    def __init__(self, oned_size, agents_list, alpha, discount):
        self.function = MainNet(oned_size, 3)
        self.function.to(DEVICE)
        self.optimizer = optim.AdamW(self.function.parameters(), lr=alpha, amsgrad=True)
        self.alpha = alpha
        self.discount = discount
        self.agents_list = agents_list

        self.batch_states = []
        self.batch_new_states = []
        self.batch_actions = []
        self.batch_adv_values = []

    def get_function_output(self, a, b, pos=None, tau=1):
        res = ten(np.concatenate([a, b, convert_position(pos)], axis=0), DEVICE).view(1, -1)
        res = self.function(res) / tau
        res = F.softmax(res, dim=1)
        return res.detach().cpu().numpy().squeeze(0).tolist()

    def get_sum_value(self, state, positions):
        sum_ = 0
        for number, agent in enumerate(self.agents_list):
            sum_ += agent.policy_value.get_value_function(np.concatenate([state["agents"], state["apples"], convert_position(positions[number])], axis=0))
        return sum_

    @abstractmethod
    def get_adv_value(self, state, positions, reward, new_state, new_positions, agent=None):
        raise NotImplementedError

    def add_experience(self, state, old_pos, new_state, new_pos, reward, action, positions, new_positions, agent=None):
        self.batch_states.append(np.concatenate([np.concatenate([state["agents"], state["apples"]], axis=0), convert_position(old_pos)], axis=0))
        self.batch_new_states.append(np.concatenate([np.concatenate([new_state["agents"], new_state["apples"]], axis=0), convert_position(new_pos)], axis=0))
        self.batch_actions.append(action)
        self.add_advantage(state, new_state, reward, positions, new_positions, agent)

    def add_advantage(self, state, new_state, reward, positions, new_positions, agent=None):
        return self.batch_adv_values.append(self.get_adv_value(state, positions, reward, new_state, new_positions, agent))

    def train(self):
        if len(self.batch_states) == 0:
            return
        states = ten(np.stack(self.batch_states, axis=0), DEVICE)
        states = states.view(states.size(0), -1)

        action_probs = self.function(states)

        actions = [2 if x == 4 else x for x in self.batch_actions]
        probs = F.softmax(action_probs, dim=1)

        dist = torch.distributions.Categorical(probs)
        # (c) turn your Python list into a LongTensor on DEVICE
        actions_tensor = ten(np.array(actions), DEVICE)

        # (d) get a tensor of log-probs, one per batch element
        log_probs = dist.log_prob(actions_tensor)      # shape [B]

        adv_values = ten(np.array(self.batch_adv_values), DEVICE)

        # policy loss + entropy bonus
        loss = - (adv_values * log_probs).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.batch_states = []
        self.batch_new_states = []
        self.batch_actions = []
        self.batch_adv_values = []
        return loss.item(), adv_values.mean().item()


class ActorNetwork(ActorNetworkBase):
    def __init__(self, oned_size, agents_list, alpha, discount):
        super().__init__(oned_size, agents_list, alpha, discount)

    def get_adv_value(self, state, positions, reward, new_state, new_positions, agent=None):
        q_value = reward + self.discount * self.get_sum_value(new_state, new_positions)
        v_value = self.get_sum_value(state, positions)
        return q_value - v_value


class ActorNetworkWithBeta(ActorNetworkBase):
    def __init__(self, oned_size, agents_list, alpha, discount):
        super().__init__(oned_size, agents_list, alpha, discount)

    def add_experience(self, state, old_pos, new_state, new_pos, reward, action, positions, new_positions, agent=None):
        super().add_experience(state, old_pos, new_state, new_pos, reward, action, positions, new_positions, agent)

    def get_adv_value(self, state, positions, reward, new_state, new_positions, agent=None):
        q_value = reward + self.discount * self.get_sum_value(new_state, new_positions)
        beta = self.agents_list[agent].beta
        return q_value - beta


class ActorNetworkCounterfactual(ActorNetworkBase):
    def __init__(self, oned_size, agents_list, alpha, discount):
        super().__init__(oned_size, agents_list, alpha, discount)

    def add_experience(self, state, old_pos, new_state, new_pos, reward, action, positions, new_positions, agent=None):
        super().add_experience(state, old_pos, new_state, new_pos, reward, action, positions, new_positions, agent)

    def get_adv_value(self, state, positions, reward, new_state, new_positions, agent=None):
        q_value = reward + self.discount * self.get_sum_value(new_state, new_positions)

        probs = self.get_function_output(state["agents"], state["apples"], positions[agent])
        counterfactual = 0
        for num, action in enumerate([0, 1, 4]):
            r = 0
            new_pos = np.clip(positions[agent] + action_vectors[action], [0, 0], state["agents"].shape-np.array([1, 1]))
            agents = state["agents"].copy()
            apples = state["apples"].copy()
            agents[new_pos[0], new_pos[1]] += 1
            agents[positions[agent][0], positions[agent][1]] -= 1
            if apples[new_pos[0], new_pos[1]] > 0:
                r = 1
                apples[new_pos[0], new_pos[1]] -= 1
            test_positions = positions.copy()
            test_positions[agent] = new_pos
            sum_ = 0.99 * self.get_sum_value({"agents": agents, "apples": apples}, test_positions) + r
            counterfactual += sum_ * probs[num]

        return q_value - counterfactual
