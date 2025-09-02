import numpy as np
from policies.random_policy import random_policy
from policies.nearest import nearest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_dtype(torch.float64)

from torch import linalg

action_vectors = [
            np.array([-1, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([0, -1]),
            np.array([0, 0])
        ]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ten(c):
    return torch.from_numpy(c).to(device).double()

def unwrap_state(state):
    return state["agents"].copy(), state["apples"].copy()

def get_closest_left_right_1d(mat, agent_pos):
    mat = list(mat)
    left = -1
    right = -1
    pos, count = agent_pos, mat[agent_pos]
    while pos > -1:
        if count > 0:
            left = agent_pos - pos
            break
        else:
            pos -= 1
            count = mat[pos]
    pos, count = agent_pos, mat[agent_pos]
    while pos < len(mat):
        if count > 0:
            right = pos - agent_pos
            break
        else:
            pos += 1
            if pos >= len(mat):
                break
            count = mat[pos]
    return left, right
def convert_input(a, b, agent_pos):
    #print(list(a.flatten()), list(b.flatten()), agent_pos)

    a[agent_pos[0], agent_pos[1]] -= 1
    a = a.flatten()
    b = b.flatten()
    #print(list(a), list(b), agent_pos)
    left1, right1 = get_closest_left_right_1d(b, agent_pos[0])
    left2, right2 = get_closest_left_right_1d(a, agent_pos[0])
    arr = [left1, right1]
    arr1 = [left2, right2]
    #print(arr, arr1)
    return [np.array(arr), np.array(arr1)]

N = 5
     # 1 agent per 5 spaces
def convert_altinput(a, b, agent_pos):
    """
    Convert to a view around the agent.
    :param a:
    :param b:
    :param agent_pos:
    :return:
    """

    ap = np.concatenate(([-1, -1, -1, -1, -1], a.flatten(), [-1, -1, -1, -1, -1]))
    bp = np.concatenate(([-1, -1, -1, -1, -1], b.flatten(), [-1, -1, -1, -1, -1]))

    leftmost = agent_pos[0]-N
    rightmost = agent_pos[0] + N
    true_a = ap[leftmost+N: rightmost+N+1]
    true_b = bp[leftmost + N: rightmost + N + 1]

    return true_a, true_b, agent_pos[0] / len(a.flatten())


class SimpleConnectedMultiple(nn.Module):
    def __init__(self, oned_size): # we work with 1-d size here.
        super(SimpleConnectedMultiple, self).__init__()
        oned_size = (N * 2 + 1)
        # self.layer1 = nn.Linear(oned_size * 2 + 1, 256)
        # self.layer2 = nn.Linear(256, 256)
        # self.layer3 = nn.Linear(256, 256)
        # self.layer4 = nn.Linear(256, 3) # [0] is move left, [1] is move right, [2] is don't move
        #self.layer1 = nn.Linear(oned_size * 2, 256)

        outl = 6 * ((oned_size * 2 + 1) - 2)
        self.layer1 = nn.Conv1d(1, 6, 3, 1)
        # self.layer2 = nn.Linear(48, 64) # 48 for an input dimension of 10 (i.e. oned size is 5)
        self.layer2 = nn.Linear(outl, 256)
        # self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 3)

        # FIRST INPUT TYPE MODEL
        #self.layer1 = nn.Conv1d(1, 6, 3, 1)
        ##self.conv_bn = nn.BatchNorm1d(19)
        ##self.layer2 = nn.Linear(48, 64) # 48 for an input dimension of 10 (i.e. oned size is 5)
        #self.layer2 = nn.Linear(114, 128)
        ##self.layer2 = nn.Linear(64, 64)
        #self.layer3 = nn.Linear(128, 128)
        #self.layer5 = nn.Linear(128, 128)
        #self.layer4 = nn.Linear(128, 3)

        # SECOND INPUT TYPE MODEL
        # self.layer1 = nn.Conv1d(1, 6, 3, 1)
        # self.layer2 = nn.Linear(12, 32)
        # self.layer3 = nn.Linear(32, 32)
        # self.layer4 = nn.Linear(32, 1)

        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
        torch.nn.init.xavier_uniform_(self.layer4.weight)
        #torch.nn.init.xavier_uniform_(self.layer5.weight)
        # print("Initialized Neural Network")

    def forward(self, a, b, pos):
        x = torch.cat((a.flatten(), b.flatten(), pos.flatten()))
        #print(x)
        #print(x)
        x = x.view(1, -1)
        #x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer1(x))
        x = x.flatten()
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        #x = F.leaky_relu(self.layer5(x))
        #  F.softmax(self.layer4(x), dim=0)
        return self.layer4(x)

counter = 0
total_reward = 0
class ActorNetwork():
    def __init__(self, oned_size, alpha, discount, beta=None, avg_alpha=None, num=0):
        self.function = SimpleConnectedMultiple(oned_size)
        self.function.to(device)
        self.optimizer = optim.AdamW(self.function.parameters(), lr=alpha, amsgrad=True)
        self.alpha = alpha
        self.discount = discount
        self.num = num
        self.beta = beta
        self.vs = 0
        self.avg_alpha = avg_alpha

        self.states = []
        self.new_states = []
        self.rewards = []
        self.poses = []
        self.actions = []
        self.action_utils_raws = []
        self.feedbacks = []
        self.usable_beta = 0

        self.critic = None

    def get_function_output(self, a, b, pos=None):
        # pose = np.array([pos[0]])

        a, b, agpos_old = convert_altinput(a, b, pos)
        return F.softmax(self.function(ten(a), ten(b), ten(np.array([agpos_old]))), dim=0).detach().cpu().numpy()

    def get_function_output_v(self, a, b, pos=None):
        poses = np.array(pos[:, 0])
        return self.function(ten(a), ten(b), ten(poses)).detach().cpu().numpy()

    def get_value_function2(self, state):
        #a, b = unwrap_state(state)
        a, b = state[0], state[1]
        return self.function(ten(a), ten(b), None).detach().cpu().numpy()

    def get_value_function(self, a, b, agents_list, pos=None):
        summ = 0
        if agents_list[0].avg_alpha is None:
            for number, agent in enumerate(agents_list):
                if number == self.num and pos is not None:
                    summ += agent.policy_value.get_sum_value(a, b, pos) * agents_list[self.num].agent_rates[number]
                else:
                    summ += agent.policy_value.get_sum_value(a, b, agent.position) * agents_list[self.num].agent_rates[number]
        else:
            for number, agent in enumerate(agents_list):
                if number == self.num and pos is not None:
                    summ += agent.get_value_function_bin(a, b, pos) #* agents_list[self.num].agent_rates[number]
                else:
                    summ += agent.get_value_function_bin(a, b, agent.position) #* agents_list[self.num].agent_rates[number]
        return summ

    def get_value_function_with_pos(self, a, b, agents_list, poses, pos):
        summ = 0
        if agents_list[0].avg_alpha is None:
            for number, agent in enumerate(agents_list):
                if number == self.num and pos is not None:
                    summ += agent.policy_value.get_sum_value(a, b, pos) * agents_list[self.num].agent_rates[number]
                else:
                    summ += agent.policy_value.get_sum_value(a, b, poses[number]) * agents_list[self.num].agent_rates[number]
        else:
            for number, agent in enumerate(agents_list):
                if number == self.num and pos is not None:
                    summ += agent.get_value_function_bin(a, b, pos) #* agents_list[self.num].agent_rates[number]
                else:
                    summ += agent.get_value_function_bin(a, b, poses[number]) #* agents_list[self.num].agent_rates[number]
        return summ

    def get_value_function_central(self, a, b, pos, agents_list):
        return agents_list[0].policy_value.get_sum_value(a, b, pos)

    def train(self, state, new_state, reward, action, agents_list):
        old_pos = np.array([state["pos"][0]])
        new_pos = np.array([new_state["pos"][0]])

        debug = False
        if debug:
            print("=========TRAINING=========")
            print(list(state["agents"].flatten()), list(state["apples"].flatten()))
            print(list(new_state["agents"].flatten()), list(new_state["apples"].flatten()))
            print(old_pos, new_pos)
            print(reward)

        a, b = unwrap_state(state)
        new_a, new_b = unwrap_state(new_state)

        a, b, agpos_old = convert_altinput(a, b, state["pos"])


        # actions = self.function(ten(a), ten(b), ten(old_pos))
        actions = self.function(ten(a), ten(b), ten(np.array([agpos_old])))

        if action == 4:
            action = 2

        prob = F.log_softmax(actions, dim=0)[action]

        if self.beta is None:
            with torch.no_grad():
                v_value = self.get_value_function(a, b, agents_list, old_pos)
        else:
            v_value = self.beta

        with torch.no_grad():
            if agents_list[0].influencers is not None:
                q_value = agents_list[0].influencers[0].get_follower_feedback(self, agents_list[self.num], action, reward, state, new_pos)
            else:
                q_value = reward + self.discount * self.get_value_function(new_a, new_b, agents_list, new_pos)

        adv = q_value - v_value
        adv = ten(adv)

        self.optimizer.zero_grad()

        loss = -1 * torch.mul(prob, adv)

        loss.backward()
        self.optimizer.step()

    def addexp(self, state, new_state, reward, action, agents_list, feedback=None, action_utils_raw=None):
        self.states.append(state)
        self.new_states.append(new_state)
        self.rewards.append(reward)
        self.actions.append(action)
        poses = []
        for i in agents_list:
            poses.append(i.position.copy())
        self.poses.append(poses)
        self.action_utils_raws.append(action_utils_raw)
        self.feedbacks.append(feedback)


    def train_with_feedback(self, state, pos, action, feedback, reward, agents_list):
        losses = []
        # old_pos = np.array([state["pos"][0]])
        old_pos = np.array([pos[0]])

        debug = False
        if debug:
            print("=========TRAINING=========")
            print(list(state["agents"].flatten()), list(state["apples"].flatten()))
            # print(list(new_state["agents"].flatten()), list(new_state["apples"].flatten()))
            print(reward)

        a, b = unwrap_state(state)
        actions_lst = self.function(ten(a), ten(b), ten(old_pos))

        if action == 4:
            action = 2

        #prob = torch.log(actions[action])
        prob = F.log_softmax(actions_lst, dim=0)[action]
        #prob = actions[action]
        v_value = agents_list[self.num].beta # + agents_list[self.num].get_util_learned(state["agents"], state["apples"], pos)

        #q_value = reward + self.discount * self.get_value_function_with_pos(new_a, new_b, agents_list, all_pos, new_pos)
        q_value = feedback + reward

        #q_value = torch.mul(actions_lst, agents_list[self.num].alphas)
        # print("Value,", v_value, "   Q Value,", q_value)

        adv = np.array([q_value - v_value])

        adv = ten(adv)

        losses.append(-1 * torch.mul(prob, adv))
        self.optimizer.zero_grad()
        loss = torch.stack(losses).sum()
        loss.backward()
        self.optimizer.step()

    def train_with_v_value(self, state, pos, action, feedback, reward, agents_list):
        losses = []
        # old_pos = np.array([state["pos"][0]])
        old_pos = np.array([pos[0]])

        debug = False
        if debug:
            print("=========TRAINING=========")
            print(list(state["agents"].flatten()), list(state["apples"].flatten()))
            # print(list(new_state["agents"].flatten()), list(new_state["apples"].flatten()))
            print(reward)

        a, b = unwrap_state(state)
        actions_lst = self.function(ten(a), ten(b), ten(old_pos))

        if action == 4:
            action = 2

        #prob = torch.log(actions[action])
        prob = F.log_softmax(actions_lst, dim=0)[action]
        #prob = actions[action]
        # v_value = agents_list[self.num].beta # + agents_list[self.num].get_util_learned(state["agents"], state["apples"], pos)
        v_value = 0
        for agent in agents_list:
            if agent.num == self.num:
                v_value += agent.get_util_learned(a, b, old_pos)
            else:
                inter_value = agent.get_util_learned(a, b, agent.position)
                if len(agent.followers) == 0:
                    v_value += inter_value * agent.agent_rates[agent.num]
                    v_value += inter_value * agents_list[agents_list[agent.num].target_influencer].agent_rates[agent.num] * agent.infl_rate

        #q_value = reward + self.discount * self.get_value_function_with_pos(new_a, new_b, agents_list, all_pos, new_pos)
        q_value = feedback + reward

        #q_value = torch.mul(actions_lst, agents_list[self.num].alphas)
        # print("Value,", v_value, "   Q Value,", q_value)

        adv = np.array([q_value - v_value])

        adv = ten(adv)

        losses.append(-1 * torch.mul(prob, adv))
        self.optimizer.zero_grad()
        loss = torch.stack(losses).sum()
        loss.backward()
        self.optimizer.step()

    def train_multiple_with_v_value(self, agents_list):
        losses = []
        states = self.states
        new_states = self.new_states
        rewards = self.rewards
        actions = self.actions
        poses = self.poses

        for it in range(len(states)):
            state = states[it]
            new_state = new_states[it]
            reward = rewards[it]
            action = actions[it]

            all_pos = poses[it]

            old_pos = np.array([state["pos"][0]])
            new_pos = np.array([new_state["pos"][0]])

            debug = False
            if debug:
                print("=========TRAINING=========")
                print(list(state["agents"].flatten()), list(state["apples"].flatten()))
                # print(list(new_state["agents"].flatten()), list(new_state["apples"].flatten()))
                print(reward)

            a, b = unwrap_state(state)
            actions_lst = self.function(ten(a), ten(b), ten(old_pos))

            if action == 4:
                action = 2

            #prob = torch.log(actions[action])
            prob = F.log_softmax(actions_lst, dim=0)[action]
            #prob = actions[action]
            # v_value = agents_list[self.num].beta # + agents_list[self.num].get_util_learned(state["agents"], state["apples"], pos)
            v_value = 0
            q_value = 0
            for agent in agents_list:
                if agent.num == self.num:
                    q, v = agents_list[self.num].value_network.training_loop(state, new_state, reward, old_pos, new_pos)
                    v_value += v
                    q_value += q
                    v_value += v * agents_list[agents_list[self.num].target_influencer].agent_rates[
                        self.num] * agent.infl_rate
                    q_value += q * agents_list[agents_list[self.num].target_influencer].agent_rates[
                        self.num] * agent.infl_rate
                else:
                    if len(agent.followers) == 0:
                        q, v = agent.value_network.training_loop(state, new_state, 0, all_pos[agent.num], all_pos[agent.num])
                        v_value += v * agent.agent_rates[self.num]
                        v_value += v * agents_list[agents_list[self.num].target_influencer].agent_rates[agent.num] * agent.infl_rate
                        q_value += q * agent.agent_rates[self.num]
                        q_value += q * agents_list[agents_list[self.num].target_influencer].agent_rates[agent.num] * agent.infl_rate


            #q_value = reward + self.discount * self.get_value_function_with_pos(new_a, new_b, agents_list, all_pos, new_pos)

            #q_value = torch.mul(actions_lst, agents_list[self.num].alphas)
            # print("Value,", v_value, "   Q Value,", q_value)

            adv = np.array([q_value - v_value])

            adv = ten(adv)

            losses.append(-1 * torch.mul(prob, adv))
        if len(states) != 0:

            self.optimizer.zero_grad()

            loss = torch.stack(losses).sum()
            loss.backward()
            self.optimizer.step()

            self.states = []
            self.new_states = []
            self.rewards = []
            self.actions = []
            self.poses = []

    def train_multiple_with_beta(self, agents_list):
        losses = []
        states = self.states
        new_states = self.new_states
        rewards = self.rewards
        actions = self.actions
        poses = self.poses
        feedbacks = self.feedbacks
        action_utils_raws = self.action_utils_raws

        if self.usable_beta == 0:
            self.usable_beta = agents_list[self.num].beta

        for it in range(len(states)):
            state = states[it]
            new_state = new_states[it]
            reward = rewards[it]
            action = actions[it]

            all_pos = poses[it]

            old_pos = np.array([state["pos"][0]])
            new_pos = np.array([new_state["pos"][0]])

            old_pos_a = np.array([state["pos"][0]])
            new_pos_a = np.array([new_state["pos"][0]])
            feedback = feedbacks[it]
            action_utils_raw = action_utils_raws[it]

            debug = False
            if debug:
                print("=========TRAINING=========")
                print(list(state["agents"].flatten()), list(state["apples"].flatten()))
                # print(list(new_state["agents"].flatten()), list(new_state["apples"].flatten()))
                print(reward)

            a, b = unwrap_state(state)
            actions_lst = self.function(ten(a), ten(b), ten(old_pos_a))

            if action == 4:
                action = 2

            #prob = torch.log(actions[action])
            prob = F.log_softmax(actions_lst, dim=0)[action]
            #prob = actions[action]
            # v_value = agents_list[self.num].beta # + agents_list[self.num].get_util_learned(state["agents"], state["apples"], pos)
            v_value = 0.0 #agents_list[self.num].beta # agents_list[self.num].beta
            q_value = 0.0
            for agent in agents_list:
                if agent.num == self.num:
                    q, v = agents_list[self.num].value_network.training_loop(state, new_state, reward, old_pos, new_pos)
                    # q, v = agents_list[self.num].value_network.just_get_q_v(state, new_state, reward, old_pos, new_pos)
                    v_value += v
                    q_value += q.item()
                    v_value += v * agents_list[agents_list[agent.num].target_influencer].agent_rates[
                        self.num] * agent.infl_rate
                    q_value += q.item() * agents_list[agents_list[agent.num].target_influencer].agent_rates[self.num] * agent.infl_rate
                else:
                    if len(agent.followers) == 0:
                        #q, v = agent.value_network.train_with_learned_util(state, new_state, 0, all_pos[agent.num], all_pos[agent.num], action_utils_raw[agent.num], agent.agent_rates[self.num])
                        # q, v = agents_list[self.num].value_network.just_get_q_v(state, new_state, reward, old_pos,
                        #                                                        new_pos)
                        q, v = agents_list[agent.num].value_network.training_loop(state, new_state, 0, all_pos[agent.num], all_pos[agent.num])
                        v_value += v * agent.agent_rates[self.num]
                        v_value += v * agents_list[agents_list[agent.num].target_influencer].agent_rates[self.num] * agent.infl_rate
                        q_value += q.item() * agent.agent_rates[self.num]
                        q_value += q.item() * agents_list[agents_list[agent.num].target_influencer].agent_rates[self.num] * agent.infl_rate

            agents_list[self.num].beta = agents_list[self.num].beta * (1 - 0.01) + q_value * 0.01

            adv = np.array([q_value - v_value])

            adv = ten(adv)

            losses.append(-1 * torch.mul(prob, adv))
        if len(states) != 0:

            self.optimizer.zero_grad()

            loss = torch.stack(losses).sum()
            loss.backward()
            self.optimizer.step()

            self.states = []
            self.new_states = []
            self.rewards = []
            self.actions = []
            self.poses = []
            self.feedbacks = []
            self.action_utils_raws = []

        self.usable_beta = agents_list[self.num].beta

    def train_multiple_with_beta_static(self, agents_list):
        losses = []
        states = self.states
        new_states = self.new_states
        rewards = self.rewards
        actions = self.actions
        poses = self.poses
        feedbacks = self.feedbacks
        action_utils_raws = self.action_utils_raws

        if self.usable_beta == 0:
            self.usable_beta = agents_list[self.num].beta

        for it in range(len(states)):
            state = states[it]
            new_state = new_states[it]
            reward = rewards[it]
            action = actions[it]

            all_pos = poses[it]

            old_pos = np.array([state["pos"][0]])
            new_pos = np.array([new_state["pos"][0]])

            old_pos_a = np.array([state["pos"][0]])
            new_pos_a = np.array([new_state["pos"][0]])
            feedback = feedbacks[it]
            action_utils_raw = action_utils_raws[it]

            debug = False
            if debug:
                print("=========TRAINING=========")
                print(list(state["agents"].flatten()), list(state["apples"].flatten()))
                # print(list(new_state["agents"].flatten()), list(new_state["apples"].flatten()))
                print(reward)

            a, b = unwrap_state(state)
            # actions_lst = self.function(ten(a), ten(b), ten(old_pos_a))

            a, b, agpos_old = convert_altinput(a, b, state["pos"])

            # actions = self.function(ten(a), ten(b), ten(old_pos))
            actions_lst = self.function(ten(a), ten(b), ten(np.array([agpos_old])))

            if action == 4:
                action = 2

            #prob = torch.log(actions[action])
            prob = F.log_softmax(actions_lst, dim=0)[action]
            #prob = actions[action]
            # v_value = agents_list[self.num].beta # + agents_list[self.num].get_util_learned(state["agents"], state["apples"], pos)
            v_value = 0.0 #agents_list[self.num].beta # agents_list[self.num].beta
            q_value = reward
            for agent in agents_list:
                if agent.num == self.num:
                    q, v = agents_list[self.num].value_network.just_get_q_v(state, new_state, reward, old_pos, new_pos)
                    # q, v = agents_list[self.num].value_network.just_get_q_v(state, new_state, reward, old_pos, new_pos)
                    v_value += v
                    q_value += q.item()
                    v_value += v * agents_list[agents_list[agent.num].target_influencer].agent_rates[
                        self.num] * agent.infl_rate
                    q_value += q.item() * agents_list[agents_list[agent.num].target_influencer].agent_rates[self.num] * agent.infl_rate
                else:
                    if len(agent.followers) == 0:
                        #q, v = agent.value_network.train_with_learned_util(state, new_state, 0, all_pos[agent.num], all_pos[agent.num], action_utils_raw[agent.num], agent.agent_rates[self.num])
                        # q, v = agents_list[self.num].value_network.just_get_q_v(state, new_state, reward, old_pos,
                        #                                                        new_pos)
                        q, v = agents_list[agent.num].value_network.just_get_q_v(state, new_state, 0, all_pos[agent.num], all_pos[agent.num])
                        v_value += v * agent.agent_rates[self.num]
                        v_value += v * agents_list[agents_list[agent.num].target_influencer].agent_rates[self.num] * agent.infl_rate
                        q_value += q.item() * agent.agent_rates[self.num]
                        q_value += q.item() * agents_list[agents_list[agent.num].target_influencer].agent_rates[self.num] * agent.infl_rate

            agents_list[self.num].beta = agents_list[self.num].beta * (1 - 0.01) + q_value * 0.01

            adv = np.array([q_value - v_value])

            adv = ten(adv)

            losses.append(-1 * torch.mul(prob, adv))
        if len(states) != 0:

            self.optimizer.zero_grad()

            loss = torch.stack(losses).sum()
            loss.backward()
            self.optimizer.step()

            self.states = []
            self.new_states = []
            self.rewards = []
            self.actions = []
            self.poses = []
            self.feedbacks = []
            self.action_utils_raws = []

        self.usable_beta = agents_list[self.num].beta

    def train_multiple_with_critic(self, agents_list):
        losses = []
        states = self.states
        new_states = self.new_states
        rewards = self.rewards
        actions = self.actions
        poses = self.poses

        for it in range(len(states)):
            state = states[it]
            new_state = new_states[it]
            reward = rewards[it]
            action = actions[it]

            all_pos = poses[it]

            old_pos = np.array([state["pos"][0]])
            new_pos = np.array([new_state["pos"][0]])

            debug = False
            if debug:
                print("=========TRAINING=========")
                print(list(state["agents"].flatten()), list(state["apples"].flatten()))
                print(list(new_state["agents"].flatten()), list(new_state["apples"].flatten()))
                print(old_pos, new_pos)
                print(reward)

            a, b = unwrap_state(state)

            actions_lst = self.function(ten(a), ten(b), ten(old_pos))

            if action == 4:
                action = 2

            prob = F.log_softmax(actions_lst, dim=0)[action]
            adv = self.get_collective_adv_basic(state, new_state, reward, old_pos, new_pos, all_pos, agents_list)


            losses.append(-1 * torch.mul(prob, adv))

        if len(states) != 0:
            # if self.critic is not None:
            #     self.critic.optimizer.zero_grad()
            #     crit_loss = torch.stack(crit_losses).sum()
            #     #crit_loss.backward(retain_graph=True)
            #     #self.critic.optimizer.step()

            self.optimizer.zero_grad()

            loss = torch.stack(losses).sum()
            loss.backward()
            # total_norm = 0
            # for p in self.function.parameters():
            #     param_norm = p.grad.detach().data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** 0.5
            #print([p.grad for p in self.function.parameters() if p.grad is not None])
            #grads = [p.grad.cpu() for p in self.function.parameters() if p.grad is not None]
            #v = linalg.vector_norm(grads, ord=2)
            # self.vs += total_norm
            #torch.nn.utils.clip_grad_value_(self.function.parameters(), -0.01, 0.01)
            self.optimizer.step()


            self.states = []
            self.new_states = []
            self.rewards = []
            self.actions = []
            self.poses = []
