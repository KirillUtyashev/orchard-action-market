import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helpers import ten, unwrap_state

torch.set_default_dtype(torch.float64)


action_vectors = [
            np.array([-1, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([0, -1]),
            np.array([0, 0])
        ]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleConnectedMultiple(nn.Module):
    def __init__(self, oned_size, num_agents=10, num_influencers=1, is_influencer=False): # we work with 1-d size here.
        super(SimpleConnectedMultiple, self).__init__()
        self.layer1 = nn.Linear(oned_size * 2 + 1, 32)
        self.layer2 = nn.Linear(32, 16)
        #self.layer3 = nn.Linear(128, 64)
        if is_influencer:
            self.layer4 = nn.Linear(16,
                                    num_agents)  # [0] is move left, [1] is move right, [2] is don't move
        else:
            #self.layer4 = nn.Linear(64, num_agents + num_influencers + 1)  # [0] is move left, [1] is move right, [2] is don't move
            self.layer4 = nn.Linear(16,
                                    num_agents + 1)  # [0] is move left, [1] is move right, [2] is don't move
        #self.layer1 = nn.Linear(oned_size * 2, 256)

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
        #torch.nn.init.xavier_uniform_(self.layer3.weight)
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
        # x = F.leaky_relu(self.layer3(x))
        #x = F.leaky_relu(self.layer5(x))
        #  F.softmax(self.layer4(x), dim=0)
        return self.layer4(x)


class ObserverNetwork():
    def __init__(self, oned_size, num_agents, alpha, discount, beta=None, avg_alpha=None, num=0, infl_net=False, num_infls=0):
        if infl_net:
            self.function = SimpleConnectedMultiple(oned_size, num_agents, is_influencer=True)
        else:
            self.function = SimpleConnectedMultiple(oned_size, num_agents, num_infls)
        self.function.to(device)
        self.optimizer = optim.AdamW(self.function.parameters(), lr=alpha, amsgrad=True)
        self.alpha = alpha
        self.discount = discount
        self.num = num
        self.beta = beta
        self.vs = 0
        self.avg_alpha = avg_alpha

        self.is_infl = infl_net

        self.states = []
        self.new_states = []
        self.rewards = []
        self.poses = []
        self.actions = []

        self.critic = None

    def get_function_output(self, a, b, pos=None, infl=False):
        pose = np.array([pos[0]])
        acts = self.function(ten(a), ten(b), ten(pose))
        if not infl:
            acts[self.num] = -100
        return F.softmax(acts, dim=0).detach().cpu().numpy()

    def get_function_output_v(self, a, b, pos=None):
        poses = np.array(pos[:, 0])
        return self.function(ten(a), ten(b), ten(poses)).detach().cpu().numpy()

    def get_value_function2(self, state):
        #a, b = unwrap_state(state)
        a, b = state[0], state[1]
        return self.function(ten(a), ten(b), None).detach().cpu().numpy()

    def get_value_function(self, a, b, agents_list, pos=None):
        if agents_list[0].influencers is not None:
            return agents_list[self.num].influencers[0].get_follower_feedback(agents_list[agent], action, reward, new_state)
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

    def get_content_value_function(self, agents_list, action):
        # here, action is the type of content.
        summ = 0
        for number, agent in enumerate(agents_list):
            if number == self.num:
                pass
                #summ += agent.policy_value.get_value_function(a, b, pos) * agents_list[self.num].agent_rates[number]
            else:
                summ += agent.get_util_pq(action, agents_list[self.num])
        summ += agents_list[0].influencers[0].get_follower_feedback(agents_list[self.num], action)
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

        actions = self.function(ten(a), ten(b), ten(old_pos))

        # if action == 4:
        #     action = 2

        #prob = torch.log(actions[action])
        det_acts = actions.detach().cpu().numpy()
        best_act = det_acts.index(min(det_acts))
        prob = F.log_softmax(actions, dim=0)[best_act]
        #prob = actions[action]
        if self.beta is None:
            with torch.no_grad():
                v_value = self.get_value_function(a, b, agents_list, old_pos)
               # v_value = self.get_value_function_central(a, b, old_pos, agents_list)
        else:
            v_value = np.sum(agents_list[self.num].alphas)

        #with torch.no_grad():
            # if agents_list[0].influencers is not None:
            #     q_value = agents_list[0].influencers[0].get_follower_feedback(self, agents_list[self.num], action, reward, state, new_pos)
            # else:
            #     q_value = reward + self.discount * self.get_value_function(new_a, new_b, agents_list, new_pos)

            q_value = reward + self.discount * self.get_value_function(new_a, new_b, new_pos, agents_list)

        adv = q_value - v_value
        adv = ten(adv)

        self.optimizer.zero_grad()

        loss = -1 * torch.mul(prob, adv)

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

    def addexp(self, state, new_state, reward, action, agents_list):
        self.states.append(state)
        self.new_states.append(new_state)
        self.rewards.append(reward)
        self.actions.append(action)
        poses = []
        for i in agents_list:
            poses.append(i.position.copy())
        self.poses.append(poses)

    def update(self, agents_list):

        if agents_list[self.num].is_new_infl:
            agents_list[self.num].is_new_infl = False
            return
        budget = agents_list[self.num].base_budget - agents_list[self.num].raw_b0_rate
        if budget < 1:
            budget = 1

        if self.is_infl:
            if budget < 20:
                budget += 5
            rates = agents_list[self.num].generate_rates_only(None, None, const_ext=True)[0:len(agents_list)]
        else:
            rates = agents_list[self.num].generate_rates_only(None, None, const_ext=True)[0:len(agents_list) + 1]

        if sum(rates) > 0:
            rates /= sum(rates)
        state = self.states[0]

        a, b = unwrap_state(state)

        actions1 = self.function(ten(a), ten(b), ten(np.array([state["pos"][0]])))

        actions1[self.num] = -100
        v_rates = F.softmax(actions1, dim=0)
        q_rates = ten(rates)
        loss = nn.functional.mse_loss(v_rates, q_rates)
        loss.backward()

        self.optimizer.step()

        self.states = [self.states[0]]
        self.new_states = []
        self.rewards = []
        self.actions = []
        self.poses = []

    def train_multiple(self, agents_list):
        losses = []
        crit_losses = []
        states = self.states

        q_rates = agents_list[self.num].generate_rates_only(None, None) / agents_list[self.num].budget
        q_rates = ten(q_rates)

        if agents_list[self.num].newly_infl_train:
            for g in self.optimizer.param_groups:
                g['lr'] = 0.8


        for it in range(len(states)):
            state = states[it]
            old_pos = np.array([state["pos"][0]])
            a, b = unwrap_state(state)

            actions1 = self.function(ten(a), ten(b), ten(old_pos))

            actions1[self.num] = -100
            v_rates = F.softmax(actions1, dim=0)

            losses.append(nn.functional.mse_loss(v_rates, q_rates))

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

            if self.critic is not None:
                self.critic.optimizer.zero_grad()
                crit_loss = torch.stack(crit_losses).sum()
                crit_loss.backward()
                self.critic.optimizer.env_step()

        if agents_list[self.num].newly_infl_train:
            for g in self.optimizer.param_groups:
                g['lr'] = self.alpha
            agents_list[self.num].newly_infl_train = False
