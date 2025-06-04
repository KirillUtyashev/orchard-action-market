from collections import deque

import numpy as np

from alloc.allocation import rate_allocate
from policies.random_policy import random_policy

import random
import torch

random.seed(352790383)
np.random.seed(3890433)

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

action_vectors = [
            np.array([-1, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([0, -1]),
            np.array([0, 0])
        ]

def calculate_ir(a, b, pos, action):
    new_pos = np.clip(pos + action_vectors[action], [0, 0], a.shape-np.array([1, 1]))
    agents = a.copy()
    apples = b.copy()
    agents[new_pos[0], new_pos[1]] += 1
    agents[pos[0], pos[1]] -= 1
    if apples[new_pos[0], new_pos[1]] > 0:
        apples[new_pos[0], new_pos[1]] -= 1
        return 1, agents, apples, new_pos
    else:
        return 0, agents, apples, new_pos

def exp_decrease_chance(topic, interest, c, rang):
    return np.exp(-np.abs(topic - interest) / c)

def lin_decrease_chance(topic, interest, c, rang):
    return (rang - np.abs(topic-interest))/rang
"""
A General "Actor-Critic" agent. In line with the theory, has a "critic" neural network and an "actor" neural network. The critic
does not necessarily only inform this actor; its Q-value is used for applying feedback to others as well.

In line with the Content Market, this agent recognizes that there is a "state", but knows that the state does not change.

This actor is a 1d agent.
"""

class ContentAgent:
    def __init__(self, policy, id=0, main_interest=0, num_agents=1, is_beta=0, is_alpha=0, is_projecting=0, b0=0, budget=1, topics=5, debug=False, influencers=None, threshold=0.00005, nonfixedpackage=None):
        self.position = np.array([0, 0])
        self.policy = policy
        self.debug = debug
        self.num = id
        """
        Policy Value: the Value function (Q-function) neural network.
        Policy Network: the Actor network.
        Observer Network: the Observing "rates".
        """
        self.policy_value = None
        self.policy_network = None
        self.observer_network = None

        self.follower_network = None
        self.influencer_network = None

        """
        Following Rates. Shouldn't be anything other than 1 unless working with is_alpha=1 (see the function to see agent rates).
        """
        self.agent_rates = np.zeros(num_agents)
        for i in range(self.agent_rates.size):
            self.agent_rates[i] = 0.048

        self.agent_rates_as_infl = np.zeros(num_agents)
        for i in range(self.agent_rates.size):
            self.agent_rates_as_infl[i] = 0.01

        self.agent_rates_as_foll = np.zeros(num_agents)
        for i in range(self.agent_rates.size):
            self.agent_rates_as_foll[i] = 0.01

        """
        Statistical Beta/Alpha values initialization.
        """
        self.alphas = np.zeros(num_agents)
        self.alpha_agents = np.zeros(num_agents)
        self.beta = 0
        self.times = 0
        self.avg_alpha = None
        self.beta_factor = 0.999
        self.is_beta = is_beta
        self.is_alpha = is_alpha
        if is_beta:
            if is_alpha:
                print("Is Beta & Alpha Agent")
            else:
                print("Is Beta Agent")

        self.proj_beta = 0
        self.proj_alpha = 0
        self.is_projecting = is_projecting

        self.alphas_raw = np.zeros(num_agents)

        self.indirect_alphas = np.zeros(num_agents)
        self.indirect_alphas_raw = np.zeros(num_agents)
        """
        Influencer Infrastructure
        "influencers" is a list. None if there are no influencers.
        """
        self.influencers = influencers
        self.infl_alphas = np.zeros(len(influencers))

        self.infl_alphas_raw = np.zeros(len(influencers))

        """
        Main Interest (Content Market)
        """
        self.interest = main_interest # the agent's main interest. Does not change.
        self.c = 6
        self.max_topics = num_agents * 2
        # External Content
        self.b0 = b0
        self.budget = budget
        self.base_budget = budget
        self.b0_rate = 1
        self.b0_adjusted_rate = 1

        self.pq = exp_decrease_chance
        self.rang = 5


        """
        For storage use
        """
        self.raw_agent_rates = np.zeros(num_agents)
        self.infl_rate = 0
        self.raw_infl_rate = 0
        self.raw_b0_rate = 0

        self.bqueue = deque([], maxlen=5) # 200

        self.aqueues = []
        self.i_aqueues = []
        for _ in range(num_agents):
            self.aqueues.append(deque([], maxlen=2)) #20 #200

        for _ in range(len(influencers)):
            self.i_aqueues.append(deque([], maxlen=500))

        self.aqueues_raw = []
        self.i_aqueues_raw = []
        for _ in range(num_agents):
            self.aqueues_raw.append(deque([], maxlen=2)) #20 #200
        for _ in range(len(influencers)):
            self.i_aqueues_raw.append(deque([], maxlen=1))

        """
        Indirect Following (Based on the content that person FOLLOWS)
        """
        self.aqueues_indirect = []
        self.aqueues_indirect_raw = []
        for dd in range(num_agents):
            self.aqueues_indirect.append(deque([], maxlen=100)) #500
            #self.aqueues_indirect[dd].append(0)
        for dd in range(num_agents):
            self.aqueues_indirect_raw.append(deque([], maxlen=100)) #500
            #self.aqueues_indirect_raw[dd].append(0)


        # Target Influencer: the Influencer that the agent believes should lead the market.
        self.target_influencer = -2

        """
        Proto-Influencer
        """
        self.followers = []
        self.max_followers = 0

        self.alphas_asinfl = np.zeros(num_agents)
        self.alphas_asinfl_raw = np.zeros(num_agents)
        self.aqueues_asinfl = []
        self.aqueues_asinfl_raw = []
        for _ in range(num_agents):
            self.aqueues_asinfl.append(deque([], maxlen=2))
        for _ in range(num_agents):
            self.aqueues_asinfl_raw.append(deque([], maxlen=2))


        self.PR = np.zeros(num_agents)
        self.PB = np.zeros(num_agents)
        self.R = np.zeros(num_agents)
        self.has_gossiped = np.zeros(num_agents)

        self.adjs = []
        self.newly_infl = True
        self.newly_infl_train = False
        self.is_new_infl = False

        self.threshold = threshold
        if threshold is None:
            self.threshold = 0 #0.0003


        self.nfp = nonfixedpackage
        if nonfixedpackage != None:
            self.base_a = self.nfp["base_a"]
            self.varAlpha = self.nfp["alpha"]
            self.varKappa = self.nfp["kappa"]
            self.n0 = self.nfp["n0"]

        self.LRF = 0
        self.lategame=False
        self.funagent=-1
        self.rate_alloc_count=0
        self.startlkdown = 0

        self.update_factor = 0.03
        self.discount_factor = 0.003
        # packs = {
        #     "base_a": base_a,
        #     "alpha": alpha_decay,
        #     "kappa": kappa_decay,
        #     "n0": threshold
        # }
    def trigger_change(self, time, ext_ra=True):
        var = np.random.random()
        if ext_ra:
            # reallocation rate externally
            mut = self.base_a * np.exp(-1 * self.varAlpha * time)
            return var < mut

        else:
            # choose influencer
            mut = (self.base_a / self.varKappa) * np.exp(-1 * self.varAlpha * time)
            return var < mut


    def identify_influencer(self, agents_list):
        threshold = self.threshold
        if not self.lategame:
            for ag in agents_list:
                if len(ag.followers) > 90:
                    self.lategame = True
                    self.funagent = ag.num

        # if self.lategame and self.target_influencer == self.funagent:
        #     return
        if self.target_influencer != self.funagent and self.lategame:
            print("Agent", self.num, "allocing without funagent")

        frontrun = 0
        for ag in agents_list:
            if len(ag.followers) > len(agents_list[frontrun].followers):
                frontrun = ag.num

        # follow the influencer with the highest OR
        # Should only occur after initial alphas have been established
        self.PR = np.copy(self.alphas) + np.copy(self.indirect_alphas)
        oldinf = self.target_influencer
        prevrep = self.R[self.target_influencer]
        if self.target_influencer == -2 or self.target_influencer == -1:
            prevrep = 0
        if self.target_influencer == -2:
            new_infl = np.argmax(self.R)

            # if new_infl != self.funagent and self.lategame:
            #     return

            agents_list[new_infl].followers.append(self.num)
            self.target_influencer = new_infl
            self.newly_infl = True
        elif self.target_influencer == -1:
            new_infl = np.argmax(self.R)
            agents_list[new_infl].followers.append(self.num)

            # if new_infl != self.funagent and self.lategame:
            #     return

            self.target_influencer = new_infl
            self.newly_infl = True
        else:
            new_infl = np.argmax(self.R)
            #if self.lategame:

            if new_infl != self.funagent and self.lategame:
                print("Agent", self.num, "has influencer", self.target_influencer, "and did not switch to", self.funagent, "instead trying to switch to", new_infl)


            if self.R[new_infl] > self.R[self.target_influencer]:
                agents_list[self.target_influencer].followers.remove(self.num)
                agents_list[new_infl].followers.append(self.num)
                self.target_influencer = new_infl

        if self.target_influencer != oldinf or self.lategame:
            newrep = self.R[self.target_influencer]
            if (self.target_influencer != oldinf or self.target_influencer != self.funagent) and self.lategame:
                print("Agent", self.num, "/", oldinf, "->", self.target_influencer, "/", prevrep, "->", newrep, "(CHANGE)")
            else:
                print("Agent", self.num, "/", oldinf, "->", self.target_influencer, "/", prevrep, "->", newrep, "/ Frontrunner:", self.R[frontrun])
        if (self.target_influencer != self.funagent) and self.lategame:
            print("Agent", self.num, "/", oldinf, "->", self.target_influencer, "/", prevrep, "->", newrep, "(/)", self.R[self.funagent])
        if len(agents_list[new_infl].followers) == 1:
            agents_list[new_infl].is_new_infl = True

    def shed_influencer(self, agents_list):
        if len(self.followers) > 0:
            if self.target_influencer != -1:
                if self.num in agents_list[self.target_influencer].followers:
                    agents_list[self.target_influencer].followers.remove(self.num)
                    self.target_influencer = -1


    def get_follower_feedback(self, actor, action, agents_list, prev=False):
        """
        Returns the (doubly) projected Q-value evaluation from its followers for an action performed by actor.

        :param actor: the agent that acted (that the influencer observed).
        :param action: the action that was made.
        :return: the (doubly) projected Q value.
        """
        #v_value = self.discount * actor.get_value_function(a, b, self.agents_list, old_pos) + reward
        feedback = 0

        for number in range(len(agents_list)):
            if number == actor.num:
                pass
                # feedback += self.discount * agent.get_value_function_bin(a, b, pos)  # * agents_list[self.num].agent_rates[number]
            elif agents_list[number].target_influencer == self.num:
                # should crash if follower did not properly set follow rate
                inter = agents_list[number].get_q_function(action) * actor.get_p_function(action) * agents_list[number].infl_rate
                feedback += inter
        if prev:
            for number in range(len(agents_list)):
                inter = agents_list[number].get_q_function(action) * actor.get_p_function(action) * \
                        agents_list[number].agent_rates[self.num]
                feedback += inter
        return feedback

    def set_functional_rates(self, a, b, pos):
        # Assume CONST EXT
        self.budget = self.base_budget - self.raw_b0_rate
        if len(self.followers) > 0:
            if self.budget < 5:
                self.budget = 5
            rates = self.influencer_network.get_function_output(a, b, pos)
            agrates = np.array(rates[0:len(self.raw_agent_rates)]) * self.budget
            inflrate = 0
        else:
            if self.budget < 1.5:
                self.budget = 1.5
            rates = self.follower_network.get_function_output(a, b, pos)
            agrates = np.array(rates[0:len(self.raw_agent_rates)]) * self.budget
            inflrate = rates[-1] * self.budget

        self.raw_agent_rates = agrates  # * self.budget
        self.raw_infl_rate = inflrate  # * self.budget
        # self.raw_b0_rate = rates[-1]  # * self.budget
        self.agent_rates = (1 - np.exp(-self.raw_agent_rates))
        self.infl_rate = 1 - np.exp(-self.raw_infl_rate)
        self.b0_rate = (1 - np.exp(-self.raw_b0_rate))

        # if len(self.followers) > 50:
        #     print("Agent", self.num, sum(self.raw_agent_rates), self.agent_rates)

    def generate_rates_only(self, a, b, const_ext=False):
        self.budget = self.base_budget
        self.LRF = len(self.followers)
         # self.raw_b0_rate / self.base_budget
        b0 = self.b0 * self.b0_rate
        if const_ext:
            self.budget -= self.raw_b0_rate
            b0 = 0
        if len(self.followers) > 0:
            # if const_ext == False:
            #     b0 = 0.3 * self.b0 * (len(self.raw_agent_rates) - len(self.followers)) / len(self.raw_agent_rates)

            if self.budget < 20:
                self.budget += 2
            self.rate_alloc_count += 1
            self.b0_adjusted_rate = self.b0_rate
            # self.b0_adjusted_rate = 0.5 * ((100-len(self.followers)) / 100) * self.b0_rate
            b0 = self.b0 * self.b0_adjusted_rate
            if self.is_new_infl is True:
                rates = np.array(rate_allocate(self.alphas + self.alphas_asinfl_raw,
                                                                         # self.alphas + self.alphas_asinfl_raw + self.R[self.num],
                                                                         np.array([0]),
                                                                         budget=self.budget,
                                                                         b0=b0))
                if const_ext == True:
                    rates = np.array(rate_allocate(self.alphas + self.alphas_asinfl_raw,
                                                                             # self.alphas + self.alphas_asinfl_raw + self.R[self.num],
                                                                             np.array([0]),
                                                                             budget=self.budget,
                                                                             b0=b0))
                    self.is_new_infl = False
            else:
                # temp = self.alphas + self.alphas_asinfl
                rates = np.array(rate_allocate(self.alphas_asinfl * 10,
                                                                         np.array([0]),
                                                                         budget=self.budget,
                                                                         b0=b0 ))
            if const_ext == False:
                print("Allocating for Agent", self.num, "/", b0, "/", sum(self.alphas_asinfl), "/", rates[-1])
        elif self.target_influencer == -1 or self.target_influencer == -2:
            rates = np.array(rate_allocate(self.alphas,
                                                                     np.array([0]),
                                                                     budget=self.budget,
                                                                     b0=b0))
        elif len(self.followers) == 0:
            infl_follow = max(self.R[self.target_influencer], self.indirect_alphas[self.target_influencer])
            rates = np.array(rate_allocate(self.alphas,
                                                                     np.array([infl_follow]),
                                                                     budget=self.budget,
                                                                     b0=b0))
        if const_ext == False:
            self.raw_b0_rate = rates[-1]
            if self.num == 28:
                print("Agent 28 EXT rate:", self.raw_b0_rate, sum(self.alphas), self.b0, rates, sum(rates))
            # self.infl_rate = rates[-2]

        return rates

    def generate_agent_rates_static(self, a, b, agents_list):
        #print("Rate gen for agent", self.num)
        # generate based on beta values, with influencer
        # self.budget = self.base_budget + 0.1 * len(self.followers)
        # if len(self.followers) > 95:
        #     self.budget = self.base_budget - 10
        # else:
        #     self.budget = self.base_budget
        self.budget = self.base_budget
        self.LRF = len(self.followers)
        b0 = self.b0 * self.b0_rate
        if len(self.followers) > 0:
            b0 = b0/5
            self.rate_alloc_count += 1
            #curb0 = b0
            #curb0 = self.b0 * (len(agents_list) - len(self.followers)) / len(agents_list)
            curb0 = (b0) * (len(agents_list) - len(self.followers)) / len(agents_list)
            if self.is_new_infl is True:
                fakesum = 0
                fakesum2 = 0
                for ag in agents_list:
                    fakesum += ag.indirect_alphas[self.num]
                    fakesum2 += ag.alphas[self.num]
                fake_alphas = np.array([(fakesum + fakesum2)/100] * 100)
                rates = np.array(rate_allocate(fake_alphas ,  #self.alphas + self.alphas_asinfl_raw + self.R[self.num],
                                                                         np.array([0]),
                                                                         budget=self.budget,
                                                                         b0=curb0))
                print(self.num, "alloc start", np.sum(self.alphas_asinfl_raw), "/", rates[-1])
                #self.is_new_infl = False
            else:
                #temp = self.alphas + self.alphas_asinfl
                #temp2 = self.alphas_asinfl_raw
                #temp2 = np.where(self.alphas_asinfl == 0, 0, temp2) + 0.00001
                temp = self.alphas_asinfl + 0.00001
                # if len(self.followers) > 75 and not self.startlkdown:
                #     self.startlkdown = True
                #     temp = self.alphas + self.alphas_asinfl_raw + self.R[self.num]

                # fakesum = 0
                # fakesum2 = 0
                # for ag in agents_list:
                #     fakesum += ag.indirect_alphas[self.num]
                #     fakesum2 += ag.alphas[self.num]
                #fake_alphas = np.array([(fakesum + fakesum2) / 100] * 100)

                rates = np.array(rate_allocate(temp,
                                                                         np.array([0]),
                                                                         budget=self.budget,
                                                                         b0=curb0))


                print(self.num, "alloc rated", np.sum(self.alphas_asinfl + self.alphas), "/", rates[-1], "/", np.sum(self.alphas_asinfl), "/", curb0, "/", self.b0_rate)
                fakesum = 0
                fakesum2 = 0
                for ag in agents_list:
                    fakesum += ag.indirect_alphas[self.num]
                    fakesum2 += ag.alphas[self.num]
                print(self.num, "fake sum", fakesum, "/", fakesum2, "/", fakesum+fakesum2)
            if (self.budget - rates[-1]) == 0:
                agrates = rates[0:len(self.agent_rates)]
            else:
                agrates = rates[0:len(agents_list)] / (self.budget - rates[-1])
            self.agent_rates_as_infl = self.agent_rates_as_infl * (1 - self.update_factor) + self.update_factor * agrates
            if rates[-1] > 2:
                rates[-1] -= random.random() * 2
            agrates = self.agent_rates_as_infl * (self.budget - rates[-1])
            rates = np.concatenate((agrates, rates[-2:]))


        elif self.target_influencer == -1 or self.target_influencer == -2:
            rates = np.array(rate_allocate(self.alphas,
                                                                     np.array([0]),
                                                                     budget=self.budget,
                                                                     b0=b0))
            if (self.budget - rates[-1]) <= 0:
                agrates = rates[0:len(self.agent_rates)]
            else:
                agrates = rates[0:len(agents_list)+1] / (self.budget - rates[-1])
            self.agent_rates_as_foll = self.agent_rates_as_foll * (1 - self.update_factor) + self.update_factor * agrates[0:len(self.agent_rates)]
            agrates = self.agent_rates_as_foll * (self.budget - rates[-1])
            rates = np.concatenate((agrates, rates[-1:]))
        elif len(self.followers) == 0:
            infl_follow = max(self.R[self.target_influencer], self.indirect_alphas[self.target_influencer])
            rates = np.array(rate_allocate(self.alphas,
                                                               np.array([infl_follow]),
                                                               budget=self.budget,
                                                               b0=b0))
            if (self.budget - rates[-1]) <= 0:
                agrates = rates[0:len(self.agent_rates)]
            else:
                agrates = rates[0:len(agents_list)+1] / (self.budget - rates[-1])
            self.agent_rates_as_foll = self.agent_rates_as_foll * (1 - self.update_factor) + self.update_factor * agrates[0:len(self.agent_rates)]
            agrates = self.agent_rates_as_foll * (self.budget - rates[-1])
            rates = np.concatenate((agrates, rates[-1:]))
        self.raw_agent_rates = np.array(rates[0:len(self.raw_agent_rates)]) #* self.budget

        self.raw_infl_rate = rates[len(self.raw_agent_rates)] #* self.budget
        self.raw_b0_rate = rates[-1] #* self.budget
        self.agent_rates = (1 - np.exp(-self.raw_agent_rates))
        self.infl_rate = 1 - np.exp(-self.raw_infl_rate)
        self.b0_rate = (1 - np.exp(-self.raw_b0_rate))
        if self.num == 28:
            print("Agent 28 EXT rate:", self.raw_b0_rate, sum(self.alphas), self.b0, rates, sum(rates))

    def generate_agent_rates_static_const_ext(self, a, b, ts):
        #print("Rate gen for agent", self.num)
        # generate based on beta values, with influencer
        # self.budget = self.base_budget + 0.1 * len(self.followers)
        # if len(self.followers) > 95:
        #     self.budget = self.base_budget - 10
        # else:
        #     self.budget = self.base_budget
        self.budget = self.base_budget - self.raw_b0_rate #+ len(self.followers) / 10
        if self.budget < 1:
            self.budget = 1
        if len(self.followers) > 0:
            if self.budget < 5 and len(self.followers) > 4:
                self.budget = 5 + random.random()
            if self.is_new_infl is True:
                rates = np.array(rate_allocate(self.alphas + self.alphas_asinfl + 0.0001,      #self.alphas_asinfl_raw,
                                                                         np.array([0]),
                                                                         budget=self.budget,
                                                                         b0=0))
                self.is_new_infl = False
            else:
                rates = np.array(rate_allocate(self.alphas_asinfl + 0.0001,
                                                                         np.array([0]),
                                                                         budget=self.budget,
                                                                         b0=0))
            if self.budget < 0:
                agrates = rates[0:len(self.agent_rates)]
            else:
                agrates = rates[0:len(self.agent_rates)] / self.budget
            var = random.random()
            if var < 0.05:
                agrates = np.zeros(len(self.agent_rates))
                for z in range(agrates.size):
                    agrates[z] = 1 / agrates.size
            self.agent_rates_as_infl = self.agent_rates_as_infl * (1 - self.update_factor) + self.update_factor * agrates
            agrates = self.agent_rates_as_infl * self.budget
            rates = np.concatenate((agrates, rates[-2:]))
        elif self.target_influencer == -1 or self.target_influencer == -2:
            rates = np.array(rate_allocate(self.alphas,
                                                                     np.array([0]),
                                                                     budget=self.budget,
                                                                     b0=0))
            if self.budget < 0:
                agrates = rates[0:len(self.agent_rates)]
            else:
                agrates = rates[0:len(self.agent_rates)+1] / self.budget
            self.agent_rates_as_foll = self.agent_rates_as_foll * (1 - self.update_factor) + self.update_factor * agrates[0:len(self.agent_rates)]
            agrates = self.agent_rates_as_foll * self.budget
            rates = np.concatenate((agrates, rates[-1:]))
        elif len(self.followers) == 0:
            infl_follow = max(self.R[self.target_influencer], self.indirect_alphas[self.target_influencer])
            rates = np.array(rate_allocate(self.alphas,
                                                               np.array([infl_follow]),
                                                               budget=self.budget,
                                                               b0=0))
            if self.budget < 0:
                agrates = rates[0:len(self.agent_rates)]
            else:
                agrates = rates[0:len(self.agent_rates)+1] / self.budget
            self.agent_rates_as_foll = self.agent_rates_as_foll * (1 - self.update_factor) + self.update_factor * agrates[0:len(self.agent_rates)]
            agrates = self.agent_rates_as_foll * self.budget
            rates = np.concatenate((agrates, rates[-1:]))
        else:
            print("Agent", self.num, "budget:", self.budget)
            rates = np.array(rate_allocate(self.alphas,
                                                               np.array([0]),
                                                               budget=self.budget,
                                                               b0=0))
        self.raw_agent_rates = np.array(rates[0:len(self.raw_agent_rates)]) #* self.budget

        self.agent_rates = (1 - np.exp(-self.raw_agent_rates))

        self.raw_infl_rate = rates[len(self.raw_agent_rates)]
        self.infl_rate = 1 - np.exp(-self.raw_infl_rate)

    def generate_agent_rates(self, a, b):
        """
        :param a:
        :param b:
        :return:
        """
        rates = self.observer_network.get_function_output(a, b, self.position)
        self.raw_agent_rates = np.array(rates[0:len(self.raw_agent_rates)]) * self.budget
        self.raw_infl_rate = rates[len(self.raw_agent_rates)]
        self.raw_b0_rate = rates[-1] * self.budget

        self.agent_rates = (1-np.exp(-self.raw_agent_rates))
        self.infl_rate = (1 - np.exp(-self.raw_infl_rate))
        self.b0_rate = (1 - np.exp(-self.raw_b0_rate))

    def get_q_function(self, observed_topic):
        return self.pq(observed_topic, self.interest, self.c, self.rang)

    def get_p_function(self, product_topic):
        return self.pq(product_topic, self.interest, self.c, self.rang)

    def get_util_pq(self, product_topic, producer):
        if producer.num == self.num:
            return 0
        return self.get_q_function(product_topic) * producer.get_p_function(product_topic) * self.agent_rates[producer.num]

    def get_util_pq_raw(self, product_topic, producer):
        if producer.num == self.num:
            return 0
        return self.get_q_function(product_topic) * producer.get_p_function(product_topic)

    def get_value_function(self, a, b, pos=None):
        if self.is_projecting:
            assert self.avg_alpha is not None
            if pos is None:
                pos = self.position
            v = self.policy_value.get_sum_value(a, b, pos)[0]
            val = (v - self.avg_alpha) / self.avg_alpha
            bound = 0.885
            val = ((val + 1) / 2) / bound
            if val > 1:
                val = 1
            elif val < 0:
                val = 0
            return np.array([val])
        else:
            if pos is None:
                pos = self.position
            return self.policy_value.get_sum_value(a, b, pos)

    def get_comm_value_function(self, a, b, agents_list, new_pos=None, debug=False, agent_poses=None):
        sum = 0
        if debug:
            assert agent_poses is not None
            for num, agent in enumerate(agents_list):
                sum += agent.policy_value.get_sum_value(a, b, np.array(agent_poses[num]))
        else:
            assert new_pos is not None
            for agent in agents_list:
                if agent.num == self.num:
                    sum += agent.policy_value.get_sum_value(a, b, new_pos)
                else:
                    sum += agent.policy_value.get_sum_value(a, b, agent.position)
        return sum

    def get_learned_action(self, state):
        a = state["agents"]
        b = state["apples"]

        #actions = [0, 1, 4]
        actions = range(0, len(state["agents"].flatten()))
        output = self.policy_network.get_function_output(a, b, self.position)

        action = random.choices(actions, weights=output)[0]
        return action

    def get_best_action(self, state, discount, agents_list):
        a = state["agents"]
        b = state["apples"]

        action = 0
        best_val = 0
        if self.debug:
            print("==========Making Decision for Agent "+str(self.num)+"===========")
            print(list(a.flatten()), list(b.flatten()))

        #for act in range(len(action_vectors)):
        for act in [0, 1, 4]:
            val, new_a, new_b, new_pos = calculate_ir(a, b, self.position, act)
            rew = val
            val += discount * self.get_comm_value_function(new_a, new_b, agents_list, new_pos=new_pos)
            if self.debug:
                print("Action " + str(action_vectors[act]) + " has expected value " + str(val) + "; immediate reward " + str(rew))
            if val > best_val:
                action = act
                best_val = val

        # self.same_actions += (action == nearest(state, self.position))
        if self.debug:
            print("Ultimately chose " + str(action_vectors[action]) + " with expected value " + str(best_val) + ".")

        return action

    def get_action(self, state, discount=0.99, agents_list=None):
        if self.policy == "value_function":
            assert agents_list is not None
            return self.get_best_action(state, discount, agents_list)
        elif self.policy == "learned_policy":
            return self.get_learned_action(state)
        elif self.policy == "random":
            length = len(state["apples"].flatten())
            return random.choice(range(length))
        else:
            return self.policy(state, self.position)





