import numpy as np
from alloc.allocation import rate_allocate

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

"""

"""

class OrchardAgent:
    def __init__(self, policy, id=0, num_agents=1, budget=1, debug=False, cm_parameters=None):
        self.position = np.array([0, 0])
        self.policy = policy
        self.debug = debug
        self.num = id
        """
        Utility Type
        """
        self.utility = "value_function"

        """
        Policy Network: the Actor network.
        Value Network: the Value network that judges (in lieu of the P/Q functions)
        Observer Network: the Observing "rates" condensed within an updating function (same with Influencer Network)
        """
        self.policy_network = None
        self.value_network = None
        self.follower_network = None
        self.influencer_network = None
        """
        Starting Following Rates. 
        """
        self.agent_rates = np.zeros(num_agents)
        for i in range(self.agent_rates.size):
            self.agent_rates[i] = 1 / budget

        self.agent_rates_as_infl = np.zeros(num_agents)
        for i in range(self.agent_rates.size):
            self.agent_rates_as_infl[i] = 1 / budget

        self.agent_rates_as_foll = np.zeros(num_agents)
        for i in range(self.agent_rates.size):
            self.agent_rates_as_foll[i] = 1 / budget

        """
        Statistical Beta/Alpha values initialization.
        """
        self.alphas = np.zeros(num_agents)
        self.beta = 0
        self.times = 0
        self.alphas_raw = np.zeros(num_agents)

        self.indirect_alphas = np.zeros(num_agents)
        self.indirect_alphas_raw = np.zeros(num_agents)

        self.base_budget = budget
        self.raw_acting_rate = budget / 2
        self.acting_rate = 1 - np.exp(-1 * budget)

        """
        For storage use
        """
        self.raw_agent_rates = np.zeros(num_agents)
        self.infl_rate = 1
        self.raw_infl_rate = 0
        self.raw_b0_rate = 0

        # Target Influencer: the Influencer that the agent believes should lead the market.
        self.target_influencer = -2

        """
        Proto-Influencer
        """
        self.followers = []
        self.max_followers = 0

        self.alphas_asinfl = np.zeros(num_agents)
        self.alphas_asinfl_raw = np.zeros(num_agents)

        self.PR = np.zeros(num_agents)
        self.PB = np.zeros(num_agents)
        self.R = np.zeros(num_agents)

        self.adjs = []
        self.newly_infl = True
        self.newly_infl_train = False
        self.is_new_infl = False

        """
        Content Market Emergence Parameters
        """
        self.cmp = cm_parameters
        if cm_parameters != None:
            self.base_a = self.cmp["base_a"]
            self.varAlpha = self.cmp["alpha"]
            self.varKappa = self.cmp["kappa"]
            self.n0 = self.cmp["n0"]

        self.LRF = 0
        self.lategame = False
        self.funagent = -1
        self.rate_alloc_count = 0
        self.startlkdown = 0

        self.update_factor = 0.01
        self.discount_factor = 0.003
        self.beta_discount_factor = 0.02

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

    def set_functional_rates(self, a, b, pos):
        """
        Sets the following rates based on functional output (i.e. the output of the influencer network / follower network).
        """
        # Assuming Constant External (or Acting) Rate
        self.budget = self.base_budget - self.raw_acting_rate
        if len(self.followers) > 0:
            if self.budget < 5:
                self.budget = 5
            rates = self.influencer_network.get_function_output(a, b, pos)
            agrates = np.array(rates[0:len(self.raw_agent_rates)]) * self.budget
            inflrate = 0 # Should not follow an influencer
        else:
            if self.budget < 1.5:
                self.budget = 1.5
            rates = self.follower_network.get_function_output(a, b, pos)
            agrates = np.array(rates[0:len(self.raw_agent_rates)]) * self.budget
            inflrate = rates[-1] * self.budget

        self.raw_agent_rates = agrates  # * self.budget
        self.raw_infl_rate = inflrate  # * self.budget
        self.agent_rates = (1 - np.exp(-self.raw_agent_rates))
        self.infl_rate = 1 - np.exp(-self.raw_infl_rate)

    def generate_rates_only(self, a, b, const_ext=False):
        """
        Gets the optimal rates; mostly used for training / alloc.'ing external (acting) rate.
        :param a: Unused.
        :param b: Unused.
        :param const_ext: Whether this generation is for a Constant External (Acting) Rate (i.e. b0 = 0) or not.
        :return:
        """
        self.budget = self.base_budget
        self.LRF = len(self.followers)
        b0 = max(5, self.beta + 5)
        if self.LRF >= 6:
            b0 = max(0, self.beta * np.power((10 - self.LRF) / 10, 2))
        if const_ext:
            self.budget -= self.raw_acting_rate
            b0 = 0
        if len(self.followers) > 0:
            self.budget = max(self.budget, 1)
            self.rate_alloc_count += 1
            if self.is_new_infl is True:
                rates = np.array(rate_allocate(self.alphas + self.alphas_asinfl_raw,
                                               np.array([0]),
                                               budget=self.budget,
                                               b0=b0))
                if const_ext:
                    rates = np.array(rate_allocate(self.alphas + self.alphas_asinfl_raw,
                                                   np.array([0]),
                                                   budget=self.budget,
                                                   b0=b0))
                    self.is_new_infl = False
            else:
                rates = np.array(rate_allocate(self.alphas_asinfl * 20,
                                               np.array([0]),
                                               budget=self.budget,
                                               b0=b0))
            if not const_ext:
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
        if not const_ext:
            self.raw_b0_rate = rates[-1]
            self.raw_acting_rate = rates[-1]
            if self.num == 28:
                print("Agent 28 EXT rate:", self.raw_b0_rate, sum(self.alphas), self.beta, rates, sum(rates))

        return rates

    def generate_agent_rates_static(self, a, b, agents_list):
        self.budget = self.base_budget
        self.LRF = len(self.followers)
        b0 = self.beta * 10
        if len(self.followers) > 0:
            self.rate_alloc_count += 1
            temp_alphas = self.alphas_asinfl + 0.00001
            if self.is_new_infl is True:
                fakesum = 0
                fakesum2 = 0
                for ag in agents_list:
                    fakesum += ag.indirect_alphas[self.num]
                    fakesum2 += ag.alphas[self.num]
                temp_alphas = np.array([(fakesum + fakesum2)/100] * 100)
            rates = np.array(rate_allocate(temp_alphas,
                                           np.array([0]),
                                           budget=self.budget,
                                           b0=b0))
            if (self.budget - rates[-1]) <= 0:
                agrates = rates[0:len(self.agent_rates)]
            else:
                agrates = rates[0:len(agents_list)] / (self.budget - rates[-1])
            self.agent_rates_as_infl = self.agent_rates_as_infl * (1 - self.update_factor) + self.update_factor * agrates
            agrates = self.agent_rates_as_infl * (self.budget - rates[-1])
            rates = np.concatenate((agrates, rates[-2:]))

        elif self.target_influencer == -1 or self.target_influencer == -2:
            if len(self.followers) == 0:
                infl_follow = max(self.R[self.target_influencer], self.indirect_alphas[self.target_influencer])
            else:
                infl_follow = 0
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
        self.raw_acting_rate = rates[-1] #* self.budget
        self.agent_rates = (1 - np.exp(-self.raw_agent_rates))
        self.infl_rate = 1 - np.exp(-self.raw_infl_rate)
        self.acting_rate = (1 - np.exp(-self.raw_acting_rate))
        if self.num == 28 and self.debug:
            print("Agent 28 EXT rate:", self.raw_acting_rate, sum(self.alphas), self.b0, rates, sum(rates))

    def get_utility(self, a, b, pos=None):
        if pos is None:
            pos = self.position

        if self.utility == "value_function":
            return self.value_network.get_value_function(a, b, pos)[0]

    def get_value_function(self, a, b, pos=None):
        if pos is None:
            pos = self.position
        return self.value_network.get_value_function(a, b, pos)
    def get_comm_value_function(self, a, b, agents_list, new_pos=None, debug=False, agent_poses=None):
        sum = 0
        if debug:
            assert agent_poses is not None
            for num, agent in enumerate(agents_list):
                sum += agent.policy_value.get_value_function(a, b, np.array(agent_poses[num]))
        else:
            assert new_pos is not None
            for agent in agents_list:
                if agent.num == self.num:
                    sum += agent.policy_value.get_value_function(a, b, new_pos)
                else:
                    sum += agent.policy_value.get_value_function(a, b, agent.position)
        return sum

    def get_learned_action(self, state):
        """
        Gets an action based off the Actor Network policy.
        :param state:
        :return:
        """
        a = state["agents"]
        b = state["apples"]

        actions = [0, 1, 4]
        #actions = range(0, 3)
        output = self.policy_network.get_function_output(a, b, self.position)


        action = random.choices(actions, weights=output)[0]
        return action

    def get_learned_action_record(self, state):
        a = state["agents"]
        b = state["apples"]
        pos = state["pos"]

        actions = [0, 1, 4]
        #actions = range(0, 3)
        output = self.policy_network.get_function_output(a, b, pos)


        # action = random.choices(actions, weights=output)[0]
        return output

    def get_best_action(self, state, discount, agents_list):
        """
        Gets the "best" action based on the Value Function.
        :param state:
        :param discount:
        :param agents_list:
        :return:
        """
        a = state["agents"]
        b = state["apples"]

        action = 0
        best_val = 0
        if self.debug:
            print("==========Making Decision for Agent "+str(self.num)+"===========")
            print(list(a.flatten()), list(b.flatten()))
        for act in [0, 1, 4]:
            val, new_a, new_b, new_pos = calculate_ir(a, b, self.position, act)
            rew = val
            val += discount * self.get_comm_value_function(new_a, new_b, agents_list, new_pos=new_pos)
            if self.debug:
                print("Action " + str(action_vectors[act]) + " has expected value " + str(val) + "; immediate reward " + str(rew))
            if val > best_val:
                action = act
                best_val = val
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



    """
    Agentic Feedback Functionality
    """