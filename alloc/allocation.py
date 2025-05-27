import numpy as np
import scipy as sp
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import math
import random

"""
Allocation Code

For the Rate Allocation problem - maximizes Sigma Q * following rate
"""


#alphas = np.array([0.3152559,  0.31565792, 0.31479956, 0.36986502])

def alloc(mat, alphas):
    return -((10000 * np.sum((1 - np.exp(-mat)) * alphas)))



def find_allocs(alphas, it=0, budget=4):

    x0 = (alphas / np.sum(alphas))
    x0 = -np.log(1-x0)
    #print(x0)

    x0 = (x0 / np.sum(x0)) * budget
    #print(x0)
    #print(x0)
    # print(x0)
    # print("Initial", alloc(x0, alphas))
    sum_constraint = lambda x: -np.sum(x) + budget
    cons = [{'type': 'eq', 'fun': sum_constraint}, {'type': 'ineq', 'fun': lambda x: x}]
    ret = np.array([budget / len(x0)] * len(x0))
    #res = minimize(alloc, x0, args=alphas, method="trust-constr", constraints=cons)
    res = minimize(alloc, x0, args=alphas, method="SLSQP", constraints=cons)
    if res.success == False:
        if it == 40:
            ret = find_allocs(alphas, it=it + 1, budget=budget)
        else:
            print("Not Success", alphas)
            return ret
    else:
        ret = res.x
        #print("Final:", alloc(ret, alphas))
    return ret

def roundabout_find_allocs(alphas1, alphas2, it=0, budget=4):
    alphas = np.concatenate((alphas1.flatten(), alphas2.flatten()))
    alphas = find_allocs(alphas, it, budget)
    #print(alphas[0:len(alphas1)], alphas[len(alphas1):])
    return alphas[0:len(alphas1)], alphas[len(alphas1):]

def roundabout_find_allocs_with_b0(alphas1, alphas2, it=0, b0=0.0, budget=4):
    # print("Budget:" , budget)

    alphas = np.concatenate((alphas1.flatten(), alphas2.flatten(), np.array([b0])))
    # print(alphas)
    alphas = find_allocs(alphas, it, budget)
    #print(alphas[0:len(alphas1)], alphas[len(alphas1):])
    summed = np.sum(alphas)
    return alphas[0:len(alphas1)], alphas[len(alphas1):len(alphas1)+len(alphas2)], alphas[-1] + budget - summed



max_thing = 0

def rate_allocate(alphas1, alphas2, it=0, b0=0.0, budget=4):
    # print("Budget:" , budget)
    budget = np.clip(np.array([budget]), 2,1000)[0]
    alphas = np.concatenate((alphas1.flatten(), np.array(alphas2.flatten()) * 2, np.array([b0])))
    global max_thing
    # addi = 0
    # for ia in alphas1:
    #     if ia > 1e-8:
    #         addi += 1
    # b0 *= (0.2 + 0.8 * (100 - addi)/100)
    if (np.sum(alphas) - b0 > max_thing):
        max_thing = np.sum(alphas) - b0
    alphas -= max_thing * 0.01 * 0.01
    alphas = np.clip(alphas, 1e-8, 1000)

    alphas = np.power(alphas, 1.1)
   # alphas = alphas / 5

    alphas = np.where(alphas < 0, 0, alphas)
    new_alphas = find_allocs(alphas, it, budget)

    new_alphas = np.where(alphas <= 0, 0, new_alphas)

    # if budget == 25:
    #     print(summed)
    if new_alphas[-1] > 1:
        new_alphas[-1] -= np.random.random()
    summed = np.sum(new_alphas)
    if summed != 0:
        new_alphas = new_alphas * budget / summed
    # if budget == 25:
    #     print(np.sum(new_alphas))
    return new_alphas
    #return np.ones(len(alphas)) * 5


if __name__ == "__main__":
    #ex_alpha = 0.0036245971366404782
    #ex_alpha = 0.01
    ex_alpha = 0.05
    no_agents = 20
    #alphas = np.zeros(100)
    alphas = np.zeros(100)
    for i in range(no_agents):
        alphas[i] = ex_alpha / no_agents
    # alphas[0] = 0.0055
    # alphas[1] = 0.001
    # alphas[2] = 0.001

    #alphas[0] = 0.0004885562105229612
    #alphas[0] = 0.0024
    print(alphas)
    print(sum(alphas))
    b0 = 100
    mar = 1
    alphas *= mar
    b0 *= mar
    rates = np.array(rate_allocate(alphas,
                                   np.array([0]),
                                   budget=25,
                                   b0=b0))
    print(np.sum(rates))
    print(rates)
    # alphas = [0.00262189, 0.00366571, 0.00552498, 0.00784672, 0.01172289, 0.01496106,
    #     0.01787477, 0.   ,      0.01641911, 0.01201591, 0.09265304, 0.02      ]
    sum_constraint = lambda x: -np.sum(x) + 1
    cons = [{'type': 'ineq', 'fun': sum_constraint}, {'type':'ineq', 'fun': lambda x: x}]
    suballoc = np.array([2.796e-04,  2.852e-04,  2.763e-04, 9.992e-01])
    #res = find_allocs(alphas)
    #print("Result:", res)
    #print((1 - np.exp(-res)))
