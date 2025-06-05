import numpy as np
from scipy.optimize import minimize

"""
Allocation Code

For the Rate Allocation problem - maximizes Sigma Q * following rate
"""


#alphas = np.array([0.3152559,  0.31565792, 0.31479956, 0.36986502])

def alloc(mat, alphas):
    """
    Allocation problem expression. Multiply by -1000 to convert minimization problem
    to maximization problem
    """
    return -(np.sum((1 - np.exp(-mat)) * alphas))


def find_allocs(alphas, agent_id, budget=1):
    """
    Find the following rates for other agents. For the agent itself, we keep the following rate of 1.
    The reason is that the agent can fully assess its own internal information and a following rate of < 1 wouldn't make sense.
    """
    alphas = np.delete(alphas, agent_id)
    if len(alphas) == 1:
        sol = np.array([budget])
    else:
        # 1) softmax over alphas
        exps = np.exp(alphas - np.max(alphas))        # subtract max for numerical stability
        foll_rates = exps / np.sum(exps)                        # now 0 < p_i < 1, sum(p)=1
        foll_rates = -np.log(1 - foll_rates)  # Convert following rates from rate to raw lambdas

        foll_rates = (foll_rates / np.sum(foll_rates)) * budget  # Normalize raw lambdas
        sum_constraint = lambda x: -np.sum(x) + budget  # Budget constraint
        cons = [{'type': 'eq', 'fun': sum_constraint}, {'type': 'ineq', 'fun': lambda x: x}]  # Optimization problem
        sol = minimize(alloc, foll_rates, args=alphas, method="SLSQP", constraints=cons)  # Solve the problem
        if not sol.success:
            print("Not Success", alphas)
            sol = np.array([budget / len(foll_rates)] * len(foll_rates))  # Equal following rates
        else:
            sol = sol.x
    return np.insert(sol, agent_id, 1)


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
