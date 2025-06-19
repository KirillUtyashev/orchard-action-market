import numpy as np
from scipy.optimize import minimize

"""
Allocation Code

For the Rate Allocation problem - maximizes Sigma Q * following rate
"""


#alphas = np.array([0.3152559,  0.31565792, 0.31479956, 0.36986502])

# def alloc(mat, alphas):
#     """
#     Allocation problem expression. Multiply by -1000 to convert minimization problem
#     to maximization problem
#     """
#     return -(np.sum((1 - np.exp(-mat)) * alphas))
#
#
# def find_allocs(alphas, agent_id, beta, budget=1.0):
#     """
#     Find the following rates for other agents. For the agent itself, we keep the following rate of 1.
#     The reason is that the agent can fully assess its own internal information and a following rate of < 1 wouldn't make sense.
#     """
#     # 1) Build “others” by removing the α for “self = agent_id”
#     others = np.delete(alphas, agent_id)   # shape = (N-1,)
#     n_others = len(others)
#
#     # 2) If there is exactly one “other” agent, we can still run the same softmax initialization,
#     #    but be cautious of log(0). We’ll clamp p slightly below 1.0 in that case.
#     if n_others == 1:
#         # Softmax([α]) = [1.0] ⇒ raw = -log(1-1.0) = ∞, so clamp:
#         p = np.array([0.999])
#         raw_others = -np.log(1.0 - p)         # a large finite number
#     else:
#         exps = np.exp(others - np.max(others))
#         p = exps / np.sum(exps)            # shape (N-1,), sum=1
#         raw_others = -np.log(1.0 - p)          # φ = (1 - e^{-λ}) ⇒ λ = -log(1 - p)
#
#     # We now have “raw_others” of length (N-1).  We want the initial guess
#     # for all λ’s to sum to 'budget' once we include λ_{A,y}.
#     # So pick λ_act_guess = budget/(n_others+1), and scale raw_others
#     # so that sum(raw_others) = budget - λ_act_guess.
#     lam_act_guess = budget / (n_others + 1.0)
#     total_raw = np.sum(raw_others)
#     if total_raw > 0:
#         scale = (budget - lam_act_guess) / total_raw
#         x0_others = raw_others * scale
#     else:
#         # If raw_others all zero, split (budget - lam_act_guess) equally
#         x0_others = np.full(n_others, (budget - lam_act_guess) / n_others)
#
#     # Build the combined initial guess vector of length (n_others + 1):
#     #   [ λ_{y,z_1}, λ_{y,z_2}, …, λ_{y,z_{N-1}},   λ_{A,y} ]
#     x0 = np.concatenate([x0_others, [lam_act_guess]])
#
#     # 3) Define the new objective that includes β·(1 - e^{-λ_act})
#     def objective(x):
#         # x[:n_others]    = λ_{y,z} for the (N-1) other agents
#         # x[n_others]     = λ_{A,y}
#         foll_vec = x[:n_others]
#         lam_act = x[n_others]
#
#         # The “follow” portion: sum_i α_i * (1 - exp(-λ_i))
#         # Our old alloc() returns negative of that:
#         val_follow = alloc(foll_vec, others)   # = -(∑ α_i·(1 - e^{-λ_i}))
#         # The “act” portion:        β * (1 - exp(-λ_act))
#         val_act = - beta * (1.0 - np.exp(-lam_act))
#
#         # Total objective to *minimize* = [–(follow_sum) – β·φ_act]
#         return val_follow + val_act
#
#     # 4) Constraints:
#     #    (a) Sum of all λ’s = budget
#     #    (b) each λ_i ≥ 0
#     cons = [
#         {
#             "type": "eq",
#             "fun": lambda x: np.sum(x) - budget
#         },
#         {
#             "type": "ineq",
#             "fun": lambda x: x    # means x[i] ≥ 0 for all i
#         }
#     ]
#
#     # 5) Solve via SLSQP
#     sol = minimize(
#         objective,
#         x0,
#         method="SLSQP",
#         constraints=cons,
#         options={"ftol": 1e-9, "maxiter": 200}
#     )
#
#     if not sol.success:
#         # Fallback: split equally among all (n_others + 1) lambdas
#         equal_val = budget / (n_others + 1.0)
#         sol_x = np.full(n_others + 1, equal_val)
#     else:
#         sol_x = sol.x
#
#     # 6) Extract results
#     lambdas_others = sol_x[:n_others]   # the optimized follow‐rates for z≠y
#     lam_act = sol_x[n_others]    # the optimized acting rate
#
#     # 7) Re‐insert “self” = 1.0 at position = agent_id, to get a length-N array:
#     full_foll = np.insert(lambdas_others, agent_id, 1.0)
#
#     return full_foll, lam_act

def alloc(mat, alphas):
    return -((10000 * np.sum((1 - np.exp(-mat)) * alphas)))


def find_allocs(alphas, it=0, budget=4):

    x0 = (alphas / np.sum(alphas))
    x0 = -np.log(1-x0)

    x0 = (x0 / np.sum(x0)) * budget

    sum_constraint = lambda x: -np.sum(x) + budget
    cons = [{'type': 'eq', 'fun': sum_constraint}, {'type': 'ineq', 'fun': lambda x: x}]
    ret = np.array([budget / len(x0)] * len(x0))
    res = minimize(alloc, x0, args=alphas, method="SLSQP", constraints=cons)
    if res.success == False:
        if it == 40:
            ret = find_allocs(alphas, it=it + 1, budget=budget)
        else:
            print("Not Success", alphas)
            return ret
    else:
        ret = res.x
    return ret


max_thing = 0


def rate_allocate(alphas1, alphas2, it=0, b0=0.0, budget=4):
    budget = np.clip(np.array([budget]), 2,1000)[0]
    alphas = np.concatenate((alphas1.flatten(), np.array(alphas2.flatten()) * 2, np.array([b0])))
    global max_thing

    if (np.sum(alphas) - b0 > max_thing):
        max_thing = np.sum(alphas) - b0
    alphas -= max_thing * 0.01 * 0.01
    alphas = np.clip(alphas, 1e-8, 1000)

    alphas = np.power(alphas, 1.1)

    alphas = np.where(alphas < 0, 0, alphas)
    new_alphas = find_allocs(alphas, it, budget)

    new_alphas = np.where(alphas <= 0, 0, new_alphas)

    if new_alphas[-1] > 1:
        new_alphas[-1] -= np.random.random()
    summed = np.sum(new_alphas)
    if summed != 0:
        new_alphas = new_alphas * budget / summed
    return new_alphas


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
