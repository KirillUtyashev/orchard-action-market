import numpy as np
from scipy.optimize import Bounds, minimize


class RateUpdater:
    def __init__(self, neigh_ids, budget):
        self.neigh_ids = list(neigh_ids) # global IDs of neighbors
        self.id2local = {gid: i for i, gid in enumerate(self.neigh_ids)}
        self.n = len(self.neigh_ids)
        self.M = float(budget)

    # accept LOCAL index (unchanged)
    def update_all(self, r_sum_by_local):
        # budget = np.clip(np.array([budget]), 2, 1000)[0]
        # alphas = np.concatenate((alphas1.flatten(), np.array(alphas2.flatten()) * 2, np.array([b0])))

        # Regularization
        # const_1 = 0.0001
        #
        # global max_thing
        # if (np.sum(alphas) - b0 > max_thing):
        #     max_thing = np.sum(alphas) - b0
        # alphas -= max_thing * const_1

        alphas = np.clip(list(r_sum_by_local.values()), 1e-8, 1000)

        # Encouraging putting more weight into strong candidates
        # const_2 = 2
        #
        # alphas = np.power(alphas, const_2)

        alphas = np.where(alphas < 0, 0, alphas)
        new_alphas = find_allocs_closed_form(alphas, self.M)

        new_alphas = np.where(alphas <= 0, 0, new_alphas)

        summed = np.sum(new_alphas)
        if summed != 0:
            new_alphas = new_alphas * self.M / summed
        return new_alphas

    # NEW: accept GLOBAL id safely
    def update_many_by_global(self, js_global, r_sum_list):
        """
        Batch update using multiple neighbors' feedback (global IDs).
        - js_global: iterable of global neighbor IDs (any order, no duplicates)
        - Rsum_list: iterable of the same length with summed rewards for each ID
        - T: ticks in the interval
        """
        # build {j_local: Rsum_j} for observed neighbors only
        r_sum_by_local = {}
        for gid in js_global:
            try:
                j_local = self.id2local[gid]
            except KeyError:
                raise ValueError(f"{gid} is not a neighbor. Valid: {self.neigh_ids}")
            r_sum_by_local[j_local] = float(r_sum_list[gid])

        # single low-variance, softmax-coupled step
        return self.update_all(r_sum_by_local)


def _obj(x, alphas, scale=1.0):
    # minimize negative to maximize original
    return -scale * np.sum((1.0 - np.exp(-x)) * alphas)


def _grad(x, alphas, scale=1.0):
    # d/dx_i = - scale * alphas_i * exp(-x_i)
    return -scale * alphas * np.exp(-x)


def find_allocs_slsqp(alphas, budget=4.0, scale=1.0):
    alphas = np.asarray(alphas, dtype=float)
    n = alphas.size
    alphas = np.clip(alphas, 0.0, np.inf)

    p = alphas / max(alphas.sum(), 1e-12)
    p = np.clip(p, 0.0, 1 - 1e-12)
    x0 = -np.log(1.0 - p)
    s = x0.sum()
    x0 = (x0 / s * budget) if s > 0 else np.full(n, budget / n)

    def c_eq(x):     # sum(x) - budget = 0
        return np.sum(x) - budget

    def c_eq_jac(x):
        return np.ones_like(x)

    bounds = Bounds(0.0, np.inf)

    def solve(start):
        return minimize(
            _obj, start, args=(alphas, scale),
            method="SLSQP",
            jac=_grad,
            bounds=bounds,
            constraints=[{"type": "eq", "fun": c_eq, "jac": c_eq_jac}],
            options={"ftol": 1e-9, "maxiter": 1000, "disp": False}
        )

    res = solve(x0)
    # tries = 0
    # while (not res.success) and (tries < max_retries):
    #     # try uniform, then random feasible starts
    #     if tries == 0:
    #         start = np.full(n, budget / n)
    #     else:
    #         noise = np.random.rand(n)
    #         start = (noise / noise.sum()) * budget
    #     res = solve(start)
    #     tries += 1

    if res.success:
        return res.x
    else:
        print(f"SLSQP failed: {res.message}")
        return
    # fallback: feasible uniform
    # return np.full(n, budget / n)


def find_allocs_closed_form(alphas, budget=4.0):
    """
    Closed-form solution to:
        max_{λ_j >= 0}  sum_j (1 - e^{-λ_j}) * α_j
        s.t.            sum_j λ_j = M

    Using: e^{-λ_j} α_j = μ  =>  λ_j = ln(α_j) - ln(μ),
    and sum_j λ_j = M  =>  ln(μ) = (sum_j ln(α_j) - M) / m   over active j.

    Handles α_j <= 0 by excluding them (λ_j = 0) and re-solving on the reduced set.
    """
    alphas = np.asarray(alphas, dtype=float)
    n = alphas.size
    if n == 0:
        return np.array([], dtype=float)

    lambdas = np.zeros(n, dtype=float)

    # Start with strictly positive α_j only; non-positive get λ_j = 0.
    active = alphas > 0.0
    if not np.any(active):
        return lambdas  # all zeros

    # Active-set loop: drop entries that become negative and re-solve.
    while True:
        idx = np.where(active)[0]
        m = idx.size
        if m == 0:
            break

        # ln(μ) = (Σ ln α_j - M) / m   over active j
        sum_log_alpha = np.sum(np.log(alphas[idx]))
        ln_mu = (sum_log_alpha - budget) / m

        # λ_j = ln(α_j) - ln(μ)
        lam = np.log(alphas[idx]) - ln_mu

        # Identify negatives; those must be set to 0 and removed from active set
        neg_mask = lam < 0.0
        if np.any(neg_mask):
            # deactivate those and continue the loop
            active[idx[neg_mask]] = False
            # (Do not fix others yet; recompute cleanly next iteration)
            continue

        # All non-negative => we're done; write them out
        lambdas[idx] = lam
        break

    # Numerical guard: ensure exact feasibility within tolerance by simple rescale (rarely needed)
    s = lambdas.sum()
    if s > 0:
        lambdas *= (budget / s)

    return lambdas
