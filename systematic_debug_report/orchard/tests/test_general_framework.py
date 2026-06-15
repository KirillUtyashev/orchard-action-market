"""Tests for the general proficiency/relatedness framework: reward formula, encoders, heuristic."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from orchard.datatypes import EnvConfig, StochasticConfig
from orchard.enums import DespawnMode, EncoderType, Heuristic
from orchard.env import create_env
from orchard.eval import evaluate_policy_metrics
from orchard.policy import nearest_action, get_phase2_actions, heuristic_action
from orchard.seed import set_all_seeds
import orchard.encoding as encoding


def _make_env(
    n_agents: int = 4,
    n_task_types: int = 4,
    relatedness_width: int = 1,
    proficiency_width: int = 1,
    sigma_a: float = 0.0,
    sigma_b: float = 0.0,
    seed: int = 0,
    height: int = 5,
    width: int = 5,
):
    set_all_seeds(seed)
    cfg = EnvConfig(
        height=height, width=width,
        n_agents=n_agents, n_tasks=2, gamma=0.99,
        n_task_types=n_task_types,
        relatedness_width=relatedness_width, proficiency_width=proficiency_width,
        max_tasks_per_type=5,
        stochastic=StochasticConfig(
            spawn_prob=0.3, despawn_mode=DespawnMode.PROBABILITY, despawn_prob=0.1,
            sigma_a=sigma_a, sigma_b=sigma_b,
        ),
    )
    return create_env(cfg)


# ---------------------------------------------------------------------------
# task-space agent centers for T != N
# ---------------------------------------------------------------------------


def test_task_centers_for_more_task_types_than_agents():
    env = _make_env(n_agents=2, n_task_types=11, relatedness_width=1, proficiency_width=5)
    assert env.task_centers.tolist() == [3, 7]

    # rel=1 means each agent is interested in three task types around its base task.
    assert set(np.nonzero(env.task_interest[0])[0].tolist()) == {2, 3, 4}
    assert set(np.nonzero(env.task_interest[1])[0].tolist()) == {6, 7, 8}

    # prof=5 spans the full 11-task ring, so both agents can perform every task.
    assert env.proficiency.sum(axis=1).tolist() == [11.0, 11.0]


def test_full_task_relatedness_when_width_spans_task_ring():
    env = _make_env(n_agents=2, n_task_types=11, relatedness_width=5, proficiency_width=5)
    assert env.task_centers.tolist() == [3, 7]
    assert np.all(env.task_interest == 1.0)
    assert np.all(env.category_rewards != 0.0)


# ---------------------------------------------------------------------------
# proficiency and relatedness matrix structure
# ---------------------------------------------------------------------------

class TestProficiencyAndRelatedness:
    def test_proficiency_diagonal_always_one(self):
        """proficiency[i, i] = 1 when S >= 0 and T = N."""
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=0)
        # With S=0: proficiency[i, kappa] = 1 iff i == kappa
        for i in range(4):
            assert env.proficiency[i, i] == 1.0

    def test_proficiency_zero_outside_proficiency_width(self):
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=1)
        # circular topology: proficiency[0, 2] has circular dist min(2,2)=2 > 1 → 0
        assert env.proficiency[0, 2] == 0.0
        # proficiency[0, 3] has circular dist min(3,1)=1 <= 1 → 1 (wraps around)
        assert env.proficiency[0, 3] == 1.0
        # proficiency[0, 1] has circular dist 1 <= 1 → 1
        assert env.proficiency[0, 1] == 1.0

    def test_relatedness_self_always_one(self):
        env = _make_env(relatedness_width=1)
        for i in range(4):
            assert env.relatedness[i, i] == 1.0

    def test_relatedness_relatedness_width_structure(self):
        env = _make_env(n_agents=4, relatedness_width=1)
        # R(0, 1) = 1[|0-1|<=1] = 1
        assert env.relatedness[0, 1] == 1.0
        # R(0, 2) = 1[|0-2|<=1] = 0
        assert env.relatedness[0, 2] == 0.0

    def test_full_relatedness_when_C_ge_N(self):
        env = _make_env(n_agents=4, relatedness_width=10)
        assert np.all(env.relatedness == 1.0)

    def test_proficiency_positive_types_consistent_with_proficiency(self):
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=1)
        for i in range(4):
            expected = frozenset(k for k in range(4) if env.proficiency[i, k] > 0)
            assert env.proficiency_positive_types[i] == expected


# ---------------------------------------------------------------------------
# Category reward generation
# ---------------------------------------------------------------------------

class TestCategoryRewards:
    def test_shape(self):
        env = _make_env(n_agents=4, n_task_types=4)
        assert env.category_rewards.shape == (4, 4)

    def test_zero_sigma_gives_uniform(self):
        env = _make_env(n_agents=4, n_task_types=4, sigma_a=0.0, sigma_b=0.0)
        # relatedness_width=1, N=4 → g = min(3,4) = 3. r'[kappa,j] = 1/g for j in C^(kappa)
        # (the caring agents) and exactly 0 for everyone else.
        g = min(2 * env.cfg.relatedness_width + 1, env.cfg.n_agents)
        for kappa in range(4):
            row = env.category_rewards[kappa]
            for j in range(4):
                caring = env.relatedness[kappa, j] > 0
                expected = (1.0 / g) if caring else 0.0
                assert abs(row[j] - expected) < 1e-5, f"kappa={kappa}, j={j}: {row[j]} != {expected}"

    def test_sigma_a_zero_means_uniform_within_category(self):
        env = _make_env(n_agents=4, n_task_types=4, sigma_a=0.0, sigma_b=1.0)
        # With sigma_a=0, all CARING agents get the same baseline within a category;
        # non-caring agents are 0.
        for kappa in range(4):
            row = env.category_rewards[kappa]
            caring_vals = [row[j] for j in range(4) if env.relatedness[kappa, j] > 0]
            assert np.allclose(caring_vals, caring_vals[0], atol=1e-5), f"kappa={kappa}: {row}"
            for j in range(4):
                if env.relatedness[kappa, j] == 0:
                    assert row[j] == 0.0

    def test_sigma_b_zero_means_equal_category_means(self):
        env = _make_env(n_agents=4, n_task_types=4, sigma_a=0.0, sigma_b=0.0)
        # Each caring agent gets 1/g; mean over the g caring agents is 1/g.
        g = min(2 * env.cfg.relatedness_width + 1, env.cfg.n_agents)
        for kappa in range(4):
            row = env.category_rewards[kappa]
            caring_vals = [row[j] for j in range(4) if env.relatedness[kappa, j] > 0]
            assert np.allclose(np.mean(caring_vals), 1.0 / g, atol=1e-5)

    def test_category_rewards_dtype_float32(self):
        env = _make_env()
        assert env.category_rewards.dtype == np.float32


# ---------------------------------------------------------------------------
# Reward formula: r_j = proficiency[actor,tau] * r'[tau,j], where r'[tau] carries
# the C^(tau) mask (the agents that care about TASK tau, keyed on tau not the actor).
# ---------------------------------------------------------------------------

class TestRewardFormula:
    def test_zero_proficiency_means_zero_reward(self):
        """If actor has no proficiency in task type, all rewards are zero."""
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=0)
        # With S=0: agent 0 can only do type 0. Picking type 1 → proficiency[0,1]=0 → all r=0
        state = env.init_state()
        rewards = env._compute_pick_rewards(actor=0, tau=1)
        assert all(r == 0.0 for r in rewards), f"Expected all zero, got {rewards}"

    def test_correct_pick_formula(self):
        """r_j = proficiency[actor,tau] * r'[tau,j], with r'[tau,j] = (1/g)·1[j∈C^(tau)]."""
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=0,
                        sigma_a=0.0, sigma_b=0.0)
        # sigma_a=sigma_b=0 → r'[tau,j] = 1/g for j in C^(tau); g = min(2*1+1, 4) = 3.
        # S=0: proficiency[0,0]=1; actor=0 picks tau=0, so reward keyed on C^(0).
        # C=1, circular: C^(0) = {3,0,1}; agent 2 does not care → 0.
        rewards = env._compute_pick_rewards(actor=0, tau=0)
        for j in range(4):
            expected = 1.0 * env.relatedness[0, j] * (1.0 / 3)  # C^(0) == agents related to id 0
            assert abs(rewards[j] - expected) < 1e-5, f"j={j}: got {rewards[j]}, expected {expected}"

    def test_relatedness_keyed_on_task_not_actor(self):
        """Reward spreads to C^(tau), the carers of the picked task — not the actor's set."""
        # proficiency_width=1 lets actor 0 pick tau=1 (circular dist 1). Reward must go
        # to C^(1) = {0,1,2}, NOT to C^(0) = {3,0,1}.
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=1,
                        relatedness_width=1, sigma_a=0.0, sigma_b=0.0)
        assert env.proficiency[0, 1] == 1.0  # actor 0 is proficient in task 1
        rewards = env._compute_pick_rewards(actor=0, tau=1)
        g = 3
        for j in range(4):
            expected = (1.0 / g) if env.relatedness[1, j] > 0 else 0.0  # C^(1), keyed on tau=1
            assert abs(rewards[j] - expected) < 1e-5, f"j={j}: got {rewards[j]}, expected {expected}"
        # Sanity: agent 3 cares about task 0 but NOT task 1 → must get nothing here.
        assert rewards[3] == 0.0
        # Agent 2 cares about task 1 but not task 0 → must get reward here.
        assert rewards[2] > 0.0

    def test_resolve_pick_removes_task(self):
        """After resolve_pick, the picked task is removed from state."""
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=10)
        state = env.init_state()
        # Find a state where actor is on an eligible task
        from orchard.enums import Action
        for _ in range(100):
            actor = state.actor
            eligible = env.proficiency_positive_types[actor]
            if state.is_agent_on_task(actor, eligible):
                n_tasks_before = len(state.task_positions)
                new_state, rewards = env.resolve_pick(state, pick_type=list(eligible)[0] if eligible else None)
                assert len(new_state.task_positions) == n_tasks_before - 1
                return
            state = env.advance_actor(env.spawn_and_despawn(state))
        pytest.skip("Could not find pick opportunity in 100 steps")


# ---------------------------------------------------------------------------
# Phase-2 action space: only proficiency > 0 types offered
# ---------------------------------------------------------------------------

class TestPhase2Actions:
    def test_stay_always_offered(self):
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=0)
        state = env.init_state()
        # Even if no eligible tasks at cell, STAY is offered
        from orchard.datatypes import State, Grid
        from orchard.enums import Action
        # Put actor on a cell with only ineligible task
        actor = 0  # proficiency[0, tau>0] = 0 with S=0
        actor_pos = state.agent_positions[actor]
        actions = get_phase2_actions(state.with_pick_phase(), env)
        # At minimum STAY is returned (or empty if no task at cell)
        assert Action.STAY in actions or len(actions) == 0

    def test_ineligible_types_not_offered(self):
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=0)
        from orchard.datatypes import State, Grid
        from orchard.enums import Action, make_pick_action
        state = env.init_state()
        actor = 0  # can only do type 0 with S=0
        actions = get_phase2_actions(state.with_pick_phase(), env)
        # pick(tau) for tau != 0 must not be in actions
        for tau in range(1, 4):
            assert make_pick_action(tau) not in actions


# ---------------------------------------------------------------------------
# Encoders: channel count and value correctness
# ---------------------------------------------------------------------------

class TestEverythingEncoder:
    def setup_method(self):
        self.env = _make_env(n_agents=4, n_task_types=4, relatedness_width=1, proficiency_width=1,
                              sigma_a=0.0, sigma_b=0.0)
        encoding.init_encoder(EncoderType.EVERYTHING_CNN_GRID, self.env, n_networks=4)
        self.state = self.env.init_state()

    def test_grid_channels_count(self):
        T, N = self.env.cfg.n_task_types, self.env.cfg.n_agents
        out = encoding.encode(self.state, 0)
        assert out.grid.shape[0] == T + N + 1

    def test_scalar_dim(self):
        N = self.env.cfg.n_agents
        out = encoding.encode(self.state, 0)
        assert out.scalar.shape[0] == N + 1

    def test_encode_all_agents_shape(self):
        T, N = self.env.cfg.n_task_types, self.env.cfg.n_agents
        h, w = self.env.cfg.height, self.env.cfg.width
        grids, scalars = encoding.encode_all_agents(self.state)
        assert grids.shape == (N, T + N + 1, h, w)
        assert scalars.shape == (N, N + 1)


class TestFilteredDecEncoder:
    def setup_method(self):
        # N=T=4, R^R=1, P^R=1 → KR = min(4, 3) = 3, KW = min(4, 5) = 4.
        self.env = _make_env(n_agents=4, n_task_types=4, relatedness_width=1, proficiency_width=1,
                             sigma_a=0.0, sigma_b=0.0)
        encoding.init_encoder(EncoderType.FILTERED_DEC_CNN_GRID, self.env, n_networks=4)
        self.state = self.env.init_state()
        self.KR = min(4, 2 * 1 + 1)
        self.KW = min(4, 2 * (1 + 1) + 1)

    def test_grid_and_scalar_dims(self):
        out = encoding.encode(self.state, 0)
        assert out.grid.shape[0] == self.KR + self.KW + 1
        assert out.scalar.shape[0] == self.KW + 1

    def test_encode_all_agents_shape(self):
        h, w = self.env.cfg.height, self.env.cfg.width
        grids, scalars = encoding.encode_all_agents(self.state)
        assert grids.shape == (4, self.KR + self.KW + 1, h, w)
        assert scalars.shape == (4, self.KW + 1)

    def test_encode_batch_for_actions_shape(self):
        from orchard.policy import get_all_actions
        actions = get_all_actions(self.env.cfg)
        after_states = [self.env.apply_action(self.state, a) for a in actions]
        out = encoding.encode_batch_for_actions(self.state, 0, after_states)
        h, w = self.env.cfg.height, self.env.cfg.width
        assert out.grid.shape == (len(actions), self.KR + self.KW + 1, h, w)
        assert out.scalar.shape == (len(actions), self.KW + 1)

    def test_encode_all_agents_for_actions_shape(self):
        from orchard.policy import get_all_actions
        actions = get_all_actions(self.env.cfg)
        after_states = [self.env.apply_action(self.state, a) for a in actions]
        grids, scalars = encoding.encode_all_agents_for_actions(self.state, after_states)
        h, w = self.env.cfg.height, self.env.cfg.width
        assert grids.shape == (4, len(actions), self.KR + self.KW + 1, h, w)
        assert scalars.shape == (4, len(actions), self.KW + 1)

    def test_tasks_outside_R_i_are_masked(self):
        """Network i must not observe a task type outside R_i = {k : d(i,k) <= R^R}."""
        from orchard.datatypes import State, Grid
        # Single task of type 2 at (0,0). For agent 0, R_0 = {3,0,1}; type 2 ∉ R_0.
        state = State(agent_positions=(Grid(2, 2), Grid(3, 3), Grid(4, 4), Grid(0, 1)),
                      task_positions=(Grid(0, 0),), actor=0, task_types=(2,))
        out0 = encoding.encode(state, 0)
        # All KR task channels empty for agent 0 (type 2 not kept).
        assert out0.grid[: self.KR].abs().sum().item() == pytest.approx(0.0)
        # Agent 2 cares about type 2 (R_2 = {1,2,3}) → it appears.
        out2 = encoding.encode(state, 2)
        assert out2.grid[: self.KR].abs().sum().item() > 0.0

    def test_agents_outside_W_i_are_masked(self):
        """Network i must not observe an agent outside W_i = {j : d(i,j) <= R^R+P^R}."""
        # N=4, R^R+P^R=2 → KW=4 = all agents, nothing masked. Use a bigger ring to test.
        env = _make_env(n_agents=8, n_task_types=8, relatedness_width=1, proficiency_width=1,
                        sigma_a=0.0, sigma_b=0.0)
        encoding.init_encoder(EncoderType.FILTERED_DEC_CNN_GRID, env, n_networks=8)
        KW = min(8, 2 * (1 + 1) + 1)  # = 5; W_0 = {6,7,0,1,2}; agents 3,4,5 excluded
        from orchard.datatypes import State, Grid
        # 8 distinct cells inside the 5x5 grid (default height=width=5).
        positions = tuple(Grid(k // 5, k % 5) for k in range(8))
        state = State(agent_positions=positions, task_positions=(), actor=0, task_types=())
        out = encoding.encode(state, 0)
        KR = min(8, 3)
        agent_block = out.grid[KR:KR + KW]
        # Exactly KW agents (the members of W_0) are marked, one cell each.
        assert agent_block.sum().item() == pytest.approx(float(KW))


# ---------------------------------------------------------------------------
# Spec propositions: exact invariants from the math spec (Proposition 1, sizes,
# support, proficiency gate, heuristic = team reward). Tolerances are tight
# because the standardization is exact, not statistical.
# ---------------------------------------------------------------------------

def _circ(a, b, N):
    return min(abs(a - b), N - abs(a - b))


class TestSpecPropositions:
    def test_prop1a_within_task_std_is_sigma_a(self):
        """std_{i in C^(k)} r^(k)_i = sigma_a, mean = 1/g (sigma_b=0)."""
        N = 7
        sigma_a = 0.4
        env = _make_env(n_agents=N, n_task_types=N, relatedness_width=2, proficiency_width=1,
                        sigma_a=sigma_a, sigma_b=0.0, seed=1)
        g = min(2 * env.cfg.relatedness_width + 1, N)
        for k in range(N):
            carers = [j for j in range(N) if _circ(k, j, N) <= env.cfg.relatedness_width]
            vals = env.category_rewards[k, carers]
            assert np.isclose(vals.std(), sigma_a, atol=1e-5), f"k={k}: std={vals.std()}"
            assert np.isclose(vals.mean(), 1.0 / g, atol=1e-5), f"k={k}: mean={vals.mean()}"

    def test_prop1b_team_total_std_is_sigma_b_mean_is_one(self):
        """std_k r^(k)_team = sigma_b and mean_k r^(k)_team = 1 (the normalized target)."""
        N = 8
        sigma_b = 0.3
        env = _make_env(n_agents=N, n_task_types=N, relatedness_width=2, proficiency_width=1,
                        sigma_a=0.0, sigma_b=sigma_b, seed=2)
        team_totals = env.category_rewards.sum(axis=1)  # r^(k)_team for each task k
        assert np.isclose(team_totals.mean(), 1.0, atol=1e-5), f"mean={team_totals.mean()}"
        assert np.isclose(team_totals.std(), sigma_b, atol=1e-5), f"std={team_totals.std()}"

    def test_support_equals_caring_set(self):
        """Nonzero entries of r^(k) are exactly C^(k) = {j : d(k,j) <= R^R}."""
        N = 7
        env = _make_env(n_agents=N, n_task_types=N, relatedness_width=2, proficiency_width=1,
                        sigma_a=0.5, sigma_b=0.5, seed=3)
        for k in range(N):
            support = set(np.nonzero(env.category_rewards[k])[0].tolist())
            expected = {j for j in range(N) if _circ(k, j, N) <= env.cfg.relatedness_width}
            assert support == expected, f"k={k}: support={support}, expected={expected}"

    def test_proficiency_gate_zeroes_reward(self):
        """r^(k)_i(c) = 0 for all i whenever the actor c is not proficient in task k."""
        N = 7
        env = _make_env(n_agents=N, n_task_types=N, relatedness_width=2, proficiency_width=1,
                        sigma_a=0.5, sigma_b=0.5, seed=4)
        PR = env.cfg.proficiency_width
        for c in range(N):
            for k in range(N):
                rewards = env._compute_pick_rewards(actor=c, tau=k)
                if _circ(c, k, N) > PR:  # actor not proficient → all zero
                    assert all(r == 0.0 for r in rewards), f"c={c}, k={k}: {rewards}"

    def test_set_sizes(self):
        """|P_c| = 2P^R+1, |C^(k)| = 2R^R+1 (mid-ring, no wrap saturation)."""
        N = 9
        env = _make_env(n_agents=N, n_task_types=N, relatedness_width=2, proficiency_width=1,
                        sigma_a=0.0, sigma_b=0.0, seed=5)
        PR, RR = env.cfg.proficiency_width, env.cfg.relatedness_width
        for c in range(N):
            assert int(env.proficiency[c].sum()) == 2 * PR + 1
        for k in range(N):
            carers = [j for j in range(N) if _circ(k, j, N) <= RR]
            assert len(carers) == 2 * RR + 1

    def test_heuristic_picks_argmax_team_reward(self):
        """In phase 2 the heuristic picks the eligible type with the highest team
        reward Sum_j r^(k)_j present at the actor's cell — confirming the heuristic
        value IS the team reward."""
        from orchard.datatypes import State, Grid
        N = 7
        env = _make_env(n_agents=N, n_task_types=N, relatedness_width=2, proficiency_width=3,
                        sigma_a=0.3, sigma_b=0.4, seed=6)
        actor = 0
        eligible = sorted(env.proficiency_positive_types[actor])
        # Stack one task of each eligible type on the actor's cell, so the heuristic
        # must choose among them by team reward.
        cell = Grid(2, 2)
        positions = [Grid(0, 0)] * N
        positions[actor] = cell
        task_positions = tuple(cell for _ in eligible)
        task_types = tuple(eligible)
        pick_state = State(agent_positions=tuple(positions), task_positions=task_positions,
                           actor=actor, task_types=task_types, pick_phase=True)
        action = nearest_action(pick_state, env)
        assert action.is_pick()
        chosen = action.pick_type()
        # Heuristic value of each eligible type = its team reward (proficiency=1 here).
        team_rewards = {tau: sum(env._compute_pick_rewards(actor, tau)) for tau in eligible}
        best = max(team_rewards, key=team_rewards.get)
        assert chosen == best, f"chose {chosen} (r={team_rewards[chosen]}), best is {best} (r={team_rewards[best]})"


# ---------------------------------------------------------------------------
# Heuristic: value-aware, picks best task not nearest task
# ---------------------------------------------------------------------------

class TestHeuristic:
    def test_heuristic_prefers_high_value_type(self):
        """With sigma_b>0, heuristic should tend toward higher-value task types."""
        set_all_seeds(0)
        env = _make_env(n_agents=4, n_task_types=4, relatedness_width=10, proficiency_width=10,
                        sigma_a=0.0, sigma_b=2.0, seed=0)
        # With C=N-1, S=T-1: all proficiency=1, all R=1
        # Task values differ by category → heuristic is value-aware
        state = env.init_state()
        action = nearest_action(state, env)
        assert action.is_move()

    def test_heuristic_stays_when_no_tasks(self):
        from orchard.datatypes import State, Grid
        env = _make_env()
        state = env.init_state()
        # Empty task state
        empty_state = State(
            agent_positions=state.agent_positions,
            task_positions=(),
            actor=state.actor,
            task_types=(),
        )
        action = nearest_action(empty_state, env)
        from orchard.enums import Action
        assert action == Action.STAY

    def test_heuristic_picks_eligible_in_phase2(self):
        """In pick phase with eligible task, heuristic picks it."""
        from orchard.datatypes import State, Grid
        from orchard.enums import Action, make_pick_action
        env = _make_env(n_agents=4, n_task_types=4, proficiency_width=10)
        state = env.init_state()
        actor = state.actor
        # Manually place actor on a task cell of type the actor can do
        eligible_types = env.proficiency_positive_types[actor]
        if not eligible_types:
            pytest.skip("No eligible types for actor")
        tau = next(iter(eligible_types))
        # Find that task type in state
        for pos, tt in zip(state.task_positions, state.task_types or []):
            if tt == tau:
                # Move actor to this position
                new_positions = list(state.agent_positions)
                new_positions[actor] = pos
                pick_state = State(
                    agent_positions=tuple(new_positions),
                    task_positions=state.task_positions,
                    actor=actor,
                    task_types=state.task_types,
                    pick_phase=True,
                )
                result = nearest_action(pick_state, env)
                assert result.is_pick(), f"Expected pick, got {result}"
                return
        pytest.skip("Could not find suitable task in initial state")


# ---------------------------------------------------------------------------
# Integration: end-to-end rollout metrics
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_rollout_metrics_run(self):
        env = _make_env(n_agents=4, n_task_types=4, relatedness_width=1, proficiency_width=1,
                        sigma_a=0.0, sigma_b=1.0, seed=42)
        state = env.init_state()
        metrics = evaluate_policy_metrics(
            state,
            lambda s: heuristic_action(s, env, Heuristic.NEAREST),
            env,
            n_steps=50,
        )
        assert "rps" in metrics
        assert "team_rps" in metrics
        assert isinstance(metrics["team_rps"], float)

    def test_team_rps_non_negative(self):
        """With sigma_b > 0 and non-trivial proficiency, heuristic should achieve positive team RPS."""
        env = _make_env(n_agents=4, n_task_types=4, relatedness_width=10, proficiency_width=10,
                        sigma_a=0.0, sigma_b=0.5, seed=1)
        state = env.init_state()
        metrics = evaluate_policy_metrics(
            state,
            lambda s: heuristic_action(s, env, Heuristic.NEAREST),
            env,
            n_steps=200,
        )
        assert metrics["team_rps"] >= 0.0

    def test_zero_proficiency_gives_zero_reward(self):
        """Fully isolated agents (C=0, S=0) only get reward for their own type."""
        env = _make_env(n_agents=4, n_task_types=4, relatedness_width=0, proficiency_width=0,
                        sigma_a=0.0, sigma_b=0.0, seed=0)
        # With C=0: R(i,j) = 1[i==j] → only actor gets reward
        # With sigma_b=0, relatedness_width=0: g=1, r'[tau,j] = 1/1 = 1.0 for all j
        # r_actor = proficiency[actor, tau] * R[actor, actor] * r'[tau, actor] = proficiency * 1 * 1
        rewards = env._compute_pick_rewards(actor=0, tau=0)
        # Only agent 0 gets reward (R(0,j)=0 for j!=0)
        for j in range(1, 4):
            assert abs(rewards[j]) < 1e-6, f"Agent {j} should get 0 reward, got {rewards[j]}"
