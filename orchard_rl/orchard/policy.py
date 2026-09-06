"""Heuristic policies and action space helpers."""

from __future__ import annotations

import itertools
import random
import numpy as np

from orchard.enums import (
    Action, ACTION_PRIORITY, DespawnMode, Heuristic, make_pick_action,
)
from orchard.datatypes import EnvConfig, State
from orchard.seed import rng


# ---------------------------------------------------------------------------
# Action space helpers
# ---------------------------------------------------------------------------
def get_all_actions(env_cfg: EnvConfig) -> list[Action]:
    """Phase-1 actions: always the 5 move actions."""
    return list(ACTION_PRIORITY)


def get_phase2_actions(state: State, env) -> list[Action]:
    """Phase-2 actions after a move landed on a task cell.

    Offers STAY + pick(κ) for each task type κ present at the actor's cell
    where phi(actor, κ) > 0 (actor has non-zero proficiency).

    env must be a BaseEnv with proficiency_positive_types attribute.
    """
    actor = state.actor
    actor_pos = state.agent_positions[actor]
    tasks_here = state.tasks_at(actor_pos)
    if not tasks_here:
        return []

    eligible_types = env.proficiency_positive_types[actor]
    types_here = sorted({tau for _, tau in tasks_here if tau in eligible_types})
    if not types_here:
        return [Action.STAY]
    actions: list[Action] = [Action.STAY]
    for tau in types_here:
        actions.append(make_pick_action(tau))
    return actions


# ---------------------------------------------------------------------------
# Value-aware nearest heuristic
# ---------------------------------------------------------------------------
def nearest_action(state: State, env) -> Action:
    """Document's greedy-optimal heuristic.

    Phase 1: move toward (q*, κ*) = argmax_{tasks} φ(actor,κ)·Σ_j r'^(κ)_j
             (team reward of picking κ; r'^(κ) carries the C^(κ) mask).
             Ties broken by Manhattan distance, then ACTION_PRIORITY.
    Phase 2: pick argmax-value eligible type present; STAY if none positive.

    env must be a BaseEnv with proficiency, category_rewards, proficiency_positive_types.
    """
    actor = state.actor

    if state.pick_phase:
        # Phase 2: pick the highest-value eligible task type at this cell
        actor_pos = state.agent_positions[actor]
        tasks_here = state.tasks_at(actor_pos)
        eligible = env.proficiency_positive_types[actor]
        best_tau = None
        best_val = 0.0
        for _, tau in tasks_here:
            if tau not in eligible:
                continue
            # Team reward = sum over caring agents; r'[tau] already carries the C^(tau) mask.
            val = float(env.proficiency[actor, tau]) * float(env.category_rewards[tau].sum())
            if val > best_val:
                best_val = val
                best_tau = tau
        if best_tau is not None:
            return make_pick_action(best_tau)
        return Action.STAY

    # Phase 1: compute per-task value and move toward best
    if not state.task_positions or state.task_types is None:
        return Action.STAY

    # task_val[k] = proficiency[actor, tau_k] * sum_j r'[tau_k, j]  (team reward;
    # r'[tau] already carries the C^(tau) mask, so the sum is over caring agents)
    task_vals = []
    for pos, tau in zip(state.task_positions, state.task_types):
        proficiency_val = float(env.proficiency[actor, tau])
        val = proficiency_val * float(env.category_rewards[tau].sum())
        task_vals.append((val, pos))

    # Find the best target: argmax value, then min distance
    ar, ac = state.agent_positions[actor]
    best_action = Action.STAY
    best_val = 0.0
    best_dist = float("inf")

    for action in ACTION_PRIORITY:
        dr, dc = action.delta
        nr = max(0, min(env.cfg.height - 1, ar + dr))
        nc = max(0, min(env.cfg.width - 1, ac + dc))
        # Find closest task and its value from this candidate position
        candidate_best_val = -float("inf")
        candidate_best_dist = float("inf")
        for val, pos in task_vals:
            d = abs(nr - pos.row) + abs(nc - pos.col)
            # Primary: maximize value; secondary: minimize distance
            if (val > candidate_best_val or
                    (val == candidate_best_val and d < candidate_best_dist)):
                candidate_best_val = val
                candidate_best_dist = d
        # Compare this action against current best
        if (candidate_best_val > best_val or
                (candidate_best_val == best_val and candidate_best_dist < best_dist)):
            best_val = candidate_best_val
            best_dist = candidate_best_dist
            best_action = action

    return best_action


# ---------------------------------------------------------------------------
# Survival-adjusted Hungarian heuristic (Baseline B)
# ---------------------------------------------------------------------------
def _agent_task_reward(env, agent: int, task_type: int) -> float:
    """Team reward produced when ``agent`` picks ``task_type``."""
    return (
        float(env.proficiency[agent, task_type])
        * float(env.category_rewards[task_type].sum())
    )


def survival_adjusted_utilities(state: State, env) -> np.ndarray:
    """Return the Baseline-B utility matrix, shaped ``(agents, tasks)``.

    A task is one concrete task instance, so stacked tasks occupy distinct
    columns. In probability-despawn environments, utility is discounted by
    ``(1 - p) ** Manhattan_distance`` exactly as specified by Baseline B.
    """
    if state.task_types is None:
        raise ValueError("Hungarian heuristic requires task_types")

    n_agents = len(state.agent_positions)
    n_tasks = len(state.task_positions)
    utilities = np.empty((n_agents, n_tasks), dtype=np.float64)

    stoch = env.cfg.stochastic
    despawn_prob = 0.0
    if stoch is not None and stoch.despawn_mode == DespawnMode.PROBABILITY:
        despawn_prob = stoch.despawn_prob
    survival_per_step = 1.0 - despawn_prob

    for agent, agent_pos in enumerate(state.agent_positions):
        for task, (task_pos, task_type) in enumerate(
            zip(state.task_positions, state.task_types)
        ):
            distance = (
                abs(agent_pos.row - task_pos.row)
                + abs(agent_pos.col - task_pos.col)
            )
            utilities[agent, task] = (
                _agent_task_reward(env, agent, task_type)
                * survival_per_step**distance
            )
    return utilities


def _assignment_from_utilities(utilities: np.ndarray) -> tuple[int | None, ...]:
    """Assign at most one positive-utility task to each agent.

    ``utilities`` is shaped ``(agents, tasks)``. The returned tuple is indexed
    by agent and contains a task index or ``None``. One zero-utility dummy
    column per agent represents staying, so the solver is never forced to
    select a negative task. Zero-utility real assignments are also converted to
    ``None``.
    """
    from scipy.optimize import linear_sum_assignment

    n_agents = utilities.shape[0]
    stay_utilities = np.zeros((n_agents, n_agents), dtype=np.float64)
    augmented = np.concatenate((utilities, stay_utilities), axis=1)

    rows, columns = linear_sum_assignment(-augmented)
    assignment: list[int | None] = [None] * n_agents
    n_tasks = utilities.shape[1]
    for agent, column in zip(rows.tolist(), columns.tolist()):
        if column < n_tasks and utilities[agent, column] > 0.0:
            assignment[agent] = column
    return tuple(assignment)


def survival_adjusted_assignment(state: State, env) -> tuple[int | None, ...]:
    """Baseline-B survival-adjusted Hungarian assignment."""
    return _assignment_from_utilities(survival_adjusted_utilities(state, env))


def _move_toward(source, target) -> Action:
    """Take one deterministic Manhattan-shortest-path step toward a target."""
    current_distance = abs(source.row - target.row) + abs(source.col - target.col)
    for action in ACTION_PRIORITY:
        dr, dc = action.delta
        candidate_distance = (
            abs(source.row + dr - target.row)
            + abs(source.col + dc - target.col)
        )
        if candidate_distance < current_distance:
            return action
    return Action.STAY


def survival_adjusted_hungarian_action(state: State, env) -> Action:
    """Centralized Baseline B policy, replanned from the full state each call."""
    actor = state.actor

    if state.pick_phase:
        actor_pos = state.agent_positions[actor]
        best_type = None
        best_reward = 0.0
        for _, task_type in state.tasks_at(actor_pos):
            reward = _agent_task_reward(env, actor, task_type)
            if reward > best_reward:
                best_reward = reward
                best_type = task_type
        return make_pick_action(best_type) if best_type is not None else Action.STAY

    assignment = survival_adjusted_assignment(state, env)
    task = assignment[actor]
    if task is None:
        return Action.STAY
    return _move_toward(
        state.agent_positions[actor],
        state.task_positions[task],
    )


# ---------------------------------------------------------------------------
# Finite-horizon rollout planners (Baseline C and D-lite)
# ---------------------------------------------------------------------------
DEFAULT_STOCHASTIC_MPC_HORIZON = 3
DEFAULT_STOCHASTIC_MPC_ROLLOUTS = 1
DEFAULT_STOCHASTIC_MPC_CANDIDATES = 3
DEFAULT_STOCHASTIC_MPC_TOP_K = 3
DEFAULT_RAW_STOCHASTIC_MPC_HORIZON = 3
DEFAULT_RAW_STOCHASTIC_MPC_ROLLOUTS = 1
DEFAULT_CLAIRVOYANT_HORIZON = 5


def _discounted_team_return(transitions) -> float:
    """Discount and sum team rewards from a simulated rollout."""
    total = 0.0
    discount = 1.0
    for transition in transitions:
        total += discount * float(sum(transition.rewards))
        discount *= transition.discount
    return total


def _sample_future_rng_states(
    live_rng_state,
    n_rollouts: int,
) -> tuple[object, ...]:
    """Create reproducible sampled futures without consuming the live RNG."""
    seed_source = random.Random()
    seed_source.setstate(live_rng_state)
    return tuple(
        random.Random(seed_source.getrandbits(128)).getstate()
        for _ in range(n_rollouts)
    )


def _survival_per_step(env) -> float:
    stoch = env.cfg.stochastic
    if stoch is not None and stoch.despawn_mode == DespawnMode.PROBABILITY:
        return 1.0 - stoch.despawn_prob
    return 1.0


def _assignment_utilities(
    state: State,
    env,
    strategy: str,
) -> np.ndarray:
    """Utility matrices used by Hungarian-guided finite-horizon planning."""
    if state.task_types is None:
        raise ValueError("Hungarian-guided planner requires task_types")

    n_agents = len(state.agent_positions)
    n_tasks = len(state.task_positions)
    utilities = np.empty((n_agents, n_tasks), dtype=np.float64)
    survival = _survival_per_step(env)

    for agent, agent_pos in enumerate(state.agent_positions):
        for task, (task_pos, task_type) in enumerate(
            zip(state.task_positions, state.task_types)
        ):
            distance = (
                abs(agent_pos.row - task_pos.row)
                + abs(agent_pos.col - task_pos.col)
            )
            reward = _agent_task_reward(env, agent, task_type)

            if strategy == "survival":
                utility = reward * survival**distance
            elif strategy == "reward":
                utility = reward
            elif strategy == "distance":
                utility = reward / (distance + 1)
            elif strategy == "nearest":
                # Mostly distance-driven, with reward as a small tie breaker.
                utility = 1.0 / (distance + 1) + 1e-3 * reward if reward > 0.0 else reward
            elif strategy == "survival_heavy":
                utility = reward * survival ** (2.0 * distance)
            elif strategy == "reward_heavy":
                utility = max(reward, 0.0) * survival ** (0.5 * distance)
            else:
                raise ValueError(f"Unknown assignment strategy: {strategy}")

            utilities[agent, task] = utility

    return utilities


def _task_signature(state: State, task: int):
    assert state.task_types is not None
    return state.task_positions[task], state.task_types[task]


def _target_exists(state: State, target) -> bool:
    position, task_type = target
    return any(
        pos == position and tau == task_type
        for pos, tau in zip(state.task_positions, state.task_types or ())
    )


def _task_set_signature(state: State) -> tuple:
    return tuple(zip(state.task_positions, state.task_types or ()))


def _assignment_to_targets(
    assignment: tuple[int | None, ...],
    state: State,
) -> list:
    return [
        None if task is None else _task_signature(state, task)
        for task in assignment
    ]


def _dedupe_assignments(
    candidates: list[tuple[int | None, ...]],
) -> list[tuple[int | None, ...]]:
    seen = set()
    unique = []
    for assignment in candidates:
        if assignment in seen:
            continue
        seen.add(assignment)
        unique.append(assignment)
    return unique


def _sample_top_k_assignment(
    utilities: np.ndarray,
    sampler: random.Random,
    top_k: int,
) -> tuple[int | None, ...]:
    """Sample a one-to-one assignment from each agent's top-k positive tasks."""
    n_agents, n_tasks = utilities.shape
    assignment: list[int | None] = [None] * n_agents
    available = set(range(n_tasks))

    agent_order = list(range(n_agents))
    agent_order.sort(
        key=lambda agent: float(np.max(utilities[agent])) if n_tasks else 0.0,
        reverse=True,
    )

    for agent in agent_order:
        positive_tasks = [
            task for task in np.argsort(-utilities[agent]).tolist()
            if task in available and utilities[agent, task] > 0.0
        ][:top_k]
        if not positive_tasks:
            continue

        weights = [max(float(utilities[agent, task]), 0.0) for task in positive_tasks]
        total = sum(weights)
        if total <= 0.0:
            task = sampler.choice(positive_tasks)
        else:
            threshold = sampler.random() * total
            cumulative = 0.0
            task = positive_tasks[-1]
            for candidate, weight in zip(positive_tasks, weights):
                cumulative += weight
                if cumulative >= threshold:
                    task = candidate
                    break
        assignment[agent] = task
        available.remove(task)

    return tuple(assignment)


def _assignment_candidates(
    state: State,
    env,
    n_candidates: int,
    top_k: int,
    live_rng_state,
) -> tuple[tuple[int | None, ...], ...]:
    """Generate plausible task-target assignments for structured MPC."""
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    base_strategies = (
        "survival",
        "reward",
        "distance",
        "nearest",
        "survival_heavy",
        "reward_heavy",
    )
    candidates = [
        _assignment_from_utilities(_assignment_utilities(state, env, strategy))
        for strategy in base_strategies[:n_candidates]
    ]
    candidates = _dedupe_assignments(candidates)

    if len(candidates) >= n_candidates:
        return tuple(candidates[:n_candidates])

    sampler = random.Random()
    sampler.setstate(live_rng_state)
    utilities = _assignment_utilities(state, env, "survival")
    attempts = 0
    max_attempts = max(10, 5 * n_candidates)
    while len(candidates) < n_candidates and attempts < max_attempts:
        attempts += 1
        sampled = _sample_top_k_assignment(utilities, sampler, top_k)
        if sampled not in candidates:
            candidates.append(sampled)

    return tuple(candidates)


def stochastic_mpc_action(
    state: State,
    env,
    horizon: int = DEFAULT_STOCHASTIC_MPC_HORIZON,
    n_rollouts: int = DEFAULT_STOCHASTIC_MPC_ROLLOUTS,
    n_candidates: int = DEFAULT_STOCHASTIC_MPC_CANDIDATES,
    top_k: int = DEFAULT_STOCHASTIC_MPC_TOP_K,
) -> Action:
    """Baseline C: Hungarian-guided finite-horizon planner.

    This is a structured approximate MPC for the 11-agent setting. Instead of
    enumerating raw movement sequences, it generates a small set of centralized
    target assignments, simulates each assignment under sampled stochastic
    futures, and executes the first one-step move from the best assignment.
    During rollout, agents keep moving toward their assigned targets; if a
    target is collected, despawns, or an agent has no target, the simulated
    policy falls back to survival-adjusted Hungarian reassignment.

    Horizons count individual agent turns in this round-robin environment.
    Runtime is O(n_candidates * n_rollouts * horizon), plus small Hungarian
    solves during candidate generation and simulated reassignments.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if n_rollouts < 1:
        raise ValueError(f"n_rollouts must be >= 1, got {n_rollouts}")
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    if state.pick_phase:
        return survival_adjusted_hungarian_action(state, env)

    from orchard.eval import rollout_trajectory

    live_rng_state = rng.getstate()
    future_rng_states = _sample_future_rng_states(
        live_rng_state,
        n_rollouts,
    )
    has_round_counter = hasattr(env, "_rounds_elapsed")
    rounds_elapsed = getattr(env, "_rounds_elapsed", None)

    fallback = survival_adjusted_hungarian_action(state, env)
    candidates = _assignment_candidates(
        state,
        env,
        n_candidates=n_candidates,
        top_k=top_k,
        live_rng_state=live_rng_state,
    )
    best_action = fallback
    best_mean_return = -float("inf")

    try:
        for assignment in candidates:
            return_sum = 0.0
            for future_rng_state in future_rng_states:
                rng.setstate(future_rng_state)
                if has_round_counter:
                    env._rounds_elapsed = rounds_elapsed

                targets = _assignment_to_targets(assignment, state)
                assigned_task_signature = _task_set_signature(state)

                def rollout_policy(sim_state: State) -> Action:
                    nonlocal assigned_task_signature
                    if sim_state.pick_phase:
                        return survival_adjusted_hungarian_action(sim_state, env)

                    actor = sim_state.actor
                    target = targets[actor]
                    current_task_signature = _task_set_signature(sim_state)
                    should_reassign = (
                        target is not None
                        and not _target_exists(sim_state, target)
                    ) or (
                        target is None
                        and current_task_signature != assigned_task_signature
                    )
                    if should_reassign:
                        targets[:] = _assignment_to_targets(
                            survival_adjusted_assignment(sim_state, env),
                            sim_state,
                        )
                        assigned_task_signature = current_task_signature
                        target = targets[actor]

                    if target is None:
                        return Action.STAY

                    target_position, _ = target
                    return _move_toward(
                        sim_state.agent_positions[actor],
                        target_position,
                    )

                return_sum += _discounted_team_return(
                    rollout_trajectory(
                        state,
                        rollout_policy,
                        env,
                        n_steps=horizon,
                    )
                )

            mean_return = return_sum / n_rollouts
            if mean_return > best_mean_return:
                best_mean_return = mean_return
                target = _assignment_to_targets(assignment, state)[state.actor]
                if target is None:
                    best_action = Action.STAY
                else:
                    target_position, _ = target
                    best_action = _move_toward(
                        state.agent_positions[state.actor],
                        target_position,
                    )
    finally:
        rng.setstate(live_rng_state)
        if has_round_counter:
            env._rounds_elapsed = rounds_elapsed

    return best_action


def raw_stochastic_mpc_action(
    state: State,
    env,
    horizon: int = DEFAULT_RAW_STOCHASTIC_MPC_HORIZON,
    n_rollouts: int = DEFAULT_RAW_STOCHASTIC_MPC_ROLLOUTS,
) -> Action:
    """Original Baseline-C planner: exhaustive open-loop action-tree MPC.

    This is the naive finite-horizon planner we used before the Hungarian-guided
    refactor. It enumerates every movement action sequence of length ``horizon``
    over consecutive round-robin actor turns, simulates each sequence under
    ``n_rollouts`` sampled stochastic futures, and executes the first action from
    the sequence with the highest mean discounted team return.

    Horizons count individual agent turns in this round-robin environment.
    Runtime is O(5**horizon * n_rollouts * horizon), so this should be used as a
    small-horizon ablation rather than the default 11-agent baseline evaluator.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if n_rollouts < 1:
        raise ValueError(f"n_rollouts must be >= 1, got {n_rollouts}")

    if state.pick_phase:
        return survival_adjusted_hungarian_action(state, env)

    from orchard.eval import rollout_trajectory

    live_rng_state = rng.getstate()
    future_rng_states = _sample_future_rng_states(
        live_rng_state,
        n_rollouts,
    )
    has_round_counter = hasattr(env, "_rounds_elapsed")
    rounds_elapsed = getattr(env, "_rounds_elapsed", None)

    fallback = survival_adjusted_hungarian_action(state, env)
    action_order = (fallback,) + tuple(
        action for action in ACTION_PRIORITY if action != fallback
    )
    best_action = fallback
    best_mean_return = -float("inf")

    try:
        for sequence in itertools.product(action_order, repeat=horizon):
            return_sum = 0.0
            for future_rng_state in future_rng_states:
                rng.setstate(future_rng_state)
                if has_round_counter:
                    env._rounds_elapsed = rounds_elapsed

                move_index = 0

                def rollout_policy(sim_state: State) -> Action:
                    nonlocal move_index
                    if sim_state.pick_phase:
                        return survival_adjusted_hungarian_action(sim_state, env)
                    action = sequence[move_index]
                    move_index += 1
                    return action

                return_sum += _discounted_team_return(
                    rollout_trajectory(
                        state,
                        rollout_policy,
                        env,
                        n_steps=horizon,
                    )
                )

            mean_return = return_sum / n_rollouts
            if mean_return > best_mean_return:
                best_mean_return = mean_return
                best_action = sequence[0]
    finally:
        rng.setstate(live_rng_state)
        if has_round_counter:
            env._rounds_elapsed = rounds_elapsed

    return best_action


def clairvoyant_rollout_action(
    state: State,
    env,
    horizon: int = DEFAULT_CLAIRVOYANT_HORIZON,
) -> Action:
    """Choose a move using exact future spawn/despawn randomness.

    All five first moves are evaluated. For each candidate, the planner restores
    the current environment RNG state, simulates the requested number of agent
    turns, and follows the survival-adjusted Hungarian policy after the
    candidate move. Thus every candidate sees the exact future random-number
    stream rather than sampled futures. The live RNG and mutable environment
    counters are restored before returning, so planning does not consume or
    alter the real future.

    This is D-lite, not proper Baseline D: first actions are searched exactly,
    while later controls use Baseline B as the tail policy.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    if state.pick_phase:
        return survival_adjusted_hungarian_action(state, env)

    from orchard.eval import rollout_trajectory

    rng_state = rng.getstate()
    has_round_counter = hasattr(env, "_rounds_elapsed")
    rounds_elapsed = getattr(env, "_rounds_elapsed", None)

    fallback = survival_adjusted_hungarian_action(state, env)
    candidates = [fallback] + [
        action for action in ACTION_PRIORITY if action != fallback
    ]
    best_action = fallback
    best_return = -float("inf")

    try:
        for candidate in candidates:
            rng.setstate(rng_state)
            if has_round_counter:
                env._rounds_elapsed = rounds_elapsed

            first_move_pending = True

            def rollout_policy(sim_state: State) -> Action:
                nonlocal first_move_pending
                if not sim_state.pick_phase and first_move_pending:
                    first_move_pending = False
                    return candidate
                return survival_adjusted_hungarian_action(sim_state, env)

            score = _discounted_team_return(
                rollout_trajectory(
                    state,
                    rollout_policy,
                    env,
                    n_steps=horizon,
                )
            )
            if score > best_return:
                best_return = score
                best_action = candidate
    finally:
        rng.setstate(rng_state)
        if has_round_counter:
            env._rounds_elapsed = rounds_elapsed

    return best_action


def heuristic_action(state: State, env, heuristic: Heuristic) -> Action:
    """Dispatch to the configured heuristic policy."""
    if heuristic == Heuristic.NEAREST:
        return nearest_action(state, env)
    elif heuristic == Heuristic.HUNGARIAN:
        return survival_adjusted_hungarian_action(state, env)
    elif heuristic == Heuristic.STOCHASTIC_MPC:
        return stochastic_mpc_action(state, env)
    elif heuristic == Heuristic.RAW_STOCHASTIC_MPC:
        return raw_stochastic_mpc_action(state, env)
    elif heuristic == Heuristic.CLAIRVOYANT_ROLLOUT:
        return clairvoyant_rollout_action(state, env)
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")
