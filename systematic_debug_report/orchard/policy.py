"""Heuristic policies and action space helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from orchard.enums import (
    Action, ACTION_PRIORITY, Heuristic, make_pick_action,
)
from orchard.datatypes import EnvConfig, Grid, State
from orchard.seed import rng


TItem = TypeVar("TItem")
ScoreMode = str
DEFAULT_EPS_NEAREST_RANDOM_PROB = 0.3


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


def random_valid_action(state: State, env) -> Action:
    """Sample a legal action for the current phase."""
    actions = get_phase2_actions(state, env) if state.pick_phase else get_all_actions(env.cfg)
    return rng.choice(actions) if actions else Action.STAY


def manhattan_distance(a: Grid, b: Grid) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


def random_argmax(items: Iterable[TItem], key: Callable[[TItem], float]) -> TItem:
    """Argmax with seeded random tie-breaking."""
    items = list(items)
    if not items:
        raise ValueError("random_argmax requires at least one item")
    scores = [key(item) for item in items]
    max_score = max(scores)
    best = [item for item, score in zip(items, scores) if score == max_score]
    return rng.choice(best)


def _team_reward(env, task_type: int) -> float:
    return float(env.category_rewards[task_type].sum())


def _is_proficient(env, actor: int, task_type: int) -> bool:
    return float(env.proficiency[actor, task_type]) > 0.0


def _score_task(reward: float, distance: int, score_mode: ScoreMode) -> float:
    if score_mode == "team_reward_over_distance":
        return reward / (1.0 + distance)
    if score_mode == "team_reward":
        return reward
    if score_mode == "inverse_distance":
        return 1.0 / (1.0 + distance)
    if score_mode == "abs_team_reward_over_distance":
        return abs(reward) / (1.0 + distance)
    raise ValueError(
        "Unknown score_mode "
        f"{score_mode!r}; expected one of "
        "'team_reward_over_distance', 'team_reward', "
        "'inverse_distance', 'abs_team_reward_over_distance'."
    )


def move_toward(state: State, env, actor_pos: Grid, target_pos: Grid) -> Action:
    """Move one step toward target, using random tie-breaking among improving moves."""
    current_dist = manhattan_distance(actor_pos, target_pos)
    improving: list[Action] = []
    for action in get_all_actions(env.cfg):
        moved = env.apply_action(state, action)
        next_pos = moved.agent_positions[state.actor]
        if manhattan_distance(next_pos, target_pos) < current_dist:
            improving.append(action)
    if improving:
        return rng.choice(improving)
    return random_valid_action(state, env)


def nearest_rewarding_task_action(
    state: State,
    env,
    *,
    epsilon: float = 0.0,
    score_mode: ScoreMode = "team_reward_over_distance",
    positive_rewards_only: bool = True,
) -> Action:
    """Move toward nearby proficient tasks with high team reward.

    The rollout has separate movement and pick phases. In movement phase this
    returns only movement actions; in pick phase it chooses among pickable tasks
    on the current cell.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    if rng.random() < epsilon:
        return random_valid_action(state, env)

    actor = state.actor
    actor_pos = state.agent_positions[actor]

    if state.pick_phase:
        pickable_here: list[tuple[int, float]] = []
        for _, task_type in state.tasks_at(actor_pos):
            if not _is_proficient(env, actor, task_type):
                continue
            reward = _team_reward(env, task_type)
            if positive_rewards_only and reward <= 0.0:
                continue
            pickable_here.append((task_type, reward))
        if pickable_here:
            best_type, _ = random_argmax(pickable_here, key=lambda item: item[1])
            return make_pick_action(best_type)
        return random_valid_action(state, env)

    if not state.task_positions or state.task_types is None:
        return random_valid_action(state, env)

    candidates: list[tuple[Grid, float]] = []
    for task_pos, task_type in zip(state.task_positions, state.task_types):
        if not _is_proficient(env, actor, task_type):
            continue
        reward = _team_reward(env, task_type)
        if positive_rewards_only and reward <= 0.0:
            continue
        distance = manhattan_distance(actor_pos, task_pos)
        score = _score_task(reward, distance, score_mode)
        candidates.append((task_pos, score))

    if not candidates:
        return random_valid_action(state, env)

    best_pos, _ = random_argmax(candidates, key=lambda item: item[1])
    return move_toward(state, env, actor_pos, best_pos)


def nearest_task_action(state: State, env) -> Action:
    """Move toward the nearest proficient task, ignoring reward values."""
    return nearest_rewarding_task_action(
        state,
        env,
        score_mode="inverse_distance",
        positive_rewards_only=False,
    )


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


def eps_nearest_action(
    state: State,
    env,
    *,
    random_action_prob: float = DEFAULT_EPS_NEAREST_RANDOM_PROB,
) -> Action:
    """Epsilon mixture of the regular nearest heuristic and random valid actions."""
    if not 0.0 <= random_action_prob <= 1.0:
        raise ValueError(f"random_action_prob must be in [0, 1], got {random_action_prob}")
    if rng.random() < random_action_prob:
        return random_valid_action(state, env)
    return nearest_action(state, env)


def heuristic_action(state: State, env, heuristic: Heuristic) -> Action:
    """Dispatch to the configured heuristic policy."""
    if heuristic == Heuristic.NEAREST:
        return nearest_action(state, env)
    if heuristic == Heuristic.EPS_NEAREST:
        return eps_nearest_action(state, env)
    if heuristic == Heuristic.NEAREST_REWARDING_TASK:
        return nearest_rewarding_task_action(state, env)
    if heuristic == Heuristic.EPS_NEAREST_REWARDING_TASK:
        return nearest_rewarding_task_action(state, env, epsilon=0.2)
    if heuristic == Heuristic.NEAREST_TASK:
        return nearest_task_action(state, env)
    if heuristic == Heuristic.RANDOM:
        return random_valid_action(state, env)
    raise ValueError(f"Unknown heuristic: {heuristic}")
