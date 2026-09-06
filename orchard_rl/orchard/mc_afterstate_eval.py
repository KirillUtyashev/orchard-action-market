"""Sample TD after-states and estimate fixed-policy MC values.

This diagnostic mirrors the value-learning trainer's after-state view:

* move after-states are encoded after an actor moves; if the actor is on an
  eligible task, the state has ``pick_phase=True``;
* pick after-states are encoded after pick/stay is resolved;
* MC continuation follows the same after-state transition structure used by TD.
"""

from __future__ import annotations

import argparse
import csv
import copy
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml

from orchard.config import _apply_overrides, _parse_env, _parse_train
from orchard.datatypes import EnvConfig, Grid, State, TrainConfig
from orchard.env import create_env
from orchard.enums import Action, Heuristic
from orchard.policy import heuristic_action
from orchard.seed import rng, set_all_seeds


@dataclass(frozen=True)
class SampledAfterState:
    state_id: int
    state: State
    source: str
    source_turn: int


@dataclass(frozen=True)
class AfterStateTransition:
    next_after_state: State
    rewards: tuple[float, ...]
    discount: float
    turns_used: int


def _load_raw_config(path: Path, overrides: Sequence[str]) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if "config" in raw:
        raw = raw["config"]
    raw = copy.deepcopy(raw)
    if overrides:
        raw = _apply_overrides(raw, list(overrides))
    return raw


def load_env_and_train_config(path: Path, overrides: Sequence[str]) -> tuple[EnvConfig, TrainConfig]:
    raw = _load_raw_config(path, overrides)
    for section in ("env", "train"):
        if section not in raw:
            raise ValueError(f"Missing required config section: {section!r}")
    return _parse_env(raw["env"]), _parse_train(raw["train"])


def _move_discount(env_cfg: EnvConfig, discount_method: str, next_actor: int) -> float:
    if discount_method == "round_steps":
        return float(env_cfg.gamma) if next_actor == 0 else 1.0
    return float(env_cfg.gamma)


def _after_move_state(state: State, env, action: Action) -> tuple[State, bool]:
    moved = env.apply_action(state, action)
    eligible_types = env.proficiency_positive_types[moved.actor]
    on_task = moved.is_agent_on_task(moved.actor, eligible_types)
    return moved.with_pick_phase() if on_task else moved, on_task


def _resolve_pick_after_state(state: State, env, action: Action) -> tuple[State, tuple[float, ...]]:
    base_state = State(
        agent_positions=state.agent_positions,
        task_positions=state.task_positions,
        actor=state.actor,
        task_types=state.task_types,
        pick_phase=False,
    )
    return env.resolve_pick(
        base_state,
        pick_type=action.pick_type() if action.is_pick() else None,
    )


def step_after_state(
    state: State,
    env,
    policy: Heuristic,
    *,
    discount_method: str,
) -> AfterStateTransition:
    """Advance one TD after-state transition under ``policy``.

    For a pick-phase after-state, this resolves pick/stay and consumes no new
    agent turn. For an ordinary after-state, this applies spawn/despawn,
    advances actor, executes that actor's move, and consumes one agent turn.
    """
    if state.pick_phase:
        action = heuristic_action(state, env, policy)
        picked, rewards = _resolve_pick_after_state(state, env, action)
        return AfterStateTransition(
            next_after_state=picked,
            rewards=rewards,
            discount=1.0,
            turns_used=0,
        )

    pre_move = env.advance_actor(env.spawn_and_despawn(state))
    action = heuristic_action(pre_move, env, policy)
    after_move, _ = _after_move_state(pre_move, env, action)
    zero_rewards = tuple(0.0 for _ in range(env.cfg.n_agents))
    return AfterStateTransition(
        next_after_state=after_move,
        rewards=zero_rewards,
        discount=_move_discount(env.cfg, discount_method, pre_move.actor),
        turns_used=1,
    )


def sample_after_states(
    env,
    policy: Heuristic,
    *,
    num_states: int,
    burn_in_steps: int,
    sample_stride: int,
) -> list[SampledAfterState]:
    if num_states < 1:
        raise ValueError("--num-states must be >= 1")
    if burn_in_steps < 0:
        raise ValueError("--burn-in-steps must be >= 0")
    if sample_stride < 1:
        raise ValueError("--sample-stride must be >= 1")

    state = env.init_state()
    samples: list[SampledAfterState] = []
    turn = 0
    afterstates_seen = 0

    while len(samples) < num_states:
        move_action = heuristic_action(state, env, policy)
        move_after, on_task = _after_move_state(state, env, move_action)

        if turn >= burn_in_steps and afterstates_seen % sample_stride == 0:
            samples.append(SampledAfterState(len(samples), move_after, "move", turn))
            if len(samples) >= num_states:
                break
        afterstates_seen += 1

        if on_task:
            pick_action = heuristic_action(move_after, env, policy)
            pick_after, _ = _resolve_pick_after_state(move_after, env, pick_action)
            if turn >= burn_in_steps and afterstates_seen % sample_stride == 0:
                samples.append(SampledAfterState(len(samples), pick_after, "pick", turn))
                if len(samples) >= num_states:
                    break
            afterstates_seen += 1
        else:
            pick_after = move_after

        state = env.advance_actor(env.spawn_and_despawn(pick_after))
        turn += 1

    return samples


def collect_after_states_for_turns(
    env,
    policy: Heuristic,
    *,
    burn_in_steps: int,
    rollout_steps: int,
    sample_stride: int,
) -> list[SampledAfterState]:
    """Collect after-states from a fixed-policy rollout window.

    ``burn_in_steps`` and ``rollout_steps`` are agent turns. The rollout window
    records the same TD after-states used for learning: one move after-state per
    agent turn, plus a pick after-state when the moved actor reaches an eligible
    task.
    """
    if burn_in_steps < 0:
        raise ValueError("--burn-in-steps must be >= 0")
    if rollout_steps < 1:
        raise ValueError("--sample-rollout-steps must be >= 1")
    if sample_stride < 1:
        raise ValueError("--sample-stride must be >= 1")

    state = env.init_state()

    for _ in range(burn_in_steps):
        move_action = heuristic_action(state, env, policy)
        move_after, on_task = _after_move_state(state, env, move_action)
        if on_task:
            pick_action = heuristic_action(move_after, env, policy)
            state, _ = _resolve_pick_after_state(move_after, env, pick_action)
        else:
            state = move_after
        state = env.advance_actor(env.spawn_and_despawn(state))

    samples: list[SampledAfterState] = []
    afterstates_seen = 0
    for rollout_turn in range(rollout_steps):
        source_turn = burn_in_steps + rollout_turn
        move_action = heuristic_action(state, env, policy)
        move_after, on_task = _after_move_state(state, env, move_action)

        if afterstates_seen % sample_stride == 0:
            samples.append(SampledAfterState(len(samples), move_after, "move", source_turn))
        afterstates_seen += 1

        if on_task:
            pick_action = heuristic_action(move_after, env, policy)
            pick_after, _ = _resolve_pick_after_state(move_after, env, pick_action)
            if afterstates_seen % sample_stride == 0:
                samples.append(SampledAfterState(len(samples), pick_after, "pick", source_turn))
            afterstates_seen += 1
        else:
            pick_after = move_after

        state = env.advance_actor(env.spawn_and_despawn(pick_after))

    return samples


def select_sample_chunk(
    samples: Sequence[SampledAfterState],
    *,
    state_start: int,
    num_states: int,
    total_sampled_states: int,
    seed: int,
) -> list[SampledAfterState]:
    if num_states < 1:
        raise ValueError("--num-states must be >= 1")
    if total_sampled_states < 1:
        raise ValueError("--total-sampled-states must be >= 1")
    if state_start + num_states > total_sampled_states:
        raise ValueError("--state-start + --num-states exceeds --total-sampled-states")
    if total_sampled_states > len(samples):
        raise ValueError(
            f"Cannot sample {total_sampled_states} states from a pool of {len(samples)} after-states"
        )

    sampler = np.random.default_rng(seed)
    selected_indices = sampler.choice(len(samples), size=total_sampled_states, replace=False)
    selected = [samples[int(idx)] for idx in selected_indices]
    return selected[state_start:state_start + num_states]


def mc_returns_for_state(
    state: State,
    env,
    policy: Heuristic,
    *,
    horizon: int,
    num_rollouts: int,
    seed: int,
    discount_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    if horizon < 1:
        raise ValueError("--horizon must be >= 1")
    if num_rollouts < 1:
        raise ValueError("--num-rollouts must be >= 1")

    returns = np.zeros((num_rollouts, env.cfg.n_agents), dtype=np.float64)
    for rollout_idx in range(num_rollouts):
        rng.seed(seed + 1_000_003 * rollout_idx)
        current = state
        total = np.zeros(env.cfg.n_agents, dtype=np.float64)
        discount = 1.0
        turns_used = 0

        horizon_agent_turns = horizon * env.cfg.n_agents

        while turns_used < horizon_agent_turns or current.pick_phase:
            transition = step_after_state(
                current,
                env,
                policy,
                discount_method=discount_method,
            )
            total += discount * np.asarray(transition.rewards, dtype=np.float64)
            discount *= transition.discount
            current = transition.next_after_state
            turns_used += transition.turns_used

        returns[rollout_idx] = total

    means = returns.mean(axis=0)
    stds = returns.std(axis=0, ddof=1) if num_rollouts > 1 else np.zeros(env.cfg.n_agents)
    ses = stds / np.sqrt(num_rollouts)
    team_returns = returns.sum(axis=1)
    team_mean = float(team_returns.mean())
    team_std = float(team_returns.std(ddof=1)) if num_rollouts > 1 else 0.0
    team_se = team_std / np.sqrt(num_rollouts)
    return means, stds, ses, team_mean, team_std, team_se


def _grid_tuple_to_str(values: Iterable[Grid]) -> str:
    return ";".join(f"{grid.row},{grid.col}" for grid in values)


def state_to_row(sample: SampledAfterState) -> dict[str, object]:
    state = sample.state
    return {
        "state_id": sample.state_id,
        "source": sample.source,
        "source_turn": sample.source_turn,
        "actor": state.actor,
        "pick_phase": int(state.pick_phase),
        "agent_positions": _grid_tuple_to_str(state.agent_positions),
        "task_positions": _grid_tuple_to_str(state.task_positions),
        "task_types": ";".join(str(task_type) for task_type in (state.task_types or ())),
        "n_tasks": len(state.task_positions),
    }


def build_fieldnames(n_agents: int) -> list[str]:
    fields = [
        "state_id",
        "source",
        "source_turn",
        "actor",
        "pick_phase",
        "agent_positions",
        "task_positions",
        "task_types",
        "n_tasks",
        "horizon_rounds",
        "horizon_agent_turns",
        "num_rollouts",
        "policy",
        "discount_method",
        "elapsed_seconds",
        "mc_team_mean",
        "mc_team_std",
        "mc_team_se",
    ]
    for prefix in ("mc_agent_mean", "mc_agent_std", "mc_agent_se"):
        fields.extend(f"{prefix}_{agent}" for agent in range(n_agents))
    return fields


def write_csv(path: Path, rows: Sequence[dict[str, object]], n_agents: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=build_fieldnames(n_agents))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in writer.fieldnames})


def evaluate_sample(
    *,
    env_cfg: EnvConfig,
    env_seed: int,
    sample: SampledAfterState,
    policy: Heuristic,
    horizon: int,
    num_rollouts: int,
    mc_seed: int,
    discount_method: str,
) -> dict[str, object]:
    """Evaluate one sampled after-state in an isolated environment."""
    set_all_seeds(env_seed)
    env = create_env(env_cfg)
    env.set_eval_mode(True)
    try:
        mc_start = time.perf_counter()
        means, stds, ses, team_mean, team_std, team_se = mc_returns_for_state(
            sample.state,
            env,
            policy,
            horizon=horizon,
            num_rollouts=num_rollouts,
            seed=mc_seed + 1_000_000_007 * sample.state_id,
            discount_method=discount_method,
        )
        elapsed_seconds = time.perf_counter() - mc_start
    finally:
        env.set_eval_mode(False)

    row = state_to_row(sample)
    row["horizon_rounds"] = int(horizon)
    row["horizon_agent_turns"] = int(horizon) * env_cfg.n_agents
    row["num_rollouts"] = int(num_rollouts)
    row["policy"] = policy.name.lower()
    row["discount_method"] = discount_method
    row["elapsed_seconds"] = elapsed_seconds
    row["mc_team_mean"] = team_mean
    row["mc_team_std"] = team_std
    row["mc_team_se"] = team_se
    for agent in range(env_cfg.n_agents):
        row[f"mc_agent_mean_{agent}"] = float(means[agent])
        row[f"mc_agent_std_{agent}"] = float(stds[agent])
        row[f"mc_agent_se_{agent}"] = float(ses[agent])
    return row


def _evaluate_sample_payload(payload: dict[str, object]) -> dict[str, object]:
    return evaluate_sample(**payload)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample value-learning after-states and compute fixed-policy MC returns.",
    )
    parser.add_argument("--metadata", type=Path, required=True, help="metadata.yaml or config YAML")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides in dot notation, e.g. env.stochastic.sigma_b=2",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-states", type=int, required=True)
    parser.add_argument(
        "--state-start",
        type=int,
        default=0,
        help="Global sampled-state offset for this chunk.",
    )
    parser.add_argument("--horizon", type=int, default=1000, help="MC horizon in full round-robin cycles")
    parser.add_argument("--num-rollouts", type=int, default=50)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for per-state MC evaluation.",
    )
    parser.add_argument("--burn-in-steps", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument(
        "--sample-rollout-steps",
        type=int,
        default=None,
        help=(
            "If set, collect after-states for this many fixed-policy agent turns "
            "after burn-in, then randomly select --total-sampled-states before chunking."
        ),
    )
    parser.add_argument(
        "--total-sampled-states",
        type=int,
        default=None,
        help="Total number of randomly selected states shared across all chunks.",
    )
    parser.add_argument(
        "--policy",
        choices=[heuristic.name.lower() for heuristic in Heuristic],
        default=None,
        help="Fixed rollout policy. Defaults to hungarian.",
    )
    parser.add_argument(
        "--discount-method",
        choices=["team_steps", "world_steps", "round_steps"],
        default=None,
        help="Defaults to train.discount_method from metadata.",
    )
    parser.add_argument(
        "--env-seed",
        type=int,
        default=None,
        help="Seed used before environment creation. Defaults to train.seed.",
    )
    parser.add_argument("--sample-seed", type=int, default=10_000)
    parser.add_argument("--state-sample-seed", type=int, default=30_000)
    parser.add_argument("--mc-seed", type=int, default=20_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    env_cfg, train_cfg = load_env_and_train_config(args.metadata, args.override)
    policy = Heuristic[args.policy.upper()] if args.policy else Heuristic.HUNGARIAN
    discount_method = args.discount_method or train_cfg.discount_method
    env_seed = train_cfg.seed if args.env_seed is None else args.env_seed
    if args.state_start < 0:
        raise ValueError("--state-start must be >= 0")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    set_all_seeds(env_seed)
    env = create_env(env_cfg)

    env.set_eval_mode(True, seed=args.sample_seed)
    try:
        if args.sample_rollout_steps is None:
            samples = sample_after_states(
                env,
                policy,
                num_states=args.state_start + args.num_states,
                burn_in_steps=args.burn_in_steps,
                sample_stride=args.sample_stride,
            )
            samples = samples[args.state_start:args.state_start + args.num_states]
            sampling_note = "first_n"
        else:
            total_sampled_states = args.total_sampled_states or args.state_start + args.num_states
            sample_pool = collect_after_states_for_turns(
                env,
                policy,
                burn_in_steps=args.burn_in_steps,
                rollout_steps=args.sample_rollout_steps,
                sample_stride=args.sample_stride,
            )
            samples = select_sample_chunk(
                sample_pool,
                state_start=args.state_start,
                num_states=args.num_states,
                total_sampled_states=total_sampled_states,
                seed=args.state_sample_seed,
            )
            sampling_note = (
                f"pool_rollout_agent_turns={args.sample_rollout_steps}, "
                f"pool_afterstates={len(sample_pool)}, "
                f"total_sampled_states={total_sampled_states}, "
                f"state_sample_seed={args.state_sample_seed}"
            )
    finally:
        env.set_eval_mode(False)

    payloads = [
        {
            "env_cfg": env_cfg,
            "env_seed": env_seed,
            "sample": sample,
            "policy": policy,
            "horizon": args.horizon,
            "num_rollouts": args.num_rollouts,
            "mc_seed": args.mc_seed,
            "discount_method": discount_method,
        }
        for sample in samples
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = build_fieldnames(env_cfg.n_agents)
    completed_rows = 0
    total_rows = len(payloads)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        def write_completed_row(row: dict[str, object]) -> None:
            nonlocal completed_rows
            writer.writerow({field: row.get(field, "") for field in fieldnames})
            f.flush()
            completed_rows += 1
            print(
                f"Wrote MC row state_id={row['state_id']} "
                f"({completed_rows}/{total_rows}) to {args.output}",
                flush=True,
            )

        if args.workers == 1:
            for payload in payloads:
                write_completed_row(_evaluate_sample_payload(payload))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(_evaluate_sample_payload, payload) for payload in payloads]
                for future in as_completed(futures):
                    write_completed_row(future.result())

    print(
        f"Wrote {completed_rows} MC after-state rows to {args.output} "
        f"(policy={policy.name.lower()}, state_start={args.state_start}, "
        f"H_rounds={args.horizon}, H_agent_turns={args.horizon * env_cfg.n_agents}, "
        f"K={args.num_rollouts}, workers={args.workers}, sampling={sampling_note})"
    )


if __name__ == "__main__":
    main()
