"""Profile one training run for N steps, breaking down time by component.

Usage:
    python -m orchard.profile_train --config <config.yaml> --steps 5000

Prints a table showing where wall-clock time is spent.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

import orchard.encoding as encoding
from orchard.config import load_config
from orchard.enums import LearningType, StoppingCondition, TDTarget, TrainMode
from orchard.env import create_env
from orchard.eval import evaluate_policy_learning
from orchard.model import create_networks
from orchard.policy import (
    ACTION_PRIORITY,
    argmax_a_Q_team,
    epsilon_greedy,
    epsilon_greedy_batched,
    nearest_apple_action,
)
from orchard.schedule import compute_schedule_value
from orchard.seed import rng, set_all_seeds
from orchard.datatypes import EncoderOutput, ExperimentConfig, State


# ── Timing infrastructure ──
_timers: dict[str, float] = defaultdict(float)
_counts: dict[str, int] = defaultdict(int)


@contextmanager
def timed(label: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    _timers[label] += elapsed
    _counts[label] += 1


def _encode_all_timed(state, n_networks, n_agents, centralized):
    with timed("encode"):
        if centralized:
            return [encoding.encode(state, 0)]
        return [encoding.encode(state, i) for i in range(n_agents)]


def _train_all_agents_timed(networks, s_encs, rewards, discount, s_next_encs, env_step):
    with timed("train_step"):
        total_loss = 0.0
        for i in range(len(networks)):
            total_loss += networks[i].train_step(
                s_enc=s_encs[i], reward=rewards[i],
                discount=discount, s_next_enc=s_next_encs[i],
                env_step=env_step,
            )
        return total_loss, len(networks)


def profile(cfg: ExperimentConfig, n_steps: int) -> None:
    set_all_seeds(cfg.train.seed)
    env = create_env(cfg.env)
    encoding.init_encoder(cfg.model.input_type, cfg.env, cfg.model.k_nearest)

    n_agents = cfg.env.n_agents
    centralized = cfg.train.learning_type == LearningType.CENTRALIZED
    n_networks = 1 if centralized else n_agents

    networks = create_networks(
        cfg.model, cfg.env, cfg.train.lr, cfg.train.total_steps,
        nstep=cfg.train.nstep, td_lambda=cfg.train.td_lambda,
        train_method=cfg.train.train_method, n_networks=n_networks,
    )

    s_t = env.init_state()
    prev_after_enc = None
    prev_reward = None
    prev_discount = None

    zero_rewards = tuple(0.0 for _ in range(n_networks))
    team_sum = cfg.train.td_target == TDTarget.AFTER_STATE

    wall_start = time.perf_counter()

    for t in range(n_steps):
        # ── Action selection ──
        with timed("action_selection"):
            if cfg.train.mode == TrainMode.POLICY_LEARNING:
                epsilon = compute_schedule_value(
                    cfg.train.policy_learning.epsilon, t, cfg.train.total_steps
                )
                if cfg.train.batch_actions:
                    action = epsilon_greedy_batched(s_t, networks, env, epsilon)
                else:
                    action = epsilon_greedy(s_t, networks, env, epsilon)
            else:
                action = nearest_apple_action(s_t, cfg.env)

        # ── Env step ──
        with timed("env_apply_action"):
            s_moved = env.apply_action(s_t, action)
            on_apple = s_moved.is_agent_on_apple(s_moved.actor)

        # ── TD update (after-state path only for brevity) ──
        if cfg.train.td_target == TDTarget.AFTER_STATE:
            if on_apple:
                move_enc = _encode_all_timed(s_moved, n_networks, n_agents, centralized)
                if prev_after_enc is not None:
                    rewards_net = prev_reward
                    _train_all_agents_timed(networks, prev_after_enc, rewards_net, prev_discount, move_enc, t)

                with timed("env_resolve_pick"):
                    s_picked, pick_rewards = env.resolve_pick(s_moved)
                pick_enc = _encode_all_timed(s_picked, n_networks, n_agents, centralized)
                team_r = (sum(pick_rewards),) if centralized else pick_rewards
                _train_all_agents_timed(networks, move_enc, team_r, 1.0, pick_enc, t)

                prev_after_enc = pick_enc
                prev_reward = zero_rewards
                prev_discount = cfg.env.gamma

                with timed("env_spawn_despawn"):
                    s_next = env.advance_actor(env.spawn_and_despawn(s_picked))
            else:
                move_enc = _encode_all_timed(s_moved, n_networks, n_agents, centralized)
                if prev_after_enc is not None:
                    _train_all_agents_timed(networks, prev_after_enc, prev_reward, prev_discount, move_enc, t)

                prev_after_enc = move_enc
                prev_reward = zero_rewards
                prev_discount = cfg.env.gamma

                with timed("env_spawn_despawn"):
                    s_next = env.advance_actor(env.spawn_and_despawn(s_moved))
        else:
            # pre-action path
            pre_enc = _encode_all_timed(s_t, n_networks, n_agents, centralized)
            if on_apple:
                move_enc = _encode_all_timed(s_moved, n_networks, n_agents, centralized)
                _train_all_agents_timed(networks, pre_enc, zero_rewards, cfg.env.gamma, move_enc, t)
                with timed("env_resolve_pick"):
                    s_picked, pick_rewards = env.resolve_pick(s_moved)
                with timed("env_spawn_despawn"):
                    s_next = env.advance_actor(env.spawn_and_despawn(s_picked))
                next_enc = _encode_all_timed(s_next, n_networks, n_agents, centralized)
                team_r = (sum(pick_rewards),) if centralized else pick_rewards
                _train_all_agents_timed(networks, move_enc, team_r, 1.0, next_enc, t)
            else:
                with timed("env_spawn_despawn"):
                    s_next = env.advance_actor(env.spawn_and_despawn(s_moved))
                next_enc = _encode_all_timed(s_next, n_networks, n_agents, centralized)
                _train_all_agents_timed(networks, pre_enc, zero_rewards, cfg.env.gamma, next_enc, t)

        s_t = s_next

        # ── Periodic eval (same freq as training) ──
        if (t + 1) % cfg.logging.main_csv_freq == 0:
            with timed("eval_policy"):
                evaluate_policy_learning(networks, env, cfg.eval.eval_steps)

    wall_total = time.perf_counter() - wall_start

    # ── Report ──
    print(f"\n{'='*65}")
    print(f"  Profile: {n_steps} steps, {wall_total:.2f}s total")
    print(f"  {n_steps/wall_total:.0f} steps/sec")
    print(f"  Config: {n_agents} agents, {cfg.env.height}x{cfg.env.width}, "
          f"{'cen' if centralized else 'dec'}, "
          f"CNN={cfg.model.conv_specs}, MLP={cfg.model.mlp_dims}")
    print(f"{'='*65}")
    print(f"  {'Component':<25} {'Total (s)':>10} {'% Wall':>8} {'Calls':>10} {'Per call (µs)':>14}")
    print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*10} {'-'*14}")

    for label in sorted(_timers, key=lambda k: -_timers[k]):
        total = _timers[label]
        pct = 100 * total / wall_total
        count = _counts[label]
        per_call_us = 1e6 * total / max(count, 1)
        print(f"  {label:<25} {total:>10.3f} {pct:>7.1f}% {count:>10} {per_call_us:>13.1f}")

    accounted = sum(_timers.values())
    other = wall_total - accounted
    print(f"  {'(other/overhead)':<25} {other:>10.3f} {100*other/wall_total:>7.1f}%")
    print(f"{'='*65}")

    # ── Breakdown of action_selection ──
    if "action_selection" in _timers:
        act_total = _timers["action_selection"]
        act_count = _counts["action_selection"]
        enc_in_act = _counts.get("encode", 0)
        print(f"\n  Action selection detail:")
        print(f"    {act_total:.2f}s for {act_count} calls")
        print(f"    ≈ {1e6*act_total/max(act_count,1):.0f} µs/step")
        print(f"    Encodes per step (action sel + training): ~{enc_in_act/max(act_count,1):.1f}")
        n_greedy = int(act_count * (1 - 0.05))
        fwd_per_greedy = 5 * n_networks
        print(f"    Estimated forward passes for action selection: "
              f"~{n_greedy * fwd_per_greedy} ({fwd_per_greedy}/greedy step × {n_greedy} greedy steps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    profile(cfg, args.steps)


if __name__ == "__main__":
    main()
