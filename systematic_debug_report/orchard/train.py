"""Training loop and CLI entry point."""
from __future__ import annotations

from pathlib import Path
import sys
sys.path.append("../")

import argparse
import os
import time

import torch

import orchard.encoding as encoding
from orchard.config import load_config
from orchard.enums import StoppingCondition, TDTarget, TrainMode
from orchard.env import create_env
from orchard.eval import (
    collect_after_state_test_states,
    collect_on_policy_test_states,
    evaluate_policy_learning,
    evaluate_value_learning,
    precompute_ground_truth,
)
from orchard.logging_ import (
    CSVLogger,
    build_detail_csv_fieldnames,
    build_main_csv_fieldnames,
    finalize_logging,
    setup_logging,
)
from orchard.model import create_networks
from orchard.policy import (
    ValueNetwork,
    argmax_a_Q_team,
    epsilon_greedy,
    nearest_apple_action,
)
from orchard.schedule import compute_schedule_value
from orchard.seed import set_all_seeds
from orchard.datatypes import EncoderOutput, ExperimentConfig

def _train_all_agents(
    networks: list[ValueNetwork],
    s_encs: list[EncoderOutput],
    rewards: tuple[float, ...],
    discount: float,
    s_next_encs: list[EncoderOutput],
    env_step: int,
) -> tuple[float, int]:
    """Train all agent networks on one transition. Returns (total_loss, count)."""
    total_loss = 0.0
    for i in range(len(networks)):
        total_loss += networks[i].train_step(
            s_enc=s_encs[i], reward=rewards[i],
            discount=discount, s_next_enc=s_next_encs[i],
            env_step=env_step,
        )
    return total_loss, len(networks)

def _save_checkpoint(
    networks: list[ValueNetwork],
    step: int,
    path: Path,
) -> None:
    """Save network and optimizer state dicts."""
    ckpt = {
        "step": step,
        "networks": [net.state_dict() for net in networks],
        "optimizers": [net.optimizer.state_dict() for net in networks],
    }
    torch.save(ckpt, path)
    

def train(cfg: ExperimentConfig) -> None:
    """Main training loop."""
    start_time = time.time()

    # --- Setup ---
    set_all_seeds(cfg.train.seed)
    env = create_env(cfg.env)
    encoding.init_encoder(cfg.model.input_type, cfg.env, cfg.model.k_nearest)
    networks = create_networks(cfg.model, cfg.env, cfg.train.lr, cfg.train.total_steps, nstep=cfg.train.nstep, td_lambda=cfg.train.td_lambda, train_method=cfg.train.train_method)
    run_dir = setup_logging(cfg)

    n_agents = cfg.env.n_agents

    # --- CSV loggers ---
    main_fields = build_main_csv_fieldnames(n_agents, cfg.train.mode)
    main_logger = CSVLogger(run_dir / "metrics.csv", main_fields)

    detail_fields = build_detail_csv_fieldnames(n_agents, networks)
    detail_logger = CSVLogger(run_dir / "details.csv", detail_fields)

    # --- Test states + ground truth (value_learning) ---
    test_states: list | None = None
    ground_truth: list | None = None
    if cfg.train.mode == TrainMode.VALUE_LEARNING:
        if cfg.train.td_target == TDTarget.PRE_ACTION:
            test_states = collect_on_policy_test_states(env, cfg.eval.n_test_states)
        else:
            test_states = collect_after_state_test_states(env, cfg.eval.n_test_states)
        ground_truth = precompute_ground_truth(test_states, env, cfg.eval, cfg.train.td_target)

    s_t = env.init_state()

    # --- After-state TD bookkeeping (only used when td_target == AFTER_STATE) ---
    prev_after_enc: list[EncoderOutput] | None = None
    prev_reward: tuple[float, ...] | None = None
    prev_discount: float | None = None

    # --- Running loss accumulator ---
    td_loss_accum: float = 0.0
    td_loss_count: int = 0
    running_max_pps: float = float("-inf")
    steps_since_improvement: int = 0

    # --- Main loop ---
    for t in range(cfg.train.total_steps):

        # --- Action selection ---
        if cfg.train.mode == TrainMode.VALUE_LEARNING:
            action = nearest_apple_action(s_t, cfg.env)
        else:
            assert cfg.train.policy_learning is not None
            epsilon = compute_schedule_value(
                cfg.train.policy_learning.epsilon, t, cfg.train.total_steps
            )
            action = epsilon_greedy(s_t, networks, env, epsilon)

        # --- TD updates ---
        s_moved = env.apply_action(s_t, action)
        on_apple = s_moved.is_agent_on_apple(s_moved.actor)

        if cfg.train.td_target == TDTarget.PRE_ACTION:
            # ============================================================
            # PRE_ACTION: train on (pre-action state → next pre-action state)
            # ============================================================
            if on_apple:
                # Transition 1: pre-action → movement after-state (discount=gamma, reward=0)
                pre_enc = [encoding.encode(s_t, i) for i in range(n_agents)]
                move_enc = [encoding.encode(s_moved, i) for i in range(n_agents)]
                loss, count = _train_all_agents(networks, pre_enc, (0.0,) * n_agents, cfg.env.gamma, move_enc, t)
                td_loss_accum += loss
                td_loss_count += count

                # Transition 2: movement after-state → next pre-action state (discount=1, reward=pick)
                s_picked, pick_rewards = env.resolve_pick(s_moved)
                s_next = env.advance_actor(env.spawn_and_despawn(s_picked))
                next_enc = [encoding.encode(s_next, i) for i in range(n_agents)]
                loss, count = _train_all_agents(networks, move_enc, pick_rewards, 1.0, next_enc, t)
                td_loss_accum += loss
                td_loss_count += count
            else:
                # Single transition: pre-action → next pre-action (discount=gamma, reward=0)
                pre_enc = [encoding.encode(s_t, i) for i in range(n_agents)]
                s_next = env.advance_actor(env.spawn_and_despawn(s_moved))
                next_enc = [encoding.encode(s_next, i) for i in range(n_agents)]
                loss, count = _train_all_agents(networks, pre_enc, (0.0,) * n_agents, cfg.env.gamma, next_enc, t)
                td_loss_accum += loss
                td_loss_count += count

        else:
            # ============================================================
            # AFTER_STATE: train on (after-state → next after-state)
            # with delayed updates. Pick after-state is PRE-SPAWN.
            # ============================================================
            if on_apple:
                move_after_enc = [encoding.encode(s_moved, i) for i in range(n_agents)]

                # Delayed update: prev after-state → movement after-state
                if prev_after_enc is not None:
                    assert prev_reward is not None and prev_discount is not None
                    loss, count = _train_all_agents(networks, prev_after_enc, prev_reward, prev_discount, move_after_enc, t)
                    td_loss_accum += loss
                    td_loss_count += count

                # Pick after-state: apple removed, PRE-SPAWN
                s_picked, pick_rewards = env.resolve_pick(s_moved)
                pick_after_enc = [encoding.encode(s_picked, i) for i in range(n_agents)]

                # Immediate update: movement after-state → pick after-state (discount=1.0)
                loss, count = _train_all_agents(networks, move_after_enc, pick_rewards, 1.0, pick_after_enc, t)
                td_loss_accum += loss
                td_loss_count += count

                # Store pick after-state for next delayed update
                prev_after_enc = pick_after_enc
                prev_reward = tuple(0.0 for _ in range(n_agents))
                prev_discount = cfg.env.gamma

                # Env response: spawn/despawn + advance actor
                s_next = env.advance_actor(env.spawn_and_despawn(s_picked))

            else:
                move_after_enc = [encoding.encode(s_moved, i) for i in range(n_agents)]

                # Delayed update: prev after-state → movement after-state
                if prev_after_enc is not None:
                    assert prev_reward is not None and prev_discount is not None
                    loss, count = _train_all_agents(networks, prev_after_enc, prev_reward, prev_discount, move_after_enc, t)
                    td_loss_accum += loss
                    td_loss_count += count

                # Store movement after-state for next delayed update
                prev_after_enc = move_after_enc
                prev_reward = tuple(0.0 for _ in range(n_agents))
                prev_discount = cfg.env.gamma

                # Env response: spawn/despawn + advance actor
                s_next = env.advance_actor(env.spawn_and_despawn(s_moved))

        s_t = s_next

        # --- Reset (value_learning only) ---
        if cfg.train.mode == TrainMode.VALUE_LEARNING:
            assert cfg.train.value_learning is not None
            if (t + 1) % cfg.train.value_learning.reset_freq == 0:
                for net in networks:
                    net.flush_nstep()
                    net.reset_traces()
                s_t = env.init_state()
                if cfg.train.td_target == TDTarget.AFTER_STATE:
                    prev_after_enc = None

        # --- Main CSV logging ---
        if (t + 1) % cfg.logging.main_csv_freq == 0:
            wall_time = time.time() - start_time
            row: dict[str, float | int | str] = {
                "step": t + 1,
                "wall_time": round(wall_time, 3),
            }

            # Value learning metrics
            if test_states is not None and ground_truth is not None:
                val_metrics = evaluate_value_learning(
                    networks, cfg.env, test_states, ground_truth
                )
                row.update(val_metrics)

            # Policy learning metrics
            if cfg.train.mode == TrainMode.POLICY_LEARNING:
                pol_metrics = evaluate_policy_learning(networks, env, cfg.eval.eval_steps)
                row.update(pol_metrics)

            # TD loss
            avg_loss = td_loss_accum / max(td_loss_count, 1)
            row["td_loss_avg"] = round(avg_loss, 8)
            td_loss_accum = 0.0
            td_loss_count = 0

            main_logger.log(row)
            
            # --- Early stopping check ---
            if (
                cfg.train.stopping_condition == StoppingCondition.RUNNING_MAX_PPS
                and "greedy_pps" in row
            ):
                pps = float(row["greedy_pps"])
                if pps > running_max_pps + cfg.train.improvement_threshold:
                    running_max_pps = pps
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += cfg.logging.main_csv_freq
                if steps_since_improvement >= cfg.train.patience_steps and (t + 1) >= cfg.train.min_steps_before_stop:
                    print(f"\nEarly stop at step {t+1}: running max PPS {running_max_pps:.4f} "
                          f"unchanged for {cfg.train.patience_steps} steps.")
                    break

            print(f"\n--- Step {t + 1} ({wall_time:.1f}s) ---")
            if "mae_avg" in row:
                print(f"  MAE avg: {row['mae_avg']:.4f}  "
                        f"Bias avg: {row['bias_avg']:.4f}  "
                        f"MAPE avg: {row['mape_avg']:.4f}")
            if "greedy_pps" in row:
                print(f"  Greedy PPS: {row['greedy_pps']:.4f}  "
                        f"Nearest PPS: {row['nearest_pps']:.4f}")

        # --- Detail CSV logging ---
        if (t + 1) % cfg.logging.detail_csv_freq == 0:
            wall_time = time.time() - start_time
            detail_row: dict[str, float | int | str] = {
                "step": t + 1,
                "wall_time": round(wall_time, 3),
            }

            # RAM usage
            try:
                import resource
                ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            except ImportError:
                ram_mb = 0.0
            detail_row["ram_mb"] = round(ram_mb, 1)

            # Current LR (from first network)
            detail_row["current_lr"] = networks[0].optimizer.param_groups[0]["lr"]

            # Current epsilon
            if cfg.train.mode == TrainMode.POLICY_LEARNING:
                assert cfg.train.policy_learning is not None
                detail_row["current_epsilon"] = compute_schedule_value(
                    cfg.train.policy_learning.epsilon, t + 1, cfg.train.total_steps
                )
            else:
                detail_row["current_epsilon"] = 0.0

            # Weight and grad norms
            for agent_idx in range(n_agents):
                w_norms = networks[agent_idx].get_weight_norms()
                g_norms = networks[agent_idx].get_grad_norms()
                for name in w_norms:
                    detail_row[f"weight_norm_agent_{agent_idx}_{name}"] = round(w_norms[name], 6)
                for name in g_norms:
                    detail_row[f"grad_norm_agent_{agent_idx}_{name}"] = round(g_norms[name], 6)

            # Step-level TD loss and value stats
            detail_row["td_loss_step"] = round(avg_loss if td_loss_count == 0 else td_loss_accum / max(td_loss_count, 1), 8)

            # Value prediction stats over test states
            if test_states is not None:
                preds: list[float] = []
                with torch.no_grad():
                    for s in test_states[:10]:  # sample for speed
                        for i in range(n_agents):
                            preds.append(networks[i](encoding.encode(s, i)).item())
                if preds:
                    detail_row["value_pred_mean"] = round(sum(preds) / len(preds), 6)
                    mean = sum(preds) / len(preds)
                    var = sum((p - mean) ** 2 for p in preds) / len(preds)
                    detail_row["value_pred_std"] = round(var ** 0.5, 6)

            detail_logger.log(detail_row)

        # --- Eval printout ---
        if cfg.eval.checkpoint_freq > 0 and (t + 1) % cfg.eval.checkpoint_freq == 0:
            _save_checkpoint(networks, t + 1, run_dir / "checkpoints" / f"step_{t + 1}.pt")
    
    # --- Final checkpoint (always saved) ---
    _save_checkpoint(networks, cfg.train.total_steps, run_dir / "checkpoints" / "final.pt")
    
    # --- Cleanup ---
    main_logger.close()
    detail_logger.close()
    finalize_logging(run_dir, start_time)
    print(f"\nRun saved to: {run_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Orchard RL Training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Override config values (dot notation): key=value",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    train(cfg)


if __name__ == "__main__":
    main()
