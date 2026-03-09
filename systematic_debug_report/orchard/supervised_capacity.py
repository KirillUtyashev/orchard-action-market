"""Supervised capacity test: can a given architecture represent the teacher's V̂?

Loads a trained teacher, rolls out its greedy policy to generate
(after-state, V̂_teacher) pairs, and trains a student with a (potentially
smaller) architecture via supervised MSE + Adam.  Reports test-set metrics.

The data stream is generated online (no epochs) from the teacher's greedy
policy.  Val and test sets are generated first (fixed), then training
streams continuously until convergence or budget exhaustion.

Usage:
    python -m orchard.supervised_capacity \
        --config path/to/base.yaml \
        --checkpoint path/to/final.pt \
        --output-dir path/to/results/ \
        [--override model.conv_specs=[[4,3],[4,3]]] \
        [--seed 9999] [--n-val 10000] [--n-test 10000] \
        [--batch-size 512] [--lr 1e-3] [--max-train 500000] \
        [--min-train 50000] [--patience 30] [--check-freq 2000]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.append("../")

import torch

import orchard.encoding as encoding
from orchard.config import load_config
from orchard.enums import LearningType, Schedule, TrainMethod
from orchard.env import create_env
from orchard.model import ValueNetwork, create_networks
from orchard.policy import argmax_a_Q_team
from orchard.eval import evaluate_policy_learning
from orchard.seed import rng, set_all_seeds
from orchard.datatypes import EncoderOutput, ScheduleConfig, State


# ---------------------------------------------------------------------------
# Teacher loading
# ---------------------------------------------------------------------------
def load_teacher(cfg, checkpoint_path):
    """Load teacher networks from checkpoint using the ORIGINAL config."""
    centralized = cfg.train.learning_type == LearningType.CENTRALIZED
    n_networks = 1 if centralized else cfg.env.n_agents
    networks = create_networks(
        cfg.model, cfg.env, cfg.train.lr, cfg.train.total_steps,
        nstep=cfg.train.nstep, td_lambda=cfg.train.td_lambda,
        train_method=cfg.train.train_method, n_networks=n_networks,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if len(ckpt["networks"]) != len(networks):
        raise ValueError(
            f"Checkpoint has {len(ckpt['networks'])} networks but config "
            f"expects {len(networks)}."
        )
    for net, sd in zip(networks, ckpt["networks"]):
        net.load_state_dict(sd, strict=True)
    for net in networks:
        net.eval()
    return networks


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def after_state_stream(env, teacher_nets, centralized):
    """Yield (net_idx, EncoderOutput, target_value) from teacher's greedy rollout.

    Produces both movement after-states and pick after-states, matching the
    distribution the TD learner was trained on.  For decentralized, each
    after-state yields one sample per agent network.
    """
    state = env.init_state()
    n_networks = len(teacher_nets)

    while True:
        action = argmax_a_Q_team(state, teacher_nets, env)
        s_moved = env.apply_action(state, action)
        on_apple = s_moved.is_agent_on_apple(s_moved.actor)

        after_states: list[State] = [s_moved]
        if on_apple:
            s_picked, _ = env.resolve_pick(s_moved)
            after_states.append(s_picked)

        for s_after in after_states:
            for i in range(n_networks):
                enc = encoding.encode(s_after, 0 if centralized else i)
                with torch.no_grad():
                    target = teacher_nets[i](enc).item()
                yield (i, enc, target)

        # Advance environment
        if on_apple:
            state = env.advance_actor(env.spawn_and_despawn(s_picked))
        else:
            state = env.advance_actor(env.spawn_and_despawn(s_moved))


def collect_fixed_set(stream, n_per_network, n_networks):
    """Collect exactly n_per_network samples for each network from the stream."""
    data = {i: [] for i in range(n_networks)}
    counts = {i: 0 for i in range(n_networks)}

    while any(counts[i] < n_per_network for i in range(n_networks)):
        net_idx, enc, target = next(stream)
        if counts[net_idx] < n_per_network:
            data[net_idx].append((enc, target))
            counts[net_idx] += 1

    return data


def batch_dataset(samples):
    """Convert list of (EncoderOutput, target) to stacked tensors.

    Returns (grid_batch | None, scalar_batch | None, target_tensor).
    """
    grids, scalars, targets = [], [], []
    for enc, t in samples:
        if enc.grid is not None:
            grids.append(enc.grid)
        if enc.scalar is not None:
            scalars.append(enc.scalar)
        targets.append(t)

    return (
        torch.stack(grids) if grids else None,
        torch.stack(scalars) if scalars else None,
        torch.tensor(targets, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(net, grid, scalar, targets):
    """Compute MSE, MAE, R² on a pre-batched dataset."""
    net.eval()
    with torch.no_grad():
        preds = net(EncoderOutput(grid=grid, scalar=scalar))
        mse = ((preds - targets) ** 2).mean().item()
        mae = (preds - targets).abs().mean().item()
        ss_res = ((targets - preds) ** 2).sum().item()
        ss_tot = ((targets - targets.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    net.train()
    return mse, mae, r2


# ---------------------------------------------------------------------------
# Student building
# ---------------------------------------------------------------------------
def build_students(model_cfg, env_cfg, n_networks, lr):
    """Build student network(s) with Adam optimizer."""
    dummy_lr = ScheduleConfig(start=lr, end=lr, schedule=Schedule.NONE)
    students = create_networks(
        model_cfg, env_cfg, dummy_lr, total_steps=1,
        nstep=1, td_lambda=0.0, train_method=TrainMethod.NSTEP,
        n_networks=n_networks,
    )
    for net in students:
        net.optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
    return students


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_training_checkpoint(path, students, samples_seen, best_avg_val_mse,
                             patience_counter, next_lr_step, train_loss_accum,
                             train_loss_count):
    """Save full training state for resumption."""
    ckpt = {
        "students": [net.state_dict() for net in students],
        "optimizers": [net.optimizer.state_dict() for net in students],
        "samples_seen": samples_seen,
        "best_avg_val_mse": best_avg_val_mse,
        "patience_counter": patience_counter,
        "next_lr_step": next_lr_step,
        "train_loss_accum": train_loss_accum,
        "train_loss_count": train_loss_count,
        "rng_state": rng.getstate(),
        "torch_rng_state": torch.random.get_rng_state(),
    }
    torch.save(ckpt, path)


def load_training_checkpoint(path, students):
    """Load full training state. Returns dict with all saved fields."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for i, net in enumerate(students):
        net.load_state_dict(ckpt["students"][i])
        net.optimizer.load_state_dict(ckpt["optimizers"][i])
    rng.setstate(ckpt["rng_state"])
    torch.random.set_rng_state(ckpt["torch_rng_state"])
    return ckpt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Supervised capacity test for architecture sizing"
    )
    parser.add_argument("--config", required=True, help="Base YAML config")
    parser.add_argument("--checkpoint", required=True, help="Teacher checkpoint (.pt)")
    parser.add_argument("--output-dir", required=True, help="Results directory")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Override student model config (dot notation)")
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--n-val", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-step-every", type=int, default=100000,
                        help="Halve LR every N training samples (per network)")
    parser.add_argument("--lr-factor", type=float, default=0.5,
                        help="LR multiplication factor at each step")
    parser.add_argument("--max-train", type=int, default=500000,
                        help="Maximum training samples per network")
    parser.add_argument("--min-train", type=int, default=50000,
                        help="Minimum training samples before early stopping")
    parser.add_argument("--patience", type=int, default=30,
                        help="Number of val checks without improvement before stopping")
    parser.add_argument("--check-freq", type=int, default=2000,
                        help="Validate every N training samples (per network)")
    parser.add_argument("--checkpoint-freq", type=int, default=50000,
                        help="Save full checkpoint every N samples (per network). 0=disabled.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load configs: teacher (original) and student (with overrides)
    # ------------------------------------------------------------------
    teacher_cfg = load_config(args.config, [])
    student_cfg = load_config(args.config, args.override)
    student_model_cfg = student_cfg.model

    centralized = teacher_cfg.train.learning_type == LearningType.CENTRALIZED
    n_agents = teacher_cfg.env.n_agents
    n_networks = 1 if centralized else n_agents

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Init env and encoder (must match teacher)
    # ------------------------------------------------------------------
    set_all_seeds(args.seed)
    env = create_env(teacher_cfg.env)
    encoding.init_encoder(
        teacher_cfg.model.input_type, teacher_cfg.env, teacher_cfg.model.k_nearest
    )

    # ------------------------------------------------------------------
    # Load teacher
    # ------------------------------------------------------------------
    teacher_nets = load_teacher(teacher_cfg, args.checkpoint)
    print(f"Teacher loaded from: {args.checkpoint}")
    print(f"  Mode: {'centralized' if centralized else 'decentralized'}")
    print(f"  Teacher arch: conv={teacher_cfg.model.conv_specs}  mlp={teacher_cfg.model.mlp_dims}")
    print(f"  Student arch: conv={student_model_cfg.conv_specs}  mlp={student_model_cfg.mlp_dims}")

    # ------------------------------------------------------------------
    # Generate val and test sets (deterministic from seed)
    # ------------------------------------------------------------------
    print(f"\nGenerating val ({args.n_val}/net) and test ({args.n_test}/net) ...")
    stream = after_state_stream(env, teacher_nets, centralized)

    val_data = collect_fixed_set(stream, args.n_val, n_networks)
    test_data = collect_fixed_set(stream, args.n_test, n_networks)

    val_batched = {}
    test_batched = {}
    for i in range(n_networks):
        val_batched[i] = batch_dataset(val_data[i])
        test_batched[i] = batch_dataset(test_data[i])
        tgt = val_batched[i][2]
        print(f"  Net {i}: val targets [{tgt.min():.4f}, {tgt.max():.4f}]  "
              f"mean={tgt.mean():.4f}  std={tgt.std():.4f}")

    # Free raw lists
    del val_data, test_data

    # ------------------------------------------------------------------
    # Build students
    # ------------------------------------------------------------------
    students = build_students(student_model_cfg, teacher_cfg.env, n_networks, args.lr)
    for i, net in enumerate(students):
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  Student net {i}: {n_params:,} parameters")

    # ------------------------------------------------------------------
    # Checkpoint directory
    # ------------------------------------------------------------------
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training log CSV
    # ------------------------------------------------------------------
    log_path = out_dir / "training_log.csv"
    log_fields = ["samples_seen", "wall_time", "ram_mb", "current_lr", "avg_train_loss", "avg_val_mse"]
    for i in range(n_networks):
        log_fields.extend([f"val_mse_{i}", f"val_mae_{i}", f"val_r2_{i}"])

    resuming = args.resume is not None
    if resuming:
        # Append to existing log
        log_file = open(log_path, "a", newline="")
        log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    else:
        log_file = open(log_path, "w", newline="")
        log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
        log_writer.writeheader()

    # ------------------------------------------------------------------
    # Training loop — online streaming, patience on avg val MSE
    # ------------------------------------------------------------------
    print(f"\nTraining (max {args.max_train}/net, patience {args.patience}, "
          f"check every {args.check_freq}) ...")
    start_time = time.time()

    samples_seen = [0] * n_networks
    batch_buffers: dict[int, list] = {i: [] for i in range(n_networks)}

    best_avg_val_mse = float("inf")
    patience_counter = 0
    stopped = False
    train_loss_accum = 0.0
    train_loss_count = 0
    next_lr_step = [args.lr_step_every] * n_networks

    # ------------------------------------------------------------------
    # Resume from checkpoint if requested
    # ------------------------------------------------------------------
    if resuming:
        print(f"\nResuming from: {args.resume}")
        ckpt = load_training_checkpoint(args.resume, students)
        samples_seen = ckpt["samples_seen"]
        best_avg_val_mse = ckpt["best_avg_val_mse"]
        patience_counter = ckpt["patience_counter"]
        next_lr_step = ckpt["next_lr_step"]
        train_loss_accum = ckpt["train_loss_accum"]
        train_loss_count = ckpt["train_loss_count"]
        print(f"  Resumed at samples_seen={samples_seen[0]}, "
              f"best_val_mse={best_avg_val_mse:.6f}, patience={patience_counter}")
        # Re-create stream from restored RNG state
        stream = after_state_stream(env, teacher_nets, centralized)  # next sample count to decay LR

    while not stopped:
        # Pull next sample from the teacher stream
        net_idx, enc, target = next(stream)
        samples_seen[net_idx] += 1

        # Manual LR decay
        if samples_seen[net_idx] >= next_lr_step[net_idx]:
            for pg in students[net_idx].optimizer.param_groups:
                pg["lr"] *= args.lr_factor
            next_lr_step[net_idx] += args.lr_step_every

        # Accumulate into batch buffer
        batch_buffers[net_idx].append((enc, target))

        # Train when batch is full
        if len(batch_buffers[net_idx]) >= args.batch_size:
            batch = batch_buffers[net_idx][:args.batch_size]
            batch_buffers[net_idx] = batch_buffers[net_idx][args.batch_size:]

            grid_b, scalar_b, target_b = batch_dataset(batch)
            students[net_idx].train()
            pred = students[net_idx](EncoderOutput(grid=grid_b, scalar=scalar_b))
            loss = ((pred - target_b) ** 2).mean()
            students[net_idx].optimizer.zero_grad()
            loss.backward()
            students[net_idx].optimizer.step()
            train_loss_accum += loss.item()
            train_loss_count += 1

        # Periodic validation (trigger on net 0's count for consistency)
        if (net_idx == 0
                and samples_seen[0] > 0
                and samples_seen[0] % args.check_freq == 0):

            elapsed = time.time() - start_time
            try:
                import resource
                ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            except ImportError:
                ram_mb = 0.0
            row = {
                "samples_seen": samples_seen[0],
                "wall_time": round(elapsed, 1),
                "ram_mb": round(ram_mb, 1),
            }

            avg_mse = 0.0
            for i in range(n_networks):
                mse, mae, r2 = evaluate(students[i], *val_batched[i])
                row[f"val_mse_{i}"] = round(mse, 8)
                row[f"val_mae_{i}"] = round(mae, 6)
                row[f"val_r2_{i}"] = round(r2, 6)
                avg_mse += mse
            avg_mse /= n_networks
            row["avg_val_mse"] = round(avg_mse, 8)
            avg_train = train_loss_accum / max(train_loss_count, 1)
            row["avg_train_loss"] = round(avg_train, 8)
            train_loss_accum = 0.0
            train_loss_count = 0

            # LR logging
            current_lr = students[0].optimizer.param_groups[0]["lr"]
            row["current_lr"] = current_lr

            log_writer.writerow(row)
            log_file.flush()

            # Print progress
            net_strs = "  ".join(
                f"net{i} mse={row[f'val_mse_{i}']:.6f} r2={row[f'val_r2_{i}']:.4f}"
                for i in range(n_networks)
            )
            print(f"  [{elapsed:6.0f}s] samples={samples_seen[0]:>7d} | "
                  f"train={avg_train:.6f} val={avg_mse:.6f} lr={current_lr:.2e} | "
                  f"ram={ram_mb:.0f}MB | {net_strs}")

            # Early stopping on avg val MSE
            if avg_mse < best_avg_val_mse:
                best_avg_val_mse = avg_mse
                patience_counter = 0
                # Save best student weights
                for i in range(n_networks):
                    torch.save(
                        students[i].state_dict(),
                        out_dir / f"best_student_{i}.pt",
                    )
            else:
                patience_counter += 1

            min_reached = samples_seen[0] >= args.min_train
            if min_reached and patience_counter >= args.patience:
                print(f"\n  Early stop: avg val MSE {best_avg_val_mse:.8f} "
                      f"unchanged for {args.patience} checks.")
                stopped = True

            if samples_seen[0] >= args.max_train:
                print(f"\n  Budget exhausted at {args.max_train} samples/net.")
                stopped = True

    log_file.close()

    # ------------------------------------------------------------------
    # Final test evaluation (load best weights)
    # ------------------------------------------------------------------
    print(f"\nFinal test evaluation (best val checkpoint):")
    results: dict[str, object] = {
        "conv_specs": str(student_model_cfg.conv_specs),
        "mlp_dims": str(student_model_cfg.mlp_dims),
        "centralized": centralized,
        "seed": args.seed,
        "n_params": sum(p.numel() for p in students[0].parameters()),
        "best_avg_val_mse": best_avg_val_mse,
    }

    for i in range(n_networks):
        best_path = out_dir / f"best_student_{i}.pt"
        if best_path.exists():
            students[i].load_state_dict(
                torch.load(best_path, map_location="cpu", weights_only=True)
            )
        students[i].eval()

        mse, mae, r2 = evaluate(students[i], *test_batched[i])
        results[f"test_mse_{i}"] = mse
        results[f"test_mae_{i}"] = mae
        results[f"test_r2_{i}"] = r2
        print(f"  Net {i}: MSE={mse:.8f}  MAE={mae:.6f}  R²={r2:.6f}")

    # Averages
    avg_test_mse = sum(results[f"test_mse_{i}"] for i in range(n_networks)) / n_networks
    avg_test_mae = sum(results[f"test_mae_{i}"] for i in range(n_networks)) / n_networks
    avg_test_r2 = sum(results[f"test_r2_{i}"] for i in range(n_networks)) / n_networks
    results["avg_test_mse"] = avg_test_mse
    results["avg_test_mae"] = avg_test_mae
    results["avg_test_r2"] = avg_test_r2

    print(f"\n  Average: MSE={avg_test_mse:.8f}  MAE={avg_test_mae:.6f}  R²={avg_test_r2:.6f}")

    # ------------------------------------------------------------------
    # PPS evaluation (greedy rollout with student networks)
    # ------------------------------------------------------------------
    print(f"\nPPS evaluation (greedy rollout with best student):")
    pps_metrics = evaluate_policy_learning(students, env, teacher_cfg.eval.eval_steps)
    results["greedy_pps"] = pps_metrics["greedy_pps"]
    results["nearest_pps"] = pps_metrics["nearest_pps"]
    print(f"  Greedy PPS: {pps_metrics['greedy_pps']:.4f}")
    print(f"  Nearest PPS: {pps_metrics['nearest_pps']:.4f}")

    results["samples_trained"] = samples_seen[0]
    results["wall_time"] = round(time.time() - start_time, 1)
    print(f"  Samples trained: {samples_seen[0]}  Wall time: {results['wall_time']}s")

    # Save summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Also save as single-row CSV for easy aggregation across jobs
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        writer.writeheader()
        writer.writerow(results)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
