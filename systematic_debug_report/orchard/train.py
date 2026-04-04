"""Training loop and CLI entry point."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import orchard.encoding as encoding
from orchard.config import load_config
from orchard.datatypes import ExperimentConfig, StoppingConfig
from orchard.enums import StoppingCondition
from orchard.env import create_env
from orchard.logging_ import (
    CSVLogger,
    build_detail_csv_fieldnames,
    build_main_csv_fieldnames,
    finalize_logging,
    setup_logging,
)
from orchard.model import create_networks
from orchard.schedule import compute_schedule_value
from orchard.seed import set_all_seeds
from orchard.trainer import create_trainer


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def _save_checkpoint(networks: list, step: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "networks": [net.state_dict() for net in networks],
    }
    torch.save(ckpt, path)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopper:
    def __init__(self, cfg: StoppingConfig, log_freq: int) -> None:
        self._condition = cfg.condition
        self._patience = cfg.patience_steps
        self._threshold = cfg.improvement_threshold
        self._min_steps = cfg.min_steps_before_stop
        self._log_freq = log_freq
        self._best: float = float("-inf")
        self._steps_since: int = 0
        self.should_stop: bool = False

    def check(self, t: int, metrics: dict[str, float | int]) -> bool:
        if self._condition == StoppingCondition.NONE:
            return False
        if self._condition == StoppingCondition.RUNNING_MAX_RPS:
            val = float(metrics.get("greedy_team_rps", 0.0))
            if val > self._best + self._threshold:
                self._best = val
                self._steps_since = 0
            else:
                self._steps_since += self._log_freq
            if self._steps_since >= self._patience and (t + 1) >= self._min_steps:
                print(f"\nEarly stop at step {t+1}: best team RPS {self._best:.6f} "
                      f"unchanged for {self._patience} steps.")
                self.should_stop = True
                return True
        return False


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(cfg: ExperimentConfig, resume_checkpoint: str | None = None) -> None:
    start_time = time.time()

    # --- Setup ---
    set_all_seeds(cfg.train.seed)
    env = create_env(cfg.env)
    encoding.init_encoder(cfg.model.encoder, cfg.env)
    networks = create_networks(cfg.model, cfg.env, cfg.train)
    n_networks = len(networks)

    # --- Resume from checkpoint ---
    if resume_checkpoint is not None:
        ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=True)
        for net, sd in zip(networks, ckpt["networks"]):
            net.load_state_dict(sd, strict=True)
        print(f"Loaded weights from: {resume_checkpoint} (step {ckpt.get('step', '?')})")

    trainer = create_trainer(cfg, networks, env)

    # --- Logging ---
    run_dir = setup_logging(cfg)
    _save_checkpoint(networks, 0, run_dir / "checkpoints" / "step_0.pt")

    heuristic_name = cfg.train.heuristic.name.lower()
    main_logger = CSVLogger(
        run_dir / "metrics.csv",
        build_main_csv_fieldnames(heuristic_name),
    )
    detail_logger = CSVLogger(
        run_dir / "details.csv",
        build_detail_csv_fieldnames(networks),
    )
    stopper = EarlyStopper(cfg.train.stopping, cfg.logging.main_csv_freq)

    state = env.init_state()

    # --- Main loop ---
    for t in range(cfg.train.total_steps):
        # ── Phase 1: Move ──
        move_action = trainer.select_move(state, t)
        s_moved = env.apply_action(state, move_action)
        on_task = s_moved.is_agent_on_task(s_moved.actor)
        trainer.train_move(s_moved, on_task, t) # note I used to do this after select_pick (but before train_pick), so select_pick is on more updated weights than the code of march 3
        # However, this is more mathematically correct as this is not true online.
        
        # ── Phase 2: Pick (only if on task) ──
        if on_task:
            pick_action = trainer.select_pick(s_moved, t)
            s_picked, pick_rewards = env.resolve_pick(
                s_moved,
                pick_type=pick_action.pick_type() if pick_action.is_pick() else None,
            )
            trainer.train_pick(s_picked, pick_rewards, t)
        else:
            s_picked = s_moved        

        # ── Advance to next turn ──
        state = env.advance_actor(env.spawn_and_despawn(s_picked))

        # ── Main CSV logging + eval ──
        if (t + 1) % cfg.logging.main_csv_freq == 0:
            trainer.sync_to_cpu()
            wall_time = time.time() - start_time
            metrics = trainer.evaluate(env, cfg.eval)
            row: dict[str, float | int | str] = {
                "step": t + 1,
                "wall_time": round(wall_time, 3),
                "td_loss_avg": round(trainer.get_td_loss(), 8),
            }
            row.update(metrics)
            main_logger.log(row)

            # Print progress
            print(f"\n--- Step {t + 1} ({wall_time:.1f}s) ---")
            print(f"  Greedy RPS: {metrics['greedy_rps']:.4f}  "
                  f"Team RPS: {metrics['greedy_team_rps']:.4f}  "
                  f"Correct PPS: {metrics['greedy_correct_pps']:.4f}  "
                  f"Wrong PPS: {metrics['greedy_wrong_pps']:.4f}")
            h_rps_key = f"{heuristic_name}_rps"
            h_team_key = f"{heuristic_name}_team_rps"
            if h_rps_key in metrics:
                print(f"  {heuristic_name} RPS: {metrics[h_rps_key]:.4f}  "
                      f"Team RPS: {metrics[h_team_key]:.4f}")

            if stopper.check(t, metrics):
                break

        # ── Detail CSV logging ──
        if (t + 1) % cfg.logging.detail_csv_freq == 0:
            trainer.sync_to_cpu()
            detail_row: dict[str, float | int | str] = {
                "step": t + 1,
                "wall_time": round(time.time() - start_time, 3),
            }
            try:
                import resource
                detail_row["ram_mb"] = round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1,
                )
            except ImportError:
                detail_row["ram_mb"] = 0.0

            if torch.cuda.is_available():
                detail_row["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
                detail_row["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
                detail_row["vram_total_mb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**2, 1,
                )

            detail_row["current_lr"] = compute_schedule_value(
                cfg.train.lr, t + 1, cfg.train.total_steps,
            )
            detail_row["current_epsilon"] = compute_schedule_value(
                cfg.train.epsilon, t + 1, cfg.train.total_steps,
            )

            for idx in range(n_networks):
                for name, val in networks[idx].get_weight_norms().items():
                    detail_row[f"weight_norm_agent_{idx}_{name}"] = round(val, 6)
                for name, val in networks[idx].get_grad_norms().items():
                    detail_row[f"grad_norm_agent_{idx}_{name}"] = round(val, 6)

            detail_row["td_loss_step"] = round(trainer.get_td_loss(), 8)
            detail_logger.log(detail_row)

        # ── Periodic checkpoint ──
        if cfg.eval.checkpoint_freq > 0 and (t + 1) % cfg.eval.checkpoint_freq == 0:
            trainer.sync_to_cpu()
            _save_checkpoint(networks, t + 1, run_dir / "checkpoints" / f"step_{t + 1}.pt")

    # --- Finalize ---
    trainer.sync_to_cpu()
    _save_checkpoint(networks, cfg.train.total_steps, run_dir / "checkpoints" / "final.pt")
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
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint (.pt) to load pretrained weights.")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Override config values: key=value (dot notation)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    train(cfg, resume_checkpoint=args.resume)


if __name__ == "__main__":
    main()
