"""Training loop and CLI entry point."""

from __future__ import annotations

import argparse
import time

import torch

from orchard.trainer.timer import TimerSection

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
from orchard.schedule import compute_schedule_value
from orchard.seed import set_all_seeds
from orchard.trainer import create_trainer


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
def train(cfg: ExperimentConfig, resume_checkpoint: str | None = None, resume_critic_only: str | None = None, resume_actor_only: str | None = None) -> None:
    start_time = time.time()

    # --- Setup ---
    set_all_seeds(cfg.train.seed)
    env = create_env(cfg.env)
    encoding.init_encoder(cfg.model.encoder, cfg.env)
    trainer = create_trainer(cfg, env)

    if resume_checkpoint is not None:
        loaded_step = trainer.load_checkpoint(resume_checkpoint)
        print(f"Loaded weights from: {resume_checkpoint} (step {loaded_step if loaded_step is not None else '?'})")
    if resume_critic_only is not None:
        loaded_step = trainer.load_critic_checkpoint(resume_critic_only)
        print(f"Loaded critic-only weights from: {resume_critic_only} (step {loaded_step if loaded_step is not None else '?'})")
    if resume_actor_only is not None:
        loaded_step = trainer.load_actor_checkpoint(resume_actor_only)
        print(f"Loaded actor-only weights from: {resume_actor_only} (step {loaded_step if loaded_step is not None else '?'})")

    # --- Logging ---
    run_dir = setup_logging(cfg)
    trainer.setup_aux_loggers(run_dir, alpha_state_log_freq=cfg.logging.alpha_state_log_freq)
    trainer.save_checkpoint(run_dir / "checkpoints" / "step_0.pt", 0)

    heuristic_name = cfg.train.heuristic.name.lower()
    main_logger = CSVLogger(
        run_dir / "metrics.csv",
        build_main_csv_fieldnames(
            heuristic_name,
            actor_critic=bool(trainer.actor_networks),
            following_rates=cfg.train.following_rates.enabled,
            influencer=cfg.train.influencer.enabled,
        ),
    )
    detail_logger = CSVLogger(
        run_dir / "details.csv",
        build_detail_csv_fieldnames(trainer.critic_networks, trainer.actor_networks),
    )
    stopper = EarlyStopper(cfg.train.stopping, cfg.logging.main_csv_freq)

    timing_logger = None
    _nvml_available = False
    _nvml_handle = None
    if cfg.logging.timing_csv_freq > 0:
        try:
            import pynvml
            pynvml.nvmlInit()
            _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            _nvml_available = True
        except Exception as e:
            print(f"[timing] pynvml unavailable ({e}), sm_util will be -1")
        timing_logger = CSVLogger(
            run_dir / "timing.csv",
            ["step", "wall_time",
             "encode_ms", "train_ms", "action_ms", "env_ms", "eval_ms",
             "total_ms",
             "sm_util_pct", "gpu_mem_util_pct",
             "vram_allocated_mb"],
        )

    state = env.init_state()
    last_completed_step = 0

    # --- Main loop ---
    for t in range(cfg.train.total_steps):
        state = trainer.step(state, t)
        last_completed_step = t + 1
        td_loss_value: float | None = None

        # ── Main CSV logging + eval ──
        if (t + 1) % cfg.logging.main_csv_freq == 0:
            trainer.sync_to_cpu()
            wall_time = time.time() - start_time
            metrics = trainer.evaluate(env, cfg.eval)
            td_loss_value = round(trainer.get_td_loss(), 8)
            row: dict[str, float | int | str] = {
                "step": t + 1,
                "wall_time": round(wall_time, 3),
                "td_loss_avg": td_loss_value,
            }
            row.update(metrics)
            row.update(trainer.get_main_metrics())
            main_logger.log(row)
            trainer.log_auxiliary(t + 1, round(wall_time, 3))

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

            for idx, net in enumerate(trainer.critic_networks):
                for name, val in net.get_weight_norms().items():
                    detail_row[f"critic_weight_norm_agent_{idx}_{name}"] = round(val, 6)
                for name, val in net.get_grad_norms().items():
                    detail_row[f"critic_grad_norm_agent_{idx}_{name}"] = round(val, 6)

            if td_loss_value is None:
                td_loss_value = round(trainer.get_td_loss(), 8)
            detail_row["td_loss_step"] = td_loss_value
            detail_row.update(trainer.get_detail_metrics())
            detail_logger.log(detail_row)

        # ── Periodic checkpoint ──
        if cfg.eval.checkpoint_freq > 0 and (t + 1) % cfg.eval.checkpoint_freq == 0:
            trainer.sync_to_cpu()
            trainer.save_checkpoint(run_dir / "checkpoints" / f"step_{t + 1}.pt", t + 1)

        # ── Timing CSV ──
        if timing_logger is not None and (t + 1) % cfg.logging.timing_csv_freq == 0:
            report = trainer._timer.report_and_reset()
            encode_ms = round(report[TimerSection.ENCODE] * 1000, 4)
            train_ms  = round(report[TimerSection.TRAIN]  * 1000, 4)
            action_ms = round(report[TimerSection.ACTION] * 1000, 4)
            env_ms    = round(report[TimerSection.ENV]    * 1000, 4)
            eval_ms   = round(report[TimerSection.EVAL]   * 1000, 4)

            sm_util = gpu_mem_util = -1
            if _nvml_available:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
                    sm_util      = util.gpu
                    gpu_mem_util = util.memory
                except Exception:
                    pass

            timing_logger.log({
                "step":             t + 1,
                "wall_time":        round(time.time() - start_time, 3),
                "encode_ms":        encode_ms,
                "train_ms":         train_ms,
                "action_ms":        action_ms,
                "env_ms":           env_ms,
                "eval_ms":          eval_ms,
                "total_ms":         round(encode_ms + train_ms + action_ms + env_ms + eval_ms, 4),
                "sm_util_pct":      sm_util,
                "gpu_mem_util_pct": gpu_mem_util,
                "vram_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1)
                                     if torch.cuda.is_available() else -1,
            })

    # --- Finalize ---
    trainer.flush_pending_updates()
    trainer.sync_to_cpu()
    trainer.save_checkpoint(run_dir / "checkpoints" / "final.pt", last_completed_step)
    main_logger.close()
    detail_logger.close()
    if timing_logger is not None:
        timing_logger.close()
    trainer.close()
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
    parser.add_argument("--resume-critic-only", type=str, default=None,
                        help="Path to checkpoint (.pt) to load critic weights only; actors train from scratch.")
    parser.add_argument("--resume-actor-only", type=str, default=None,
                        help="Path to checkpoint (.pt) to load actor weights only; critics train from scratch.")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Override config values: key=value (dot notation)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    train(cfg, resume_checkpoint=args.resume,
      resume_critic_only=args.resume_critic_only,
      resume_actor_only=args.resume_actor_only)


if __name__ == "__main__":
    main()
