import os
import random
import time
import cProfile
import pstats

import torch

from debug.code.training.helpers import env_step, random_policy
from debug.code.core.log import finalize_logging


class LearningLoopMixin:
    _PIPELINE_PROFILE_STAGES = (
        "action_select",
        "env_step",
        "train_update",
        "diagnostics",
        "eval",
        "other",
        "loop_total",
    )

    def _reset_pipeline_profile(self) -> None:
        self.pipeline_profile_stage_totals = {stage: 0.0 for stage in self._PIPELINE_PROFILE_STAGES}
        self.pipeline_profile_stage_snapshot = {stage: 0.0 for stage in self._PIPELINE_PROFILE_STAGES}
        self.pipeline_profile_last_step = 0
        self.pipeline_profile_last_wall_time = 0.0

    def _pipeline_wall_time(self) -> float:
        if self.train_start_time is None:
            return 0.0
        return max(0.0, float(time.time() - self.train_start_time))

    def _add_pipeline_time(self, stage: str, elapsed_seconds: float) -> None:
        if not self.pipeline_profile_enabled:
            return
        if stage not in self.pipeline_profile_stage_totals:
            return
        self.pipeline_profile_stage_totals[stage] += max(0.0, float(elapsed_seconds))

    def _maybe_log_pipeline_profile(self, *, step: int, force: bool = False) -> None:
        if not self.pipeline_profile_enabled or self.pipeline_profile_logger is None:
            return
        freq = max(1, int(self.pipeline_profile_freq))
        if not force and step > 0 and step % freq != 0:
            return

        wall_time = self._pipeline_wall_time()
        delta_by_stage = {
            stage: self.pipeline_profile_stage_totals[stage] - self.pipeline_profile_stage_snapshot[stage]
            for stage in self._PIPELINE_PROFILE_STAGES
        }
        loop_total = max(0.0, delta_by_stage["loop_total"])

        steps_since_last = max(0, int(step) - int(self.pipeline_profile_last_step))
        wall_since_last = max(0.0, wall_time - self.pipeline_profile_last_wall_time)
        steps_per_second = (steps_since_last / wall_since_last) if wall_since_last > 0 else 0.0
        avg_step_seconds = (loop_total / steps_since_last) if steps_since_last > 0 else 0.0

        def _pct(stage_name: str) -> float:
            if loop_total <= 0:
                return 0.0
            return 100.0 * delta_by_stage[stage_name] / loop_total

        self.pipeline_profile_logger.log(
            {
                "step": int(step),
                "wall_time": round(wall_time, 6),
                "steps_since_last": int(steps_since_last),
                "wall_since_last": round(wall_since_last, 6),
                "steps_per_second": round(steps_per_second, 6),
                "avg_step_seconds": round(avg_step_seconds, 6),
                "action_select_s": round(delta_by_stage["action_select"], 6),
                "env_step_s": round(delta_by_stage["env_step"], 6),
                "train_update_s": round(delta_by_stage["train_update"], 6),
                "diagnostics_s": round(delta_by_stage["diagnostics"], 6),
                "eval_s": round(delta_by_stage["eval"], 6),
                "other_s": round(delta_by_stage["other"], 6),
                "loop_total_s": round(loop_total, 6),
                "action_select_pct": round(_pct("action_select"), 3),
                "env_step_pct": round(_pct("env_step"), 3),
                "train_update_pct": round(_pct("train_update"), 3),
                "diagnostics_pct": round(_pct("diagnostics"), 3),
                "eval_pct": round(_pct("eval"), 3),
                "other_pct": round(_pct("other"), 3),
            }
        )

        self.pipeline_profile_stage_snapshot = dict(self.pipeline_profile_stage_totals)
        self.pipeline_profile_last_step = int(step)
        self.pipeline_profile_last_wall_time = wall_time

    def _save_cprofile_outputs(self, profiler: cProfile.Profile) -> None:
        stats_path = self.data_dir / "cprofile.pstats"
        top_path = self.data_dir / "cprofile_top.txt"
        profiler.dump_stats(str(stats_path))

        sort_by = str(self.cprofile_sort_by)
        with open(top_path, "w") as f:
            stats = pstats.Stats(profiler, stream=f)
            try:
                stats.sort_stats(sort_by)
            except KeyError:
                f.write(f"Invalid cprofile_sort_by='{sort_by}'. Falling back to 'cumulative'.\n\n")
                stats.sort_stats("cumulative")
            stats.print_stats(int(self.cprofile_top_n))

    def _supervised_target(self, encoded_state, network_idx: int):
        if not self.supervised_enabled:
            return None
        if network_idx < 0 or network_idx >= len(self.supervised_networks):
            raise IndexError(
                f"Missing supervised teacher for student index {network_idx}. "
                f"Teachers={len(self.supervised_networks)}."
            )
        with torch.no_grad():
            pred = self.supervised_networks[network_idx].model(encoded_state).squeeze()
        return float(pred.item())

    def _run_initial_evaluation(self) -> None:
        if self.supervised_enabled:
            self.eval_performance(0)
        elif self.exp_config.algorithm.random_policy:
            self.evaluate_networks(step=0, plot=True, store_last=True)
        elif self.exp_config.reward.reward_learning:
            self.evaluate_networks_reward(step=0, plot=True, store_last=True)
        else:
            self.eval_performance(0)
        self._maybe_log_action_probabilities(step=0)
        self._maybe_log_tracked_state_values(step=0)
        self._maybe_log_weight_samples(step=0)

    def _select_training_action(self, curr_state: dict, actor_idx: int):
        if self.exp_config.algorithm.random_policy or self.exp_config.reward.reward_learning:
            return random_policy(
                curr_state["agent_positions"][actor_idx],
                width=self.width,
                length=self.length,
            )
        return self.agent_controller.agent_get_action(self.env, actor_idx)

    def _train_centralized_transition(self, curr_state, s_moved, s_next, pick_rewards, on_apple):
        enc_t = self.encoder.encode(curr_state, 0)
        enc_moved = self.encoder.encode(s_moved, 0)
        enc_next = self.encoder.encode(s_next, 0)
        net = self.critic_networks[0]
        reward = sum(pick_rewards)
        true_t = self._supervised_target(enc_t, 0)
        true_moved = self._supervised_target(enc_moved, 0)

        if on_apple:
            net.add_experience(
                enc_t,
                enc_moved,
                0,
                discount_factor=self.discount_factor,
                true_value=true_t,
            )
            net.add_experience(
                enc_moved,
                enc_next,
                reward,
                discount_factor=1.0,
                true_value=true_moved,
            )
        else:
            net.add_experience(
                enc_t,
                enc_next,
                reward,
                discount_factor=self.discount_factor,
                true_value=true_t,
            )

        net.train()

    def _train_decentralized_transition(self, curr_state, s_moved, s_next, pick_rewards, on_apple):
        for i in range(len(self.critic_networks)):
            enc_t = self.encoder.encode(curr_state, i)
            enc_moved = self.encoder.encode(s_moved, i)
            enc_next = self.encoder.encode(s_next, i)
            true_t = self._supervised_target(enc_t, i)
            true_moved = self._supervised_target(enc_moved, i)

            if on_apple:
                self.critic_networks[i].add_experience(
                    enc_t,
                    enc_moved,
                    0,
                    discount_factor=self.discount_factor,
                    true_value=true_t,
                )
                self.critic_networks[i].add_experience(
                    enc_moved,
                    enc_next,
                    pick_rewards[i],
                    discount_factor=1.0,
                    true_value=true_moved,
                )
            else:
                self.critic_networks[i].add_experience(
                    enc_t,
                    enc_next,
                    pick_rewards[i],
                    discount_factor=self.discount_factor,
                    true_value=true_t,
                )

            self.critic_networks[i].train()

    def _run_periodic_evaluation(self, step: int) -> None:
        print(f"Running evaluation at step {step}/{self.trajectory_length}")
        if self.supervised_enabled:
            self.eval_performance(step)
        elif not self.exp_config.reward.reward_learning:
            if self.exp_config.algorithm.random_policy:
                plot = step == self.trajectory_length
                self.evaluate_networks(step=step, plot=plot, store_last=True)
            else:
                self.eval_performance(step)
        else:
            plot = step == self.trajectory_length
            self.evaluate_networks_reward(step=step, plot=plot, store_last=True)

        self._maybe_log_action_probabilities(step=step)
        self._maybe_log_tracked_state_values(step=step)

    def step_and_collect_observation(self) -> None:
        start = time.perf_counter()
        self._run_initial_evaluation()
        dt = time.perf_counter() - start
        eval_stage = "eval" if self.pipeline_profile_include_eval else "other"
        self._add_pipeline_time(eval_stage, dt)
        self._add_pipeline_time("loop_total", dt)
        self._maybe_log_pipeline_profile(step=0, force=True)

        curr_state = dict(self.env.get_state())
        curr_state["actor_id"] = 0
        actor_idx = 0

        for sec in range(self.trajectory_length):
            step_start = time.perf_counter()
            measured = 0.0

            start = time.perf_counter()
            new_pos = self._select_training_action(curr_state, actor_idx)
            dt = time.perf_counter() - start
            measured += dt
            self._add_pipeline_time("action_select", dt)

            start = time.perf_counter()
            s_moved, s_next, pick_rewards, on_apple, next_actor_idx = env_step(
                self.env,
                actor_idx,
                new_pos,
                self.num_agents,
            )
            dt = time.perf_counter() - start
            measured += dt
            self._add_pipeline_time("env_step", dt)

            start = time.perf_counter()
            if self.exp_config.algorithm.centralized:
                self._train_centralized_transition(curr_state, s_moved, s_next, pick_rewards, on_apple)
            else:
                self._train_decentralized_transition(curr_state, s_moved, s_next, pick_rewards, on_apple)
            dt = time.perf_counter() - start
            measured += dt
            self._add_pipeline_time("train_update", dt)

            curr_state = s_next
            actor_idx = next_actor_idx

            start = time.perf_counter()
            self._maybe_log_weight_samples(step=(sec + 1))
            dt = time.perf_counter() - start
            measured += dt
            self._add_pipeline_time("diagnostics", dt)

            if (sec + 1) % self.exp_config.logging.main_csv_freq == 0:
                start = time.perf_counter()
                self._run_periodic_evaluation(step=(sec + 1))
                dt = time.perf_counter() - start
                measured += dt
                eval_stage = "eval" if self.pipeline_profile_include_eval else "other"
                self._add_pipeline_time(eval_stage, dt)

            step_elapsed = time.perf_counter() - step_start
            self._add_pipeline_time("loop_total", step_elapsed)
            self._add_pipeline_time("other", max(0.0, step_elapsed - measured))
            self._maybe_log_pipeline_profile(step=(sec + 1))

    def _close_loggers(self) -> None:
        self.main_logger.close()
        for logger in self.action_prob_loggers.values():
            logger.close()
        for logger in self.value_track_loggers.values():
            logger.close()
        for logger in self.weight_sample_loggers.values():
            logger.close()
        if self.pipeline_profile_logger is not None:
            self.pipeline_profile_logger.close()

    def train(self):
        start_time = time.time()
        self.train_start_time = start_time
        self._reset_pipeline_profile()
        profiler = cProfile.Profile() if self.cprofile_enabled else None
        if profiler is not None:
            profiler.enable()

        try:
            start = time.perf_counter()
            self.build_experiment()
            dt = time.perf_counter() - start
            self._add_pipeline_time("other", dt)
            self._add_pipeline_time("loop_total", dt)

            self.step_and_collect_observation()

            start = time.perf_counter()
            self._write_last_greedy_position_heatmaps()
            dt = time.perf_counter() - start
            self._add_pipeline_time("other", dt)
            self._add_pipeline_time("loop_total", dt)

            start = time.perf_counter()
            self.save_networks(self.data_dir / "weights")
            dt = time.perf_counter() - start
            self._add_pipeline_time("other", dt)
            self._add_pipeline_time("loop_total", dt)

            start = time.perf_counter()
            finalize_logging(self.data_dir, start_time)
            dt = time.perf_counter() - start
            self._add_pipeline_time("other", dt)
            self._add_pipeline_time("loop_total", dt)
        finally:
            if profiler is not None:
                profiler.disable()
            self._maybe_log_pipeline_profile(step=self.trajectory_length, force=True)
            if profiler is not None:
                self._save_cprofile_outputs(profiler)
            self._close_loggers()

    def save_networks(self, path):
        os.makedirs(path, exist_ok=True)
        print("Saving networks: ", random.getstate()[1][0])
        payload = {"critics": []}
        for crit in self.critic_networks:
            payload["critics"].append({"blob": crit.export_net_state()})

        dst = path / "weights.pt"
        torch.save(payload, dst)
