import os
import random
import time

import torch

from debug.code.core.enums import NUM_AGENTS
from debug.code.training.helpers import env_step, random_policy
from debug.code.core.log import finalize_logging


class LearningLoopMixin:
    def _run_initial_evaluation(self) -> None:
        if self.exp_config.algorithm.random_policy:
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
            return random_policy(curr_state["agent_positions"][actor_idx])
        return self.agent_controller.agent_get_action(self.env, actor_idx)

    def _train_centralized_transition(self, curr_state, s_moved, s_next, pick_rewards, on_apple):
        enc_t = self.encoder.encode(curr_state, 0)
        enc_moved = self.encoder.encode(s_moved, 0)
        enc_next = self.encoder.encode(s_next, 0)
        net = self.critic_networks[0]
        reward = sum(pick_rewards)

        if on_apple:
            net.add_experience(enc_t, enc_moved, 0, discount_factor=self.discount_factor)
            net.add_experience(enc_moved, enc_next, reward, discount_factor=1.0)
        else:
            net.add_experience(enc_t, enc_next, reward, discount_factor=self.discount_factor)

        net.train()

    def _train_decentralized_transition(self, curr_state, s_moved, s_next, pick_rewards, on_apple):
        for i in range(len(self.critic_networks)):
            enc_t = self.encoder.encode(curr_state, i)
            enc_moved = self.encoder.encode(s_moved, i)
            enc_next = self.encoder.encode(s_next, i)

            if on_apple:
                self.critic_networks[i].add_experience(
                    enc_t,
                    enc_moved,
                    0,
                    discount_factor=self.discount_factor,
                )
                self.critic_networks[i].add_experience(
                    enc_moved,
                    enc_next,
                    pick_rewards[i],
                    discount_factor=1.0,
                )
            else:
                self.critic_networks[i].add_experience(
                    enc_t,
                    enc_next,
                    pick_rewards[i],
                    discount_factor=self.discount_factor,
                )

            self.critic_networks[i].train()

    def _run_periodic_evaluation(self, step: int) -> None:
        print(f"Running evaluation at step {step}/{self.trajectory_length}")
        if not self.exp_config.reward.reward_learning:
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
        self._run_initial_evaluation()

        curr_state = dict(self.env.get_state())
        curr_state["actor_id"] = 0
        actor_idx = 0

        for sec in range(self.trajectory_length):
            new_pos = self._select_training_action(curr_state, actor_idx)
            s_moved, s_next, pick_rewards, on_apple, next_actor_idx = env_step(
                self.env,
                actor_idx,
                new_pos,
                NUM_AGENTS,
            )

            if self.exp_config.algorithm.centralized:
                self._train_centralized_transition(curr_state, s_moved, s_next, pick_rewards, on_apple)
            else:
                self._train_decentralized_transition(curr_state, s_moved, s_next, pick_rewards, on_apple)

            curr_state = s_next
            actor_idx = next_actor_idx
            self._maybe_log_weight_samples(step=(sec + 1))

            if (sec + 1) % self.exp_config.logging.main_csv_freq == 0:
                self._run_periodic_evaluation(step=(sec + 1))

    def _close_loggers(self) -> None:
        self.main_logger.close()
        for logger in self.action_prob_loggers.values():
            logger.close()
        for logger in self.value_track_loggers.values():
            logger.close()
        for logger in self.weight_sample_loggers.values():
            logger.close()

    def train(self):
        start_time = time.time()
        self.train_start_time = start_time
        self.build_experiment()
        self.step_and_collect_observation()
        self._write_last_greedy_position_heatmaps()
        self.save_networks(self.data_dir / "weights")
        finalize_logging(self.data_dir, start_time)
        self._close_loggers()

    def save_networks(self, path):
        os.makedirs(path, exist_ok=True)
        print("Saving networks: ", random.getstate()[1][0])
        payload = {"critics": []}
        for crit in self.critic_networks:
            payload["critics"].append({"blob": crit.export_net_state()})

        dst = path / "weights.pt"
        torch.save(payload, dst)
