import random

import numpy as np
import torch

from debug.code.training.helpers import eval_performance, set_all_seeds


class LearningEvalMixin:
    def eval_performance(self, step):
        self.save_rng_state()
        set_all_seeds(42069)
        self.agent_controller.epsilon = 0
        with torch.no_grad():
            results = eval_performance(
                agent_controller=self.agent_controller,
                reward_module=self.reward_module,
                d_apple=self.env.d_apple,
                p_apple=self.env.p_apple,
                num_agents=self.num_agents,
                width=self.width,
                length=self.length,
                max_apples=self.env.max_apples,
                capture_greedy_positions=True,
            )
        greedy_positions = results.pop("greedy_agent_positions", None)
        results["step"] = step
        results["current_lr"] = self.critic_networks[0].get_lr()
        self.agent_controller.epsilon = self.exp_config.train.epsilon
        self.main_logger.log(results)
        if greedy_positions is not None:
            self._save_greedy_eval_positions(step=step, positions=greedy_positions)
        self.restore_rng_state()

    def restore_rng_state(self):
        if self.rng_state is not None:
            random.setstate(self.rng_state["python"])
            np.random.set_state(self.rng_state["numpy"])
            torch.set_rng_state(self.rng_state["torch"])

    def save_rng_state(self):
        self.rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }

    def evaluate_networks(self, *, step: int | None = None, plot: bool = False, store_last: bool = True):
        errors_by_agent = {i: [] for i in range(self.num_agents)}
        ape_by_agent = {i: [] for i in range(self.num_agents)}
        in_ci_by_agent = {i: [] for i in range(self.num_agents)}
        eps = 1e-8

        for eval_state in self.evaluation_states:
            mc_values = eval_state["mc"]
            ci_low = eval_state.get("ci95_low", None)
            ci_high = eval_state.get("ci95_high", None)
            rewards = self.reward_module.get_reward(
                eval_state,
                eval_state["actor_id"],
                eval_state["agent_positions"][eval_state["actor_id"]],
                eval_state["mode"],
            )

            for i in range(self.num_agents):
                encoded = self.encoder.encode(eval_state, i)
                if not self.exp_config.algorithm.centralized:
                    pred = float(self.critic_networks[eval_state["actor_id"]].get_value_function(encoded))
                else:
                    pred = float(self.critic_networks[0].get_value_function(encoded))

                true = float(mc_values[i]) if not self.exp_config.reward.reward_learning else float(rewards[i])
                err = true - pred
                errors_by_agent[i].append(err)
                ape_by_agent[i].append((abs(err) / abs(true)) * 100.0 if abs(true) > eps else abs(err))

                if (not self.exp_config.reward.reward_learning) and (ci_low is not None) and (ci_high is not None):
                    lo, hi = float(ci_low[i]), float(ci_high[i])
                    in_ci_by_agent[i].append(1.0 if (pred >= lo and pred <= hi) else 0.0)

        if store_last:
            self._last_eval_errors_by_agent = ape_by_agent

        careful_pred_this_eval = np.full((self.num_agents, len(self.careful_distances)), np.nan, dtype=float)

        for j, d in enumerate(self.careful_distances):
            item = self.careful_actor_states[j]
            if item is None:
                continue
            st = item["init_state"] if isinstance(item, dict) and "init_state" in item else item
            if st.get("actor_id", None) != 0:
                raise RuntimeError(f"Expected actor_id=0, got {st.get('actor_id')} at distance {d}")

            for eval_agent_id in range(self.num_agents):
                encoded = self.encoder.encode(st, eval_agent_id)
                if not self.exp_config.algorithm.centralized:
                    pred = float(self.critic_networks[eval_agent_id].get_value_function(encoded))
                else:
                    pred = float(self.critic_networks[0].get_value_function(encoded))
                careful_pred_this_eval[eval_agent_id, j] = pred

        if step is not None:
            self.careful_eval_steps.append(int(step))
            for eval_agent_id in range(self.num_agents):
                for j in range(len(self.careful_distances)):
                    self.careful_pred_history_actor0[eval_agent_id][j].append(
                        float(careful_pred_this_eval[eval_agent_id, j])
                    )

        return errors_by_agent

    def evaluate_networks_reward(self, *, step: int | None = None, plot: bool = False, store_last: bool = True):
        errors_by_agent = {i: [] for i in range(self.num_agents)}
        abs_errors = []
        correct_flags = []
        tol = 0.10

        if not isinstance(self.evaluation_states, list):
            self.evaluation_states = []

        with torch.no_grad():
            for eval_state in self.evaluation_states:
                true_rewards = eval_state.get("true_rewards")
                if true_rewards is None:
                    true_rewards = self.reward_module.get_reward(
                        eval_state,
                        eval_state["actor_id"],
                        eval_state["agent_positions"][eval_state["actor_id"]],
                        mode=1,
                    )
                true_rewards = np.asarray(true_rewards, dtype=np.float32)

                if self.exp_config.algorithm.centralized:
                    encoded = self.encoder.encode(eval_state, 0)
                    pred = float(self.critic_networks[0].get_value_function(encoded))
                    actor_id = int(eval_state["actor_id"])
                    true = float(sum(true_rewards))
                    err = true - pred
                    errors_by_agent[actor_id].append(err)
                    abs_errors.append(abs(err))
                    correct_flags.append(1.0 if abs(err) <= tol else 0.0)
                else:
                    for agent_id in range(self.num_agents):
                        encoded = self.encoder.encode(eval_state, agent_id)
                        pred = float(self.critic_networks[agent_id].get_value_function(encoded))
                        true = float(true_rewards[agent_id])
                        err = true - pred
                        errors_by_agent[agent_id].append(err)
                        abs_errors.append(abs(err))
                        correct_flags.append(1.0 if abs(err) <= tol else 0.0)

        reward_mae_mean = float(np.mean(abs_errors)) if abs_errors else None
        reward_acc_mean = float(np.mean(correct_flags)) if correct_flags else None

        if step is not None:
            self.main_logger.log(
                {
                    "step": int(step),
                    "current_lr": self.critic_networks[0].get_lr() if self.critic_networks else None,
                    "reward_mae_mean": reward_mae_mean,
                    "reward_acc_mean": reward_acc_mean,
                }
            )

        return errors_by_agent
