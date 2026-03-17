import numpy as np

from debug.code.env.environment import Orchard
from debug.code.training.helpers import env_step, nearest_apple_policy, random_policy


class LearningStateGenerationMixin:
    def _reward_eval_target_counts(self) -> tuple[int, int]:
        total = max(1, int(self.reward_eval_num_states))
        zero_target = int(round(total * float(self.reward_eval_zero_frac)))
        zero_target = min(max(zero_target, 0), total)
        reward_target = total - zero_target
        return zero_target, reward_target

    @staticmethod
    def _reward_eval_snapshot_source(
        curr_state: dict,
        s_moved: dict,
        pick_rewards,
    ) -> tuple[dict, str]:
        rewards = np.asarray(pick_rewards, dtype=np.float32)
        if np.allclose(rewards, 0.0):
            return curr_state, "curr_state"
        return s_moved, "s_moved"

    def _make_reward_eval_state(
        self,
        curr_state: dict,
        s_moved: dict,
        actor_id: int,
        pick_rewards,
    ) -> dict:
        rewards = np.asarray(pick_rewards, dtype=np.float32)
        snapshot_source, source_name = self._reward_eval_snapshot_source(
            curr_state,
            s_moved,
            rewards,
        )
        eval_state = self._snapshot_state(snapshot_source)
        eval_state["actor_id"] = int(actor_id)
        eval_state["true_rewards"] = rewards
        eval_state["reward_eval_source"] = source_name
        return eval_state

    def _generate_evaluation_states_supervised(self) -> None:
        p_apple = self.exp_config.algorithm.q_agent / float(self.width**2)
        d_apple = 1 / self.exp_config.env.apple_life
        burnin = max(100, self.num_agents * 10)

        eval_env = Orchard(
            self.length,
            self.width,
            self.num_agents,
            self.reward_module,
            p_apple=p_apple,
            d_apple=d_apple,
            max_apples=self.exp_config.env.max_apples,
        )
        eval_env.set_positions()

        actor_idx = 0
        curr_state = dict(eval_env.get_state())
        states = []
        total_steps = burnin + self.supervised_eval_num_states

        for t in range(total_steps):
            new_pos = random_policy(
                curr_state["agent_positions"][actor_idx],
                width=self.width,
                length=self.length,
            )
            s_moved, s_next, _, _, next_actor_idx = env_step(
                eval_env,
                actor_idx,
                new_pos,
                self.num_agents,
            )

            if t >= burnin:
                eval_state = self._snapshot_state(s_moved)
                eval_state["actor_id"] = actor_idx
                states.append(eval_state)

            curr_state = s_next
            actor_idx = next_actor_idx

        self.supervised_evaluation_states = states

    def _generate_evaluation_states_reward_learning(self) -> None:
        p_apple = self.exp_config.algorithm.q_agent / float(self.width**2)
        d_apple = 1 / self.exp_config.env.apple_life
        burnin = max(100, self.num_agents * 10)
        zero_target, reward_target = self._reward_eval_target_counts()

        eval_env = Orchard(
            self.length,
            self.width,
            self.num_agents,
            self.reward_module,
            p_apple=p_apple,
            d_apple=d_apple,
            max_apples=self.exp_config.env.max_apples,
        )
        eval_env.set_positions()

        actor_idx = 0
        curr_state = dict(eval_env.get_state())
        states = []
        zero_count = 0
        reward_count = 0
        max_steps = burnin + max(self.reward_eval_num_states * 500, 20_000)

        for t in range(max_steps):
            new_pos = nearest_apple_policy(
                curr_state["agent_positions"][actor_idx],
                curr_state["apples"],
            )
            s_moved, s_next, pick_rewards, _, next_actor_idx = env_step(
                eval_env, actor_idx, new_pos, self.num_agents
            )

            if t >= burnin:
                eval_state = self._make_reward_eval_state(
                    curr_state,
                    s_moved,
                    actor_idx,
                    pick_rewards,
                )
                if eval_state["reward_eval_source"] == "curr_state":
                    if zero_count < zero_target:
                        states.append(eval_state)
                        zero_count += 1
                elif reward_count < reward_target:
                    states.append(eval_state)
                    reward_count += 1

                if zero_count >= zero_target and reward_count >= reward_target:
                    break

            curr_state = s_next
            actor_idx = next_actor_idx
        else:
            raise RuntimeError(
                "Unable to generate the requested reward-evaluation state mix. "
                f"Collected zero_reward={zero_count}/{zero_target}, "
                f"nonzero_reward={reward_count}/{reward_target} after {max_steps} rollout steps."
            )

        self.evaluation_states = states
