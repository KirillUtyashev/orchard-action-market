import numpy as np

from debug.code.env.environment import Orchard
from debug.code.training.helpers import env_step, random_policy


class LearningStateGenerationMixin:
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
        total_steps = burnin + self.reward_eval_num_states

        for t in range(total_steps):
            new_pos = random_policy(
                curr_state["agent_positions"][actor_idx],
                width=self.width,
                length=self.length,
            )
            s_moved, s_next, pick_rewards, _, next_actor_idx = env_step(
                eval_env, actor_idx, new_pos, self.num_agents
            )

            if t >= burnin:
                eval_state = self._snapshot_state(s_moved)
                eval_state["actor_id"] = actor_idx
                eval_state["true_rewards"] = np.asarray(pick_rewards, dtype=np.float32)
                states.append(eval_state)

            curr_state = s_next
            actor_idx = next_actor_idx

        self.evaluation_states = states
