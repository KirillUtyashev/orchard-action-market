import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from debug.code.core.enums import DISCOUNT_FACTOR, data_dir
from debug.code.env.environment import Orchard
from debug.code.training.helpers import env_step, random_policy
from debug.code.training.monte_carlo import generate_careful_distance_series, generate_initial_state_full


def _worker_generate_state(args):
    reward_module, run_id, seed, discount_factor, p_apple, d_apple, q_agent, tau, num_agents, width, length = args
    return generate_initial_state_full(
        reward_module=reward_module,
        run_id=run_id,
        seed=seed,
        discount_factor=discount_factor,
        p_apple=p_apple,
        d_apple=d_apple,
        q_agent=q_agent,
        tau=tau,
        num_agents=num_agents,
        width=width,
        length=length,
        save=True,
    )


def _worker_generate_careful(arg):
    reward_module, seed, discount_factor, p_apple, d_apple, agent_id, distance, num_agents, width, length = arg
    return generate_careful_distance_series(
        reward_module=reward_module,
        seed=seed,
        discount_factor=discount_factor,
        p_apple=p_apple,
        d_apple=d_apple,
        distances=(distance,),
        self_id=agent_id,
        num_agents=num_agents,
        width=width,
        length=length,
    )


def _state_path(picker_r: float, run_id: int) -> Path:
    return data_dir / "states" / "full" / f"init_state_reward_{picker_r}_{run_id}.npz"


def _careful_state_path(agent_id: int, seed: int, distance: int) -> Path:
    out_dir = data_dir / "states" / "careful"
    return out_dir / f"careful_agent{agent_id}_seed{seed}_d{distance}.npz"


def _load_state_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as f:
        return f["dict"].item()


class LearningStateGenerationMixin:
    def _generate_evaluation_states(self, p_apple, d_apple, sequential: bool = False, processes: int = 8):
        start = time.time()

        out_dir = data_dir / "states" / "full"
        out_dir.mkdir(parents=True, exist_ok=True)

        num = self.num_eval_states
        results = [None] * num
        missing_args = []
        missing_indices = []

        for i in range(num):
            path = _state_path(self.reward_module.picker_r, i)
            if path.exists():
                results[i] = _load_state_npz(path)
            else:
                missing_indices.append(i)
                missing_args.append(
                    (
                        self.reward_module,
                        i,
                        i,
                        DISCOUNT_FACTOR,
                        p_apple,
                        d_apple,
                        self.exp_config.algorithm.q_agent,
                        self.exp_config.env.apple_life,
                        self.num_agents,
                        self.width,
                        self.length,
                    )
                )

        if missing_args:
            if sequential:
                generated = [_worker_generate_state(arg) for arg in missing_args]
            else:
                with mp.Pool(processes=processes) as pool:
                    generated = pool.map(_worker_generate_state, missing_args)
            for idx, state in zip(missing_indices, generated):
                results[idx] = state

        self.evaluation_states = results

        careful_dir = data_dir / "states" / "careful"
        careful_dir.mkdir(parents=True, exist_ok=True)

        distances = self.careful_distances
        careful_seed = 42069
        self.careful_evals = [[] for _ in range(self.num_agents)]
        missing_args, missing_keys = [], []

        for agent_id in range(self.num_agents):
            for d in distances:
                path = _careful_state_path(agent_id, careful_seed, d)
                if path.exists():
                    self.careful_evals[agent_id].append(_load_state_npz(path))
                else:
                    self.careful_evals[agent_id].append(None)
                    missing_keys.append((agent_id, d, path))
                    missing_args.append(
                        (
                            self.reward_module,
                            careful_seed,
                            DISCOUNT_FACTOR,
                            p_apple,
                            d_apple,
                            agent_id,
                            d,
                            self.num_agents,
                            self.width,
                            self.length,
                        )
                    )

        if missing_args:
            if sequential:
                generated = [_worker_generate_careful(arg) for arg in missing_args]
            else:
                with mp.Pool(processes=processes) as pool:
                    generated = pool.map(_worker_generate_careful, missing_args)
            for (agent_id, d, _), item in zip(missing_keys, generated):
                j = distances.index(d)
                self.careful_evals[agent_id][j] = item

        self.careful_actor_states = self.careful_evals[0]
        print(f"Generated/loaded {num} eval states in {time.time() - start:.3f}s")

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
