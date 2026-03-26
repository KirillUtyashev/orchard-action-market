import hashlib
import time

import numpy as np
import torch

from debug.code.agents.actor_critic_agent import ACAgentRates
from debug.code.env.environment import MoveAction, Orchard
from debug.code.training.helpers import env_step, random_policy, set_all_seeds
from debug.code.core.log import CSVLogger, build_weight_sample_csv_fieldnames

ACTION_NAMES = ("LEFT", "RIGHT", "UP", "DOWN", "STAY")
DELTA_TO_ACTION = {tuple(a.vector.tolist()): a.name for a in MoveAction}


class LearningDiagnosticsMixin:
    def _following_rate_agents(self) -> list[ACAgentRates]:
        return [agent for agent in self.agents if isinstance(agent, ACAgentRates)]

    def _effective_follow_weight(self, observer: ACAgentRates, target_id: int) -> float:
        influencer = getattr(self, "external_influencer", None)
        return float(observer.get_effective_observing_probability(target_id, influencer))

    def _compute_following_rate_stats(self) -> dict[str, float | None]:
        rate_agents = self._following_rate_agents()
        if not rate_agents:
            return {
                "alpha_mean": None,
                "alpha_positive_frac": None,
                "following_weight_mean": None,
                "active_follow_edges_mean": None,
                "beta_mean": None,
                "influencer_weight_mean": None,
                "follower_to_influencer_weight_mean": None,
                "effective_follow_weight_mean": None,
            }

        alpha_rows = np.stack([np.asarray(agent.agent_alphas, dtype=float) for agent in rate_agents], axis=0)
        weight_rows = np.stack(
            [np.asarray(agent.agent_observing_probabilities, dtype=float) for agent in rate_agents],
            axis=0,
        )
        flat_alphas = []
        flat_weights = []
        for observer_id in range(alpha_rows.shape[0]):
            mask = np.ones(self.num_agents, dtype=bool)
            mask[observer_id] = False
            flat_alphas.append(alpha_rows[observer_id, mask])
            flat_weights.append(weight_rows[observer_id, mask])
        flat_alphas = np.concatenate(flat_alphas) if flat_alphas else np.array([], dtype=float)
        flat_weights = np.concatenate(flat_weights) if flat_weights else np.array([], dtype=float)
        flat_effective_weights = []
        for agent in rate_agents:
            for target_id in range(self.num_agents):
                if target_id == agent.id:
                    continue
                flat_effective_weights.append(self._effective_follow_weight(agent, target_id))
        flat_effective_weights = np.asarray(flat_effective_weights, dtype=float)
        follower_to_influencer_weights = np.asarray(
            [agent.influencer_observing_probability for agent in rate_agents],
            dtype=float,
        )
        influencer = getattr(self, "external_influencer", None)
        influencer_weights = (
            np.asarray(influencer.outgoing_weights, dtype=float) if influencer is not None else np.array([], dtype=float)
        )
        influencer_beta = np.asarray(influencer.beta, dtype=float) if influencer is not None else np.array([], dtype=float)

        # Count active off-diagonal edges per observer.
        per_observer_active = []
        for observer_id, row in enumerate(weight_rows):
            mask = np.ones(self.num_agents, dtype=bool)
            mask[observer_id] = False
            per_observer_active.append(float(np.count_nonzero(row[mask] > 1e-12)))

        return {
            "alpha_mean": float(np.mean(flat_alphas)) if flat_alphas.size else None,
            "alpha_positive_frac": float(np.mean(flat_alphas > 0.0)) if flat_alphas.size else None,
            "following_weight_mean": float(np.mean(flat_weights)) if flat_weights.size else None,
            "active_follow_edges_mean": float(np.mean(per_observer_active)) if per_observer_active else None,
            "beta_mean": float(np.mean(influencer_beta)) if influencer_beta.size else None,
            "influencer_weight_mean": float(np.mean(influencer_weights)) if influencer_weights.size else None,
            "follower_to_influencer_weight_mean": (
                float(np.mean(follower_to_influencer_weights)) if follower_to_influencer_weights.size else None
            ),
            "effective_follow_weight_mean": (
                float(np.mean(flat_effective_weights)) if flat_effective_weights.size else None
            ),
        }

    def _refresh_following_rate_stats(self) -> None:
        self._last_following_rate_stats = self._compute_following_rate_stats()

    def _stable_tensor_seed(self, agent_id: int, tensor_name: str) -> int:
        digest = hashlib.blake2b(tensor_name.encode("utf-8"), digest_size=8).digest()
        name_hash = int.from_bytes(digest, byteorder="little", signed=False)
        base_seed = int(self.exp_config.train.seed)
        return (base_seed + (agent_id + 1) * 1_000_003 + name_hash) % (2**63 - 1)

    @staticmethod
    def _sample_stratified_indices(numel: int, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        if numel <= 0:
            return np.array([], dtype=np.int64)
        k = min(num_samples, numel)
        if k == numel:
            return np.arange(numel, dtype=np.int64)
        out = np.empty(k, dtype=np.int64)
        for i in range(k):
            start = (i * numel) // k
            end = ((i + 1) * numel) // k
            if end <= start:
                out[i] = start
            else:
                out[i] = int(rng.integers(start, end))
        return out

    def _init_weight_sample_indices(self) -> None:
        self.weight_sample_indices = {}
        for logger in self.weight_sample_loggers.values():
            logger.close()
        self.weight_sample_loggers = {}
        if not self.weight_samples_enabled:
            return

        weight_fields = build_weight_sample_csv_fieldnames()
        for agent_id, network in enumerate(self.critic_networks):
            self.weight_sample_loggers[agent_id] = CSVLogger(
                self.data_dir / f"weight_samples_network_{agent_id}.csv",
                weight_fields,
            )
            tensor_samples: dict[str, np.ndarray] = {}
            for tensor_name, param in network.model.named_parameters():
                if not tensor_name.endswith("weight") or param.dim() <= 1:
                    continue
                numel = int(param.numel())
                if numel <= 0:
                    continue
                rng = np.random.default_rng(self._stable_tensor_seed(agent_id, tensor_name))
                idx = self._sample_stratified_indices(numel, self.weight_samples_per_tensor, rng)
                if idx.size > 0:
                    tensor_samples[tensor_name] = idx
            self.weight_sample_indices[agent_id] = tensor_samples

    def _log_weight_samples(self, step: int) -> None:
        if not self.weight_samples_enabled:
            return
        wall_time = round(float(time.time() - self.train_start_time), 3) if self.train_start_time else 0.0
        for agent_id, network in enumerate(self.critic_networks):
            logger = self.weight_sample_loggers.get(agent_id)
            if logger is None:
                continue
            param_dict = dict(network.model.named_parameters())
            for tensor_name, sample_indices in self.weight_sample_indices.get(agent_id, {}).items():
                param = param_dict.get(tensor_name)
                if param is None:
                    continue
                flat = param.detach().flatten()
                for sample_id, flat_idx in enumerate(sample_indices):
                    idx = int(flat_idx)
                    logger.log(
                        {
                            "step": int(step),
                            "wall_time": wall_time,
                            "tensor_name": tensor_name,
                            "sample_id": int(sample_id),
                            "flat_index": idx,
                            "value": float(flat[idx].item()),
                        }
                    )

    def _maybe_log_weight_samples(self, step: int) -> None:
        if not self.weight_samples_enabled:
            return
        freq = self.weight_samples_freq if self.weight_samples_freq > 0 else self.exp_config.logging.main_csv_freq
        if step == 0:
            self._log_weight_samples(step)
            return
        if freq > 0 and step % freq == 0:
            self._log_weight_samples(step)

    @staticmethod
    def _snapshot_state(state: dict) -> dict:
        return {
            "agents": state["agents"].copy(),
            "apples": state["apples"].copy(),
            "agent_positions": state["agent_positions"].copy(),
        }

    @staticmethod
    def _decode_action(old_pos: np.ndarray, new_pos: np.ndarray) -> str:
        dr = int(new_pos[0] - old_pos[0])
        dc = int(new_pos[1] - old_pos[1])
        return DELTA_TO_ACTION.get((dr, dc), "STAY")

    def _build_env_from_state(self, state: dict) -> Orchard:
        return Orchard(
            self.length,
            self.width,
            self.num_agents,
            self.reward_module,
            p_apple=self.env.p_apple,
            d_apple=self.env.d_apple,
            max_apples=self.env.max_apples,
            start_agents_map=state["agents"],
            start_apples_map=state["apples"],
            start_agent_positions=state["agent_positions"],
        )

    def _preferred_cells(self) -> list[tuple[int, int]]:
        cells = []
        for r in range(1, self.width - 1):
            for c in range(1, self.length - 1):
                cells.append((r, c))
        for r in range(self.width):
            for c in range(self.length):
                if r in {0, self.width - 1} or c in {0, self.length - 1}:
                    cells.append((r, c))
        return cells

    def _build_tracking_state(
        self,
        actor_id: int,
        actor_pos: tuple[int, int],
        apple_positions: list[tuple[int, int]],
        other_anchor_positions: list[tuple[int, int]] | None = None,
    ) -> dict:
        if other_anchor_positions is None:
            other_anchor_positions = []

        actor_pos = (int(actor_pos[0]), int(actor_pos[1]))
        agent_positions = np.full((self.num_agents, 2), -1, dtype=int)
        agent_positions[actor_id] = np.array(actor_pos, dtype=int)
        used_positions = {actor_pos}

        other_ids = [i for i in range(self.num_agents) if i != actor_id]
        for other_id, pos in zip(other_ids, other_anchor_positions):
            pos_t = (int(pos[0]), int(pos[1]))
            if pos_t in used_positions:
                continue
            agent_positions[other_id] = np.array(pos_t, dtype=int)
            used_positions.add(pos_t)

        fill_candidates = self._preferred_cells()
        for other_id in other_ids:
            if agent_positions[other_id, 0] != -1:
                continue
            for pos in fill_candidates:
                if pos not in used_positions:
                    agent_positions[other_id] = np.array(pos, dtype=int)
                    used_positions.add(pos)
                    break

        if np.any(agent_positions < 0):
            raise RuntimeError("Could not assign valid positions for all agents in value-tracking state.")

        agents = np.zeros((self.width, self.length), dtype=int)
        for r, c in agent_positions:
            agents[r, c] += 1

        apples = np.zeros((self.width, self.length), dtype=int)
        for r, c in apple_positions:
            if 0 <= r < self.width and 0 <= c < self.length:
                apples[r, c] = 1

        return {
            "agents": agents,
            "apples": apples,
            "agent_positions": agent_positions,
            "actor_id": actor_id,
        }

    def _dense_apple_positions(self, excluded_positions: set[tuple[int, int]]) -> list[tuple[int, int]]:
        cap = int(max(1, self.exp_config.env.max_apples))
        out = []
        for pos in self._preferred_cells():
            if pos in excluded_positions:
                continue
            out.append(pos)
            if len(out) >= cap:
                break
        return out

    def _generate_value_tracking_states_for_agent(self, agent_id: int) -> list[dict]:
        mid_r = max(1, min(self.width - 2, self.width // 2))
        mid_c = max(1, min(self.length - 2, self.length // 2))
        actor_mid = (mid_r, mid_c)
        right_1 = (mid_r, min(self.length - 1, mid_c + 1))
        right_2 = (mid_r, min(self.length - 1, mid_c + 2))
        up_1 = (max(0, mid_r - 1), mid_c)
        left_1 = (mid_r, max(0, mid_c - 1))
        corner = (0, 0)
        corner_right = (0, 1 if self.length > 1 else 0)
        edge_top = (0, mid_c)
        far_bottom = (self.width - 1, mid_c)
        other_far = (max(0, self.width - 2), max(0, self.length - 2))

        states = [
            self._build_tracking_state(agent_id, actor_mid, [], [other_far]),
            self._build_tracking_state(agent_id, actor_mid, [actor_mid], [other_far]),
            self._build_tracking_state(agent_id, actor_mid, [other_far], [other_far]),
            self._build_tracking_state(agent_id, actor_mid, [right_1], [other_far]),
            self._build_tracking_state(agent_id, (1, 1), [(self.width - 1, self.length - 1)], [other_far]),
            self._build_tracking_state(agent_id, actor_mid, [right_1], [right_2]),
            self._build_tracking_state(agent_id, actor_mid, [up_1, left_1], [other_far]),
            self._build_tracking_state(agent_id, corner, [corner_right], [other_far]),
            self._build_tracking_state(agent_id, edge_top, [far_bottom], [other_far]),
            self._build_tracking_state(agent_id, actor_mid, [], [other_far]),
        ]

        dense_excluded = {tuple(states[-1]["agent_positions"][i]) for i in range(self.num_agents)}
        states[-1]["apples"][:, :] = 0
        for r, c in self._dense_apple_positions(dense_excluded):
            states[-1]["apples"][r, c] = 1

        return states[: self.value_track_num_states]

    def _ensure_value_track_states(self) -> None:
        if self.value_track_states_by_agent is not None:
            return
        if self.value_track_num_states <= 0:
            self.value_track_states_by_agent = {agent_id: [] for agent_id in range(self.num_agents)}
            return
        self.value_track_states_by_agent = {
            agent_id: self._generate_value_tracking_states_for_agent(agent_id) for agent_id in range(self.num_agents)
        }

    def _predict_state_value(self, state: dict, agent_id: int) -> float:
        if self.exp_config.algorithm.centralized:
            encoded = self.encoder.encode(state, 0)
            return float(self.critic_networks[0].get_value_function(encoded))
        encoded = self.encoder.encode(state, agent_id)
        return float(self.critic_networks[agent_id].get_value_function(encoded))

    def evaluate_tracked_state_values(self, step: int) -> None:
        if not self.value_track_loggers:
            return
        if self.value_track_num_states <= 0 or self.encoder is None or not self.critic_networks:
            return

        self._ensure_value_track_states()
        wall_time = round(float(time.time() - self.train_start_time), 3) if self.train_start_time else 0.0

        with torch.no_grad():
            for agent_id in range(self.num_agents):
                logger = self.value_track_loggers.get(agent_id)
                if logger is None:
                    continue
                states = self.value_track_states_by_agent.get(agent_id, [])
                row = {"step": int(step), "wall_time": wall_time}
                for idx in range(self.value_track_num_states):
                    if idx < len(states):
                        row[f"state_{idx}"] = self._predict_state_value(states[idx], agent_id)
                    else:
                        row[f"state_{idx}"] = float("nan")
                logger.log(row)

    def _maybe_log_tracked_state_values(self, step: int) -> None:
        if self.exp_config.reward.reward_learning:
            return
        self.evaluate_tracked_state_values(step=step)

    def _log_following_rate_snapshots(self, step: int) -> None:
        if not getattr(self, "following_rate_loggers", None):
            return
        wall_time = round(float(time.time() - self.train_start_time), 3) if self.train_start_time else 0.0
        for observer_id, agent in enumerate(self.agents):
            if not isinstance(agent, ACAgentRates):
                continue
            logger = self.following_rate_loggers.get(observer_id)
            if logger is None:
                continue
            row = {"step": int(step), "wall_time": wall_time}
            alphas = np.asarray(agent.agent_alphas, dtype=float)
            rates = np.asarray(agent.following_rates, dtype=float)
            weights = np.asarray(agent.agent_observing_probabilities, dtype=float)
            for target_id in range(self.num_agents):
                row[f"alpha_to_{target_id}"] = float(alphas[target_id])
            for target_id in range(self.num_agents):
                row[f"lambda_to_{target_id}"] = float(rates[target_id])
            for target_id in range(self.num_agents):
                row[f"weight_to_{target_id}"] = float(weights[target_id])
            row["lambda_to_influencer"] = float(agent.following_rate_to_influencer)
            row["weight_to_influencer"] = float(agent.influencer_observing_probability)
            row["influencer_value"] = float(agent.influencer_value)
            logger.log(row)

    def _log_influencer_snapshot(self, step: int) -> None:
        logger = getattr(self, "influencer_logger", None)
        influencer = getattr(self, "external_influencer", None)
        if logger is None or influencer is None:
            return
        wall_time = round(float(time.time() - self.train_start_time), 3) if self.train_start_time else 0.0
        row = {"step": int(step), "wall_time": wall_time}
        beta = np.asarray(influencer.beta, dtype=float)
        rates = np.asarray(influencer.outgoing_rates, dtype=float)
        weights = np.asarray(influencer.outgoing_weights, dtype=float)
        for target_id in range(self.num_agents):
            row[f"beta_to_actor_{target_id}"] = float(beta[target_id])
        for target_id in range(self.num_agents):
            row[f"lambda_to_actor_{target_id}"] = float(rates[target_id])
        for target_id in range(self.num_agents):
            row[f"weight_to_actor_{target_id}"] = float(weights[target_id])
        logger.log(row)

    def _maybe_log_following_rate_snapshots(self, step: int) -> None:
        if self.exp_config.reward.reward_learning:
            return
        self._log_following_rate_snapshots(step=step)
        self._log_influencer_snapshot(step=step)

    def _sample_action_probability_states(self, num_states: int) -> list[dict]:
        if num_states <= 0:
            return []

        sample_env = Orchard(
            self.length,
            self.width,
            self.num_agents,
            self.reward_module,
            p_apple=self.env.p_apple,
            d_apple=self.env.d_apple,
            max_apples=self.env.max_apples,
        )
        sample_env.set_positions()

        actor_idx = 0
        curr_state = dict(sample_env.get_state())
        states = []
        burnin = self.action_prob_burnin
        stride = self.action_prob_stride
        total_steps = burnin + stride * num_states

        for t in range(total_steps):
            new_pos = random_policy(
                curr_state["agent_positions"][actor_idx],
                width=self.width,
                length=self.length,
            )
            _, s_next, _, _, actor_idx = env_step(sample_env, actor_idx, new_pos, self.num_agents)
            curr_state = s_next
            if t >= burnin and ((t - burnin) % stride == 0):
                states.append(self._snapshot_state(curr_state))
                if len(states) >= num_states:
                    break

        return states

    def _ensure_action_probability_states(self) -> None:
        if self.action_prob_eval_states is not None:
            return
        if self.action_prob_num_states <= 0 or self.env is None:
            self.action_prob_eval_states = []
            return

        self.save_rng_state()
        set_all_seeds(self.action_prob_seed)
        self.action_prob_eval_states = self._sample_action_probability_states(self.action_prob_num_states)
        self.restore_rng_state()

    def evaluate_action_probabilities(self, step: int) -> None:
        if not self.action_prob_loggers:
            return
        if self.action_prob_num_states <= 0 or self.agent_controller is None or self.env is None:
            return

        self._ensure_action_probability_states()
        sampled_states = self.action_prob_eval_states
        if not sampled_states:
            return

        if bool(getattr(self.exp_config.algorithm, "actor_critic", False)):
            summed_probs_by_agent = {agent_id: np.zeros(len(MoveAction), dtype=np.float64) for agent_id in range(self.num_agents)}
            with torch.no_grad():
                for state in sampled_states:
                    for agent_id in range(self.num_agents):
                        probs = np.asarray(
                            self.agent_controller.get_action_probabilities_from_state(state, agent_id),
                            dtype=np.float64,
                        )
                        summed_probs_by_agent[agent_id] += probs

            wall_time = round(float(time.time() - self.train_start_time), 3) if self.train_start_time else 0.0
            denom = float(len(sampled_states))
            for agent_id in range(self.num_agents):
                probs = summed_probs_by_agent[agent_id] / denom
                self.action_prob_loggers[agent_id].log(
                    {
                        "step": int(step),
                        "wall_time": wall_time,
                        "left": probs[MoveAction.LEFT.idx],
                        "right": probs[MoveAction.RIGHT.idx],
                        "up": probs[MoveAction.UP.idx],
                        "down": probs[MoveAction.DOWN.idx],
                        "stay": probs[MoveAction.STAY.idx],
                    }
                )
            return

        counts_by_agent = {agent_id: {name: 0 for name in ACTION_NAMES} for agent_id in range(self.num_agents)}
        decisions_by_agent = {agent_id: 0 for agent_id in range(self.num_agents)}

        with torch.no_grad():
            for state in sampled_states:
                for agent_id in range(self.num_agents):
                    env_for_eval = self._build_env_from_state(state)
                    old_pos = env_for_eval.agent_positions[agent_id].copy()
                    new_pos = self.agent_controller.agent_get_action(env_for_eval, agent_id, epsilon=0.0)
                    action = self._decode_action(old_pos, new_pos)
                    counts_by_agent[agent_id][action] += 1
                    decisions_by_agent[agent_id] += 1

        wall_time = round(float(time.time() - self.train_start_time), 3) if self.train_start_time else 0.0
        for agent_id in range(self.num_agents):
            decisions = decisions_by_agent[agent_id]
            if decisions == 0:
                continue
            counts = counts_by_agent[agent_id]
            self.action_prob_loggers[agent_id].log(
                {
                    "step": int(step),
                    "wall_time": wall_time,
                    "left": counts["LEFT"] / decisions,
                    "right": counts["RIGHT"] / decisions,
                    "up": counts["UP"] / decisions,
                    "down": counts["DOWN"] / decisions,
                    "stay": counts["STAY"] / decisions,
                }
            )

    def _maybe_log_action_probabilities(self, step: int) -> None:
        if self.exp_config.reward.reward_learning:
            return
        self.evaluate_action_probabilities(step=step)

    def _save_greedy_eval_positions(self, *, step: int, positions: np.ndarray) -> None:
        arr = np.asarray(positions, dtype=np.int16)
        if arr.ndim != 3 or arr.shape[1] != self.num_agents or arr.shape[2] != 2:
            raise ValueError(f"Unexpected greedy position shape: {arr.shape}, expected (T, {self.num_agents}, 2)")

        out_path = self.agent_positions_dir / f"greedy_eval_step_{int(step):09d}.npz"
        np.savez_compressed(out_path, positions=arr)
        self.last_greedy_positions_path = out_path
        self.last_greedy_eval_step = int(step)

    def _write_last_greedy_position_heatmaps(self) -> None:
        if self.last_greedy_positions_path is None or not self.last_greedy_positions_path.exists():
            return

        with np.load(self.last_greedy_positions_path) as data:
            positions = np.asarray(data["positions"], dtype=np.int64)

        if positions.ndim != 3 or positions.shape[1] != self.num_agents or positions.shape[2] != 2:
            return

        import matplotlib.pyplot as plt

        for agent_id in range(self.num_agents):
            visits = np.zeros((self.width, self.length), dtype=np.int64)
            for r, c in positions[:, agent_id]:
                if 0 <= r < self.width and 0 <= c < self.length:
                    visits[r, c] += 1

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(visits, cmap="hot", origin="upper")
            step_label = self.last_greedy_eval_step if self.last_greedy_eval_step is not None else "unknown"
            ax.set_title(f"Agent {agent_id} greedy-eval visit heatmap (step {step_label})")
            ax.set_xlabel("col")
            ax.set_ylabel("row")
            fig.colorbar(im, ax=ax, label="visits")
            fig.tight_layout()
            fig.savefig(self.agent_positions_dir / f"agent_{agent_id}_heatmap.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
