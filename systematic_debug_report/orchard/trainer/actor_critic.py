"""Actor-critic trainers for orchard."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch

import orchard.encoding as encoding
from orchard.actor_critic import (
    PolicyNetwork,
    build_phase1_legal_mask,
    build_phase1_policy_prob_row,
    build_phase2_legal_mask,
    build_phase2_policy_prob_row,
    generate_phase2_policy_eval_states,
    policy_index_to_action,
    sample_phase1_policy_eval_states,
)
from orchard.batched_training import BatchedTrainer
from orchard.datatypes import (
    EncoderOutput,
    EvalConfig,
    FollowingRatesConfig,
    InfluencerConfig,
    ScheduleConfig,
    State,
)
from orchard.enums import Action, Heuristic, PickMode
from orchard.env.base import BaseEnv
from orchard.eval import evaluate_policy_metrics
from orchard.following_rates import (
    FollowingRateAgentState,
    build_following_rate_snapshot_row,
    compute_following_rate_stats,
    initial_following_rate_to_influencer,
    initial_following_rate_vector,
)
from orchard.influencer import (
    ExternalInfluencer,
    build_influencer_snapshot_row,
    initial_influencer_outgoing_rates,
)
from orchard.logging_ import (
    CSVLogger,
    build_following_rate_csv_fieldnames,
    build_influencer_csv_fieldnames,
    build_phase1_policy_prob_csv_fieldnames,
    build_phase2_policy_prob_csv_fieldnames,
)
from orchard.model import ValueNetwork
from orchard.policy import heuristic_action
from orchard.schedule import compute_schedule_value
from orchard.trainer.base import TrainerBase
from orchard.trainer.timer import Timer, TimerSection


def _move_encoder_output_to_device(enc: EncoderOutput, device: torch.device) -> EncoderOutput:
    scalar = enc.scalar.to(device) if enc.scalar is not None else None
    grid = enc.grid.to(device) if enc.grid is not None else None
    return EncoderOutput(scalar=scalar, grid=grid)


def _sanitize_metric(value: float | int | None) -> float | int | str:
    if value is None:
        return ""
    return value


class ActorCriticTrainerBase(TrainerBase):
    """Shared orchard-native actor-critic runtime."""

    def __init__(
        self,
        critic_networks: list[ValueNetwork],
        actor_networks: list[PolicyNetwork],
        env: BaseEnv,
        gamma: float,
        critic_lr_schedule: ScheduleConfig,
        actor_lr_schedule: ScheduleConfig,
        total_steps: int,
        heuristic: Heuristic,
        freeze_critic: bool,
        following_rates_cfg: FollowingRatesConfig,
        influencer_cfg: InfluencerConfig,
        timer: Timer | None = None,
    ) -> None:
        self._critic_networks_list = critic_networks
        self._actor_networks_list = actor_networks
        self._env = env
        self._gamma = float(gamma)
        self._critic_lr_schedule = critic_lr_schedule
        self._actor_lr_schedule = actor_lr_schedule
        self._total_steps = int(total_steps)
        self._heuristic = heuristic
        self._freeze_critic = bool(freeze_critic)
        self._timer = timer or Timer()
        self._n_agents = env.cfg.n_agents
        self._decision_count = 0
        self._zero_rewards = tuple(0.0 for _ in range(self._n_agents))
        self._critic_prev_after: Any = None

        self._td_loss_accum = 0.0
        self._td_loss_count = 0
        self._actor_loss_accum = 0.0
        self._actor_loss_count = 0
        self._advantage_accum = 0.0
        self._advantage_count = 0
        self._entropy_accum = 0.0
        self._entropy_count = 0

        self._following_rates_cfg = following_rates_cfg
        self._influencer_cfg = influencer_cfg
        self._following_states: list[FollowingRateAgentState] = []
        self._influencer: ExternalInfluencer | None = None
        if self._following_rates_cfg.enabled:
            self._following_states = [
                FollowingRateAgentState(
                    agent_id=agent_id,
                    agent_alphas=np.zeros(self._n_agents, dtype=float),
                    budget=self._following_rates_cfg.budget,
                    following_rates=initial_following_rate_vector(
                        self._n_agents,
                        agent_id,
                        self._following_rates_cfg.budget,
                        influencer_enabled=self._influencer_cfg.enabled,
                    ),
                    following_rate_to_influencer=initial_following_rate_to_influencer(
                        self._n_agents,
                        self._following_rates_cfg.budget,
                        influencer_enabled=self._influencer_cfg.enabled,
                    ),
                    rate_solver_name=self._following_rates_cfg.solver,
                    influencer_value=0.0,
                )
                for agent_id in range(self._n_agents)
            ]
        if self._influencer_cfg.enabled:
            self._influencer = ExternalInfluencer(
                budget=self._influencer_cfg.budget,
                num_agents=self._n_agents,
                init_outgoing_rates=initial_influencer_outgoing_rates(
                    self._n_agents,
                    self._influencer_cfg.budget,
                    influencer_enabled=True,
                ),
                rate_solver_name=self._following_rates_cfg.solver,
            )
        self._refresh_influencer_beta()
        self._refresh_follower_influencer_values()

        self._phase1_logger: CSVLogger | None = None
        self._phase2_logger: CSVLogger | None = None
        self._following_loggers: dict[int, CSVLogger] = {}
        self._influencer_logger: CSVLogger | None = None
        self._phase1_eval_states: list[State] | None = None
        self._phase2_eval_states: list[tuple[str, State]] | None = None

    # ------------------------------------------------------------------
    # Abstract critic hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def _encode_all_critics(self, state: State) -> Any:
        ...

    @abstractmethod
    def _critic_values(self, encoded_state: Any) -> list[float]:
        ...

    @abstractmethod
    def _critic_td_step(
        self,
        prev: Any,
        rewards: tuple[float, ...],
        discount: float,
        current: Any,
        t: int,
    ) -> float:
        ...

    # ------------------------------------------------------------------
    # Turn stepping
    # ------------------------------------------------------------------
    def step(self, state: State, t: int) -> State:
        move_action, _, move_mask = self._sample_action(state, phase2=False)
        s_moved = self._env.apply_action(state, move_action)
        actor = state.actor

        if self._env.cfg.pick_mode == PickMode.CHOICE and s_moved.is_agent_on_task(actor):
            pick_state = s_moved.with_pick_phase()
            self._train_decision(
                state=state,
                action=move_action,
                legal_mask=move_mask,
                discount=self._gamma,
                t=t,
            )

            pick_action, _, pick_mask = self._sample_action(pick_state, phase2=True)
            s_picked, pick_rewards = self._env.resolve_pick(
                s_moved,
                pick_type=pick_action.pick_type() if pick_action.is_pick() else None,
            )
            self._train_decision(
                state=pick_state,
                action=pick_action,
                legal_mask=pick_mask,
                discount=1.0,
                t=t,
            )
            next_state = self._env.advance_actor(self._env.spawn_and_despawn(s_picked))
            return next_state

        if s_moved.is_agent_on_task(actor):
            self._train_decision(
                state=state,
                action=move_action,
                legal_mask=move_mask,
                discount=self._gamma,
                t=t,
            )
            s_picked, rewards = self._env.resolve_pick(s_moved)
            self._train_critic_after_transition(s_picked, rewards, 1.0, t)
        else:
            self._train_decision(
                state=state,
                action=move_action,
                legal_mask=move_mask,
                discount=self._gamma,
                t=t,
            )
            s_picked = s_moved
        next_state = self._env.advance_actor(self._env.spawn_and_despawn(s_picked))
        return next_state

    # ------------------------------------------------------------------
    # Actor policy
    # ------------------------------------------------------------------
    def _actor_device(self, actor_id: int) -> torch.device:
        return next(self._actor_networks_list[actor_id].parameters()).device

    def _encode_actor_state(self, state: State, actor_id: int) -> EncoderOutput:
        enc = encoding.encode(state, actor_id)
        return _move_encoder_output_to_device(enc, self._actor_device(actor_id))

    def _legal_mask(self, state: State, phase2: bool) -> np.ndarray:
        if phase2:
            return build_phase2_legal_mask(state, self._env.cfg)
        return build_phase1_legal_mask(state, self._env.cfg)

    def _actor_probabilities(self, state: State, phase2: bool) -> np.ndarray:
        actor_id = state.actor
        actor_state = self._encode_actor_state(state, actor_id)
        legal_mask = self._legal_mask(state, phase2)
        return self._actor_networks_list[actor_id].get_action_probabilities(actor_state, legal_mask)

    def _sample_action(self, state: State, phase2: bool) -> tuple[Action, np.ndarray, np.ndarray]:
        actor_id = state.actor
        actor_state = self._encode_actor_state(state, actor_id)
        legal_mask = self._legal_mask(state, phase2)
        action, probs = self._actor_networks_list[actor_id].sample_action(actor_state, legal_mask)
        return action, probs, legal_mask

    def _greedy_action(self, state: State, phase2: bool) -> Action:
        probs = self._actor_probabilities(state, phase2)
        return policy_index_to_action(int(np.argmax(probs)))

    # ------------------------------------------------------------------
    # Following-rate state
    # ------------------------------------------------------------------
    def _refresh_influencer_beta(self) -> None:
        if self._influencer is None:
            return
        self._influencer.recompute_beta(self._following_states)

    def _refresh_follower_influencer_values(self) -> None:
        if not self._following_states:
            return
        if self._influencer is None:
            for state in self._following_states:
                state.set_influencer_value(0.0)
            return
        outgoing_weights = self._influencer.outgoing_weights
        for state in self._following_states:
            state.set_influencer_value(float((outgoing_weights * state.agent_alphas).sum()))

    def _should_reallocate_following_rates(self) -> bool:
        if not self._following_rates_cfg.enabled or self._following_rates_cfg.fixed:
            return False
        freq = max(1, int(self._following_rates_cfg.reallocation_freq))
        return self._decision_count % freq == 0

    def _reallocate_following_rates(self) -> None:
        if not self._following_states:
            return
        if self._influencer is not None:
            self._refresh_influencer_beta()
            self._influencer.update_outgoing_rates()
            self._refresh_follower_influencer_values()
            for state in self._following_states:
                state.update_following_rates(influencer_value=state.influencer_value)
            self._refresh_influencer_beta()
            self._refresh_follower_influencer_values()
            return
        for state in self._following_states:
            state.update_following_rates()

    def _effective_observer_weight(self, observer_id: int, actor_id: int) -> float:
        observer_state = self._following_states[observer_id]
        return float(observer_state.get_effective_observing_probability(actor_id, self._influencer))

    def _current_following_rate_stats(self) -> dict[str, float | None]:
        if not self._following_states:
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
        return compute_following_rate_stats(self._following_states, self._influencer)

    # ------------------------------------------------------------------
    # After-state action valuation
    # ------------------------------------------------------------------
    @staticmethod
    def _clear_pick_phase(state: State) -> State:
        if not state.pick_phase:
            return state
        return State(
            agent_positions=state.agent_positions,
            task_positions=state.task_positions,
            actor=state.actor,
            task_types=state.task_types,
            pick_phase=False,
        )

    def _after_state_for_action(
        self,
        state: State,
        action: Action,
        phase2: bool,
    ) -> tuple[State, tuple[float, ...]]:
        if phase2:
            base_state = self._clear_pick_phase(state)
            after_state, rewards = self._env.resolve_pick(
                base_state,
                pick_type=action.pick_type() if action.is_pick() else None,
            )
            return after_state, rewards

        s_moved = self._env.apply_action(state, action)
        if s_moved.is_agent_on_task(s_moved.actor):
            return s_moved.with_pick_phase(), self._zero_rewards
        return s_moved, self._zero_rewards

    def _action_objective(
        self,
        rewards: tuple[float, ...],
        after_values: list[float],
        actor_id: int,
        discount: float,
    ) -> float:
        if self._following_states:
            total = float(rewards[actor_id]) + float(discount) * float(after_values[actor_id])
            for observer_id in range(self._n_agents):
                if observer_id == actor_id:
                    continue
                weight = self._effective_observer_weight(observer_id, actor_id)
                total += weight * (
                    float(rewards[observer_id]) + float(discount) * float(after_values[observer_id])
                )
            return total
        return float(sum(rewards) + float(discount) * sum(after_values))

    def _enumerate_action_objectives(
        self,
        state: State,
        legal_mask: np.ndarray,
        discount: float,
    ) -> tuple[np.ndarray, dict[int, tuple[float, ...]], dict[int, list[float]]]:
        phase2 = bool(state.pick_phase)
        q_values = np.zeros(int(legal_mask.shape[0]), dtype=float)
        rewards_by_action: dict[int, tuple[float, ...]] = {}
        after_values_by_action: dict[int, list[float]] = {}

        for action_idx in np.flatnonzero(legal_mask):
            action = policy_index_to_action(int(action_idx))
            after_state, rewards = self._after_state_for_action(state, action, phase2=phase2)
            encoded_after_state = self._encode_all_critics(after_state)
            with torch.no_grad():
                after_values = self._critic_values(encoded_after_state)
            rewards_by_action[int(action_idx)] = rewards
            after_values_by_action[int(action_idx)] = after_values
            q_values[int(action_idx)] = self._action_objective(
                rewards,
                after_values,
                state.actor,
                discount,
            )

        return q_values, rewards_by_action, after_values_by_action

    # ------------------------------------------------------------------
    # Decision training
    # ------------------------------------------------------------------
    def _train_critic_after_transition(
        self,
        after_state: State,
        rewards: tuple[float, ...],
        discount: float,
        t: int,
    ) -> None:
        self._timer.start(TimerSection.ENCODE)
        current_after = self._encode_all_critics(after_state)
        self._timer.stop()

        if self._critic_prev_after is not None:
            if not self._freeze_critic:
                self._timer.start(TimerSection.TRAIN)
                critic_loss = self._critic_td_step(
                    self._critic_prev_after,
                    rewards,
                    discount,
                    current_after,
                    t,
                )
                self._td_loss_accum += critic_loss
                self._td_loss_count += self._n_agents
                self._timer.stop()

        self._critic_prev_after = current_after

    def _train_decision(
        self,
        state: State,
        action: Action,
        legal_mask: np.ndarray,
        discount: float,
        t: int,
    ) -> None:
        actor_id = state.actor
        action_idx = int(action.value)

        actor_state = self._encode_actor_state(state, actor_id)
        actor_net = self._actor_networks_list[actor_id]
        probs = actor_net.get_action_probabilities(actor_state, legal_mask)
        q_values, rewards_by_action, after_values_by_action = self._enumerate_action_objectives(
            state,
            legal_mask,
            discount,
        )
        selected_q_value = float(q_values[action_idx])
        baseline_value = float(np.dot(probs, q_values))
        advantage = selected_q_value - baseline_value

        if self._following_states:
            rho = float(self._following_rates_cfg.rho)
            for observer_id, observer_state in enumerate(self._following_states):
                if observer_id == actor_id:
                    continue
                selected_rewards = rewards_by_action[action_idx]
                selected_after_values = after_values_by_action[action_idx]
                observer_q = (
                    float(selected_rewards[observer_id]) +
                    float(discount) * float(selected_after_values[observer_id])
                )
                observer_state.update_alpha(actor_id, observer_q, rho)
            self._refresh_influencer_beta()
            self._refresh_follower_influencer_values()
            self._decision_count += 1
            if self._should_reallocate_following_rates():
                self._reallocate_following_rates()
        else:
            self._decision_count += 1

        selected_after_state, _ = self._after_state_for_action(
            state,
            action,
            phase2=bool(state.pick_phase),
        )
        self._train_critic_after_transition(
            selected_after_state,
            rewards_by_action[action_idx],
            discount,
            t,
        )

        self._timer.start(TimerSection.TRAIN)
        actor_lr = compute_schedule_value(self._actor_lr_schedule, t, self._total_steps)
        actor_net.set_lr(actor_lr)
        actor_net.add_experience(actor_state, legal_mask, action, advantage)
        actor_metrics = actor_net.train_batch()
        self._timer.stop()

        if actor_metrics is not None:
            self._actor_loss_accum += float(actor_metrics["loss"])
            self._actor_loss_count += 1
            self._advantage_accum += float(actor_metrics["advantage_mean"])
            self._advantage_count += 1
            self._entropy_accum += float(actor_metrics["entropy_mean"])
            self._entropy_count += 1

    # ------------------------------------------------------------------
    # Evaluation and logging
    # ------------------------------------------------------------------
    def evaluate(self, env: BaseEnv, eval_cfg: EvalConfig) -> dict[str, float | int]:
        eval_start = env.init_state()

        def greedy_policy(s: State, phase2: bool = False) -> Action:
            return self._greedy_action(s, phase2=phase2)

        def baseline_policy(s: State, phase2: bool = False) -> Action:
            return heuristic_action(s, env.cfg, self._heuristic, phase2=phase2)

        heuristic_name = self._heuristic.name.lower()
        greedy_metrics = evaluate_policy_metrics(eval_start, greedy_policy, env, eval_cfg.eval_steps)
        baseline_metrics = evaluate_policy_metrics(eval_start, baseline_policy, env, eval_cfg.eval_steps)
        return {
            "greedy_rps": greedy_metrics["rps"],
            "greedy_team_rps": greedy_metrics["team_rps"],
            "greedy_correct_pps": greedy_metrics["correct_pps"],
            "greedy_wrong_pps": greedy_metrics["wrong_pps"],
            f"{heuristic_name}_rps": baseline_metrics["rps"],
            f"{heuristic_name}_team_rps": baseline_metrics["team_rps"],
            f"{heuristic_name}_correct_pps": baseline_metrics["correct_pps"],
            f"{heuristic_name}_wrong_pps": baseline_metrics["wrong_pps"],
        }

    def get_td_loss(self) -> float:
        avg = self._td_loss_accum / max(self._td_loss_count, 1)
        self._td_loss_accum = 0.0
        self._td_loss_count = 0
        return avg

    def get_main_metrics(self) -> dict[str, float | int | str]:
        actor_loss_mean = (
            self._actor_loss_accum / self._actor_loss_count
            if self._actor_loss_count > 0
            else None
        )
        advantage_mean = (
            self._advantage_accum / self._advantage_count
            if self._advantage_count > 0
            else None
        )
        entropy_mean = (
            self._entropy_accum / self._entropy_count
            if self._entropy_count > 0
            else None
        )
        self._actor_loss_accum = 0.0
        self._actor_loss_count = 0
        self._advantage_accum = 0.0
        self._advantage_count = 0
        self._entropy_accum = 0.0
        self._entropy_count = 0

        row: dict[str, float | int | str] = {
            "actor_lr": self._actor_networks_list[0].get_lr() if self._actor_networks_list else "",
            "actor_loss_mean": _sanitize_metric(actor_loss_mean),
            "advantage_mean": _sanitize_metric(advantage_mean),
            "policy_entropy_mean": _sanitize_metric(entropy_mean),
        }
        if self._following_rates_cfg.enabled:
            stats = self._current_following_rate_stats()
            allowed_keys = {
                "alpha_mean",
                "alpha_positive_frac",
                "following_weight_mean",
                "active_follow_edges_mean",
                "follower_to_influencer_weight_mean",
                "effective_follow_weight_mean",
            }
            if self._influencer_cfg.enabled:
                allowed_keys.update({"beta_mean", "influencer_weight_mean"})
            for key, value in stats.items():
                if key not in allowed_keys:
                    continue
                row[key] = _sanitize_metric(value)
        return row

    def get_detail_metrics(self) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            "current_actor_lr": self._actor_networks_list[0].get_lr() if self._actor_networks_list else "",
        }
        for idx, actor_net in enumerate(self._actor_networks_list):
            for name, val in actor_net.get_weight_norms().items():
                row[f"actor_weight_norm_agent_{idx}_{name}"] = round(val, 6)
            for name, val in actor_net.get_grad_norms().items():
                row[f"actor_grad_norm_agent_{idx}_{name}"] = round(val, 6)
        return row

    def setup_aux_loggers(self, run_dir: Path) -> None:
        self._phase1_logger = CSVLogger(
            run_dir / "phase1_policy_probabilities.csv",
            build_phase1_policy_prob_csv_fieldnames(self._env.cfg.n_task_types),
        )
        self._phase2_logger = CSVLogger(
            run_dir / "phase2_policy_probabilities.csv",
            build_phase2_policy_prob_csv_fieldnames(self._env.cfg.n_task_types),
        )
        self._phase1_eval_states = sample_phase1_policy_eval_states(self._env.cfg)
        self._phase2_eval_states = generate_phase2_policy_eval_states(self._env.cfg)

        if self._following_rates_cfg.enabled:
            fieldnames = build_following_rate_csv_fieldnames(self._n_agents)
            self._following_loggers = {
                idx: CSVLogger(run_dir / f"following_rates_agent_{idx}.csv", fieldnames)
                for idx in range(self._n_agents)
            }
        if self._influencer is not None:
            self._influencer_logger = CSVLogger(
                run_dir / "external_influencer.csv",
                build_influencer_csv_fieldnames(self._n_agents),
            )

    def log_auxiliary(self, step: int, wall_time: float) -> None:
        if self._phase1_logger is not None and self._phase1_eval_states is not None:
            for idx, state in enumerate(self._phase1_eval_states):
                probs = self._actor_probabilities(state, phase2=False)
                self._phase1_logger.log(
                    build_phase1_policy_prob_row(step, wall_time, idx, state, probs, self._env.cfg)
                )
        if self._phase2_logger is not None and self._phase2_eval_states is not None:
            for idx, (label, state) in enumerate(self._phase2_eval_states):
                probs = self._actor_probabilities(state, phase2=True)
                self._phase2_logger.log(
                    build_phase2_policy_prob_row(step, wall_time, idx, label, state, probs, self._env.cfg)
                )
        for idx, state in enumerate(self._following_states):
            logger = self._following_loggers.get(idx)
            if logger is not None:
                logger.log(build_following_rate_snapshot_row(step, wall_time, state))
        if self._influencer_logger is not None and self._influencer is not None:
            self._influencer_logger.log(build_influencer_snapshot_row(step, wall_time, self._influencer))

    def close(self) -> None:
        if self._phase1_logger is not None:
            self._phase1_logger.close()
        if self._phase2_logger is not None:
            self._phase2_logger.close()
        for logger in self._following_loggers.values():
            logger.close()
        if self._influencer_logger is not None:
            self._influencer_logger.close()

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: Path, step: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "algorithm": "actor_critic",
            "step": int(step),
            "critics": [net.state_dict() for net in self._critic_networks_list],
            "actors": [net.state_dict() for net in self._actor_networks_list],
            "following_rates": [
                {
                    "agent_id": state.agent_id,
                    "agent_alphas": state.agent_alphas.tolist(),
                    "budget": state.budget,
                    "following_rates": state.following_rates.tolist(),
                    "following_rate_to_influencer": state.following_rate_to_influencer,
                    "rate_solver_name": state.rate_solver_name,
                    "influencer_value": state.influencer_value,
                }
                for state in self._following_states
            ],
            "influencer": None,
        }
        if self._influencer is not None:
            ckpt["influencer"] = {
                "budget": self._influencer.budget,
                "num_agents": self._influencer.num_agents,
                "outgoing_rates": self._influencer.outgoing_rates.tolist(),
                "beta": self._influencer.beta.tolist(),
                "rate_solver_name": self._influencer.rate_solver_name,
            }
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str | Path) -> int | None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self._critic_prev_after = None

        # Allow actor-critic runs to initialize critics from value checkpoints.
        if "networks" in ckpt:
            for net, sd in zip(self._critic_networks_list, ckpt["networks"]):
                net.load_state_dict(sd, strict=True)
            return ckpt.get("step")

        if ckpt.get("algorithm") == "value":
            for net, sd in zip(self._critic_networks_list, ckpt["critics"]):
                net.load_state_dict(sd, strict=True)
            return ckpt.get("step")

        if ckpt.get("algorithm") != "actor_critic":
            raise ValueError(
                f"Checkpoint algorithm mismatch: expected actor_critic or value, got {ckpt.get('algorithm')!r}."
            )
        for net, sd in zip(self._critic_networks_list, ckpt["critics"]):
            net.load_state_dict(sd, strict=True)
        for net, sd in zip(self._actor_networks_list, ckpt["actors"]):
            net.load_state_dict(sd, strict=True)

        if self._following_states and ckpt.get("following_rates"):
            for state, payload in zip(self._following_states, ckpt["following_rates"]):
                state.agent_alphas = np.asarray(payload["agent_alphas"], dtype=float)
                state.agent_alphas[state.agent_id] = 0.0
                state.set_following_rates(payload["following_rates"])
                state.set_influencer_rate(payload.get("following_rate_to_influencer", 0.0))
                state.set_influencer_value(payload.get("influencer_value", 0.0))

        if self._influencer is not None and ckpt.get("influencer") is not None:
            influencer_payload = ckpt["influencer"]
            self._influencer.set_outgoing_rates(influencer_payload["outgoing_rates"])
            self._influencer.set_beta(influencer_payload["beta"])
        self._refresh_influencer_beta()
        self._refresh_follower_influencer_values()
        return ckpt.get("step")

    def load_critic_checkpoint(self, path: str | Path) -> int | None:
        """Load critic weights only; actor networks remain randomly initialised."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self._critic_prev_after = None
        if "networks" in ckpt:  # legacy value checkpoint
            for net, sd in zip(self._critic_networks_list, ckpt["networks"]):
                net.load_state_dict(sd, strict=True)
        elif "critics" in ckpt:  # value or actor_critic checkpoint
            for net, sd in zip(self._critic_networks_list, ckpt["critics"]):
                net.load_state_dict(sd, strict=True)
        else:
            raise ValueError(f"No critic weights found in checkpoint. Keys: {list(ckpt.keys())}")
        return ckpt.get("step")

    def load_actor_checkpoint(self, path: str | Path) -> int | None:
        """Load actor weights only; critic networks remain as-is."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if "actors" not in ckpt:
            raise ValueError(f"No actor weights found in checkpoint. Keys: {list(ckpt.keys())}")
        for net, sd in zip(self._actor_networks_list, ckpt["actors"]):
            net.load_state_dict(sd, strict=True)
        return ckpt.get("step")
    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def critic_networks(self) -> list[ValueNetwork]:
        return self._critic_networks_list

    @property
    def actor_networks(self) -> list[PolicyNetwork]:
        return self._actor_networks_list


class ActorCriticCpuTrainer(ActorCriticTrainerBase):
    """CPU actor-critic trainer."""

    def _encode_all_critics(self, state: State) -> list[EncoderOutput]:
        return [encoding.encode(state, i) for i in range(self._n_agents)]

    def _critic_values(self, encoded_state: list[EncoderOutput]) -> list[float]:
        return [
            float(net(enc).item())
            for net, enc in zip(self._critic_networks_list, encoded_state)
        ]

    def _critic_td_step(
        self,
        prev: list[EncoderOutput],
        rewards: tuple[float, ...],
        discount: float,
        current: list[EncoderOutput],
        t: int,
    ) -> float:
        alpha = compute_schedule_value(self._critic_lr_schedule, t, self._total_steps)
        total_loss = 0.0
        for idx, net in enumerate(self._critic_networks_list):
            delta = net.td_step(
                s_enc=prev[idx],
                reward=float(rewards[idx]),
                discount=discount,
                s_next_enc=current[idx],
                alpha=alpha,
            )
            total_loss += delta ** 2
        return total_loss

    def sync_to_cpu(self) -> None:
        pass


class ActorCriticGpuTrainer(ActorCriticTrainerBase):
    """GPU actor-critic trainer with batched critic TD updates."""

    def __init__(
        self,
        critic_networks: list[ValueNetwork],
        actor_networks: list[PolicyNetwork],
        bt: BatchedTrainer,
        env: BaseEnv,
        gamma: float,
        critic_lr_schedule: ScheduleConfig,
        actor_lr_schedule: ScheduleConfig,
        total_steps: int,
        heuristic: Heuristic,
        freeze_critic: bool,
        following_rates_cfg: FollowingRatesConfig,
        influencer_cfg: InfluencerConfig,
        timer: Timer | None = None,
    ) -> None:
        super().__init__(
            critic_networks=critic_networks,
            actor_networks=actor_networks,
            env=env,
            gamma=gamma,
            critic_lr_schedule=critic_lr_schedule,
            actor_lr_schedule=actor_lr_schedule,
            total_steps=total_steps,
            heuristic=heuristic,
            freeze_critic=freeze_critic,
            following_rates_cfg=following_rates_cfg,
            influencer_cfg=influencer_cfg,
            timer=timer,
        )
        self._bt = bt

    def _encode_all_critics(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        return encoding.encode_all_agents(state)

    def _critic_values(self, encoded_state: tuple[torch.Tensor, torch.Tensor]) -> list[float]:
        grids, scalars = encoded_state
        return self._bt.forward_single_batched(grids, scalars).detach().cpu().tolist()

    def _critic_td_step(
        self,
        prev: tuple[torch.Tensor, torch.Tensor],
        rewards: tuple[float, ...],
        discount: float,
        current: tuple[torch.Tensor, torch.Tensor],
        t: int,
    ) -> float:
        grids_t, scalars_t = prev
        grids_next, scalars_next = current
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        return self._bt.td_lambda_step_batched(
            grids_t,
            scalars_t,
            rewards_t,
            discount,
            grids_next,
            scalars_next,
            alpha=compute_schedule_value(self._critic_lr_schedule, t, self._total_steps),
        )

    def sync_to_cpu(self) -> None:
        self._bt.sync_to_networks()

    def load_checkpoint(self, path: str | Path) -> int | None:
        step = super().load_checkpoint(path)
        self._bt.sync_from_networks()
        return step

    def load_critic_checkpoint(self, path: str | Path) -> int | None:
            step = super().load_critic_checkpoint(path)
            self._bt.sync_from_networks()
            return step

    def load_actor_checkpoint(self, path: str | Path) -> int | None:
        step = super().load_actor_checkpoint(path)
        # actors live on their own devices, no bt sync needed
        return step