"""Learned-actor trainer: one shared masked actor, swappable team critic.

The actor is updated from the team value only (sum_i V_i for a decentralized
critic, V_cen for a centralized one) through `TeamCritic.team_values`, so the
critic type is a config switch (train.learning_type) and the actor code is the
same in both cases. Critic outputs used for advantages are computed under
no_grad, so the actor loss never reaches critic parameters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import orchard.encoding as encoding
from orchard.actor_critic import (
    build_phase1_legal_mask,
    build_phase2_legal_mask,
    policy_index_to_action,
)
from orchard.datatypes import (
    EncoderOutput,
    EvalConfig,
    LearnedActorConfig,
    ScheduleConfig,
    State,
)
from orchard.enums import Action, Heuristic
from orchard.env.base import BaseEnv
from orchard.eval import evaluate_policy_metrics
from orchard.learned_actor.actor import SharedActor
from orchard.learned_actor.critic import TeamCritic
from orchard.model import ValueNetwork
from orchard.policy import heuristic_action
from orchard.schedule import compute_schedule_value
from orchard.trainer.base import TrainerBase
from orchard.trainer.timer import Timer, TimerSection


class LearnedActorTrainer(TrainerBase):
    """Shared-actor actor-critic trainer (CPU)."""

    reports_sampled_eval = False

    def __init__(
        self,
        critic: TeamCritic,
        actor: SharedActor,
        env: BaseEnv,
        gamma: float,
        critic_lr_schedule: ScheduleConfig,
        actor_lr_schedule: ScheduleConfig,
        total_steps: int,
        heuristic: Heuristic,
        freeze_critic: bool,
        warmup_steps: int,
        learned_actor_cfg: LearnedActorConfig,
        discount_method: str = "round_steps",
        timer: Timer | None = None,
    ) -> None:
        self._critic = critic
        self._actor = actor
        self._env = env
        self._gamma = float(gamma)
        self._critic_lr_schedule = critic_lr_schedule
        self._actor_lr_schedule = actor_lr_schedule
        self._total_steps = int(total_steps)
        self._heuristic = heuristic
        self._freeze_critic = bool(freeze_critic)
        self._warmup_steps = max(0, int(warmup_steps))
        self._cfg = learned_actor_cfg
        self._discount_method = discount_method
        self._timer = timer or Timer()
        self._n_agents = env.cfg.n_agents
        self.reports_sampled_eval = bool(learned_actor_cfg.eval_sampled)

        # Afterstate chain used for the critic TD update.
        self._prev_after: list[EncoderOutput] | None = None
        # Actor experience collected during the current round.
        self._batch_states: list[EncoderOutput] = []
        self._batch_masks: list[np.ndarray] = []
        self._batch_actions: list[int] = []
        self._batch_advantages: list[float] = []

        self._td_loss_accum = 0.0
        self._td_loss_count = 0
        self._actor_loss_accum = 0.0
        self._advantage_accum = 0.0
        self._entropy_accum = 0.0
        self._actor_count = 0.0

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    def _legal_mask(self, state: State) -> np.ndarray:
        if state.pick_phase:
            return build_phase2_legal_mask(state, self._env.cfg, self._env.proficiency_positive_types)
        return build_phase1_legal_mask(state, self._env.cfg)

    def _probs(self, state: State) -> tuple[EncoderOutput, np.ndarray, np.ndarray]:
        enc = encoding.encode(state, state.actor)
        mask = self._legal_mask(state)
        return enc, mask, self._actor.get_action_probabilities(enc, mask)

    @staticmethod
    def _sample_index(probs: np.ndarray) -> int:
        p = np.asarray(probs, dtype=np.float64)
        return int(np.random.choice(len(p), p=p / p.sum()))

    def _greedy_action(self, state: State) -> Action:
        _, _, probs = self._probs(state)
        return policy_index_to_action(int(np.argmax(probs)))

    def _sampled_action(self, state: State) -> Action:
        _, _, probs = self._probs(state)
        return policy_index_to_action(self._sample_index(probs))

    # ------------------------------------------------------------------
    # Afterstates
    # ------------------------------------------------------------------
    def _after_state_for_action(
        self, state: State, action: Action,
    ) -> tuple[State, tuple[float, ...]]:
        if state.pick_phase:
            base = State(
                agent_positions=state.agent_positions,
                task_positions=state.task_positions,
                actor=state.actor,
                task_types=state.task_types,
                pick_phase=False,
            )
            return self._env.resolve_pick(
                base, pick_type=action.pick_type() if action.is_pick() else None,
            )
        s_moved = self._env.apply_action(state, action)
        zeros = tuple(0.0 for _ in range(self._n_agents))
        if s_moved.is_agent_on_task(s_moved.actor, self._env.proficiency_positive_types[s_moved.actor]):
            return s_moved.with_pick_phase(), zeros
        return s_moved, zeros

    # ------------------------------------------------------------------
    # One decision: advantage, critic TD step, actor experience
    # ------------------------------------------------------------------
    def _decide(
        self,
        state: State,
        action: Action,
        actor_enc: EncoderOutput,
        mask: np.ndarray,
        probs: np.ndarray,
        discount: float,
        t: int,
    ) -> None:
        legal = [int(i) for i in np.flatnonzero(mask)]
        outcomes = [
            self._after_state_for_action(state, policy_index_to_action(i)) for i in legal
        ]
        after_states = [o[0] for o in outcomes]
        team_rewards = torch.tensor([float(sum(o[1])) for o in outcomes], dtype=torch.float32)
        pos = legal.index(int(action.value))

        self._timer.start(TimerSection.TRAIN)
        # Q(s, a) = r_team(a) + discount * V_team(afterstate_a). The afterstate
        # value excludes the transition reward, so r is added exactly once.
        q = team_rewards + discount * self._critic.team_values(state, after_states)
        selected_after_enc = self._critic.encode(after_states[pos])

        if self._cfg.advantage == "afterstate":
            p = torch.as_tensor(probs[legal], dtype=torch.float32)
            advantage: float | None = float((q[pos] - (p * q).sum()).item())
        else:
            # td_error: the critic's TD error on the sampled transition,
            # r + discount * V_team(s') - V_team(previous afterstate).
            advantage = None
            if self._prev_after is not None:
                advantage = float(q[pos].item()) - self._critic.team_value_of(self._prev_after)

        self._update_critic(selected_after_enc, outcomes[pos][1], discount, t)

        if advantage is not None and t >= self._warmup_steps:
            self._batch_states.append(actor_enc)
            self._batch_masks.append(np.asarray(mask, dtype=bool))
            self._batch_actions.append(int(action.value))
            self._batch_advantages.append(advantage)
        self._timer.stop()

    def _update_critic(
        self,
        current: list[EncoderOutput],
        rewards: tuple[float, ...],
        discount: float,
        t: int,
    ) -> None:
        if t < self._warmup_steps:
            self._prev_after = None
            return
        if self._freeze_critic:
            # No critic update, but keep the chain so td_error advantages work.
            self._prev_after = current
            return
        if self._prev_after is not None:
            alpha = compute_schedule_value(self._critic_lr_schedule, t, self._total_steps)
            loss = self._critic.td_step(self._prev_after, rewards, discount, current, alpha)
            self._td_loss_accum += loss
            self._td_loss_count += len(self._critic.networks)
        self._prev_after = current

    def _flush_actor_batch(self, t: int) -> None:
        if not self._batch_states:
            return
        self._actor.set_lr(compute_schedule_value(self._actor_lr_schedule, t, self._total_steps))
        metrics = self._actor.update(
            self._batch_states,
            np.stack(self._batch_masks, axis=0),
            self._batch_actions,
            torch.tensor(self._batch_advantages, dtype=torch.float32),
            entropy_coef=self._cfg.entropy_coef,
            normalize_advantage=self._cfg.normalize_advantage,
        )
        n = metrics["sample_count"]
        self._actor_loss_accum += metrics["loss"] * n
        self._advantage_accum += metrics["advantage_mean"] * n
        self._entropy_accum += metrics["entropy_mean"] * n
        self._actor_count += n
        self._batch_states, self._batch_masks = [], []
        self._batch_actions, self._batch_advantages = [], []

    def flush_pending_updates(self) -> None:
        self._batch_states, self._batch_masks = [], []
        self._batch_actions, self._batch_advantages = [], []

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def _move_discount(self, state: State) -> float:
        if self._discount_method == "round_steps":
            # gamma once per full round: only the first agent's move carries it.
            return self._gamma if state.actor == 0 else 1.0
        return self._gamma

    def step(self, state: State, t: int) -> State:
        self._timer.step_begin()
        move_discount = self._move_discount(state)

        self._timer.start(TimerSection.ACTION)
        enc, mask, probs = self._probs(state)
        move_action = policy_index_to_action(self._sample_index(probs))
        self._timer.stop()

        self._timer.start(TimerSection.ENV)
        s_moved = self._env.apply_action(state, move_action)
        actor = state.actor
        on_task = s_moved.is_agent_on_task(actor, self._env.proficiency_positive_types[actor])
        self._timer.stop()

        self._decide(state, move_action, enc, mask, probs, move_discount, t)

        if on_task:
            pick_state = s_moved.with_pick_phase()
            self._timer.start(TimerSection.ACTION)
            pick_enc, pick_mask, pick_probs = self._probs(pick_state)
            pick_action = policy_index_to_action(self._sample_index(pick_probs))
            self._timer.stop()
            self._timer.start(TimerSection.ENV)
            s_after, _ = self._env.resolve_pick(
                s_moved, pick_type=pick_action.pick_type() if pick_action.is_pick() else None,
            )
            self._timer.stop()
            self._decide(pick_state, pick_action, pick_enc, pick_mask, pick_probs, 1.0, t)
        else:
            s_after = s_moved

        self._timer.start(TimerSection.ENV)
        next_state = self._env.advance_actor(self._env.spawn_and_despawn(s_after))
        self._timer.stop()

        # One actor update per round of N agents, on the current policy's samples.
        if next_state.actor == 0:
            self._timer.start(TimerSection.TRAIN)
            self._flush_actor_batch(t)
            self._timer.stop()
        return next_state

    # ------------------------------------------------------------------
    # Evaluation and metrics
    # ------------------------------------------------------------------
    def evaluate(self, env: BaseEnv, eval_cfg: EvalConfig) -> dict[str, float | int]:
        env.set_eval_mode(True, seed=eval_cfg.eval_seed)
        try:
            start = env.init_state()
            name = self._heuristic.name.lower()
            greedy = evaluate_policy_metrics(start, self._greedy_action, env, eval_cfg.eval_steps)
            base = evaluate_policy_metrics(
                start, lambda s: heuristic_action(s, env, self._heuristic), env, eval_cfg.eval_steps,
            )
            out: dict[str, float | int] = {
                "greedy_rps": greedy["rps"],
                "greedy_team_rps": greedy["team_rps"],
                f"{name}_rps": base["rps"],
                f"{name}_team_rps": base["team_rps"],
            }
            if self._cfg.eval_sampled:
                sampled = evaluate_policy_metrics(start, self._sampled_action, env, eval_cfg.eval_steps)
                out["sampled_rps"] = sampled["rps"]
                out["sampled_team_rps"] = sampled["team_rps"]
            return out
        finally:
            env.set_eval_mode(False)

    def get_td_loss(self) -> float:
        avg = self._td_loss_accum / max(self._td_loss_count, 1)
        self._td_loss_accum = 0.0
        self._td_loss_count = 0
        return avg

    def get_main_metrics(self) -> dict[str, float | int | str]:
        n = self._actor_count
        row: dict[str, float | int | str] = {
            "actor_lr": self._actor.get_lr(),
            "actor_loss_mean": self._actor_loss_accum / n if n > 0 else "",
            "advantage_mean": self._advantage_accum / n if n > 0 else "",
            "policy_entropy_mean": self._entropy_accum / n if n > 0 else "",
        }
        self._actor_loss_accum = self._advantage_accum = self._entropy_accum = 0.0
        self._actor_count = 0.0
        return row

    def get_detail_metrics(self) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {"current_actor_lr": self._actor.get_lr()}
        for name, val in self._actor.get_weight_norms().items():
            row[f"actor_weight_norm_agent_0_{name}"] = round(val, 11)
        for name, val in self._actor.get_grad_norms().items():
            row[f"actor_grad_norm_agent_0_{name}"] = round(val, 11)
        return row

    def setup_aux_loggers(self, run_dir: Path, alpha_state_log_freq: int = 0, env_trace: bool = False) -> None:
        del run_dir, alpha_state_log_freq, env_trace

    def sync_to_cpu(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Checkpoints (same layout as the existing actor_critic checkpoints)
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: Path, step: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "actor_critic",
                "step": int(step),
                "critics": [net.state_dict() for net in self._critic.networks],
                "actors": [self._actor.state_dict()],
            },
            path,
        )

    def _load_critics(self, ckpt: dict) -> None:
        key = "networks" if "networks" in ckpt else "critics"
        if key not in ckpt:
            raise ValueError(f"No critic weights found in checkpoint. Keys: {list(ckpt.keys())}")
        for net, sd in zip(self._critic.networks, ckpt[key]):
            net.load_state_dict(sd, strict=True)
        self._prev_after = None

    def load_checkpoint(self, path: str | Path) -> int | None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self._load_critics(ckpt)
        if "actors" in ckpt:
            self._actor.load_state_dict(ckpt["actors"][0], strict=True)
        return ckpt.get("step")

    def load_critic_checkpoint(self, path: str | Path) -> int | None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self._load_critics(ckpt)
        return ckpt.get("step")

    def load_actor_checkpoint(self, path: str | Path) -> int | None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if "actors" not in ckpt:
            raise ValueError(f"No actor weights found in checkpoint. Keys: {list(ckpt.keys())}")
        self._actor.load_state_dict(ckpt["actors"][0], strict=True)
        return ckpt.get("step")

    @property
    def critic_networks(self) -> list[ValueNetwork]:
        return self._critic.networks

    @property
    def actor_networks(self) -> list[SharedActor]:
        return [self._actor]
