"""Value trainer base: shared logic for CPU and GPU value-learning trainers.

CpuValueTrainer and GpuValueTrainer override only:
  _encode_all(state) → encoding in subclass-specific format
  _td_step(prev, rewards, discount, current, t) → sum of δ²
  _compute_team_values(state, after_states, actor) → per-action team values
  sync_to_cpu()
"""

from __future__ import annotations

import csv as _csv
from abc import abstractmethod
from pathlib import Path
from typing import Any

import torch

from orchard.datatypes import EvalConfig, ScheduleConfig, State
from orchard.enums import Action, Heuristic, PickMode
from orchard.env.base import BaseEnv
from orchard.model import ValueNetwork
from orchard.policy import get_all_actions, get_phase2_actions, heuristic_action
from orchard.schedule import compute_schedule_value
from orchard.seed import rng
from orchard.trainer.base import TrainerBase
from orchard.trainer.timer import Timer, TimerSection


class ValueTrainerBase(TrainerBase):
    """Shared logic for value-function trainers (CPU and GPU).

    Subclasses only implement encoding, TD update, and value computation.
    """

    def __init__(
        self,
        network_list: list[ValueNetwork],
        env: BaseEnv,
        gamma: float,
        epsilon_schedule: ScheduleConfig,
        lr_schedule: ScheduleConfig,
        total_steps: int,
        heuristic: Heuristic,
        timer: Timer | None = None,
    ) -> None:
        self._networks_list = network_list
        self._env = env
        self._n_agents = env.cfg.n_agents
        self._n_networks = len(network_list)
        self._centralized = (self._n_networks == 1)
        self._gamma = gamma
        self._epsilon_schedule = epsilon_schedule
        self._lr_schedule = lr_schedule
        self._total_steps = total_steps
        self._heuristic = heuristic
        self._timer = timer or Timer()

        self._zero_rewards = tuple(0.0 for _ in range(self._n_networks))

        # After-state TD bookkeeping (opaque: subclass determines format)
        self._prev: Any = None
        self._move: Any = None

        # Cached encoding of the selected after-state from the most recent greedy
        # action selection. Set by _greedy_action via _cache_selected_enc(); consumed
        # and cleared by train_move/train_pick to avoid re-encoding the same state.
        self._cached_enc: Any = None

        # Loss accumulator
        self._td_loss_accum: float = 0.0
        self._td_loss_count: int = 0

        # Env trace (opened by setup_aux_loggers, closed by close)
        self._trace_f: Any = None
        self._trace_w: Any = None

    # ------------------------------------------------------------------
    # Auxiliary logging
    # ------------------------------------------------------------------
    def setup_aux_loggers(self, run_dir: Path, alpha_state_log_freq: int = 0, env_trace: bool = True) -> None:
        if not env_trace:
            return
        fields = (
            ["step", "actor", "epsilon", "action", "on_task", "pick_happened", "pick_task_type"]
            + [f"reward_{i}" for i in range(self._n_agents)]
            + ["n_tasks_before_spawn", "tasks_despawned", "tasks_spawned",
               "n_tasks_after", "task_positions_after", "agent_positions"]
        )
        self._trace_f = open(run_dir / "env_trace.csv", "w", newline="")
        self._trace_w = _csv.DictWriter(self._trace_f, fieldnames=fields)
        self._trace_w.writeheader()
        self._trace_f.flush()

    def close(self) -> None:
        if self._trace_f is not None:
            self._trace_f.close()
            self._trace_f = None
            self._trace_w = None

    # ------------------------------------------------------------------
    # Abstract methods — subclass must implement
    # ------------------------------------------------------------------
    @abstractmethod
    def _encode_all(self, state: State) -> Any:
        """Encode state for all networks. Format is subclass-specific."""
        ...

    @abstractmethod
    def _td_step(
        self, prev: Any, rewards: tuple[float, ...],
        discount: float, current: Any, t: int,
    ) -> float:
        """Run TD update. Returns sum of δ² for loss tracking."""
        ...

    @abstractmethod
    def _compute_team_values(
        self, state: State, after_states: list[State],
    ) -> list[float]:
        """Compute sum_j V_j(after_state) for each candidate after-state."""
        ...

    def _cache_selected_enc(self, best_idx: int) -> None:
        """Optionally cache the encoding at best_idx for reuse in train_move/train_pick.

        Default no-op. GpuTrainer overrides this to slice and store the encoding
        already computed inside _compute_team_values, avoiding a redundant encode call.
        """

    # ------------------------------------------------------------------
    # Turn stepping
    # ------------------------------------------------------------------
    def step(self, state: State, t: int) -> State:
        self._timer.step_begin()

        _trace_actor = state.actor
        move_action = self.select_move(state, t)
        _trace_eps = compute_schedule_value(self._epsilon_schedule, t, self._total_steps)

        self._timer.start(TimerSection.ENV)
        s_moved = self._env.apply_action(state, move_action)
        on_task = s_moved.is_agent_on_task(s_moved.actor)
        self._timer.stop()

        self.train_move(s_moved, on_task, t)

        _trace_pick_happened = False
        _trace_pick_type = -1
        _trace_pick_rewards: tuple[float, ...] = tuple(0.0 for _ in range(self._n_agents))

        if on_task:
            self._timer.start(TimerSection.ENV)
            if self._env.cfg.pick_mode == PickMode.FORCED:
                # Forced mode: always pick, no epsilon selection (matches old force_pick behavior)
                s_picked, pick_rewards = self._env.resolve_pick(s_moved)
                _trace_pick_type = 0
            else:
                pick_action = self.select_pick(s_moved.with_pick_phase(), t)
                s_picked, pick_rewards = self._env.resolve_pick(
                    s_moved,
                    pick_type=pick_action.pick_type() if pick_action.is_pick() else None,
                )
                _trace_pick_type = pick_action.pick_type() if pick_action.is_pick() else -1
            self._timer.stop()
            self.train_pick(s_picked, pick_rewards, t)
            _trace_pick_happened = True
            _trace_pick_rewards = pick_rewards
        else:
            s_picked = s_moved

        self._timer.start(TimerSection.ENV)
        _s_pre_spawn = s_picked
        _s_post_spawn = self._env.spawn_and_despawn(s_picked)
        result = self._env.advance_actor(_s_post_spawn)
        self._timer.stop()

        if self._trace_w is not None:
            _fmt = lambda ps: ";".join(f"{p.row},{p.col}" for p in sorted(ps))
            _tasks_before = set(_s_pre_spawn.task_positions)
            _tasks_after = set(_s_post_spawn.task_positions)
            _row: dict = {
                "step": t,
                "actor": _trace_actor,
                "epsilon": round(_trace_eps, 6),
                "action": move_action.name,
                "on_task": on_task,
                "pick_happened": _trace_pick_happened,
                "pick_task_type": _trace_pick_type,
                "n_tasks_before_spawn": len(_tasks_before),
                "tasks_despawned": _fmt(_tasks_before - _tasks_after),
                "tasks_spawned": _fmt(_tasks_after - _tasks_before),
                "n_tasks_after": len(_tasks_after),
                "task_positions_after": _fmt(_tasks_after),
                "agent_positions": _fmt(result.agent_positions),
            }
            for i in range(self._n_agents):
                _row[f"reward_{i}"] = _trace_pick_rewards[i] if i < len(_trace_pick_rewards) else 0.0
            self._trace_w.writerow(_row)
            self._trace_f.flush()

        return result

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_move(self, state: State, t: int) -> Action:
        self._timer.start(TimerSection.ACTION)
        eps = compute_schedule_value(self._epsilon_schedule, t, self._total_steps)
        actions = get_all_actions(self._env.cfg)
        if rng.random() < eps:
            action = actions[rng.randint(0, len(actions) - 1)]
        else:
            action = self._greedy_action(state)
        self._timer.stop()
        return action

    def select_pick(self, state: State, t: int) -> Action:
        self._timer.start(TimerSection.ACTION)
        eps = compute_schedule_value(self._epsilon_schedule, t, self._total_steps)
        actions = get_phase2_actions(state, self._env.cfg)
        if rng.random() < eps:
            action = actions[rng.randint(0, len(actions) - 1)]
        else:
            action = self._greedy_action(state)
        self._timer.stop()
        return action

    def _greedy_action(self, state: State) -> Action:
        """Argmax over Q_team = r_team(s,a) + sum_j V_j(after_state)."""
        phase2 = state.pick_phase
        all_actions = get_phase2_actions(state, self._env.cfg) if phase2 else get_all_actions(self._env.cfg)

        # Build after-states and immediate rewards for each candidate action
        after_states: list[State] = []
        immediate_rewards: list[float] = []
        self._timer.start(TimerSection.ACTION_ENV)
        for a in all_actions:
            if phase2 and a.is_pick():
                s_after, rewards = self._env.resolve_pick(state, pick_type=a.pick_type())
                after_states.append(s_after)
                immediate_rewards.append(sum(rewards))
            elif phase2:
                after_states.append(state)
                immediate_rewards.append(0.0)
            else: #
                s = self._env.apply_action(state, a)
                if s.is_agent_on_task(s.actor):
                    after_states.append(s.with_pick_phase())
                else:
                    after_states.append(s)
                immediate_rewards.append(0.0)
        self._timer.stop()

        team_values = self._compute_team_values(state, after_states)

        best_idx = 0
        best_val = team_values[0] + immediate_rewards[0]
        for k in range(1, len(all_actions)):
            val = team_values[k] + immediate_rewards[k]
            if val > best_val:
                best_val = val
                best_idx = k
        if not phase2:
            self._cache_selected_enc(best_idx)
        return all_actions[best_idx]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_move(self, s_moved: State, on_task: bool, t: int) -> None:
        """Encode move after-state. TD update: prev_after →[r=0, γ=γ]→ move_after."""
        self._timer.start(TimerSection.ENCODE)
        if self._cached_enc is not None:
            self._move = self._cached_enc
            self._cached_enc = None
        else:
            enc_state = s_moved.with_pick_phase() if on_task else s_moved
            self._move = self._encode_all(enc_state)
        self._timer.stop()

        if self._prev is not None:
            self._timer.start(TimerSection.TRAIN)
            loss = self._td_step(self._prev, self._zero_rewards, self._gamma,
                                 self._move, t)
            self._td_loss_accum += loss
            self._td_loss_count += self._n_networks
            self._timer.stop()

        if not on_task:
            self._prev = self._move

    def train_pick(self, s_picked: State, rewards: tuple[float, ...], t: int) -> None:
        """Encode pick after-state. TD update: move_after →[r=rewards, γ=1]→ pick_after."""
        self._timer.start(TimerSection.ENCODE)
        pick_enc = self._encode_all(s_picked)
        self._timer.stop()

        train_rewards = (sum(rewards),) if self._centralized else rewards

        self._timer.start(TimerSection.TRAIN)
        loss = self._td_step(self._move, train_rewards, 1.0, pick_enc, t)
        self._td_loss_accum += loss
        self._td_loss_count += self._n_networks
        self._timer.stop()

        self._prev = pick_enc

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, env: BaseEnv, eval_cfg: EvalConfig) -> dict[str, float | int]:
        from orchard.eval import evaluate_policy_metrics
        eval_start = env.init_state()

        def greedy_policy(s: State) -> Action:
            return self._greedy_action(s)

        def baseline_policy(s: State) -> Action:
            return heuristic_action(s, env.cfg, self._heuristic)

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

    # ------------------------------------------------------------------
    # Loss tracking
    # ------------------------------------------------------------------
    def get_td_loss(self) -> float:
        avg = self._td_loss_accum / max(self._td_loss_count, 1)
        self._td_loss_accum = 0.0
        self._td_loss_count = 0
        return avg

    @property
    def critic_networks(self) -> list[ValueNetwork]:
        return self._networks_list

    @property
    def networks(self) -> list[ValueNetwork]:
        return self.critic_networks

    def save_checkpoint(self, path: Path, step: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": "value",
                "step": int(step),
                "critics": [net.state_dict() for net in self._networks_list],
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> int | None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if "networks" in ckpt:
            for net, sd in zip(self._networks_list, ckpt["networks"]):
                net.load_state_dict(sd, strict=True)
            return ckpt.get("step")

        if ckpt.get("algorithm") != "value":
            raise ValueError(
                f"Checkpoint algorithm mismatch: expected value, got {ckpt.get('algorithm')!r}."
            )
        for net, sd in zip(self._networks_list, ckpt["critics"]):
            net.load_state_dict(sd, strict=True)
        return ckpt.get("step")
