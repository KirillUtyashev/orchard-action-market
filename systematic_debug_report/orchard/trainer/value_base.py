"""Value trainer base: shared logic for CPU and GPU value-learning trainers.

CpuValueTrainer and GpuValueTrainer override only:
  _encode_all(state) → encoding in subclass-specific format
  _td_step(prev, rewards, discount, current, t) → sum of δ²
  _compute_team_values(state, after_states, actor) → per-action team values
  sync_to_cpu()
"""

from __future__ import annotations

import csv as _csv
import dataclasses
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
        train_only_teammates: bool = False,
        per_type_seeds: tuple[int, ...] | None = None,
        simulate_stranger_gap: int = 0,
        greedy_own_type_only: bool = False,
        discount_method: str = "team_steps",
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

        # Per-agent epsilon RNGs — one per agent, mapped from per_type_seeds.
        # Agent i uses the RNG for its first assigned task type.
        # None → fall back to global rng (existing behaviour).
        if per_type_seeds is not None and env.cfg.task_assignments is not None:
            import random as _random
            assert all(len(env.cfg.task_assignments[i]) == 1 for i in range(env.cfg.n_agents)), \
                "per_type_seeds requires each agent to have exactly one task type"
            self._agent_rngs: list[_random.Random] | None = [
                _random.Random(per_type_seeds[env.cfg.task_assignments[i][0]])
                for i in range(env.cfg.n_agents)
            ]
        else:
            self._agent_rngs = None

        # Precompute per-agent teammate index lists (including self) for train_only_teammates
        if train_only_teammates and env.cfg.task_assignments is not None:
            self._teammate_sets: list[list[int]] | None = [
                [j for j in range(env.cfg.n_agents)
                 if set(env.cfg.task_assignments[i]) & set(env.cfg.task_assignments[j])]
                for i in range(env.cfg.n_agents)
            ]
        else:
            self._teammate_sets = None

        self._discount_method: str = discount_method

        if discount_method == "team_steps" and self._centralized:
            raise ValueError("discount_method='team_steps' is not supported for centralized training")

        # Per-team TD trace: one _prev per team so stranger steps don't contaminate
        # team k's discount chain. Only active when train_only_teammates is set.
        if self._teammate_sets is not None and env.cfg.task_assignments is not None:
            assert all(
                len(env.cfg.task_assignments[i]) == 1 for i in range(env.cfg.n_agents)
            ), "per-team TD requires each agent to have exactly one task type"
            self._agent_team_idx: list[int] | None = [
                env.cfg.task_assignments[i][0] for i in range(env.cfg.n_agents)
            ]
            self._prev_per_team: list[Any] = [None] * env.cfg.n_task_types
            if discount_method in ("world_steps", "round_steps"):
                self._n_team_agents: list[int] = [
                    sum(1 for i in range(env.cfg.n_agents) if self._agent_team_idx[i] == k)
                    for k in range(env.cfg.n_task_types)
                ]
                self._team_step_count: list[int] = [0] * env.cfg.n_task_types
                if discount_method == "world_steps":
                    # Per-team accumulated discount factor.
                    # When team k's agent acts, eff_gamma = _gamma_accum[k] * gamma.
                    # After use, resets to gamma^simulate_stranger_gap (1.0 in T=M;
                    # gamma^n_strangers in T=1 to simulate stranger steps that don't exist).
                    # Every time any OTHER team's agent takes a move step,
                    # _gamma_accum[k] *= gamma for all k not in that team.
                    # This makes dec V_i = E[sum_t gamma^t r_i(t)] over ALL world steps,
                    # so sum_i V_i_dec ≈ V_cen.
                    self._gamma_accum_per_team: list[float] | None = [1.0] * env.cfg.n_task_types
                    self._simulate_stranger_gap: int = simulate_stranger_gap
                else:  # round_steps: no stranger accumulation needed
                    self._gamma_accum_per_team = None
                    self._simulate_stranger_gap = 0
            else:  # "team_steps": plain gamma per own-team step, no stranger accumulation
                self._gamma_accum_per_team = None
                self._simulate_stranger_gap = 0
                self._n_team_agents = None
                self._team_step_count = None
        else:
            self._agent_team_idx = None
            self._prev_per_team = None
            self._gamma_accum_per_team = None
            self._simulate_stranger_gap = 0
            self._n_team_agents = None
            self._team_step_count = None

        # Round boundary counter for centralized round_steps: tracks steps within
        # the current global round (0 = start of new round → use gamma).
        self._global_round_step_count: int = 0

        self._greedy_own_type_only = greedy_own_type_only

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

        # Debug fields populated each step for env_trace
        self._dbg_was_greedy: bool = False
        self._dbg_best_val: float = 0.0
        self._dbg_td_delta_sq: float = 0.0
        self._dbg_enc_was_cached: bool = False  # True = move enc from _cached_enc, False = fresh encode_all

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
               "n_tasks_after", "task_positions_after", "task_types_after",
               "agent_positions", "agent_positions_indexed",
               "was_greedy", "best_val", "td_delta_sq",
               "enc_grid_l2", "enc_scalar",
               "enc_ch_l2", "enc_was_cached"]
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
        teammate_indices: list[int] | None = None,
    ) -> float:
        """Run TD update. Returns sum of δ² for loss tracking.

        teammate_indices: if set, only train those agent networks; others skipped.
        """
        ...

    @abstractmethod
    def _compute_team_values(
        self, state: State, after_states: list[State],
        teammate_indices: list[int] | None = None,
    ) -> list[float]:
        """Compute sum_j V_j(after_state) for each candidate after-state.

        teammate_indices: if set, only sum over these agent indices.
        """
        ...

    def _cache_selected_enc(self, best_idx: int) -> None:
        """Optionally cache the encoding at best_idx for reuse in train_move/train_pick.

        Default no-op. GpuTrainer overrides this to slice and store the encoding
        already computed inside _compute_team_values, avoiding a redundant encode call.
        """

    def _enc_grid_l2_for_actor(self, actor: int) -> float:
        """L2 norm of actor's grid encoding from self._move (format-agnostic)."""
        if self._move is None:
            return float('nan')
        m = self._move
        try:
            if isinstance(m, list):  # CPU: list[EncoderOutput]
                enc = m[actor] if actor < len(m) else m[0]
                return enc.grid.norm().item() if enc.grid is not None else float('nan')
            else:  # GPU: (grids (N,C,H,W), scalars (N,S))
                return m[0][actor].norm().item()
        except Exception:
            return float('nan')

    def _enc_grid_per_ch_l2_for_actor(self, actor: int) -> str:
        """Per-channel L2 norms for actor's grid, comma-separated."""
        if self._move is None:
            return ''
        m = self._move
        try:
            g = m[actor].grid if isinstance(m, list) else m[0][actor]
            if g is None:
                return ''
            return ','.join(f'{g[c].norm().item():.6f}' for c in range(g.shape[0]))
        except Exception:
            return ''

    def _enc_scalar_for_actor(self, actor: int) -> str:
        """Scalar encoding for actor from self._move as comma-separated string."""
        if self._move is None:
            return ''
        m = self._move
        try:
            if isinstance(m, list):  # CPU
                enc = m[actor] if actor < len(m) else m[0]
                return ','.join(f'{x:.6f}' for x in enc.scalar.tolist()) if enc.scalar is not None else ''
            else:  # GPU
                return ','.join(f'{x:.6f}' for x in m[1][actor].cpu().tolist())
        except Exception:
            return ''

    # ------------------------------------------------------------------
    # Turn stepping
    # ------------------------------------------------------------------
    def step(self, state: State, t: int) -> State:
        self._timer.step_begin()

        _trace_actor = state.actor
        teammate_indices = self._teammate_sets[state.actor] if self._teammate_sets is not None else None

        _team_idx = self._agent_team_idx[state.actor] if self._agent_team_idx is not None else None
        if _team_idx is not None:
            self._prev = self._prev_per_team[_team_idx]
            if self._discount_method == "world_steps":
                # Compute effective gamma: accumulated stranger discount * own move step.
                _eff_gamma = self._gamma_accum_per_team[_team_idx] * self._gamma
                # Reset accumulator. The gap is only applied at the END of the team's
                # round (after all n_team_agents have acted consecutively). Mid-round
                # resets to 1.0 because no strangers acted between teammates.
                self._team_step_count[_team_idx] += 1
                if self._team_step_count[_team_idx] >= self._n_team_agents[_team_idx]:
                    # End of team's round: load simulated stranger gap for T=1 testing,
                    # or 1.0 in T=M (strangers accumulate naturally via the loop below).
                    self._gamma_accum_per_team[_team_idx] = (
                        self._gamma ** self._simulate_stranger_gap
                        if self._simulate_stranger_gap > 0 else 1.0
                    )
                    self._team_step_count[_team_idx] = 0
                else:
                    # Mid-round: teammates are about to act — no strangers in between.
                    self._gamma_accum_per_team[_team_idx] = 1.0
            elif self._discount_method == "round_steps":
                # gamma at the start of team k's sub-round (first step after strangers
                # finished); 1.0 for all subsequent steps within the sub-round.
                _eff_gamma = self._gamma if self._team_step_count[_team_idx] == 0 else 1.0
                self._team_step_count[_team_idx] += 1
                if self._team_step_count[_team_idx] >= self._n_team_agents[_team_idx]:
                    self._team_step_count[_team_idx] = 0
            else:  # "team_steps": plain gamma per own-team step, no stranger accumulation
                _eff_gamma = self._gamma
        else:
            if self._discount_method == "round_steps":
                # Centralized round_steps: gamma at first step of each global round,
                # 1.0 for all other steps within the round. Both cen and dec use this.
                _eff_gamma = self._gamma if self._global_round_step_count == 0 else 1.0
                self._global_round_step_count += 1
                if self._global_round_step_count >= self._n_agents:
                    self._global_round_step_count = 0
            else:
                _eff_gamma = self._gamma

        move_action = self.select_move(state, t)
        _trace_eps = compute_schedule_value(self._epsilon_schedule, t, self._total_steps)

        self._timer.start(TimerSection.ENV)
        s_moved = self._env.apply_action(state, move_action)
        _my_types = (
            frozenset(self._env.cfg.task_assignments[s_moved.actor])
            if self._env.cfg.task_assignments is not None else None
        )
        on_task = s_moved.is_agent_on_task(s_moved.actor, _my_types)
        self._timer.stop()

        self.train_move(s_moved, on_task, t, teammate_indices, discount=_eff_gamma)

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
            self.train_pick(s_picked, pick_rewards, t, teammate_indices)
            _trace_pick_happened = True
            _trace_pick_rewards = pick_rewards
        else:
            s_picked = s_moved

        if _team_idx is not None:
            self._prev_per_team[_team_idx] = self._prev
            if self._discount_method == "world_steps":
                # One move step just happened for team _team_idx.
                # All other teams accumulate gamma for this step — their next TD
                # update will account for this stranger step in the discount chain.
                for _k in range(len(self._gamma_accum_per_team)):
                    if _k != _team_idx:
                        self._gamma_accum_per_team[_k] *= self._gamma

        self._timer.start(TimerSection.ENV)
        _s_pre_spawn = s_picked
        _s_post_spawn = self._env.spawn_and_despawn(s_picked)
        result = self._env.advance_actor(_s_post_spawn)
        self._timer.stop()

        if self._trace_w is not None:
            _fmt = lambda ps: ";".join(f"{p.row},{p.col}" for p in sorted(ps))
            _tasks_before = set(_s_pre_spawn.task_positions)
            _tasks_after = set(_s_post_spawn.task_positions)
            _post_pairs = sorted(
                zip(_s_post_spawn.task_positions, _s_post_spawn.task_types or []),
                key=lambda x: (x[0].row, x[0].col, x[1]),
            )
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
                "task_positions_after": ";".join(f"{p.row},{p.col}" for p, _ in _post_pairs),
                "task_types_after": ";".join(str(tt) for _, tt in _post_pairs),
                "agent_positions": _fmt(result.agent_positions),
                "agent_positions_indexed": ";".join(f"{p.row},{p.col}" for p in result.agent_positions),
                "was_greedy": self._dbg_was_greedy,
                "best_val": round(self._dbg_best_val, 8),
                "td_delta_sq": round(self._dbg_td_delta_sq, 10),
                "enc_grid_l2": round(self._enc_grid_l2_for_actor(_trace_actor), 8),
                "enc_scalar": self._enc_scalar_for_actor(_trace_actor),
                "enc_ch_l2": self._enc_grid_per_ch_l2_for_actor(_trace_actor),
                "enc_was_cached": self._dbg_enc_was_cached,
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
        _arng = self._agent_rngs[state.actor] if self._agent_rngs is not None else rng
        roll = _arng.random()
        if roll < eps:
            action = actions[_arng.randint(0, len(actions) - 1)]
            self._dbg_was_greedy = False
            self._dbg_best_val = 0.0
        else:
            action = self._greedy_action(state)
            self._dbg_was_greedy = True
        self._timer.stop()
        return action

    def select_pick(self, state: State, t: int) -> Action:
        self._timer.start(TimerSection.ACTION)
        eps = compute_schedule_value(self._epsilon_schedule, t, self._total_steps)
        actions = get_phase2_actions(state, self._env.cfg)
        _arng = self._agent_rngs[state.actor] if self._agent_rngs is not None else rng
        if _arng.random() < eps:
            action = actions[_arng.randint(0, len(actions) - 1)]
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
        _actor_types = (
            frozenset(self._env.cfg.task_assignments[state.actor])
            if self._env.cfg.task_assignments is not None else None
        )
        self._timer.start(TimerSection.ACTION_ENV)
        for a in all_actions:
            if phase2 and a.is_pick():
                s_after, rewards = self._env.resolve_pick(state, pick_type=a.pick_type())
                after_states.append(s_after)
                immediate_rewards.append(sum(rewards))
            elif phase2:
                # STAY exits pick phase: after-state has pick_phase=False.
                # All pick phase actions (pick or stay) return to move phase next.
                # Using pick_phase=True here would make Q(STAY) != Q(pick(stranger))
                # under blind encoding, breaking T=1 ≡ T=M for choice pick.
                after_states.append(dataclasses.replace(state, pick_phase=False))
                immediate_rewards.append(0.0)
            else:
                s = self._env.apply_action(state, a)
                if s.is_agent_on_task(s.actor, _actor_types):
                    after_states.append(s.with_pick_phase())
                else:
                    after_states.append(s)
                immediate_rewards.append(0.0)
        self._timer.stop()

        _greedy_tm = (
            self._teammate_sets[state.actor]
            if self._greedy_own_type_only and self._teammate_sets is not None
            else None
        )
        team_values = self._compute_team_values(state, after_states, _greedy_tm)

        best_idx = 0
        best_val = team_values[0] + immediate_rewards[0]
        for k in range(1, len(all_actions)):
            val = team_values[k] + immediate_rewards[k]
            if val > best_val:
                best_val = val
                best_idx = k
        if not phase2:
            self._cache_selected_enc(best_idx)
        self._dbg_best_val = team_values[best_idx] + immediate_rewards[best_idx]
        return all_actions[best_idx]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_move(self, s_moved: State, on_task: bool, t: int,
                   teammate_indices: list[int] | None = None,
                   discount: float | None = None) -> None:
        """Encode move after-state. TD update: prev_after →[r=0, γ=eff_γ]→ move_after.

        discount: effective gamma for this step, accounting for accumulated stranger
        discount from _gamma_accum_per_team. Defaults to self._gamma if not provided.
        """
        _gamma = discount if discount is not None else self._gamma
        self._timer.start(TimerSection.ENCODE)
        if self._cached_enc is not None:
            self._move = self._cached_enc
            self._cached_enc = None
            self._dbg_enc_was_cached = True
        else:
            enc_state = s_moved.with_pick_phase() if on_task else s_moved
            self._move = self._encode_all(enc_state)
            self._dbg_enc_was_cached = False
        self._timer.stop()

        if self._prev is not None:
            self._timer.start(TimerSection.TRAIN)
            loss = self._td_step(self._prev, self._zero_rewards, _gamma,
                                 self._move, t, teammate_indices)
            n_trained = len(teammate_indices) if teammate_indices is not None else self._n_networks
            self._td_loss_accum += loss
            self._td_loss_count += n_trained
            self._dbg_td_delta_sq = loss
            self._timer.stop()
        else:
            self._dbg_td_delta_sq = 0.0

        if not on_task:
            self._prev = self._move

    def train_pick(self, s_picked: State, rewards: tuple[float, ...], t: int,
                   teammate_indices: list[int] | None = None) -> None:
        """Encode pick after-state. TD update: move_after →[r=rewards, γ=1]→ pick_after."""
        self._timer.start(TimerSection.ENCODE)
        pick_enc = self._encode_all(s_picked)
        self._timer.stop()

        train_rewards = (sum(rewards),) if self._centralized else rewards

        self._timer.start(TimerSection.TRAIN)
        loss = self._td_step(self._move, train_rewards, 1.0, pick_enc, t, teammate_indices)
        n_trained = len(teammate_indices) if teammate_indices is not None else self._n_networks
        self._td_loss_accum += loss
        self._td_loss_count += n_trained
        self._timer.stop()

        self._prev = pick_enc

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, env: BaseEnv, eval_cfg: EvalConfig) -> dict[str, float | int]:
        from orchard.eval import evaluate_policy_metrics
        env.set_eval_mode(True)
        try:
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
        finally:
            env.set_eval_mode(False)

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
