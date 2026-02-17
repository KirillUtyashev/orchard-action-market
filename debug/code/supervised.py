import json
import os
import random
import time
from pathlib import Path

import numpy as np
import multiprocessing as mp

import torch

from debug.code.forward_view import TorchRLCritic, VNet
from debug.code.main_net import MainNet
from debug.code.td_lambda import TDLambda
from utils import ten
from debug.code.controllers import ViewController
from debug.code.library_value_function import EligibilityCritic
from debug.code.monte_carlo import generate_careful_distance_series, \
    generate_initial_state_full, \
    generate_initial_state_supervised, \
    iid_supervised, monte_carlo_supervised, run
from debug.code.simple_agent import SimpleAgent
from config import (
    NUM_AGENTS,
    W,
    L,
    PROBABILITY_APPLE,
    DISCOUNT_FACTOR,
    DEVICE,
    data_dir
)

from debug.code.environment import Orchard
from debug.code.helpers import random_policy, teleport, transition
from debug.code.reward import Reward
from debug.code.value import Value
from debug.code.value_function import VNetwork
import matplotlib.pyplot as plt


def _worker_generate_state(args):
    reward_module, run_id, seed, discount_factor, p_apple, d_apple = args
    state = generate_initial_state_full(
        reward_module=reward_module,
        run_id=run_id,
        seed=seed,
        discount_factor=discount_factor,
        p_apple=p_apple,
        d_apple=d_apple,
        save=True,  # writes the npz so next run can load
    )
    return state  # you can also return (state, mc) if you want


def _worker_generate_careful(arg):
    # arg = (reward_module, seed, discount_factor, p_apple, d_apple, agent_id, distance)
    reward_module, seed, discount_factor, p_apple, d_apple, agent_id, distance = arg

    # Make a single-distance “series” generator, or call a helper that builds one init_state.
    # Here’s the clean approach: reuse your series builder but request just one distance.
    res = generate_careful_distance_series(
        reward_module=reward_module,
        seed=seed,
        discount_factor=discount_factor,
        p_apple=p_apple,
        d_apple=d_apple,
        distances=(distance,),
        self_id=agent_id
    )
    return res


def _state_path(picker_r: float, run_id: int) -> Path:
    return data_dir / "states" / "full" / f"init_state_reward_{picker_r}_{run_id}.npz"


def _load_state_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as f:  # needed for object arrays / dicts [web:350]
        return f["dict"].item()


def _careful_state_path(agent_id: int, seed: int, distance: int):
    out_dir = data_dir / "states" / "careful"
    return out_dir / f"careful_agent{agent_id}_seed{seed}_d{distance}.npz"


class Learning:
    def __init__(self, exp_config):
        self.env = None
        self.agent_controller = None
        self.input_type = None
        self.agents = []
        self.critic_networks = []
        self.reward_module = Reward(exp_config.train_config.picker_r, NUM_AGENTS)
        self.trajectory_length = exp_config.train_config.timesteps
        self.exp_config = exp_config

        self.num_eval_states = exp_config.train_config.num_eval_states

        self.theoretical_val = Value(exp_config.train_config.picker_r, NUM_AGENTS, DISCOUNT_FACTOR, PROBABILITY_APPLE, exp_config.train_config.variance)

        self._networks_for_eval = []

        self.eval_history = []  # list of {"step": int, "mae_pct_overall": float, "mae_pct_by_state": {...}}

        type_ = "supervised" if exp_config.train_config.supervised else \
            "forward" if exp_config.train_config.forward else \
                "reward_learning" if exp_config.train_config.reward_learning else \
                    "eligibility" if exp_config.train_config.eligibility else "td0"

        self.input_dim = 0
        if self.exp_config.train_config.input_dim != 3 and  self.exp_config.train_config.input_dim != 326:
            self.input_dim = 3 + 3 * NUM_AGENTS + 4 * self.exp_config.train_config.top_k_num_apples
        else:
            self.input_dim = self.exp_config.train_config.input_dim

        self.data_dir = data_dir / type_ / str(exp_config.train_config.picker_r) / str(self.input_dim)

        default_final_path = data_dir / type_ / str(exp_config.train_config.picker_r) / str(self.input_dim) / str(exp_config.train_config.variance) / f"final_eval_errors_{exp_config.train_config.hidden_dimensions}_{exp_config.train_config.num_seeds}_{exp_config.train_config.alpha}_{exp_config.train_config.schedule_lr}_{exp_config.train_config.lmda}.json"

        self.final_eval_errors_path = Path(
            getattr(exp_config.train_config, "final_eval_errors_path", default_final_path)
        )

        default_hist_path = data_dir / type_ / str(exp_config.train_config.picker_r) / str(self.input_dim) / str(exp_config.train_config.variance)

        self.mae_history_path = Path(getattr(exp_config.train_config, "mae_history_path", default_hist_path / f"mae_pct_history_{exp_config.train_config.hidden_dimensions}_{exp_config.train_config.num_seeds}_{exp_config.train_config.alpha}_{exp_config.train_config.schedule_lr}_{exp_config.train_config.lmda}.json"))
        self.coverage_plot_path = Path(getattr(exp_config.train_config, "mae_history_path", default_hist_path / f"coverage_{exp_config.train_config.hidden_dimensions}_{exp_config.train_config.num_seeds}_{exp_config.train_config.alpha}_{exp_config.train_config.schedule_lr}_{exp_config.train_config.lmda}.png"))

        default_plot_path = self.mae_history_path.with_suffix(".png")
        self.mae_plot_path = Path(getattr(exp_config.train_config, "mae_plot_path", default_plot_path))
        self._last_eval_errors_by_state = None

        self.discount_factor = DISCOUNT_FACTOR

        self.careful_evals = []

        self.focus_actor_id = 0
        self.careful_distances = (4, 1)

        # actor-0 careful states by distance (same states used to evaluate every agent)
        self.careful_actor_states = [None for _ in range(len(self.careful_distances))]

        # history: predictions of each eval-agent on the same actor-0 state at each distance
        self.careful_eval_steps = []
        self.careful_pred_history_actor0 = [
            [[] for _ in range(len(self.careful_distances))]
            for _ in range(NUM_AGENTS)
        ]

        self.careful_dir = data_dir / type_ / str(exp_config.train_config.picker_r) / "careful"
        self.careful_plot_dir = self.careful_dir / "plots"
        self.careful_json_path = self.careful_dir / "careful_eval_history.json"

    def _init_critic_networks(self):

        for _ in range(NUM_AGENTS):
            if self.exp_config.train_config.eligibility:
                self.critic_networks.append(EligibilityCritic(MainNet(self.input_dim, 1, self.exp_config.train_config.hidden_dimensions), self.exp_config.train_config.alpha, DISCOUNT_FACTOR, lambda_coeff=self.exp_config.train_config.lmda, num_training_steps=self.trajectory_length))
            elif self.exp_config.train_config.forward:
                self.critic_networks.append(TDLambda(self.input_dim, 1, self.exp_config.train_config.alpha, self.discount_factor, self.exp_config.train_config.hidden_dimensions,
                                                     self.exp_config.train_config.num_layers, self.trajectory_length, self.exp_config.train_config.schedule_lr, self.exp_config.train_config.lmda))
            else:
                self.critic_networks.append(VNetwork(self.input_dim, 1, self.exp_config.train_config.alpha, self.discount_factor,
                                                     self.exp_config.train_config.hidden_dimensions, self.exp_config.train_config.num_layers, self.trajectory_length, self.exp_config.train_config.schedule_lr))

    def _init_agents_for_training(self):
        for i in range(NUM_AGENTS):
            self.agents.append(SimpleAgent(teleport(W) if not self.exp_config.train_config.random_policy else random_policy, i, self.critic_networks[i]))

    def _generate_evaluation_states(self, p_apple, d_apple, sequential: bool = False, processes: int = 8):
        start = time.time()

        # ----- existing full-state logic unchanged -----
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
                missing_args.append((self.reward_module, i, i, self.discount_factor, p_apple, d_apple))

        if missing_args:
            if sequential:
                generated = [_worker_generate_state(arg) for arg in missing_args]
            else:
                with mp.Pool(processes=processes) as pool:
                    generated = pool.map(_worker_generate_state, missing_args)

            for idx, state in zip(missing_indices, generated):
                results[idx] = state

        self.evaluation_states = results

        # ----- NEW careful-state logic -----
        careful_dir = data_dir / "states" / "careful"
        careful_dir.mkdir(parents=True, exist_ok=True)

        distances = self.careful_distances
        careful_seed = 42069

        self.careful_evals = [[] for _ in range(NUM_AGENTS)]

        missing_args, missing_keys = [], []

        for agent_id in range(NUM_AGENTS):
            for d in distances:
                path = _careful_state_path(agent_id, careful_seed, d)
                if path.exists():
                    self.careful_evals[agent_id].append(_load_state_npz(path))   # whole dict
                else:
                    self.careful_evals[agent_id].append(None)
                    missing_keys.append((agent_id, d, path))
                    missing_args.append((self.reward_module, careful_seed, self.discount_factor, p_apple, d_apple, agent_id, d))

        if missing_args:
            if sequential:
                generated = [_worker_generate_careful(arg) for arg in missing_args]
            else:
                with mp.Pool(processes=processes) as pool:
                    generated = pool.map(_worker_generate_careful, missing_args)

            for (agent_id, d, path), item in zip(missing_keys, generated):
                j = distances.index(d)
                self.careful_evals[agent_id][j] = item

        self.careful_actor_states = self.careful_evals[0]

        end = time.time()
        print(f"Generated/loaded {num} eval states in {end - start:.3f}s")

    def _generate_evaluation_states_reward_learning(self):
        if not hasattr(self, 'evaluation_states'):
            self.evaluation_states = {
                'Z1': [],
                'Y11': [],
                'Y10': []
            }

        # State types to iterate over
        state_types = ['Z1', 'Y11', 'Y10']

        for state_type in state_types:
            for _ in range(self.num_eval_states):
                state_dict, agent_positions = generate_initial_state_supervised(self.reward_module, state_type, save=False)

                state_dict["agent_positions"] = agent_positions

                # Save to class attribute
                self.evaluation_states[state_type].append(state_dict)

    def build_experiment(self):
        # 1. Initialize our CNN critic network.
        self._init_critic_networks()

        # 2.
        self._init_agents_for_training()

        p_apple = (self.exp_config.train_config.q_agent * NUM_AGENTS) / (W ** 2)
        d_apple = 1 / self.exp_config.train_config.apple_life
        # p_apple = self.exp_config.train_config.q_agent / (W ** 2)
        # d_apple = 1 / (self.exp_config.train_config.apple_life * NUM_AGENTS)

        if not self.exp_config.train_config.reward_learning:
            self._generate_evaluation_states(p_apple, d_apple)
        else:
            self._generate_evaluation_states_reward_learning()

        # 3. Initialize OUR agent controller. ignore test flag.
        self.agent_controller = ViewController(self.input_dim, self.exp_config.train_config.top_k_num_apples)

        # 4. Create the environment.
        self.env = Orchard(
            W,
            L,
            NUM_AGENTS,
            self.reward_module,
            p_apple,
            d_apple
        )

        self.env.set_positions()

        # 5. Set up the network for evaluation, consistent with the parent class.
        self._networks_for_eval = self.critic_networks

    def step_and_collect_observation(self) -> None:
        """
        Takes the result of a single, clean environment step and adds the
        corresponding experience to the training buffer.
        """
        eval_intervals = [self.trajectory_length // 5 * (i + 1) for i in range(5)]
        if self.exp_config.train_config.reward_learning is False:
            self.evaluate_networks(step=0, plot=True, store_last=True)
        else:
            self.evaluate_networks_reward(step=0, plot=True, store_last=True)
        curr_state = None
        actor_idx = None
        for sec in range(self.trajectory_length):
            for step in range(-1, NUM_AGENTS):
                final_state, semi_state, res, actor_idx = transition(step, curr_state, self.env, actor_idx)

                if step != -1:
                    for i in range(NUM_AGENTS):
                        processed_old_state = self.agent_controller(curr_state, i)
                        processed_intermediate_state = self.agent_controller(semi_state, i)
                        processed_final_state = self.agent_controller(final_state, i)

                        if self.exp_config.train_config.reward_learning:
                            self.critic_networks[i].add_experience(
                                processed_old_state, None, 0, discount_factor=self.discount_factor
                            )
                            self.critic_networks[i].add_experience(
                                processed_intermediate_state, None, res.reward_vector[i],
                                discount_factor=self.discount_factor
                            )
                        else:
                            # Mode 0 → Mode 1: use gamma discount
                            self.critic_networks[i].add_experience(
                                processed_old_state, processed_intermediate_state, 0,
                                discount_factor=self.discount_factor  # γ
                            )
                            # Mode 1 → Mode 0: NO discount (discount = 1.0)
                            self.critic_networks[i].add_experience(
                                processed_intermediate_state, processed_final_state,
                                res.reward_vector[i],
                                discount_factor=1.0  # No discount!
                            )

                curr_state = final_state

            # Train NNs
            for i in range(NUM_AGENTS):
                if self.exp_config.train_config.supervised:
                    self.critic_networks[i].train_supervised()
                elif self.exp_config.train_config.reward_learning:
                    self.critic_networks[i].train_reward_supervised()
                else:
                    self.critic_networks[i].train()

            # Eval NNs
            if (sec + 1) in eval_intervals:
                print(f"Running evaluation at step {sec + 1}/{self.trajectory_length}")
                if self.exp_config.train_config.reward_learning is False:
                    if sec == self.trajectory_length - 1:
                        self.evaluate_networks(step=(sec + 1), plot=True, store_last=True)
                    else:
                        self.evaluate_networks(step=(sec + 1), plot=False, store_last=True)
                else:
                    if sec == self.trajectory_length - 1:
                        self.evaluate_networks_reward(step=(sec + 1), plot=True, store_last=True)
                    else:
                        self.evaluate_networks_reward(step=(sec + 1), plot=False, store_last=True)

    def evaluate_networks(self, *, step: int | None = None, plot: bool = False, store_last: bool = True):
        errors_by_agent = {i: [] for i in range(NUM_AGENTS)}
        ape_by_agent = {i: [] for i in range(NUM_AGENTS)}

        # NEW: CI coverage tracking
        in_ci_by_agent = {i: [] for i in range(NUM_AGENTS)}

        eps = 1e-8  # avoids blowups when true value is 0

        for eval_state in self.evaluation_states:
            mc_values = eval_state["mc"]

            # NEW: only if present (lets you roll this out incrementally)
            ci_low = eval_state.get("ci95_low", None)
            ci_high = eval_state.get("ci95_high", None)

            rewards = self.reward_module.get_reward(
                eval_state,
                eval_state["actor_id"],
                eval_state["agent_positions"][eval_state["actor_id"]],
                eval_state["mode"],
            )

            for i in range(NUM_AGENTS):
                input_ = self.agent_controller(eval_state, i)
                if self.exp_config.train_config.eligibility or self.exp_config.train_config.forward:
                    obs = ten(input_, DEVICE).view(-1)
                    pred = float(self.critic_networks[i].get_value_function(obs).cpu().item())
                else:
                    pred = float(self.critic_networks[i].get_value_function(input_))

                true = float(mc_values[i]) if not self.exp_config.train_config.reward_learning else float(rewards[i])
                err = true - pred
                errors_by_agent[i].append(err)

                if abs(true) > eps:
                    ape_by_agent[i].append((abs(err) / abs(true)) * 100.0)
                else:
                    ape_by_agent[i].append(abs(err))

                # NEW: coverage check (using MC CI only when mc target is used)
                if (not self.exp_config.train_config.reward_learning) and (ci_low is not None) and (ci_high is not None):
                    lo = float(ci_low[i])
                    hi = float(ci_high[i])
                    in_ci_by_agent[i].append(1.0 if (pred >= lo and pred <= hi) else 0.0)

        if store_last:
            self._last_eval_errors_by_agent = ape_by_agent

        # Overall MAE% across all agents/samples
        all_ape = [x for xs in ape_by_agent.values() for x in xs]
        mae_pct_overall = float(np.mean(all_ape)) if all_ape else float("nan")

        # Per-agent MAE%
        mae_pct_by_agent = {i: (float(np.mean(xs)) if len(xs) else None) for i, xs in ape_by_agent.items()}

        # NEW: coverage summaries (mean of 0/1 indicators = fraction inside)
        all_in_ci = [x for xs in in_ci_by_agent.values() for x in xs]
        coverage_overall = float(np.mean(all_in_ci)) if all_in_ci else None
        coverage_by_agent = {i: (float(np.mean(xs)) if len(xs) else None) for i, xs in in_ci_by_agent.items()}

        if step is not None:
            self.eval_history.append({
                "step": int(step),
                "mae_pct_overall": mae_pct_overall,
                "mae_pct_by_agent": mae_pct_by_agent,
                # NEW:
                "coverage_overall": coverage_overall,
                "coverage_by_agent": coverage_by_agent,
            })
            self._plot_mae_history(save_path=self.mae_plot_path, per_agent=True)
            # NEW:
            self._plot_coverage_history(save_path=self.coverage_plot_path, per_agent=True)

        if plot:
            self._plot_errors(errors_by_agent)

        careful_pred_this_eval = np.full(
            (NUM_AGENTS, len(self.careful_distances)),
            np.nan,
            dtype=float
        )

        for j, d in enumerate(self.careful_distances):
            item = self.careful_actor_states[j]
            if item is None:
                continue

            st = item["init_state"] if isinstance(item, dict) and "init_state" in item else item

            # sanity check: ensure we're focusing on actor 0
            if st.get("actor_id", None) != 0:
                raise RuntimeError(f"Expected actor_id=0, got {st.get('actor_id')} at distance {d}")

            # evaluate all agents' critics on this same state
            for eval_agent_id in range(NUM_AGENTS):
                input_ = self.agent_controller(st, eval_agent_id)

                if self.exp_config.train_config.eligibility or self.exp_config.train_config.forward:
                    obs = ten(input_, DEVICE).view(-1)
                    pred = float(self.critic_networks[eval_agent_id].get_value_function(obs).cpu().item())
                else:
                    pred = float(self.critic_networks[eval_agent_id].get_value_function(input_))

                careful_pred_this_eval[eval_agent_id, j] = pred

        if step is not None:
            self.careful_eval_steps.append(int(step))
            for eval_agent_id in range(NUM_AGENTS):
                for j in range(len(self.careful_distances)):
                    self.careful_pred_history_actor0[eval_agent_id][j].append(
                        float(careful_pred_this_eval[eval_agent_id, j])
                    )

            self._plot_careful_history_actor0()

        return errors_by_agent

    def evaluate_networks_reward(self, *, step: int | None = None, plot: bool = False, store_last: bool = True):
        errors_by_state = {
            "Z0": [], "Z1": [], "Y11": [], "Y10": [], "Y00": [], "Y01": []
        }
        ape_by_state = {k: [] for k in errors_by_state.keys()}  # absolute % error per sample

        eps = 1e-8  # avoids blowups when true value is 0

        for state in self.evaluation_states:
            for eval_state in self.evaluation_states[state]:
                theoretical_values = self.theoretical_val.theoretical_value(
                    eval_state, eval_state["actor_id"], eval_state["agent_positions"], eval_=True
                )
                rewards = self.reward_module.get_reward(
                    eval_state,
                    eval_state["actor_id"],
                    eval_state["agent_positions"][eval_state["actor_id"]],
                    eval_state["mode"],
                )

                for i in range(NUM_AGENTS):
                    input_ = self.agent_controller(eval_state, i)
                    if self.exp_config.train_config.eligibility or self.exp_config.train_config.forward:
                        obs = ten(input_, DEVICE).view(-1)
                        pred = float(self.critic_networks[i].get_value_function(obs).cpu().item())
                    else:
                        pred = float(self.critic_networks[i].get_value_function(input_))

                    if not self.exp_config.train_config.reward_learning:
                        true = float(theoretical_values[i])
                    else:
                        true = float(rewards[i])

                    err = true - pred

                    agent_state = state
                    if eval_state["actor_id"] != i:
                        if state == "Z1":
                            agent_state = "Z0"
                        elif state == "Y11":
                            agent_state = "Y01"
                        else:
                            agent_state = "Y00"

                    errors_by_state[agent_state].append(err)

                    if abs(true) > eps:
                        ape_by_state[agent_state].append((abs(err) / abs(true)) * 100.0)
                    else:
                        ape_by_state[agent_state].append(abs(err))

        if store_last:
            self._last_eval_errors_by_state = ape_by_state

        # ---- NEW: compute MAE% (mean absolute % error) summary ----
        mae_pct_by_state = {}
        all_apes = []
        for k, v in ape_by_state.items():
            if len(v) == 0:
                mae_pct_by_state[k] = None
            else:
                mae_pct_by_state[k] = float(np.mean(v))
                all_apes.extend(v)

        mae_pct_overall = float(np.mean(all_apes)) if len(all_apes) > 0 else None

        if step is not None:
            self.eval_history.append({
                "step": int(step),
                "mae_pct_overall": mae_pct_overall,
                "mae_pct_by_state": mae_pct_by_state,
            })
            # Only overall MAE curve by default
            self._plot_mae_history_reward(save_path=self.mae_plot_path, per_state=False)

        if plot:
            self._plot_errors(errors_by_state)

        return errors_by_state

    def _plot_coverage_history(self, save_path: str, per_agent: bool = True):
        if not self.eval_history:
            return

        steps = np.array([h["step"] for h in self.eval_history], dtype=int)
        cov = np.array([
            (h.get("coverage_overall", np.nan) if h.get("coverage_overall", None) is not None else np.nan)
            for h in self.eval_history
        ], dtype=float)

        plt.figure(figsize=(8, 4))
        plt.plot(steps, cov, label="Overall coverage", linewidth=2)
        plt.ylim(0.0, 1.0)
        plt.xlabel("Step")
        plt.ylabel("Fraction inside MC 95% CI")
        plt.title("NN coverage vs MC 95% CI")
        plt.grid(True, alpha=0.3)

        if per_agent:
            for i in range(NUM_AGENTS):
                series = []
                for h in self.eval_history:
                    by_agent = h.get("coverage_by_agent", None) or {}
                    series.append(by_agent.get(i, np.nan))
                series = np.array(series, dtype=float)
                plt.plot(steps, series, alpha=0.5, linewidth=1, label=f"Agent {i}")

        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _plot_careful_history_actor0(self):
        if not self.careful_eval_steps:
            return

        plot_dir = getattr(self, "careful_plot_dir", None)
        if plot_dir is None:
            plot_dir = (data_dir / "careful" / "plots")
        plot_dir.mkdir(parents=True, exist_ok=True)

        x = list(self.careful_eval_steps)
        distances = self.careful_distances
        hist = self.careful_pred_history_actor0

        for eval_agent_id in range(NUM_AGENTS):
            plt.figure(figsize=(8, 4.5))

            for j, d in enumerate(distances):
                y = hist[eval_agent_id][j]
                if not y:
                    continue
                plt.plot(x, y, marker="o", linewidth=1.5, label=f"d={d}")

            plt.xlabel("Training step (evaluation point)")
            plt.ylabel("Predicted value")
            plt.title(f"Actor=0 state predictions over time (Eval agent {eval_agent_id})")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=9, ncol=3)

            out = plot_dir / f"careful_actor0_predictions_evalagent{eval_agent_id}.png"
            plt.tight_layout()
            plt.savefig(out, dpi=250, bbox_inches="tight")
            plt.close()

    def _plot_mae_history(self, save_path: Path, *, per_agent: bool = False):
        if len(self.eval_history) == 0:
            return

        steps = [h["step"] for h in self.eval_history]
        overall = [h["mae_pct_overall"] for h in self.eval_history]

        plt.figure(figsize=(9, 5))
        plt.plot(steps, overall, marker="o", label="Overall MAE%")

        if per_agent:
            for i in range(NUM_AGENTS):
                ys, ok_steps = [], []
                for h in self.eval_history:
                    y = h.get("mae_pct_by_agent", {}).get(i, None)
                    if y is not None:
                        ok_steps.append(h["step"])
                        ys.append(y)
                if ys:
                    plt.plot(ok_steps, ys, marker=".", linewidth=1, alpha=0.6, label=f"Agent {i}")

        plt.xlabel("Training step (evaluation point)")
        plt.ylabel("MAE % of true value")
        plt.title("Evaluation MAE% over training")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=3 if per_agent else 1, fontsize=9)

        plt.ylim(bottom=0)
        ymax = max([v for v in overall if v == v], default=0.0)  # ignore NaNs
        if per_agent:
            for h in self.eval_history:
                for v in h.get("mae_pct_by_agent", {}).values():
                    if v is not None:
                        ymax = max(ymax, float(v))

        top = int(np.ceil(ymax / 10.0) * 10) if ymax > 0 else 10
        plt.yticks(np.arange(0, top + 1, 10))

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close()

    def _plot_mae_history_reward(self, save_path: Path, *, per_state: bool = False):
        if len(self.eval_history) == 0:
            return

        steps = [h["step"] for h in self.eval_history]
        overall = [h["mae_pct_overall"] for h in self.eval_history]

        plt.figure(figsize=(9, 5))
        plt.plot(steps, overall, marker="o", label="Overall MAE%")

        if per_state:
            state_types = ["Z0", "Z1", "Y11", "Y10", "Y00", "Y01"]
            for st in state_types:
                ys, ok_steps = [], []
                for h in self.eval_history:
                    y = h["mae_pct_by_state"].get(st, None)
                    if y is not None:
                        ok_steps.append(h["step"])
                        ys.append(y)
                if ys:
                    plt.plot(ok_steps, ys, marker=".", linewidth=1, alpha=0.6, label=st)

        plt.xlabel("Training step (evaluation point)")
        plt.ylabel("MAE % of true value")
        plt.title("Evaluation MAE% over training")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=3 if per_state else 1, fontsize=9)

        # ---- add this ----
        plt.ylim(bottom=0)  # ensure 0 is visible [web:223]
        ymax = max(overall) if overall else 0.0
        if per_state:
            for h in self.eval_history:
                for v in h.get("mae_pct_by_state", {}).values():
                    if v is not None:
                        ymax = max(ymax, float(v))
        top = int(np.ceil(ymax / 10.0) * 10)  # next multiple of 10
        plt.yticks(np.arange(0, top + 1, 10))  # 0,10,20,... [web:219]
        # -------------------

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close()

    def save_final_evaluation_errors(self):
        """Save ONLY the most recent evaluation errors to disk (JSON)."""
        if getattr(self, "_last_eval_errors_by_agent", None) is None:
            raise RuntimeError("No evaluation has been run yet; cannot save final errors.")

        self.final_eval_errors_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "errors_by_agent": self._last_eval_errors_by_agent,
            "counts_by_agent": {str(k): len(v) for k, v in self._last_eval_errors_by_agent.items()},
        }
        with open(self.final_eval_errors_path, "w") as f:
            json.dump(payload, f)

        print(f"Final evaluation errors saved to: {self.final_eval_errors_path}")

    def train(self):
        self.build_experiment()
        self.step_and_collect_observation()
        # Final evaluation at the last step
        # self.save_final_evaluation_errors()

        self.save_networks(self.data_dir / "weights")

        # Save MAE% history to JSON (optional but useful)
        self.mae_history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.mae_history_path, "w") as f:
            json.dump({"eval_history": self.eval_history}, f)
        print(f"MAE% history saved to: {self.mae_history_path}")
        print(f"MAE% plot saved to: {self.mae_plot_path}")

        # ---- NEW: return the history to the caller ----
        return self.eval_history

    def _plot_errors(self, errors_by_agent):
        import matplotlib.pyplot as plt
        from datetime import datetime
        import math

        plots_dir = data_dir / "supervised"
        plots_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        agent_ids = sorted(errors_by_agent.keys())
        n_agents = len(agent_ids)

        n_cols = min(3, n_agents) if n_agents > 0 else 1
        n_rows = int(math.ceil(n_agents / n_cols)) if n_agents > 0 else 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(-1)

        for j, agent_id in enumerate(agent_ids):
            ax = axes[j]
            errors = np.asarray(errors_by_agent[agent_id], dtype=float)

            if errors.size > 0:
                mean_error = float(np.mean(errors))
                std_error = float(np.std(errors))

                ax.hist(errors, bins=30, alpha=0.7, color=f"C{j % 10}", edgecolor="black", linewidth=0.5)
                ax.axvline(mean_error, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_error:.4f}")
                ax.axvline(mean_error + std_error, color="orange", linestyle=":", linewidth=1.5, alpha=0.7)
                ax.axvline(mean_error - std_error, color="orange", linestyle=":", linewidth=1.5, alpha=0.7)

                stats_text = f"Mean: {mean_error:.4f}\nStd: {std_error:.4f}\nN: {errors.size}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

                ax.set_title(f"Agent {agent_id} Error Distribution")
                ax.set_xlabel("Error (True - Predicted)")
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f"No data for\nAgent {agent_id}",
                        transform=ax.transAxes, ha="center", va="center", fontsize=12)
                ax.set_title(f"Agent {agent_id} Error Distribution")

        # Hide unused subplots
        for k in range(n_agents, len(axes)):
            axes[k].axis("off")

        plt.tight_layout()
        plot_path = plots_dir / f"error_distributions_by_agent_{timestamp}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Error distribution plots saved to: {plots_dir}")

    def save_networks(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        print("Saving networks: ", random.getstate()[1][0])

        payload = {
            "critics": [],  # list of {name, blob}
        }

        # critics (unique, deduped)
        for crit in self.critic_networks:
            payload["critics"].append({"blob": crit.export_net_state()})

        dst = os.path.join(path, f"weights_{self.exp_config.train_config.hidden_dimensions}_{self.exp_config.train_config.alpha}_{self.exp_config.train_config.lmda}.pt")
        torch.save(payload, dst)
