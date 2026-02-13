import json
import random
import time
from pathlib import Path

import numpy as np
import multiprocessing as mp
from debug.code.forward_view import TorchRLCritic, VNet
from debug.code.main_net import MainNet
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
    # res is a list with one element
    return res[0]


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

        type_ = "supervised" if exp_config.train_config.supervised else "forward" if exp_config.train_config.forward else "eligibility" if exp_config.train_config.eligibility else "td0"

        default_final_path = data_dir / type_ / str(exp_config.train_config.picker_r) / str(self.exp_config.train_config.input_dim) / str(exp_config.train_config.variance) / f"final_eval_errors_{exp_config.train_config.hidden_dimensions}_{exp_config.train_config.num_seeds}_{exp_config.train_config.alpha}_{exp_config.train_config.schedule_lr}_{exp_config.train_config.lmda}.json"
        self.final_eval_errors_path = Path(
            getattr(exp_config.train_config, "final_eval_errors_path", default_final_path)
        )

        default_hist_path = data_dir / type_ / str(exp_config.train_config.picker_r) / str(self.exp_config.train_config.input_dim) / str(exp_config.train_config.variance) / f"mae_pct_history_{exp_config.train_config.hidden_dimensions}_{exp_config.train_config.num_seeds}_{exp_config.train_config.alpha}_{exp_config.train_config.schedule_lr}_{exp_config.train_config.lmda}.json"

        self.mae_history_path = Path(getattr(exp_config.train_config, "mae_history_path", default_hist_path))

        default_plot_path = self.mae_history_path.with_suffix(".png")
        self.mae_plot_path = Path(getattr(exp_config.train_config, "mae_plot_path", default_plot_path))
        self._last_eval_errors_by_state = None

        self.discount_factor = DISCOUNT_FACTOR ** (1 / (2 * NUM_AGENTS))

        self.careful_evals = []

        self.careful_eval_steps = []

        self.careful_distances = (4, 0)
        self.careful_ape_history = [
            [[] for _ in range(len(self.careful_distances))]
            for _ in range(NUM_AGENTS)
        ]
        type_ = "supervised" if exp_config.train_config.supervised else \
            "forward" if exp_config.train_config.forward else \
                "eligibility" if exp_config.train_config.eligibility else "td0"

        self.careful_dir = data_dir / type_ / str(exp_config.train_config.picker_r) / "careful"
        self.careful_plot_dir = self.careful_dir / "plots"
        self.careful_json_path = self.careful_dir / "careful_eval_history.json"

    def _init_critic_networks(self):
        for _ in range(NUM_AGENTS):
            if self.exp_config.train_config.eligibility:
                self.critic_networks.append(EligibilityCritic(MainNet(self.exp_config.train_config.input_dim, 1, self.exp_config.train_config.hidden_dimensions), self.exp_config.train_config.alpha, DISCOUNT_FACTOR, lambda_coeff=self.exp_config.train_config.lmda, num_training_steps=self.trajectory_length))
            elif self.exp_config.train_config.forward:
                self.critic_networks.append(TorchRLCritic(VNet(self.exp_config.train_config.input_dim, self.exp_config.train_config.hidden_dimensions), self.exp_config.train_config.alpha, DISCOUNT_FACTOR, 1000, num_training_steps=self.trajectory_length, lambda_coeff=self.exp_config.train_config.lmda))
            else:
                self.critic_networks.append(VNetwork(self.exp_config.train_config.input_dim, 1, self.exp_config.train_config.alpha, self.discount_factor,
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

        end = time.time()
        print(f"Generated/loaded {num} eval states in {end - start:.3f}s")

    def build_experiment(self):
        # 1. Initialize our CNN critic network.
        self._init_critic_networks()

        # 2.
        self._init_agents_for_training()

        p_apple = self.exp_config.train_config.q_agent / (W ** 2)
        d_apple = 1 / (self.exp_config.train_config.apple_life * NUM_AGENTS)

        self._generate_evaluation_states(p_apple, d_apple)

        # 3. Initialize OUR agent controller. ignore test flag.
        self.agent_controller = ViewController(self.exp_config.train_config.input_dim, 1)

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

        self.evaluate_networks(step=0, plot=True, store_last=True)
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
                            self.critic_networks[i].add_experience(processed_old_state, None, 0)
                            self.critic_networks[i].add_experience(processed_intermediate_state, None, res.reward_vector[i])
                        else:
                            self.critic_networks[i].add_experience(processed_old_state, processed_intermediate_state, 0)
                            self.critic_networks[i].add_experience(processed_intermediate_state, processed_final_state, res.reward_vector[i])

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
                if sec == self.trajectory_length - 1:
                    self.evaluate_networks(step=(sec + 1), plot=True, store_last=True)
                else:
                    self.evaluate_networks(step=(sec + 1), plot=False, store_last=True)

    def evaluate_networks(self, *, step: int | None = None, plot: bool = False, store_last: bool = True):
        errors_by_agent = {i: [] for i in range(NUM_AGENTS)}
        ape_by_agent = {i: [] for i in range(NUM_AGENTS)}

        eps = 1e-8  # avoids blowups when true value is 0

        for eval_state in self.evaluation_states:
            mc_values = eval_state["mc"]
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

        if store_last:
            self._last_eval_errors_by_agent = ape_by_agent

        # Overall MAE% across all agents/samples
        all_ape = [x for xs in ape_by_agent.values() for x in xs]
        mae_pct_overall = float(np.mean(all_ape)) if all_ape else float("nan")

        # Per-agent MAE%
        mae_pct_by_agent = {
            i: (float(np.mean(xs)) if len(xs) else None)
            for i, xs in ape_by_agent.items()
        }

        if step is not None:
            self.eval_history.append({
                "step": int(step),
                "mae_pct_overall": mae_pct_overall,
                "mae_pct_by_agent": mae_pct_by_agent,
            })
            self._plot_mae_history(save_path=self.mae_plot_path, per_agent=True)

        if plot:
            self._plot_errors(errors_by_agent)

        careful_pred_this_eval = np.full(
            (NUM_AGENTS, len(self.careful_distances)),
            np.nan,
            dtype=float
        )

        for agent_id in range(NUM_AGENTS):
            for j, d in enumerate(self.careful_distances):
                item = self.careful_evals[agent_id][j]
                if item is None:
                    continue

                st = item["init_state"] if isinstance(item, dict) and "init_state" in item else item

                input_ = self.agent_controller(st, agent_id)
                if self.exp_config.train_config.eligibility or self.exp_config.train_config.forward:
                    obs = ten(input_, DEVICE).view(-1)
                    pred = float(self.critic_networks[agent_id].get_value_function(obs).cpu().item())
                else:
                    pred = float(self.critic_networks[agent_id].get_value_function(input_))

                careful_pred_this_eval[agent_id, j] = pred

        if step is not None:
            self.careful_eval_steps.append(int(step))
            for agent_id in range(NUM_AGENTS):
                for j in range(len(self.careful_distances)):
                    self.careful_ape_history[agent_id][j].append(float(careful_pred_this_eval[agent_id, j]))

            self._plot_careful_history()  # now plots raw predictions

        return errors_by_agent

    def _plot_careful_history(self):
        """
        Plot raw predictions for careful states across eval steps.

        Expects a nested list structure:
          hist[agent_id][dist_idx] = list of predicted scalars over evaluations
        and:
          self.careful_eval_steps = list of eval steps (same length as each series).
        """
        if not getattr(self, "careful_eval_steps", None):
            return

        # Use pred history if you created it; otherwise reuse the existing container
        hist = getattr(self, "careful_pred_history", None)
        if hist is None:
            hist = getattr(self, "careful_ape_history", None)
        if hist is None:
            raise RuntimeError("No careful prediction history found (need careful_pred_history or careful_ape_history).")

        distances = getattr(self, "careful_distances", None)
        if distances is None:
            # fallback if you hard-coded elsewhere
            distances = (4, 3, 2, 1, 0)

        plot_dir = getattr(self, "careful_plot_dir", None)
        if plot_dir is None:
            plot_dir = (data_dir / "careful" / "plots")
        plot_dir.mkdir(parents=True, exist_ok=True)

        x = list(self.careful_eval_steps)

        for agent_id in range(NUM_AGENTS):
            plt.figure(figsize=(8, 4.5))

            for j, d in enumerate(distances):
                y = hist[agent_id][j]
                if not y:
                    continue

                # If any NaNs, matplotlib will break the line automatically
                plt.plot(x, y, marker="o", linewidth=1.5, label=f"d={d}")

            plt.xlabel("Training step (evaluation point)")
            plt.ylabel("Predicted value")
            plt.title(f"Careful-state predictions vs distance (Agent {agent_id})")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=9, ncol=3)

            out = plot_dir / f"careful_predictions_agent{agent_id}.png"
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
        self.save_final_evaluation_errors()

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
