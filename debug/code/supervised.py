import json
import random
from pathlib import Path

import numpy as np
from config import data_dir
from debug.code.controllers import ViewController
from debug.code.monte_carlo import generate_initial_state_supervised
from debug.code.simple_agent import SimpleAgent
from config import (
    NUM_AGENTS,
    W,
    L,
    PROBABILITY_APPLE,
    DISCOUNT_FACTOR
)

from debug.code.environment import Orchard
from debug.code.helpers import teleport
from debug.code.reward import Reward
from debug.code.value import Value
from models.value_function import VNetwork


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

        self.theoretical_val = Value(exp_config.train_config.picker_r, NUM_AGENTS, DISCOUNT_FACTOR, PROBABILITY_APPLE)

        self._networks_for_eval = []

        default_final_path = data_dir / "supervised" / str(exp_config.train_config.picker_r) / str(self.exp_config.train_config.input_dim) / f"final_eval_errors_{exp_config.train_config.hidden_dimensions}.json"
        self.final_eval_errors_path = Path(
            getattr(exp_config.train_config, "final_eval_errors_path", default_final_path)
        )
        self._last_eval_errors_by_state = None

    def _init_critic_networks(self):
        for _ in range(NUM_AGENTS):
            self.critic_networks.append(VNetwork(self.exp_config.train_config.input_dim, 1, self.exp_config.train_config.alpha, DISCOUNT_FACTOR,
                                                 self.exp_config.train_config.hidden_dimensions, self.exp_config.train_config.num_layers, supervised=self.exp_config.train_config.supervised))

    def _init_agents_for_training(self):
        for i in range(NUM_AGENTS):
            self.agents.append(SimpleAgent(teleport(W), i, self.critic_networks[i]))

    def _generate_evaluation_states(self):
        # Initialize class attribute to store states
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

        self._generate_evaluation_states()

        # 3. Initialize OUR agent controller. ignore test flag.
        self.agent_controller = ViewController()

        # 4. Create the environment.
        self.env = Orchard(
            W,
            L,
            NUM_AGENTS,
            self.reward_module,
            PROBABILITY_APPLE,
        )

        self.env.set_positions()

        # 5. Set up the network for evaluation, consistent with the parent class.
        self._networks_for_eval = self.critic_networks

    def step_and_collect_observation(self) -> None:
        """
        Takes the result of a single, clean environment step and adds the
        corresponding experience to the training buffer.
        """
        curr_state = self.env.get_state()
        actor_idx = random.randint(0, NUM_AGENTS - 1)
        curr_state["actor_id"] = actor_idx
        curr_state["mode"] = 0

        eval_intervals = [self.trajectory_length // 5 * (i + 1) for i in range(5)]

        for step in range(self.trajectory_length):
            self.env.process_action(actor_idx, teleport(W), mode=0)

            semi_state = self.env.get_state()
            semi_state["actor_id"] = actor_idx
            semi_state["mode"] = 1

            res = self.env.process_action(actor_idx, None, mode=1)

            final_state = self.env.get_state()
            actor_idx = random.randint(0, NUM_AGENTS - 1)
            final_state["actor_id"] = actor_idx
            final_state["mode"] = 0

            theoretical_values_z = self.theoretical_val.theoretical_value(curr_state, curr_state["actor_id"], curr_state["agent_positions"])
            theoretical_values_y = self.theoretical_val.theoretical_value(semi_state, semi_state["actor_id"], semi_state["agent_positions"])

            for i in range(NUM_AGENTS):
                processed_old_state, processed_intermediate_state, processed_final_state = self.agent_controller(curr_state, i), self.agent_controller(semi_state, i), self.agent_controller(final_state, i)
                if self.exp_config.train_config.supervised:
                    self.critic_networks[i].add_experience(processed_old_state, None, None, theoretical_values_z[i])
                    self.critic_networks[i].add_experience(processed_intermediate_state, None, None, theoretical_values_y[i])
                    self.critic_networks[i].train_supervised()
                elif self.exp_config.train_config.reward_learning:
                    self.critic_networks[i].add_experience(processed_old_state, None, 0)
                    # self.critic_networks[i].train_reward_supervised()

                    self.critic_networks[i].add_experience(processed_intermediate_state, None, res.reward_vector[i])
                    self.critic_networks[i].train_reward_supervised()
                else:
                    self.critic_networks[i].add_experience(processed_old_state, processed_intermediate_state, 0)
                    self.critic_networks[i].add_experience(processed_intermediate_state, processed_final_state, res.reward_vector[i])
                    self.critic_networks[i].train()

            curr_state = final_state

            if (step + 1) in eval_intervals:
                print(f"Running evaluation at step {step + 1}/{self.trajectory_length}")
                self.evaluate_networks()

    def evaluate_networks(self, *, plot: bool = False, store_last: bool = True):
        errors_by_state = {
            "Z0": [], "Z1": [], "Y11": [], "Y10": [], "Y00": [], "Y01": []
        }

        for state in self.evaluation_states:
            for eval_state in self.evaluation_states[state]:
                theoretical_values = self.theoretical_val.theoretical_value(
                    eval_state, eval_state["actor_id"], eval_state["agent_positions"]
                )
                rewards = self.reward_module.get_reward(
                    eval_state,
                    eval_state["actor_id"],
                    eval_state["agent_positions"][eval_state["actor_id"]],
                    eval_state["mode"],
                )

                for i in range(NUM_AGENTS):
                    input_ = self.agent_controller(eval_state, i)
                    pred = self.critic_networks[i].get_value_function(input_)

                    if not self.exp_config.train_config.reward_learning:
                        error = theoretical_values[i] - pred
                    else:
                        error = rewards[i] - pred

                    agent_state = state
                    if eval_state["actor_id"] != i:
                        if state == "Z1":
                            agent_state = "Z0"
                        elif state == "Y11":
                            agent_state = "Y01"
                        else:
                            agent_state = "Y00"

                    errors_by_state[agent_state].append(float(error.item()))

        if store_last:
            self._last_eval_errors_by_state = errors_by_state

        if plot:
            self._plot_errors(errors_by_state)

        return errors_by_state

    def save_final_evaluation_errors(self):
        """Save ONLY the most recent evaluation errors to disk (JSON)."""
        if self._last_eval_errors_by_state is None:
            raise RuntimeError("No evaluation has been run yet; cannot save final errors.")

        self.final_eval_errors_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "errors_by_state": self._last_eval_errors_by_state,
            "counts_by_state": {k: len(v) for k, v in self._last_eval_errors_by_state.items()},
        }
        with open(self.final_eval_errors_path, "w") as f:
            json.dump(payload, f)

        print(f"Final evaluation errors saved to: {self.final_eval_errors_path}")

    def train(self):
        self.build_experiment()
        self.step_and_collect_observation()

        # Run ONE final evaluation and save it.
        self.evaluate_networks(plot=True, store_last=True)
        self.save_final_evaluation_errors()

    def _plot_errors(self, errors_by_state):
        import matplotlib.pyplot as plt
        from datetime import datetime

        # Create supervised plots directory
        plots_dir = data_dir / "supervised"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Get current timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create a figure with subplots for each state type
        state_types = ["Z0", "Z1", "Y11", "Y10", "Y00", "Y01"]
        n_states = len(state_types)

        # Calculate grid dimensions for subplots
        n_cols = 3
        n_rows = 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        axes = axes.flatten()

        # Plot histogram for each state type
        for i, state_type in enumerate(state_types):
            ax = axes[i]
            errors = errors_by_state[state_type]

            if len(errors) > 0:
                errors = np.array(errors)

                # Calculate statistics
                mean_error = np.mean(errors)
                std_error = np.std(errors)

                # Create histogram
                ax.hist(errors, bins=30, alpha=0.7, color=f'C{i}', edgecolor='black', linewidth=0.5)

                # Add vertical lines for mean
                ax.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.4f}')
                ax.axvline(mean_error + std_error, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
                ax.axvline(mean_error - std_error, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)

                # Add statistics text box
                stats_text = f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}\nN: {len(errors)}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                ax.set_title(f'State {state_type} Error Distribution')
                ax.set_xlabel('Error (Theoretical - Predicted)')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)

            else:
                # No data for this state
                ax.text(0.5, 0.5, f'No data for\nState {state_type}',
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title(f'State {state_type} Error Distribution')

        plt.tight_layout()

        # Save the combined plot
        plot_filename = f"error_distributions_{timestamp}.png"
        plot_path = plots_dir / plot_filename
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Also create individual plots for each state type
        for state_type in state_types:
            errors = errors_by_state[state_type]

            if len(errors) > 0:
                errors = np.array(errors)
                mean_error = np.mean(errors)
                std_error = np.std(errors)

                fig, ax = plt.subplots(figsize=(8, 6))

                # Create histogram
                ax.hist(errors, bins=30, alpha=0.7, color=f'C{state_types.index(state_type)}',
                       edgecolor='black', linewidth=0.5)

                # Add vertical lines for mean and std
                ax.axvline(mean_error, color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {mean_error:.4f}')
                ax.axvline(mean_error + std_error, color='orange', linestyle=':',
                          linewidth=1.5, alpha=0.7, label=f'+1 Std')
                ax.axvline(mean_error - std_error, color='orange', linestyle=':',
                          linewidth=1.5, alpha=0.7, label=f'-1 Std')

                # Add statistics text box
                stats_text = f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}\nN: {len(errors)}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                ax.set_title(f'State {state_type} Error Distribution\n(Theoretical Value - Predicted Value)')
                ax.set_xlabel('Error')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                ax.legend()

                plt.tight_layout()

                # Save individual plot
                individual_filename = f"error_distribution_{state_type}_{timestamp}.png"
                individual_path = plots_dir / individual_filename
                fig.savefig(individual_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

        print(f"Error distribution plots saved to: {plots_dir}")

        # Print summary statistics
        print("\nError Statistics Summary:")
        print("-" * 50)
        for state_type in state_types:
            errors = errors_by_state[state_type]
            if len(errors) > 0:
                errors = np.array(errors)
                mean_error = np.mean(errors)
                std_error = np.std(errors)
                print(f"{state_type:>4}: Mean = {mean_error:>8.4f}, Std = {std_error:>8.4f}, N = {len(errors):>4}")
            else:
                print(f"{state_type:>4}: No data")
