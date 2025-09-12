import os
from pathlib import Path
from collections import deque

from agents.simple_agent import SimpleAgent
from algorithm import Algorithm
from config import CHECKPOINT_DIR
from matplotlib import pyplot as plt
from actor_critic.actor_critic import ActorCritic
from agents.actor_critic_agent import ACAgentRates, ACAgentRatesFixed
from configs.config import ExperimentConfig
import numpy as np
from helpers.helpers import get_discounted_value
from helpers.controllers import AgentControllerActorCriticRates, \
    AgentControllerActorCriticRatesAdvantage, \
    AgentControllerActorCriticRatesFixed, AgentControllerCentralized, \
    ViewController
from plots import plot_smoothed
from models.actor_network import ActorNetwork
from models.value_function import VNetwork

linestyles = ["-", "--", "-.", ":"]


def plot_agent_rewards(collected_rewards_over_time,
                       title="Collected rewards over time",
                       xlabel="Step",
                       ylabel="Reward",
                       out_path=None,  # e.g. "graphs/rewards_over_time.png"
                       show=True):
    """
    Plot all agent reward series from a (T, N) array where:
      T = number of time steps (rows)
      N = number of agents (columns).
    """
    arr = np.asarray(collected_rewards_over_time)
    if arr.size == 0 or arr.shape[0] == 0:
        print("Nothing to plot: array is empty (no time steps).")
        return

    T, N = arr.shape
    x = np.arange(T)

    plt.figure(figsize=(10, 5))
    for j in range(N):
        y = arr[:, j]
        # Handle NaNs gracefully so a single NaN doesn't break the plot
        if np.isnan(y).all():
            continue
        plt.plot(x, y, label=f"Agent {j}", linestyle=linestyles[j % len(linestyles)])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")

    if show:
        plt.show()
    else:
        plt.close()


def load_alphas(filepath: str):
    """
    Load a .npz file saved with save_follow_rates_arrays() and
    reconstruct the same dict-of-arrays structure.

    Returns:
        foll_rate_hist : dict[int, np.ndarray]
            Mapping agent_id -> (timesteps, n_agents) array
    """
    data = np.load(filepath, allow_pickle=False)
    alpha_hist = {}

    for key in data.files:  # e.g. "agent_0", "agent_1", ...
        agent_id = int(key.split("_")[1])
        alpha_hist[agent_id] = data[key]

    return alpha_hist


def load_follow_rates(filepath: str):
    """
    Load a .npz file saved with save_follow_rates_arrays() and
    reconstruct the same dict-of-arrays structure.

    Returns:
        foll_rate_hist : dict[int, np.ndarray]
            Mapping agent_id -> (timesteps, n_agents) array
    """
    data = np.load(filepath, allow_pickle=False)
    foll_rate_hist = {}

    for key in data.files:  # e.g. "agent_0", "agent_1", ...
        agent_id = int(key.split("_")[1])
        foll_rate_hist[agent_id] = data[key]

    return foll_rate_hist


class ActorCriticRates(ActorCritic):
    def __init__(self, config: ExperimentConfig, name=None):
        if name is None:
            name = f"""ActorCriticRates-<{config.train_config.num_agents}>_agents-_length-<{config.env_config.length}>_width-<{config.env_config.width}>_s_target-<{config.env_config.s_target}>-alpha-<{config.train_config.alpha}>-apple_mean_lifetime-<{config.env_config.apple_mean_lifetime}>-<{config.train_config.hidden_dimensions}>-<{config.train_config.num_layers}>-vision-<{config.train_config.vision}>-batch_size-<{config.train_config.batch_size}>-actor_alpha-<{config.train_config.actor_alpha}>-actor_hidden-<{config.train_config.hidden_dimensions_actor}>-actor_layers-<{config.train_config.num_layers_actor}>-beta-<{config.train_config.beta_rate}>-budget-<{config.train_config.budget}>"""
        super().__init__(config, name)

        N = self.train_config.num_agents

        # === NEW: lightweight RAM buffers (no vstack during training) ===
        # Use float16 for 4x smaller rows; tune if you prefer float32.
        self._hist_dtype = np.float16
        self._log_every = 100                   # (2) log every 100 steps
        self._hist_window = 20000               # rolling window to bound RAM; adjust as needed

        self.foll_rate_hist = {i: deque(maxlen=self._hist_window) for i in range(N)}
        self.agent_distance_hist = {i: deque(maxlen=self._hist_window) for i in range(N)}
        self.alpha_ema = {i: deque(maxlen=self._hist_window) for i in range(N)}
        self.collected_rewards_over_time = deque(maxlen=self._hist_window)

    # === helpers to append one row efficiently ===
    def _append_row(self, dq: deque, row_like):
        arr = np.asarray(row_like, dtype=self._hist_dtype).reshape(1, -1)
        dq.append(arr)

    def _record_rates(self, step: int, agent_i: int, agent_rates, alphas):
        """Append a snapshot only on logging steps."""
        if step % self._log_every != 0:
            return
        if len(agent_rates) != self.train_config.num_agents or len(alphas) != self.train_config.num_agents:
            raise ValueError("agent_rates/alphas must have length num_agents")
        self._append_row(self.foll_rate_hist[agent_i], agent_rates)
        self._append_row(self.alpha_ema[agent_i], alphas)

    def _save_agent_distances(self, step: int, agent_i: int):
        if step % self._log_every != 0:
            return
        # distances to all agents (1 x N row)
        distances = [np.linalg.norm(agent.position - self.agents_list[agent_i].position)
                     for agent in self.agents_list]
        self._append_row(self.agent_distance_hist[agent_i], distances)

    def _record_collected_rewards(self, step: int):
        if step % self._log_every != 0:
            return
        # record ONCE per env step (1 x N row)
        row = [self.agents_list[i].collected_apples for i in range(self.train_config.num_agents)]
        self._append_row(self.collected_rewards_over_time, row)

    # === save to disk (convert deques -> arrays only here), compressed ===
    def _save_follow_rates_arrays(self):
        path = os.path.join(CHECKPOINT_DIR, self.name)
        os.makedirs(path, exist_ok=True)
        out = {}
        for i in range(self.train_config.num_agents):
            if len(self.foll_rate_hist[i]) == 0:
                arr = np.zeros((0, self.train_config.num_agents), dtype=self._hist_dtype)
            else:
                arr = np.vstack(self.foll_rate_hist[i])  # (K, N)
            out[f"agent_{i}"] = arr
        np.savez_compressed(os.path.join(path, "follow_rates.npz"), **out)

    def _save_alpha_arrays(self):
        path = os.path.join(CHECKPOINT_DIR, self.name)
        os.makedirs(path, exist_ok=True)
        out = {}
        for i in range(self.train_config.num_agents):
            if len(self.alpha_ema[i]) == 0:
                arr = np.zeros((0, self.train_config.num_agents), dtype=self._hist_dtype)
            else:
                arr = np.vstack(self.alpha_ema[i])  # (K, N)
            out[f"agent_{i}"] = arr
        np.savez_compressed(os.path.join(path, "alphas.npz"), **out)

    # === plotting helpers ===
    def _save_follow_rates_for_agent(self, agent_i: int):
        # Safely get (K, N) array view for plotting
        if len(self.foll_rate_hist[agent_i]) == 0:
            return
        arr = np.vstack(self.foll_rate_hist[agent_i])
        if arr.shape[0] == 0:
            return

        fig, ax = plt.subplots(figsize=(10, 4))
        for j in range(self.train_config.num_agents):
            series = arr[:, j]
            if series.size > 0:
                ax.plot(series, label=f"Following rate of agent {j}")
        ax.legend()
        ax.set_title(f"Follow Rates of Agent {agent_i}")
        ax.set_xlabel(f"Training Step (logged every {self._log_every} steps)")
        ax.set_ylabel("Following Rates")
        fig.savefig(self.graphs_out_path / f"Follow_Rates_agent_{agent_i}.png")
        plt.close(fig)

    def training_step(self, step):
        super().training_step(step)

        # record per-agent logs only when needed
        for num, agent in enumerate(self.agents_list):
            self._record_rates(step, num, agent.agent_rates, agent.agent_alphas)
            self._save_agent_distances(step, num)

        # record rewards ONCE per step (and only when logging)
        self._record_collected_rewards(step)

        if step > 5000 and (step % 1000) == 0:
            for agent in self.agents_list:
                agent.learn_rates()

    def collect_observation(self, step):
        try:
            for tick in range(self.train_config.num_agents):
                s, new_s, r, agent, positions, action = self.env_step(tick)
                if action is not None:
                    for each_agent in range(len(self.agents_list)):
                        curr_pos = self.agents_list[each_agent].position
                        reward = r if each_agent == agent else 0
                        processed_state = self.view_controller.process_state(s, positions[each_agent])
                        processed_new_state = self.view_controller.process_state(new_s, curr_pos)
                        self.agents_list[each_agent].add_experience(
                            processed_state, processed_new_state, reward)
                        if each_agent == agent:
                            new_positions = [self.agents_list[j].position for j in range(len(self.agents_list))]
                            advantage = reward + self.agent_controller.get_collective_advantage(
                                s, positions, new_s, new_positions, agent)
                            self.agents_list[each_agent].policy_network.add_experience(
                                processed_state, processed_new_state, r, action, advantage)
        except Exception as e:
            self.logger.error(f"Error collecting observations: {e}")
            raise

    def init_agents_for_eval(self):
        a_list = []
        for ii in range(len(self.agents_list)):
            trained_agent = ACAgentRates("learned_policy", len(self.agents_list),
                                         self.train_config.beta_rate, ii, self.train_config.budget)
            trained_agent.policy_network = self.p_network_list[ii]
            a_list.append(trained_agent)
        return a_list

    def log_progress(self, sample_state, sample_state5, sample_state6):
        super().log_progress(sample_state, sample_state5, sample_state6)

        for agent_id in range(self.train_config.num_agents):
            self._save_follow_rates_for_agent(agent_id)

            # Alpha/Q curves
            if len(self.alpha_ema[agent_id]) > 0:
                arr = np.vstack(self.alpha_ema[agent_id])
                fig, ax = plt.subplots(figsize=(10, 4))
                for other_agent in range(self.train_config.num_agents):
                    series = arr[:, other_agent]
                    if series.size > 0:
                        ax.plot(series, label=f"Q-value from Agent {other_agent}",
                                linestyle=linestyles[other_agent % len(linestyles)])
                ax.legend()
                ax.set_title(f"Observed Q-values for Agent {agent_id}")
                ax.set_xlabel(f"Training Step (logged every {self._log_every})")
                ax.set_ylabel("Q-value")
                fig.savefig(self.graphs_out_path / f"Q_values_agent_{agent_id}.png")
                plt.close(fig)

            # Distances (smoothed plot)
            if len(self.agent_distance_hist[agent_id]) > 0:
                arr = np.vstack(self.agent_distance_hist[agent_id])  # (K, N)
                series_list = [arr[:, j] for j in range(self.train_config.num_agents)]
                labels = [f"Distance to {j}" for j in range(self.train_config.num_agents)]
                fig = plot_smoothed(series_list, labels,
                                    title=f"Distance to Agent {agent_id}",
                                    xlabel=f"Training Step (logged every {self._log_every})",
                                    ylabel="Distance")
                fig.savefig(self.graphs_out_path / f"Distance_agent_{agent_id}.png")
                plt.close(fig)

        # Save compressed snapshots of histories
        self._save_follow_rates_arrays()
        self._save_alpha_arrays()
        plt.close("all")

        # Rewards plot (if any)
        if len(self.collected_rewards_over_time) > 0:
            rewards_arr = np.vstack(self.collected_rewards_over_time).astype(np.float32)  # for plotting
            plot_agent_rewards(rewards_arr,
                               title="Agent rewards",
                               out_path=self.graphs_out_path / "agent_rewards.png",
                               show=False)

    def restore_all(self):
        # self.foll_rate_hist = load_follow_rates(str(os.path.join(CHECKPOINT_DIR, self.name, "follow_rates.npz")))
        # self.alpha_ema = load_alphas(str(os.path.join(CHECKPOINT_DIR, self.name, "alphas.npz")))
        # After loading from npz (arrays), keep last row for runtime values
        # for num, agent in enumerate(self.agents_list):
        #     last_rates = np.asarray(self.foll_rate_hist[num][-1], dtype=float) if len(self.foll_rate_hist[num]) else np.zeros(self.train_config.num_agents)
        #     last_alphas = np.asarray(self.alpha_ema[num][-1], dtype=float) if len(self.alpha_ema[num]) else np.zeros(self.train_config.num_agents)
        #     agent.agent_rates = last_rates.copy()
        #     agent.agent_alphas = last_alphas.copy()
        name = f"""ActorCriticFixedRates-<{self.train_config.num_agents}>_agents-_length-<{self.env_config.length}>_width-<{self.env_config.width}>_s_target-<{self.env_config.s_target}>-alpha-<{self.train_config.alpha}>-apple_mean_lifetime-<{self.env_config.apple_mean_lifetime}>-<{self.train_config.hidden_dimensions}>-<{self.train_config.num_layers}>-vision-<{self.train_config.vision}>-batch_size-<{self.train_config.batch_size}>-actor_alpha-<{self.train_config.actor_alpha}>-actor_hidden-<{self.train_config.hidden_dimensions_actor}>-actor_layers-<{self.train_config.num_layers_actor}>-beta-<0.0>-budget-<{self.train_config.budget}>"""
        self.load_networks(name)
        agent_pos, apples = self._load_env_state()
        return agent_pos, apples

    def run(self):
        try:
            self.view_controller = ViewController(self.train_config.vision)
            self.agent_controller = AgentControllerActorCriticRates(self.agents_list, self.view_controller)

            for nummer in range(self.train_config.num_agents):
                agent = ACAgentRates("learned_policy", self.train_config.num_agents,
                                     self.train_config.beta_rate, nummer, self.train_config.budget)
                agent.policy_network, agent.policy_value = self.init_networks()
                self.agents_list.append(agent)
                self.v_network_list.append(agent.policy_value)
                self.p_network_list.append(agent.policy_network)
            self.network_for_eval = self.p_network_list
            return self.training_loop() if not self.train_config.skip else self.training_loop(*self.restore_all())
        except Exception as e:
            self.logger.error(f"Failed to run decentralized training: {e}")
            raise


class ActorCriticRatesAdvantage(ActorCriticRates):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config, f"""ActorCriticRatesAdv-<{config.train_config.num_agents}>_agents-_length-<{config.env_config.length}>_width-<{config.env_config.width}>_s_target-<{config.env_config.s_target}>-alpha-<{config.train_config.alpha}>-apple_mean_lifetime-<{config.env_config.apple_mean_lifetime}>-<{config.train_config.hidden_dimensions}>-<{config.train_config.num_layers}>-vision-<{config.train_config.vision}>-batch_size-<{config.train_config.batch_size}>-actor_alpha-<{config.train_config.actor_alpha}>-actor_hidden-<{config.train_config.hidden_dimensions_actor}>-actor_layers-<{config.train_config.num_layers_actor}>-beta-<{config.train_config.beta_rate}>-budget-<{config.train_config.budget}>""")

    def run(self):
        try:
            self.view_controller = ViewController(self.train_config.vision)
            self.agent_controller = AgentControllerActorCriticRatesAdvantage(self.agents_list, self.view_controller)
            for nummer in range(self.train_config.num_agents):
                agent = ACAgentRates("learned_policy", self.train_config.num_agents, self.train_config.beta_rate, nummer, self.train_config.budget)
                agent.policy_network, agent.policy_value = self.init_networks()
                self.agents_list.append(agent)
                self.v_network_list.append(agent.policy_value)
                self.p_network_list.append(agent.policy_network)
            self.network_for_eval = self.p_network_list
            return self.training_loop() if not self.train_config.skip else self.training_loop(*self.restore_all())
        except Exception as e:
            self.logger.error(f"Failed to run decentralized training: {e}")
            raise

    def log_progress(self, sample_state, sample_state5, sample_state6):
        Algorithm.log_progress(self, sample_state, sample_state5, sample_state6)
        for agent_id in range(self.train_config.num_agents):
            self._save_follow_rates_for_agent(agent_id)

            # Alpha/Q curves
            if len(self.alpha_ema[agent_id]) > 0:
                arr = np.vstack(self.alpha_ema[agent_id])
                fig, ax = plt.subplots(figsize=(10, 4))
                for other_agent in range(self.train_config.num_agents):
                    series = arr[:, other_agent]
                    if series.size > 0:
                        ax.plot(series, label=f"Advantage Value from Agent {other_agent}",
                                linestyle=linestyles[other_agent % len(linestyles)])
                ax.legend()
                ax.set_title(f"Observed Advantage values for Agent {agent_id}")
                ax.set_xlabel(f"Training Step (logged every {self._log_every})")
                ax.set_ylabel("Advantage value")
                fig.savefig(self.graphs_out_path / f"Q_values_agent_{agent_id}.png")
                plt.close(fig)

            # Distances (smoothed plot)
            if len(self.agent_distance_hist[agent_id]) > 0:
                arr = np.vstack(self.agent_distance_hist[agent_id])  # (K, N)
                series_list = [arr[:, j] for j in range(self.train_config.num_agents)]
                labels = [f"Distance to {j}" for j in range(self.train_config.num_agents)]
                fig = plot_smoothed(series_list, labels,
                                    title=f"Distance to Agent {agent_id}",
                                    xlabel=f"Training Step (logged every {self._log_every})",
                                    ylabel="Distance")
                fig.savefig(self.graphs_out_path / f"Distance_agent_{agent_id}.png")
                plt.close(fig)

        # Save compressed snapshots of histories
        self._save_follow_rates_arrays()
        self._save_alpha_arrays()
        plt.close("all")

        # Rewards plot (if any)
        if len(self.collected_rewards_over_time) > 0:
            rewards_arr = np.vstack(self.collected_rewards_over_time).astype(np.float32)  # for plotting
            plot_agent_rewards(rewards_arr,
                               title="Agent rewards",
                               out_path=self.graphs_out_path / "agent_rewards.png",
                               show=False)


class ActorCriticRatesFixed(ActorCritic):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config, f"""ActorCriticFixedRates-<{config.train_config.num_agents}>_agents-_length-<{config.env_config.length}>_width-<{config.env_config.width}>_s_target-<{config.env_config.s_target}>-alpha-<{config.train_config.alpha}>-apple_mean_lifetime-<{config.env_config.apple_mean_lifetime}>-<{config.train_config.hidden_dimensions}>-<{config.train_config.num_layers}>-vision-<{config.train_config.vision}>-batch_size-<{config.train_config.batch_size}>-actor_alpha-<{config.train_config.actor_alpha}>-actor_hidden-<{config.train_config.hidden_dimensions_actor}>-actor_layers-<{config.train_config.num_layers_actor}>-beta-<{config.train_config.beta_rate}>-budget-<{config.train_config.budget}>""")

    def init_agents_for_eval(self):
        a_list = []
        for ii in range(len(self.agents_list)):
            trained_agent = ACAgentRatesFixed("learned_policy", self.train_config.num_agents, ii, self.train_config.budget)
            trained_agent.policy_network = self.p_network_list[ii]
            a_list.append(trained_agent)
        return a_list

    def collect_observation(self, step):
        try:
            for tick in range(self.train_config.num_agents):
                s, new_s, r, agent, positions, action = self.env_step(tick)
                if action is not None:
                    for each_agent in range(len(self.agents_list)):
                        curr_pos = self.agents_list[each_agent].position
                        reward = r if each_agent == agent else 0
                        processed_state = self.view_controller.process_state(s, positions[each_agent])
                        processed_new_state = self.view_controller.process_state(new_s, curr_pos)
                        self.agents_list[each_agent].add_experience(
                            processed_state, processed_new_state, reward)
                        if each_agent == agent:
                            new_positions = []
                            for j in range(len(self.agents_list)):
                                new_positions.append(self.agents_list[j].position)
                            advantage = reward + self.train_config.discount * self.agent_controller.collective_value_from_state(new_s, new_positions, agent) - self.agent_controller.collective_value_from_state(s, positions, agent)
                            self.agents_list[each_agent].policy_network.add_experience(processed_state, processed_new_state, r, action, advantage)
        except Exception as e:
            self.logger.error(f"Error collecting observations: {e}")
            raise

    def run(self):
        try:
            self.view_controller = ViewController(self.train_config.vision)
            self.agent_controller = AgentControllerActorCriticRatesFixed(self.agents_list, self.view_controller)
            for nummer in range(self.train_config.num_agents):
                agent = ACAgentRatesFixed("learned_policy", self.train_config.num_agents, nummer, self.train_config.budget)
                agent.policy_network, agent.policy_value = self.init_networks()
                self.agents_list.append(agent)
                self.v_network_list.append(agent.policy_value)
                self.p_network_list.append(agent.policy_network)
            self.network_for_eval = self.p_network_list
            return self.training_loop() if not self.train_config.skip else self.training_loop(*self.restore_all())
        except Exception as e:
            self.logger.error(f"Failed to run decentralized training: {e}")
            raise

    def build_experiment(self, view_controller_cls=ViewController,
                         agent_controller_cls=AgentControllerCentralized,
                         agent_type=SimpleAgent, value_network_cls=VNetwork,
                         actor_network_cls=ActorNetwork):
        pass
