import os
from pathlib import Path
from config import CHECKPOINT_DIR
from matplotlib import pyplot as plt
from actor_critic_kirill import ActorCritic
from agents.actor_critic_agent import ACAgentRates
from configs.config import ExperimentConfig
import numpy as np
from helpers import get_discounted_value
from value_function_learning.controllers import AgentControllerActorCriticRates, ViewController
from main import plot_smoothed

linestyles = ["-", "--", "-.", ":"]


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


def load_beta_arrays(filepath: str):
    data = np.load(filepath, allow_pickle=False)
    beta_hist = {}
    for key in data.files:  # "agent_0", "agent_1", ...
        agent_id = int(key.split("_")[1])
        beta_hist[agent_id] = data[key]
    return beta_hist


class ActorCriticRates(ActorCritic):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config, f"""ActorCriticRates-<{config.train_config.num_agents}>_agents-_length-<{config.env_config.length}>_width-<{config.env_config.width}>_s_target-<{config.env_config.s_target}>-alpha-<{config.train_config.alpha}>-apple_mean_lifetime-<{config.env_config.apple_mean_lifetime}>-<{config.train_config.hidden_dimensions}>-<{config.train_config.num_layers}>-vision-<{config.train_config.vision}>-batch_size-<{config.train_config.batch_size}>-actor_alpha-<{config.train_config.actor_alpha}>-actor_hidden-<{config.train_config.hidden_dimensions_actor}>-actor_layers-<{config.train_config.num_layers_actor}>-beta-<{config.train_config.beta_rate}>-budget-<{config.train_config.budget}>""")

        # The following rate history of each agent for plotting
        self.foll_rate_hist = {i: np.zeros((0, self.train_config.num_agents), dtype=float) for i in range(self.train_config.num_agents)}
        self.agent_distance_hist = {i: np.zeros((0, self.train_config.num_agents), dtype=float) for i in range(self.train_config.num_agents)}
        self.alpha_ema = {i: np.zeros((0, self.train_config.num_agents), dtype=float) for i in range(self.train_config.num_agents)}
        self.beta_hist = {i: np.zeros((0,), dtype=float) for i in range(self.train_config.num_agents)}

    def _record_rates(self, agent_i: int, agent_rates, alphas):
        """
        Append a snapshot of agent_i's follow rates to all agents.
        rates_global_vec: iterable length n_agents, in GLOBAL index order.
        """
        # for j in range(self.train_config.num_agents):
        #     new_list = np.append(self.foll_rate_hist[agent_i][j], float(agent_rates[j]))
        #     self.foll_rate_hist[agent_i][j] = new_list
        # make sure we have a 1 x n_agents row
        row = np.asarray(agent_rates, dtype=float).reshape(1, -1)
        row_2 = np.asarray(alphas, dtype=float).reshape(1, -1)
        if row.shape[1] != self.train_config.num_agents:
            raise ValueError(
                f"agent_rates must have length {self.train_config.num_agents}, "
                f"got shape {row.shape}"
            )

        # append the new row to the (K, n_agents) history for this agent
        self.foll_rate_hist[agent_i] = np.vstack([self.foll_rate_hist[agent_i], row])
        self.alpha_ema[agent_i] = np.vstack([self.alpha_ema[agent_i], row_2])

    def _save_agent_distances(self, agent_i):
        stack = []
        for id_, agent in enumerate(self.agents_list):
            distance = np.linalg.norm(agent.position - self.agents_list[agent_i].position)
            stack.append(distance)
        self.agent_distance_hist[agent_i] = np.vstack([self.agent_distance_hist[agent_i], stack])

    def _record_beta(self, agent_i: int, beta_value: float):
        """Append one scalar beta for agent_i (dynamic, from empty)."""
        self.beta_hist[agent_i] = np.append(self.beta_hist[agent_i], float(beta_value))

    def _save_beta_arrays(self):
        """
        Save all agents' beta histories into a single .npz file.
        Keys: agent_0, agent_1, ..., each a 1-D array of length = #records.
        """
        path = os.path.join(CHECKPOINT_DIR, self.name)
        if not os.path.isdir(path):
            print("new_path")
            os.makedirs(path)
        path = os.path.join(str(path), "betas.npz")
        payload = {f"agent_{i}": self.beta_hist[i] for i in range(self.train_config.num_agents)}
        np.savez(path, **payload)

    def _save_beta_for_agent(self, agent_i: int):
        """Save a PNG line plot of beta history for a single agent."""
        series = self.beta_hist[agent_i]  # 1-D array, length = #records
        if series.size == 0:
            return  # nothing to plot yet

        plt.figure(figsize=(10, 3))
        plt.plot(series)
        plt.title(f"Beta of Agent {agent_i}")
        plt.xlabel("Training Step")
        plt.ylabel("β")
        out_path = self.graphs_out_path / f"Beta_agent_{agent_i}.png"
        plt.savefig(out_path)
        plt.close()

    def _save_all_betas(self):
        """Save one PNG per agent for all beta histories."""
        for i in range(self.train_config.num_agents):
            self._save_beta_for_agent(i)

    def _save_follow_rates_for_agent(self, agent_i: int):
        """
        Save a PNG of agent_i's follow rates at training step `graph_step`.
        """

        arr = self.foll_rate_hist[agent_i]  # shape (K, n_agents)
        # empty-safe: nothing recorded yet
        if arr.shape[0] == 0:
            return

        plt.figure(figsize=(10, 4))
        for j in range(self.train_config.num_agents):
            series = arr[:, j]  # <-- column j, not row j
            if series.size > 0:
                plt.plot(series, label=f"Following rate of agent {j}")

        plt.legend()
        plt.title(f"Follow Rates of Agent {agent_i}")
        plt.xlabel("Training Step")
        plt.ylabel("Following Rates")
        out_path = self.graphs_out_path / f"Follow_Rates_agent_{agent_i}.png"
        plt.savefig(out_path)
        plt.close()

    def _save_follow_rates_arrays(self):
        """
        Save all agents' follow-rate histories into a single .npz file.
        Each agent_i's history is an array of shape (timesteps, n_agents).
        """
        path = os.path.join(CHECKPOINT_DIR, self.name)
        if not os.path.isdir(path):
            print("new_path")
            os.makedirs(path)
        path = os.path.join(str(path), "follow_rates.npz")
        np.savez(path, **{f"agent_{i}": self.foll_rate_hist[i] for i in range(self.train_config.num_agents)})

    def _save_alpha_arrays(self):
        """
        Save all agents' follow-rate histories into a single .npz file.
        Each agent_i's history is an array of shape (timesteps, n_agents).
        """
        path = os.path.join(CHECKPOINT_DIR, self.name)
        if not os.path.isdir(path):
            print("new_path")
            os.makedirs(path)
        path = os.path.join(str(path), "alphas.npz")
        np.savez(path, **{f"agent_{i}": self.alpha_ema[i] for i in range(self.train_config.num_agents)})

    def update_critic(self):
        super().update_critic()
        for num, agent in enumerate(self.agents_list):
            # Update betas
            agent.update_beta()

            # Record rates for plotting
            self._record_rates(num, agent.agent_rates, agent.agent_alphas)
            self._record_beta(num, agent.beta)
            self._save_agent_distances(num)

    def training_step(self, step):
        super().training_step(step)
        if step > 5000:
            if (step % 1000) == 0:
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
                            new_positions = []
                            for j in range(len(self.agents_list)):
                                new_positions.append(self.agents_list[j].position)
                            total_feedback = reward + self.train_config.discount * self.agent_controller.collective_value_from_state(new_s, new_positions, agent)
                            self.agents_list[each_agent].beta_temp_batch.append(total_feedback)
                            advantage_for_actor = total_feedback - self.agents_list[each_agent].beta
                            # advantage_for_actor = total_feedback - self.agent_controller.collective_value_from_state(s, positions, agent)
                            self.agents_list[each_agent].policy_network.add_experience(processed_state, processed_new_state, r, action, advantage_for_actor)

        except Exception as e:
            self.logger.error(f"Error collecting observations: {e}")
            raise

    def init_agents_for_eval(self):
        a_list = []
        for ii in range(len(self.agents_list)):
            trained_agent = ACAgentRates("learned_policy", len(self.agents_list), self.train_config.beta_rate, ii, self.train_config.budget)
            trained_agent.policy_network = self.p_network_list[ii]
            a_list.append(trained_agent)
        return a_list

    def log_progress(self, sample_state, sample_state5, sample_state6):
        super().log_progress(sample_state, sample_state5, sample_state6)

        for agent_id in range(self.train_config.num_agents):
            self._save_follow_rates_for_agent(agent_id)
            plt.figure(figsize=(10, 4))
            arr = self.alpha_ema[agent_id]
            for other_agent in range(self.train_config.num_agents):
                series = arr[:, other_agent]  # <-- column j, not row j
                if series.size > 0:
                    plt.plot(series, label=f"Q-value from Agent {other_agent}", linestyle=linestyles[other_agent % len(linestyles)])
            plt.legend()
            plt.title(f"Observed Q-values for Agent {agent_id}")
            plt.xlabel("Training Step")
            plt.ylabel("Q-value")
            plt.savefig(self.graphs_out_path / f"Q_values_agent_{agent_id}.png")
            plt.close()

            arr = self.agent_distance_hist[agent_id]  # shape [T, num_agents]
            series_list = [arr[:, j] for j in range(self.train_config.num_agents)]
            labels = [f"Distance to {j}" for j in range(self.train_config.num_agents)]
            plot = plot_smoothed(series_list, labels, title=f"Distance to Agent {agent_id}", xlabel="Training Step", ylabel="Distance")
            plot.savefig(self.graphs_out_path / f"Distance_agent_{agent_id}.png")

        self._save_follow_rates_arrays()
        self._save_all_betas()
        self._save_beta_arrays()
        self._save_alpha_arrays()

    def restore_all(self):
        self.foll_rate_hist = load_follow_rates(str(os.path.join(CHECKPOINT_DIR, self.name, "follow_rates.npz")))
        self.beta_hist = load_beta_arrays(str(os.path.join(CHECKPOINT_DIR, self.name, "betas.npz")))
        self.alpha_ema = load_alphas(str(os.path.join(CHECKPOINT_DIR, self.name, "alphas.npz")))

        for num, agent in enumerate(self.agents_list):
            agent.agent_rates = np.asarray(self.foll_rate_hist[num][-1], dtype=float).copy()
            agent.agent_alphas = np.asarray(self.alpha_ema[num][-1], dtype=float).copy()
            agent.beta = self.beta_hist[num][-1]
        return super().restore_all()

    def run(self):
        try:
            self.view_controller = ViewController(self.train_config.vision)
            self.agent_controller = AgentControllerActorCriticRates(self.agents_list, self.view_controller)
            for nummer in range(self.train_config.num_agents):
                agent = ACAgentRates("learned_policy", self.train_config.num_agents, self.train_config.beta_rate, nummer, self.train_config.budget)
                agent.policy_network, agent.policy_value = self.init_networks()
                self.agents_list.append(agent)
                self.v_network_list.append(agent.policy_value)
                self.p_network_list.append(agent.policy_network)
            self.network_for_eval = self.p_network_list
            return self.train() if not self.train_config.skip else self.train(*self.restore_all())
        except Exception as e:
            self.logger.error(f"Failed to run decentralized training: {e}")
            raise
