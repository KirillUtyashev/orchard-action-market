import logging
from abc import abstractmethod
from pathlib import Path
from typing import Optional
import torch
from numpy import floating
from config import CHECKPOINT_DIR
from main import run_environment_1d
from plots import add_to_plots, graph_plots
from orchard.environment import *
from helpers import generate_sample_states
import os
import time
from policies.random_policy import random_policy
import psutil
from dataclasses import dataclass
from typing import Tuple


@dataclass
class EvalResult:
    total_apples: int
    total_picked: int
    picked_per_agent: float
    per_agent: float
    average_distance: float
    apple_per_sec: float
    nearest_actions: int
    idle_actions: int

    def log(self, logger):
        """Log all evaluation metrics"""
        logger.info(f"Picked per agents: {self.picked_per_agent}")
        logger.info(f"Ratio picked: {self.per_agent}")
        logger.info(f"Mean distance: {self.average_distance}")
        logger.info(f"Total apples: {self.total_apples}")
        logger.info(f"Total picked: {self.total_picked}")
        logger.info(f"Apple per sec: {self.apple_per_sec}")
        logger.info(f"Nearest actions: {self.nearest_actions}")
        logger.info(f"Idle actions: {self.idle_actions}")

    @property
    def as_tuple(self) -> Tuple:
        """Convert to tuple for backwards compatibility"""
        return (self.total_apples, self.total_picked, self.picked_per_agent,
                self.per_agent, self.average_distance, self.apple_per_sec,
                self.nearest_actions, self.idle_actions)


def memory_snapshot(label="mem", show_children=False, top_n=5):
    """
    Print a memory usage summary for the current process (and optionally top child processes).

    Parameters
    ----------
    label : str
        Tag to include in the printed line (e.g., 'step=1000', 'eval', etc.).
    show_children : bool
        If True, prints top-N child processes by RSS.
    top_n : int
        Number of child processes to display when show_children=True.
    """
    proc = psutil.Process(os.getpid())
    try:
        mi = proc.memory_info()
    except psutil.NoSuchProcess:
        return

    rss = mi.rss  # resident set size (bytes)
    vms = mi.vms  # virtual mem size (bytes)

    children = proc.children(recursive=True)
    rss_children = 0
    child_stats = []
    for ch in children:
        try:
            chi = ch.memory_info()
        except psutil.NoSuchProcess:
            continue
        rss_children += chi.rss
        if show_children:
            child_stats.append((chi.rss, ch.pid, " ".join(ch.cmdline()[:3]) or ch.name()))

    total_rss = rss + rss_children

    print(
        f"[{time.strftime('%H:%M:%S')}] {label}: "
        f"RSS_self={rss/1e6:.1f}MB | RSS_children={rss_children/1e6:.1f}MB | TOTAL={total_rss/1e6:.1f}MB | VMS={vms/1e6:.1f}MB",
        flush=True,
    )

    if show_children and child_stats:
        child_stats.sort(reverse=True)  # largest first
        print("  Top child processes (RSS MB):", flush=True)
        for rss_bytes, pid, cmd in child_stats[:top_n]:
            print(f"    PID {pid}: {rss_bytes/1e6:.1f}MB  {cmd}", flush=True)


class Algorithm:
    def __init__(self, config, name):
        self.train_config = config.train_config
        self.env_config = config.env_config
        self.env = None
        self.name = name
        self.debug = config.debug
        self.rng_state = None

        log_folder = Path("logs")
        log_folder.mkdir(parents=True, exist_ok=True)

        filename = log_folder / f"{name}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=str(filename),
            filemode='a'
        )

        self.logger = logging.getLogger(self.name)

        self.agents_list = []

        self.loss_plot = []
        self.loss_plot5 = []
        self.loss_plot6 = []
        self.weights_plot = {}
        self.critic_loss = []

        self.max_ratio = 0

        # Network(s) used for eval_network at the middle and end of training
        self.network_for_eval = []
        self.view_controller = None
        self.agent_controller = None

        if self.train_config.test:
            self.count_random_actions = 0

    def create_env(self):
        self.env = Orchard(self.env_config.length, self.env_config.width, self.train_config.num_agents, self.agents_list, spawn_algo=self.env_config.spawn_algo, despawn_algo=self.env_config.despawn_algo, s_target=self.env_config.s_target, apple_mean_lifetime=self.env_config.apple_mean_lifetime)
        self.env.initialize(self.agents_list)

    @abstractmethod
    def update_critic(self):
        raise NotImplementedError

    @abstractmethod
    def collect_observation(self, step):
        raise NotImplementedError

    def save_rng_state(self):
        """Save all random states"""
        self.rng_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }

    def restore_rng_state(self):
        """Restore all random states"""
        if self.rng_state is not None:
            random.setstate(self.rng_state['python'])
            np.random.set_state(self.rng_state['numpy'])
            torch.set_rng_state(self.rng_state['torch'])

    def train_batch(self) -> None:
        """Train on a batch of experiences with error handling."""
        try:
            self.critic_loss.append(self.update_critic())
        except Exception as e:
            self.logger.error(f"Error during batch training: {e}")
            raise

    def log_progress(self, sample_state, sample_state5, sample_state6):
        agent_obs = []
        for i in range(self.train_config.num_agents):
            agent_obs.append(self.view_controller.process_state(sample_state, sample_state["poses"][i]))
        v_value = self.agent_controller.get_collective_value(agent_obs, 0)
        agent_obs = []
        for i in range(self.train_config.num_agents):
            agent_obs.append(self.view_controller.process_state(sample_state5, sample_state5["poses"][i]))
        v_value5 = self.agent_controller.get_collective_value(agent_obs, 0)
        agent_obs = []
        for i in range(self.train_config.num_agents):
            agent_obs.append(self.view_controller.process_state(sample_state6, sample_state6["poses"][i]))
        v_value6 = self.agent_controller.get_collective_value(agent_obs, 0)

        add_to_plots(self.network_for_eval[0].function.state_dict(), self.weights_plot)

        print("P", v_value)
        self.loss_plot.append(v_value.item())
        self.loss_plot5.append(v_value5.item())
        self.loss_plot6.append(v_value6.item())

    @abstractmethod
    def update_lr(self, step):
        raise NotImplementedError

    def evaluate_checkpoint(self, step: int, seed: int) -> EvalResult:
        """Evaluate the current checkpoint"""
        print(f"=====Eval at {step} steps======")
        result = self.eval_network(seed)
        print("=====Completed Evaluation=====")
        return result

    @abstractmethod
    def update_actor(self):
        raise NotImplementedError

    def eval_network(self, seed: int) -> EvalResult:
        """Run network evaluation"""

        self.save_rng_state()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        agents_list = self.init_agents_for_eval()

        with torch.no_grad():
            results = run_environment_1d(
                num_agents=self.train_config.num_agents,
                side_length=self.env_config.length,
                width=self.env_config.width,
                S=None,
                phi=None,
                name=self.name,
                agents_list=agents_list,
                spawn_algo=self.env_config.spawn_algo,
                despawn_algo=self.env_config.despawn_algo,
                timesteps=10000,
                vision=self.train_config.vision,
                s_target=self.env_config.s_target,
                apple_mean_lifetime=self.env_config.apple_mean_lifetime,
                epsilon=self.train_config.epsilon
            )

        # Create EvalResult from returned tuple
        eval_result = EvalResult(*results)

        # Save networks
        self._save_best_networks()

        self.restore_rng_state()

        return eval_result

    def agent_get_action(self, agent_id: int) -> int:
        if random.random() < self.train_config.epsilon:
            action = random_policy(self.env.available_actions)
            if self.train_config.test:
                self.count_random_actions += 1
        else:
            with torch.no_grad():
                action = self.agent_controller.get_best_action(self.env.get_state(),
                                                               agent_id,
                                                               self.env.available_actions)
        return action

    def _save_best_networks(self):
        """Save the current best networks"""
        print("saving best")
        path = os.path.join(CHECKPOINT_DIR, self.name)
        if not os.path.isdir(path):
            print("new_path")
            os.makedirs(path)
        self.save_networks(path)

    def env_step(self, tick):
        agent_id = random.randint(0, self.train_config.num_agents - 1)
        state = self.env.get_state()  # this is assumed to be a dict with "agents" and "apples"
        old_pos = self.agents_list[agent_id].position
        positions = []
        for i in range(self.train_config.num_agents):
            positions.append(self.agents_list[i].position)
        action = self.agent_get_action(agent_id)
        reward, new_pos = self.env.main_step(self.agents_list[agent_id].position.copy(), action)
        self.agents_list[agent_id].position = new_pos.copy()
        if tick == self.train_config.num_agents - 1:
            self.env.apples_despawned += self.env.despawn_algorithm(self.env, self.env.despawn_rate)
            self.env.total_apples += self.env.spawn_algorithm(self.env, self.env.spawn_rate)
        return self._format_env_step_return(state, self.env.get_state(), reward, agent_id, positions, action, old_pos)

    @abstractmethod
    def _format_env_step_return(self, state, new_state, reward, agent_id, positions, action, old_pos):
        raise NotImplementedError

    @abstractmethod
    def init_agents_for_eval(self):
        raise NotImplementedError

    @abstractmethod
    def save_networks(self, path):
        raise NotImplementedError

    def train(self) -> Tuple[floating, ...] | None:
        """Train the value function."""
        try:
            self.create_env()

            sample_state, sample_state5, sample_state6 = generate_sample_states(
                self.env.length, self.env.width, self.train_config.num_agents)

            for step in range(self.train_config.timesteps):
                # Collect and process observations
                self.collect_observation(step)

                # Train if enough samples collected
                if len(self.agents_list[0].policy_value.batch_states) >= self.train_config.batch_size:
                    self.update_critic()

                if hasattr(self.agents_list[0], "policy_network"):
                    for i in range(self.train_config.num_agents):
                        if len(self.agents_list[i].policy_network.batch_states) >= self.train_config.batch_size:
                            self.agents_list[i].policy_network.train()

                # Log progress and update a learning rate
                if step % (0.02 * self.train_config.timesteps) == 0:
                    self.log_progress(sample_state, sample_state5, sample_state6)
                    if self.debug:
                        memory_snapshot(label=f"step={step}", show_children=True)
                self.update_lr(step)

                # Periodic evaluation
                if (step % (self.train_config.timesteps * 0.1) == 0) and (step != self.train_config.timesteps - 1):
                    self.evaluate_checkpoint(step, self.train_config.seed).log(self.logger)
                    graph_plots(self.name, self.weights_plot, self.critic_loss, self.loss_plot, self.loss_plot5, self.loss_plot6)
            # Final evaluation
            graph_plots(self.name, self.weights_plot, self.critic_loss, self.loss_plot, self.loss_plot5, self.loss_plot6)
            return self._evaluate_final()
        except Exception as e:
            self.logger.error(f"Failed during training: {e}")
            return None

    def _evaluate_final(self) -> Tuple[floating, ...]:
        """Perform final evaluation."""
        mean_metrics = {
            'total_apples': [], 'total_picked': [], 'picked_per_agent': [],
            'per_agent': [], 'average_distance': [], 'apple_per_sec': [],
            'nearest_actions': [], 'idle_actions': []
        }

        for k in range(3):
            result = self.evaluate_checkpoint(self.train_config.timesteps - 1, self.train_config.seed + k)
            for i, key in enumerate(mean_metrics.keys()):
                mean_metrics[key].append(getattr(result, key))

        # Log final averages
        self.logger.info(f"Ratio picked: {np.mean(mean_metrics['per_agent'])}")
        self.logger.info(f"Mean distance: {np.mean(mean_metrics['average_distance'])}")
        self.logger.info(f"Total apples: {np.mean(mean_metrics['total_apples'])}")
        self.logger.info(f"Total picked: {np.mean(mean_metrics['total_picked'])}")
        self.logger.info(f"Picked per agents: {np.mean(mean_metrics['picked_per_agent'])}")

        return tuple(np.mean(val) for val in mean_metrics.values())

    @abstractmethod
    def run(self):
        raise NotImplementedError


#         elif "AC" in name:
#         for nummer, netwk in enumerate(network_list):
#             torch.save(netwk.function.state_dict(),
#                        path + "/" + name + "_" + str(nummer) + "_it_" + str(
#                            iteration) + ".pt")
#     else:
#     torch.save(network_list[0].function.state_dict(),
#                path + "/" + name + "_cen_it_" + str(iteration) + ".pt")
# maxi = max(maxi, val)
# return maxi, ratio
