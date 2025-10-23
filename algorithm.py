import csv
import glob
import logging
import re
from pathlib import Path
import torch
from matplotlib import pyplot as plt
from numpy import floating

from agents.agent import Agent, AgentInfo
from agents.simple_agent import SimpleAgent
from config import DEVICE, OUT_DIR
from configs.config import EnvironmentConfig, ExperimentConfig, TrainingConfig
from helpers.controllers import (
    AgentController,
    AgentControllerCentralized,
    ViewController,
    ViewControllerOrchardSelfless,
)
from main import eval_performance
from models import network, reward_network
from models.actor_network import ActorNetwork
from models.cnn import CNN
from models.reward_network import RewardNetwork
from models.value_function import VNetwork
from plots import add_to_plots, graph_plots, plot_hybrid_smoothed
from orchard.environment import *
from helpers.helpers import create_env, generate_sample_states
import os
import time
import psutil
from dataclasses import dataclass
from typing import Any, Sequence, Tuple


times = 0


ENV_MAP = {
    "OrchardBasic": OrchardBasic,
    "OrchardSelfless": OrchardSelfless,
    "OrchardIDs": OrchardIDs,
    "OrchardMineNoReward": OrchardMineNoReward,
    "OrchardMineAllRewards": OrchardMineAllRewards,
    "OrchardEuclideanRewards": OrchardEuclideanRewards,
    "OrchardEuclideanNegativeRewards": OrchardEuclideanNegativeRewards,
}

ENV_MAP_NEW_DYNAMIC = {
    "OrchardBasic": OrchardBasicNewDynamic,
    "OrchardEuclideanRewards": OrchardEuclideanRewardsNewDynamic,
    "OrchardEuclideanNegativeRewards": OrchardEuclideanNegativeRewardsNewDynamic,
}


VIEW_CONTROLLER_MAP = {
    OrchardBasic: ViewController,
    OrchardBasicNewDynamic: ViewController,
    OrchardEuclideanRewardsNewDynamic: ViewController,
    OrchardIDs: ViewController,
    OrchardEuclideanNegativeRewardsNewDynamic: ViewController,
    OrchardMineNoReward: ViewControllerOrchardSelfless,
    OrchardSelfless: ViewControllerOrchardSelfless,
    OrchardMineAllRewards: ViewControllerOrchardSelfless,
    OrchardEuclideanRewards: ViewController,
    OrchardEuclideanNegativeRewards: ViewController,
}


@dataclass
class EnvStep:
    """Represents a single step in the environment.

    Attributes:
        old_state: dict with "agents" and "apples" keys
        new_state: dict with "agents" and "apples" keys
        acting_agent_id: The ID of the agent that took the action.
        old_positions: List of all agents' positions before the action.
        action: The action taken by the acting agent. 0=left, 1=right, 2=stay, 3=up, 4=down
        reward_vector: Array where index i is the amount of apples collected by agent i.
        picked: Whether an apple was picked during this action.
    """

    old_state: dict
    new_state: dict
    acting_agent_id: int
    old_positions: list
    action: int
    reward_vector: np.ndarray
    picked: bool


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
        return (
            self.total_apples,
            self.total_picked,
            self.picked_per_agent,
            self.per_agent,
            self.average_distance,
            self.apple_per_sec,
            self.nearest_actions,
            self.idle_actions,
        )


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
            child_stats.append(
                (chi.rss, ch.pid, " ".join(ch.cmdline()[:3]) or ch.name())
            )

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
            print(f"PID {pid}: {rss_bytes/1e6:.1f}MB  {cmd}", flush=True)


class Algorithm:
    """
    Key methods:
    - create_env
    - collect_observation
    - init_agents_for_eval
    - eval_network
    - env_step
    - training_step
    - training_loop
    """

    def create_folders(self):
        key_params = (
            f"h{self.train_config.hidden_dimensions}_"
            f"l{self.train_config.num_layers}_"
            f"a{self.train_config.num_agents}_"
            f"{self.env_config.width}x{self.env_config.length}_"
        )
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        # unique id so folders don't overwrite each other.
        self.run_id = f"{self.name}_{key_params}_{timestamp}_{os.getpid()}"
        self.run_output_dir = OUT_DIR / self.run_id

        self.log_folder = self.run_output_dir / "algo_logs"

        self.log_folder.mkdir(parents=True, exist_ok=True)
        self.run_output_dir.mkdir(parents=True, exist_ok=True)

        self.graphs_out_path = self.run_output_dir / "graphs" / self.name
        self.graphs_out_path.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.run_output_dir / "checkpoints"

    def __init__(self, config: ExperimentConfig, name):
        self.train_config: TrainingConfig = config.train_config
        self.env_config: EnvironmentConfig = config.env_config
        self.name = name
        self.debug = config.debug
        self.rng_state = None

        self.create_folders()
        # Create simple file containing all detailed parameters of this run
        self.params_file = self.run_output_dir / "params.txt"
        with open(self.params_file, "w") as f:
            f.write(f"Experiment Name: {self.name}\n")
            f.write(f"[Training Config]\n")
            for field_name, value in vars(self.train_config).items():
                f.write(f"{field_name}: {value}\n")
            f.write(f"[Environment Config]\n")
            for field_name, value in vars(self.env_config).items():
                f.write(f"{field_name}: {value}\n")

        filename = self.log_folder / f"{self.name}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=str(filename),
            filemode="a",
        )

        self.logger = logging.getLogger(self.name)

        self._agents_list: list[SimpleAgent] = []

        self.loss_plot: list[float] = []
        self.loss_plot5: list[float] = []
        self.loss_plot6: list[float] = []
        self.weights_plot: dict[str, Any] = {}
        self.critic_loss = []
        self.apple_count_history: list[int] = []
        self.despawn_to_pick_ratio_history: list[float] = []
        self.apples_spawned_per_step_history: list[int] = []

        self.max_ratio = 0

        # Network(s) used for eval_network at the middle and end of training
        self.network_for_eval = []
        self.v_weights = {}
        self.agent_info = AgentInfo(
            policy=self.train_config.policy, num_agents=self.train_config.num_agents
        )
        self.env_cls = (
            ENV_MAP[self.env_config.env_cls]
            if self.train_config.new_dynamic is False
            else ENV_MAP_NEW_DYNAMIC[self.env_config.env_cls]
        )

        if self.train_config.test:
            self.count_random_actions = 0

    def generate_plots(
        self,
    ):  # NOTE The reason for this is because different algorithms generate plots
        # differently so we cannot use a general graph_plots function for every algorithm.
        """
        Generates base plots. Subclasses can generate more plots.
        """

        # --- 1. Plot the Critic's Training Loss ---
        if self.critic_loss:
            fig = plot_hybrid_smoothed(
                [self.critic_loss],
                labels=["Critic Training MSE Loss"],
                title="Critic Training Loss (Smoothed)",
                xlabel="Training Step",
                ylabel="MSE Loss",
            )
            ax = fig.gca()
            ax.set_yscale("log")
            ax.grid(True)
            fig.savefig(self.graphs_out_path / "Critic_Training_Loss.png")
            plt.close(fig)

        # --- 2. Plot the Sample State Convergence (Value Function) ---
        if self.loss_plot:
            plt.figure(figsize=(10, 6))
            plt.plot(self.loss_plot, label="Sample State 1 Value")
            plt.plot(self.loss_plot5, label="Sample State 2 Value")
            plt.plot(self.loss_plot6, label="Sample State 3 Value")
            plt.title("Value of Fixed Sample States During Training")
            plt.xlabel("Logging Step")
            plt.ylabel("Predicted Value")
            plt.legend()
            plt.grid(True)
            plt.savefig(self.graphs_out_path / "Sample_State_Values.png")
            plt.close()

        # --- 3. Plot the Apple Count History ---
        if self.apple_count_history:
            fig = plot_hybrid_smoothed(
                [self.apple_count_history],
                labels=["Apple Count"],
                title="Number of Apples in Orchard During Training",
                xlabel="Training Step",
                ylabel="Total Apples",
            )
            ax = fig.gca()
            ax.grid(True)
            fig.savefig(self.graphs_out_path / "Apple_Count_During_Training.png")
            plt.close(fig)

        # --- 4. Plot the Despawn-to-Pick Ratio ---
        if self.despawn_to_pick_ratio_history:
            fig = plot_hybrid_smoothed(
                [self.despawn_to_pick_ratio_history],
                labels=["Despawn-to-Pick Ratio"],
                title="Apple Collection Efficiency During Training",
            )
            ax = fig.gca()
            ax.set_ylabel("Apples Despawned / Apples Picked")
            ax.grid(True)
            fig.savefig(self.graphs_out_path / "Despawn_to_Pick_Ratio.png")
            plt.close(fig)

        if self.apples_spawned_per_step_history:
            fig = plot_hybrid_smoothed(
                [self.apples_spawned_per_step_history],
                labels=["Apples Spawned"],
                title="Apples Spawned Per Timestep",
            )
            ax = fig.gca()
            ax.set_ylabel("Count")
            ax.grid(True)
            fig.savefig(self.graphs_out_path / "Apples_Spawned_Per_Step.png")
            plt.close(fig)

    @property
    def agents_list(self) -> Sequence[Agent]:
        """This is necessary because most classes have agent_list type Agent, but things like RewardAgent have the type more specific
        so we can override the type later.

        Returns:
            Sequence[Agent]: Sequence allows types to be differenced, e.g. List[RewardAgent] is a Sequence[Agent]
        """
        return self._agents_list

    @abstractmethod
    def step_and_collect_observation(self, step):
        raise NotImplementedError

    def save_rng_state(self):
        """Save all random states"""
        self.rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }

    def restore_rng_state(self):
        """Restore all random states"""
        if self.rng_state is not None:
            random.setstate(self.rng_state["python"])
            np.random.set_state(self.rng_state["numpy"])
            torch.set_rng_state(self.rng_state["torch"])

    def log_progress(self, sample_state, sample_state5, sample_state6):
        agent_obs = []
        for i in range(self.train_config.num_agents):
            agent_obs.append(
                self.critic_view_controller.state_to_nn_input(
                    sample_state, sample_state["poses"][i], i + 1
                )
            )
        v_value = self.agent_controller.get_collective_value(agent_obs, 0)
        agent_obs = []
        for i in range(self.train_config.num_agents):
            agent_obs.append(
                self.critic_view_controller.state_to_nn_input(
                    sample_state5, sample_state5["poses"][i], i + 1
                )
            )
        v_value5 = self.agent_controller.get_collective_value(agent_obs, 0)
        agent_obs = []
        for i in range(self.train_config.num_agents):
            agent_obs.append(
                self.critic_view_controller.state_to_nn_input(
                    sample_state6, sample_state6["poses"][i], i + 1
                )
            )
        v_value6 = self.agent_controller.get_collective_value(agent_obs, 0)

        # add_to_plots(self.network_for_eval[0].function.state_dict(), self.weights_plot)

        print("P", v_value)
        self.loss_plot.append(v_value.item())
        self.loss_plot5.append(v_value5.item())
        self.loss_plot6.append(v_value6.item())

    def update_lr(self, step):
        pass

    def evaluate_checkpoint(self, step: int, seed: int) -> EvalResult:
        """Evaluate agents based on current model parameters and plot the graphs for it."""
        print(f"=====Eval at {step} steps======")
        result = self.eval_network(seed)
        print("=====Completed Evaluation=====")
        return result

    def run_inference(self):
        """Run inference using current model weights and plot graphs

        Returns:
            Eval Results. see class for details.
        """
        agents_list, agent_controller = self.init_agents_for_eval()

        env = create_env(
            self.env_config,
            self.train_config.num_agents,
            None,
            None,
            agents_list,
            self.env_cls,
        )

        with torch.no_grad():
            results = eval_performance(
                num_agents=self.train_config.num_agents,
                agent_controller=agent_controller,
                env=env,
                name=self.name,
                agents_list=agents_list,
                timesteps=self.train_config.eval_timesteps,
                epsilon=self.train_config.epsilon,
            )

        # Create EvalResult from returned tuple
        return EvalResult(*results)

    def eval_network(self, seed: int) -> EvalResult:
        """Run network evaluation"""

        self.save_rng_state()
        print("Before eval: ", random.getstate()[1][0])

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            eval_result = self.run_inference()

            print("After eval: ", random.getstate()[1][0])

            self.restore_rng_state()
            print("Back to initial: ", random.getstate()[1][0])

            # Save networks
            self._save_best_networks()

            return eval_result
        except Exception as e:
            self.logger.error(f"Error during eval_network: {e}")
            raise

    def _save_best_networks(self):
        """Save the current best networks"""
        print("saving best")
        path = self.checkpoint_path
        if not os.path.isdir(path):
            print("new_path")
            os.makedirs(path)
        self.save_networks(str(path))
        self._save_agent_positions()
        self._save_apples()

    def _save_agent_positions(self, when: str = "final") -> None:
        """
        Save current agents' positions to CHECKPOINT_DIR/<algo-name>.
        Writes both a .npy (fast to load) and a .csv (human-readable).
        """
        if not self._agents_list:
            self.logger.error("No agents to save positions for.")
            return

        positions = np.asarray(
            [a.position for a in self._agents_list], dtype=np.int32
        )  # shape: [num_agents, 2] (or whatever your position shape is)
        out_dir = self.checkpoint_path
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / f"agent_positions_{when}.npy", positions)
        np.savetxt(
            out_dir / f"agent_positions_{when}.csv", positions, fmt="%d", delimiter=","
        )

    def _save_apples(self, when: str = "final") -> None:
        """
        Save current environment state (agents and apples) to CHECKPOINT_DIR/<algo-name>.
        Each is saved as both .npy (fast to load) and .csv (human-readable).
        """
        if not self.env:
            self.logger.error("No environment to save state for.")
            return

        state = self.env.get_state()

        if "agents" not in state or "apples" not in state:
            self.logger.error("Environment state missing 'agents' or 'apples' keys.")
            return

        out_dir = self.checkpoint_path
        out_dir.mkdir(parents=True, exist_ok=True)

        # Apples
        apples_arr = np.asarray(state["apples"], dtype=np.int32)
        np.save(out_dir / f"apples_{when}.npy", apples_arr)
        np.savetxt(out_dir / f"apples_{when}.csv", apples_arr, fmt="%d", delimiter=",")

        self.logger.info(f"Saved env state (agents + apples) to {out_dir}")

    def _load_env_state(self, when: str = "final"):
        """
        Load saved environment state (agents + apples) from CHECKPOINT_DIR/<algo-name>.
        Returns (agents_array, apples_array) as np.int32 arrays.
        """
        out_dir = Path(CHECKPOINT_DIR) / self.name
        agents_path = out_dir / f"agent_positions_{when}.npy"
        apples_path = out_dir / f"apples_{when}.npy"

        if not agents_path.exists() or not apples_path.exists():
            self.logger.error(f"Missing agents/apples state files in {out_dir}")
            return None, None

        agents_arr = np.load(agents_path)
        apples_arr = np.load(apples_path)

        self.logger.info(f"Loaded env state from {out_dir} ({when})")
        return agents_arr.astype(np.int32), apples_arr.astype(np.int32)

    def _collect_unique_critics(self):
        """Return (layout, critics_list) where layout is 'centralized' or 'decentralized'.
        critics_list is a list of (name, critic_obj) with duplicates removed."""
        seen = {}
        critics = []
        for i, a in enumerate(self._agents_list):
            if hasattr(a, "policy_value") and a.policy_value is not None:
                key = id(a.policy_value)
                if key not in seen:
                    seen[key] = f"critic_shared" if not critics else f"critic_{i}"
                    critics.append((seen[key], a.policy_value))
        layout = "centralized" if len(critics) == 1 else "decentralized"
        return layout, critics

    def single_agent_env_step(self, tick, agent_id=None) -> EnvStep:
        """Simulates one agent taking a single step in the environment, and returns
        the resulting transition information. Picks a random agent to act if agent_id not specified.

        Args:
            tick: A counter within a larger timestep, used to trigger periodic
                environment updates (e.g., after N ticks, where N is the
                number of agents).

        Returns:
            An EnvStep object containing the complete transition information,
            including the state before and after the action, the acting
            agent's ID, the action taken, and the resulting reward vector.
        """
        if agent_id is None:
            agent_id = random.randint(0, self.train_config.num_agents - 1)
        state = (
            self.env.get_state()
        )  # this is assumed to be a dict with "agents" and "apples"
        positions = []
        for i in range(self.train_config.num_agents):
            positions.append(self._agents_list[i].position)
        action = self.agent_controller.agent_get_action(
            self.env, agent_id, self.train_config.epsilon
        )
        action_result = self.env.process_action(
            agent_id, self._agents_list[agent_id].position.copy(), action
        )

        self._agents_list[agent_id].collected_apples += action_result.reward_vector[
            agent_id
        ]
        return EnvStep(
            old_state=state,
            new_state=self.env.get_state(),
            acting_agent_id=agent_id,
            old_positions=positions,
            action=action,
            reward_vector=action_result.reward_vector,
            picked=action_result.picked,
        )

    @abstractmethod
    def init_agents_for_eval(self):
        raise NotImplementedError

    def save_networks(self, path: str, global_step: int | None = None) -> None:
        os.makedirs(path, exist_ok=True)
        layout, critics = self._collect_unique_critics()

        self.save_rng_state()

        print("Saving networks: ", random.getstate()[1][0])

        payload = {
            "step": global_step,
            "layout": layout,  # 'centralized' or 'decentralized'
            "rng_state": self.rng_state,  # <<--- new
            "critics": [],  # list of {name, blob}
            "actors": [],  # list aligned to agents_list (None if missing)
        }

        # critics (unique, deduped)
        for name, crit in critics:
            payload["critics"].append({"name": name, "blob": crit.export_net_state()})

        # actors (one per agent, keep alignment)
        for a in self._agents_list:
            if hasattr(a, "policy_network") and a.policy_network is not None:
                pn = a.policy_network
                payload["actors"].append(pn.export_net_state())
            else:
                payload["actors"].append(None)

        global times
        dst = os.path.join(path, f"{self.name}_ckpt_{times}.pt")
        times += 1
        torch.save(payload, dst)

    def load_networks(self, name: str) -> int:
        path = os.path.join(CHECKPOINT_DIR, name)

        # 1) Find the newest ckpt_{number}.pt by numeric suffix
        candidates = glob.glob(os.path.join(path, "*_ckpt_*.pt"))

        latest_step = None
        latest_path = None

        for f in candidates:
            m = re.search(r"ckpt_(\d+)\.pt$", os.path.basename(f))
            if m:
                step = int(m.group(1))
                if latest_step is None or step > latest_step:
                    latest_step, latest_path = step, f

        # Fallback to legacy file if no step-tagged snapshot is present
        if latest_path is None:
            latest_path = os.path.join(path, f"{name}_ckpt.pt")

        ckpt = torch.load(latest_path, map_location="cpu")

        # 2) Set global 'times' to the detected step (or ckpt['step'] if present)
        step_in_ckpt = ckpt.get("step")
        final_step = (
            step_in_ckpt if isinstance(step_in_ckpt, int) else (latest_step or 0)
        )
        global times
        times = final_step

        # map critics back
        layout_saved = ckpt.get("layout", "centralized")
        crit_blobs = ckpt.get("critics", [])
        layout_now, critics_now = self._collect_unique_critics()

        if layout_saved == "centralized":
            if crit_blobs and critics_now:
                blob = crit_blobs[0]["blob"]
                _, obj = critics_now[0]
                obj.import_net_state(blob)
        else:
            for (name_now, obj_now), saved in zip(critics_now, crit_blobs):
                blob = saved["blob"]
                obj_now.import_net_state(blob)

        # actors (aligned with agents_list)
        act_blobs = ckpt.get("actors", [])
        for agent, blob in zip(self._agents_list, act_blobs):
            if (
                blob
                and hasattr(agent, "policy_network")
                and agent.policy_network is not None
            ):
                pn = agent.policy_network
                pn.import_net_state(blob, device=DEVICE)

        # restore RNG if present
        rng = ckpt.get("rng_state")
        if rng is not None:
            self.rng_state = rng
            self.restore_rng_state()

        print(f"Restoring from: {latest_path}")
        print("Restoring networks: ", random.getstate()[1][0])
        return final_step

    def restore_all(self):
        self.load_networks(self.name)
        agent_pos, apples = self._load_env_state()
        return agent_pos, apples

    def train_agent(self, agent) -> None:
        """
        Handles the "how" of training. It will train EVERY valid network
        that belongs to the agent.
        """
        # Train the critic/value network if it exists
        if hasattr(agent, "policy_value") and agent.policy_value is not None:
            network = agent.policy_value
            loss = None
            # This also correctly handles the CNN's special train method name
            if isinstance(network, CNN):
                loss = network.train_batch()
            else:
                loss = network.train()
            if loss is not None:
                self.critic_loss.append(loss)

        # Train the actor network if it exists
        if hasattr(agent, "policy_network") and agent.policy_network is not None:
            agent.policy_network.train()

    def training_step(self, step: int) -> None:
        """
        1. For this step/second, on a random subset of agents, the agents act and store the state info into neural net input
        2. The environment updates (spawn/despawn apples)
        3. Then for all agents i, if agent i has observed enough samples, train it.

        Args:
            step: The current training step.
        """
        if self.debug:
            return
        # Collect and process observations
        self.step_and_collect_observation(step)

        self.env.spawn_despawn()

        self.apple_count_history.append(self.env.apples.sum())

        for (
            agent
        ) in self._agents_list:  # IMPORTANT test for mlp where the primary_network is
            agent_network = agent.get_primary_network()
            if (
                agent_network
                and len(agent_network.batch_states) >= self.train_config.batch_size
            ):
                self.train_agent(agent)
        return

    def training_loop(self) -> Tuple[floating, ...] | None:
        """Train the value function."""
        try:
            log_constant = 0.02 * self.train_config.timesteps
            eval_constant = (
                self.train_config.eval_interval * self.train_config.timesteps
            )

            sample_state, sample_state5, sample_state6 = generate_sample_states(
                self.env.length, self.env.width, self.train_config.num_agents
            )

            for step in range(self.train_config.timesteps):
                self.training_step(step)

                self.record_spawn_despawn_stats()

                # Log progress and update a learning rate
                if step % log_constant == 0:
                    self.log_progress(sample_state, sample_state5, sample_state6)
                    memory_snapshot(label=f"step={step}", show_children=True)
                    # self._save_best_networks()
                self.update_lr(step)

                # Periodic evaluation
                if step > 0:
                    if (step % eval_constant == 0) and (
                        step != self.train_config.timesteps - 1
                    ):
                        self.evaluate_checkpoint(step, self.train_config.seed).log(
                            self.logger
                        )
                        # self.generate_plots()
            # Final evaluation
            self.generate_plots()
            if not self.debug:
                return self._evaluate_final()
            else:
                return None
        except Exception as e:
            self.logger.error(f"Failed during training: {e}")
            return None

    def record_spawn_despawn_stats(self):
        apples_despawned_before = self.env.total_despawned
        apples_spawned_before = (
            self.env.total_apples
        )  # Using total_apples for spawn count

        # 2. Run the environment's spawn/despawn logic
        # This will now correctly increment the cumulative counters inside the env
        self.env.spawn_despawn()
        apples_despawned_this_step = self.env.total_despawned - apples_despawned_before
        apples_spawned_this_step = self.env.total_apples - apples_spawned_before

        self.apples_spawned_per_step_history.append(apples_spawned_this_step)
        if self.env.total_picked > 0:
            ratio = self.env.total_despawned / self.env.total_picked
            self.despawn_to_pick_ratio_history.append(ratio)
        else:
            self.despawn_to_pick_ratio_history.append(0.0)

    def _evaluate_final(self) -> Tuple[floating, ...]:
        """Perform final evaluation."""
        mean_metrics = {
            "total_apples": [],
            "total_picked": [],
            "picked_per_agent": [],
            "per_agent": [],
            "average_distance": [],
            "apple_per_sec": [],
            "nearest_actions": [],
            "idle_actions": [],
        }

        for k in range(3):
            result = self.evaluate_checkpoint(
                self.train_config.timesteps - 1, self.train_config.seed + k
            )
            for i, key in enumerate(mean_metrics.keys()):
                mean_metrics[key].append(getattr(result, key))
        avg_total_apples = np.mean(mean_metrics["total_apples"])
        avg_total_picked = np.mean(mean_metrics["total_picked"])
        avg_picked_per_agent = np.mean(mean_metrics["picked_per_agent"])
        avg_per_agent = np.mean(mean_metrics["per_agent"])
        avg_average_distance = np.mean(mean_metrics["average_distance"])
        # Log final averages
        self.logger.info(f"Ratio picked: {avg_per_agent}")
        self.logger.info(f"Mean distance: {avg_average_distance}")
        self.logger.info(f"Total apples: {avg_total_apples}")
        self.logger.info(f"Total picked: {avg_total_picked}")
        self.logger.info(f"Picked per agents: {avg_picked_per_agent}")

        self.log_results_to_csv(
            avg_total_apples,
            avg_total_picked,
            avg_picked_per_agent,
            avg_per_agent,
            avg_average_distance,
        )
        return tuple(np.mean(val) for val in mean_metrics.values())

    @abstractmethod
    def build_experiment(
        self,
        view_controller_cls=ViewController,
        agent_controller_cls=AgentControllerCentralized,
        agent_type=SimpleAgent,
        value_network_cls=VNetwork,
        actor_network_cls=ActorNetwork,
        test=False,
        **kwargs,
    ):
        """the build experiment method should not have such strict parameters. Use kwargs for compatibility with other designs."""

        self.critic_view_controller: ViewController = view_controller_cls(
            self.train_config.critic_vision, self.train_config.new_input
        )
        self.actor_view_controller: ViewController = view_controller_cls(
            self.train_config.actor_vision, self.train_config.new_input
        )
        self.agent_controller: AgentController = agent_controller_cls(
            self._agents_list, self.critic_view_controller, self.actor_view_controller
        )
        self._init_agents_for_training(
            agent_type,
            self._init_critic_networks(value_network_cls),
            self._init_actor_networks(actor_network_cls),
            self._init_reward_networks(),
        )
        if not test:
            # NOTE this was causing problems for me so I remove it temporarily.
            # agent_pos = *self.restore_all() if self.train_config.skip else (None, None)
            self.env: Orchard = create_env(
                self.env_config,
                self.train_config.num_agents,
                None,
                None,
                self._agents_list,
                self.env_cls,
                debug=self.debug,
            )

    def log_results_to_csv(
        self,
        avg_total_apples,
        avg_total_picked,
        avg_picked_per_agent,
        avg_per_agent,
        avg_average_distance,
    ):
        """Appends the key results of an evaluation to a master CSV file."""
        results_file = OUT_DIR / "results_summary.csv"

        # Prepare the data row
        data_row = {
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": self.name,
            "num_agents": self.train_config.num_agents,
            "learning_rate": self.train_config.alpha,
            "discount_factor": self.train_config.discount,
            "batch_size": self.train_config.batch_size,
            "training_timesteps": self.train_config.timesteps,
            "eval_timesteps": self.train_config.eval_timesteps,
            "width": self.env_config.width,
            "length": self.env_config.length,
            "hidden_dim": self.train_config.hidden_dimensions,
            "num_layers": self.train_config.num_layers,
            "env_cls": self.env_config.env_cls,
            "ratio_picked": avg_per_agent,
            "apples_per_agent": avg_picked_per_agent,
            "total_picked": avg_total_picked,
            "total_spawned": avg_total_apples,
            "avg_distance": avg_average_distance,
        }

        # Write to CSV
        file_exists = results_file.exists()
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data_row.keys())
            if not file_exists:
                writer.writeheader()  # Write header only once
            writer.writerow(data_row)

        print(f"--- Results logged to {results_file} ---")

    def _init_agents_for_training(
        self, agent_cls, value_networks, actor_networks, reward_networks
    ):
        """Create agents of type agent_cls using provided parameters as info, and store in self._agents_list.

        Args:
            agent_cls: The class of the agent to create.
            value_networks: The value networks to assign to the agents.
            actor_networks: The actor networks to assign to the agents.
            reward_networks: The reward networks to assign to the agents.
        """
        info = self.agent_info
        for num in range(self.train_config.num_agents):
            info.agent_id = num
            agent = agent_cls(info)
            if hasattr(agent, "policy_value"):
                agent.policy_value = value_networks[num]
            if hasattr(agent, "policy_network"):
                agent.policy_network = actor_networks[num]
            if hasattr(agent, "reward_network"):
                agent.reward_network = reward_networks[num]
            self._agents_list.append(agent)

    def train(self):
        self.build_experiment(view_controller_cls=VIEW_CONTROLLER_MAP[self.env_cls])
        self.training_loop()

    def _init_critic_networks(self, value_network_cls=VNetwork):
        if value_network_cls is None:
            return []
        critic_networks = []
        for _ in range(self.train_config.num_agents):
            # Get critic network vision
            if self.train_config.critic_vision != 0:
                if self.env_config.width != 1:
                    critic_input_dim = 2 * self.train_config.critic_vision**2 + 2
                else:
                    critic_input_dim = 2 * self.train_config.critic_vision + 1
            elif self.train_config.new_input:
                critic_input_dim = 3 * self.env_config.length * self.env_config.width
            else:
                critic_input_dim = (
                    2 * self.env_config.length * self.env_config.width + 2
                )
            critic_networks.append(
                value_network_cls(
                    critic_input_dim,
                    1,
                    self.train_config.alpha,
                    self.train_config.discount,
                    self.train_config.hidden_dimensions,
                    self.train_config.num_layers,
                )
            )
        return critic_networks

    def _init_actor_networks(self, actor_network_cls=ActorNetwork):
        if actor_network_cls is None:
            return []
        actor_networks = []
        for _ in range(self.train_config.num_agents):
            # Get actor network vision
            if self.train_config.actor_vision != 0:
                if self.env_config.width != 1:
                    actor_input_dim = self.train_config.actor_vision**2 + 1
                else:
                    actor_input_dim = self.train_config.actor_vision + 1
            elif self.train_config.new_input:
                # This branch was missing. Add it to handle the new input format.
                actor_input_dim = 3 * self.env_config.length * self.env_config.width
            else:
                actor_input_dim = self.env_config.length * self.env_config.width + 1
            actor_networks.append(
                actor_network_cls(
                    actor_input_dim,
                    5 if self.env_config.width > 1 else 3,
                    self.train_config.actor_alpha,
                    self.train_config.discount,
                    self.train_config.hidden_dimensions_actor,
                    self.train_config.num_layers_actor,
                )
            )
        return actor_networks

    def _init_reward_networks(self, reward_network_cls=RewardNetwork):
        reward_networks = []
        for _ in range(self.train_config.num_agents):
            # Get actor network vision
            if self.train_config.critic_vision != 0:
                if self.env_config.width != 1:
                    critic_input_dim = 2 * self.train_config.critic_vision**2 + 2
                else:
                    critic_input_dim = 2 * self.train_config.critic_vision + 1
            elif self.train_config.new_input:
                critic_input_dim = 3 * self.env_config.length * self.env_config.width
            else:
                critic_input_dim = (
                    2 * self.env_config.length * self.env_config.width + 2
                )
            reward_networks.append(
                reward_network_cls(
                    critic_input_dim,
                    1,
                    self.train_config.alpha,
                    self.train_config.discount,
                    self.train_config.hidden_dimensions,
                    self.train_config.num_layers,
                )
            )
        return reward_networks
