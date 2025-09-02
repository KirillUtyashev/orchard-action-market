import glob
import logging
import re
from pathlib import Path
import torch
from numpy import floating

from agents.agent import AgentInfo
from agents.simple_agent import SimpleAgent
from config import CHECKPOINT_DIR, DEVICE
from helpers.controllers import AgentControllerCentralized, ViewController
from main import eval_performance
from models.actor_network import ActorNetwork
from models.value_function import VNetwork
from plots import add_to_plots, graph_plots
from orchard.environment import *
from helpers.helpers import generate_sample_states
import os
import time
from policies.random_policy import random_policy
import psutil
from dataclasses import dataclass
from typing import Tuple

times = 0


@dataclass
class EnvStep:
    old_state: dict
    new_state: dict
    picker_reward: int
    acting_agent_id: int
    old_positions: list
    action: int
    apple_owner_id: Optional[int] = None
    apple_owner_reward: Optional[int] = None


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
    def __init__(self, config, name):
        self.train_config = config.train_config
        self.env_config = config.env_config
        self.env = None
        self.name = name
        self.debug = config.debug
        self.rng_state = None


        log_folder = Path("logs")
        log_folder.mkdir(parents=True, exist_ok=True)

        graph_folder = Path("graphs")
        graph_folder.mkdir(parents=True, exist_ok=True)

        name_folder = graph_folder / self.name
        name_folder.mkdir(parents=True, exist_ok=True)

        self.graphs_out_path = name_folder

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
        self.v_weights = {}
        self.critic_view_controller = None
        self.actor_view_controller = None
        self.agent_controller = None
        self.agent_info = AgentInfo(
            policy=self.train_config.policy,
            num_agents=self.train_config.num_agents
        )
        self.env_cls = OrchardBasic if self.env_config.env_cls == "OrchardBasic" else OrchardSelfless

        if self.train_config.test:
            self.count_random_actions = 0

    def create_env(self, agent_pos, apples, env_cls=OrchardBasic):
        env = env_cls(self.env_config.length, self.env_config.width, self.train_config.num_agents, self.agents_list, s_target=self.env_config.s_target, apple_mean_lifetime=self.env_config.apple_mean_lifetime)
        env.initialize(self.agents_list, agent_pos=agent_pos, apples=apples)
        return env

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

    def log_progress(self, sample_state, sample_state5, sample_state6):
        agent_obs = []
        for i in range(self.train_config.num_agents):
            agent_obs.append(self.critic_view_controller.process_state(sample_state, sample_state["poses"][i]))
        v_value = self.agent_controller.get_collective_value(agent_obs, 0)
        agent_obs = []
        for i in range(self.train_config.num_agents):
            agent_obs.append(self.critic_view_controller.process_state(sample_state5, sample_state5["poses"][i]))
        v_value5 = self.agent_controller.get_collective_value(agent_obs, 0)
        agent_obs = []
        for i in range(self.train_config.num_agents):
            agent_obs.append(self.critic_view_controller.process_state(sample_state6, sample_state6["poses"][i]))
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

    def eval_network(self, seed: int) -> EvalResult:
        """Run network evaluation"""

        self.save_rng_state()
        print("Before eval: ", random.getstate()[1][0])

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        agents_list, agent_controller = self.init_agents_for_eval()

        env = self.create_env(None, None, self.env_cls)

        with torch.no_grad():
            results = eval_performance(
                num_agents=self.train_config.num_agents,
                agent_controller=agent_controller,
                env=env,
                name=self.name,
                agents_list=agents_list,
                timesteps=10000,
                epsilon=self.train_config.epsilon
            )

        # Create EvalResult from returned tuple
        eval_result = EvalResult(*results)
        print("After eval: ", random.getstate()[1][0])

        self.restore_rng_state()
        print("Back to initial: ", random.getstate()[1][0])

        # Save networks
        self._save_best_networks()

        return eval_result

    def _save_best_networks(self):
        """Save the current best networks"""
        print("saving best")
        path = os.path.join(CHECKPOINT_DIR, self.name)
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
        if not self.agents_list:
            self.logger.error("No agents to save positions for.")
            return

        positions = np.asarray([a.position for a in self.agents_list], dtype=np.int32)  # shape: [num_agents, 2] (or whatever your position shape is)
        out_dir = Path(CHECKPOINT_DIR) / self.name
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / f"agent_positions_{when}.npy", positions)
        np.savetxt(out_dir / f"agent_positions_{when}.csv", positions, fmt="%d", delimiter=",")

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

        out_dir = Path(CHECKPOINT_DIR) / self.name
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
        for i, a in enumerate(self.agents_list):
            if hasattr(a, "policy_value") and a.policy_value is not None:
                key = id(a.policy_value)
                if key not in seen:
                    seen[key] = f"critic_shared" if not critics else f"critic_{i}"
                    critics.append((seen[key], a.policy_value))
        layout = "centralized" if len(critics) == 1 else "decentralized"
        return layout, critics

    def env_step(self, tick):
        agent_id = random.randint(0, self.train_config.num_agents - 1)
        state = self.env.get_state()  # this is assumed to be a dict with "agents" and "apples"
        positions = []
        for i in range(self.train_config.num_agents):
            positions.append(self.agents_list[i].position)
        action = self.agent_controller.agent_get_action(self.env, agent_id, self.train_config.epsilon)
        action_result = self.env.process_action(agent_id, self.agents_list[agent_id].position.copy(), action)
        self.agents_list[agent_id].position = action_result.new_position.copy()
        if tick == self.train_config.num_agents - 1:
            self.env.apples_despawned += self.env.despawn_algorithm(self.env, self.env.despawn_rate)
            self.env.total_apples += self.env.spawn_algorithm(self.env, self.env.spawn_rate)
        self.agents_list[agent_id].collected_apples += action_result.picker_reward
        return EnvStep(
            old_state=state,
            new_state=self.env.get_state(),
            picker_reward=action_result.picker_reward,
            acting_agent_id=agent_id,
            old_positions=positions,
            action=action,
            apple_owner_id=action_result.owner_id,
            apple_owner_reward=action_result.owner_reward
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
            "layout": layout,                    # 'centralized' or 'decentralized'
            "rng_state": self.rng_state,         # <<--- new
            "critics": [],                       # list of {name, blob}
            "actors": [],                        # list aligned to agents_list (None if missing)
        }

        # critics (unique, deduped)
        for name, crit in critics:
            payload["critics"].append({"name": name, "blob": crit.export_net_state()})

        # actors (one per agent, keep alignment)
        for a in self.agents_list:
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
        final_step = step_in_ckpt if isinstance(step_in_ckpt, int) else (latest_step or 0)
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
        for agent, blob in zip(self.agents_list, act_blobs):
            if blob and hasattr(agent, "policy_network") and agent.policy_network is not None:
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

    def training_step(self, step):
        # Collect and process observations
        self.collect_observation(step)

        # Train if enough samples collected
        if hasattr(self.agents_list[0], "policy_value"):
            for i in range(self.train_config.num_agents):
                if len(self.agents_list[i].policy_value.batch_states) >= self.train_config.batch_size:
                    self.agents_list[i].policy_value.train()

        if hasattr(self.agents_list[0], "policy_network"):
            for i in range(self.train_config.num_agents):
                if len(self.agents_list[i].policy_network.batch_states) >= self.train_config.batch_size:
                    self.agents_list[i].policy_network.train()

    def training_loop(self) -> Tuple[floating, ...] | None:
        """Train the value function."""
        try:
            # if self.train_config.timesteps < 2000000:
            #     log_constant = 0.02 * 2000000
            #     eval_constant = 0.1 * 2000000
            # else:
            log_constant = 0.02 * self.train_config.timesteps
            eval_constant = 0.1 * self.train_config.timesteps

            sample_state, sample_state5, sample_state6 = generate_sample_states(
                self.env.length, self.env.width, self.train_config.num_agents)

            for step in range(self.train_config.timesteps):
                self.training_step(step)

                # Log progress and update a learning rate
                if step % log_constant == 0:
                    self.log_progress(sample_state, sample_state5, sample_state6)
                    if self.debug:
                        memory_snapshot(label=f"step={step}", show_children=True)
                    # self._save_best_networks()
                self.update_lr(step)

                # Periodic evaluation
                if (step % eval_constant == 0) and (step != self.train_config.timesteps - 1):
                    self.evaluate_checkpoint(step, self.train_config.seed).log(self.logger)
                    graph_plots(self.name, self.weights_plot, self.critic_loss, self.loss_plot, self.loss_plot5, self.loss_plot6, self.v_weights)
            # Final evaluation
            graph_plots(self.name, self.weights_plot, self.critic_loss, self.loss_plot, self.loss_plot5, self.loss_plot6, self.v_weights)
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
    def build_experiment(self, view_controller_cls=ViewController, agent_controller_cls=AgentControllerCentralized,
                         agent_type=SimpleAgent, value_network_cls=VNetwork, actor_network_cls=ActorNetwork):
        self.critic_view_controller = view_controller_cls(self.train_config.critic_vision)
        self.actor_view_controller = view_controller_cls(self.train_config.actor_vision)
        self.agent_controller = agent_controller_cls(self.agents_list, self.critic_view_controller, self.actor_view_controller)
        self._init_agents_for_training(agent_type, self._init_critic_networks(value_network_cls), self._init_actor_networks(actor_network_cls))
        self.env = self.create_env(*self.restore_all() if self.train_config.skip else (None, None), self.env_cls)

    def _init_agents_for_training(self, agent_cls, value_networks, actor_networks):
        info = self.agent_info
        for num in range(self.train_config.num_agents):
            info.agent_id = num
            agent = agent_cls(info)
            if hasattr(agent, "policy_value"):
                agent.policy_value = value_networks[num]
            if hasattr(agent, "policy_network"):
                agent.policy_network = actor_networks[num]
            self.agents_list.append(agent)

    def train(self):
        self.build_experiment()
        self.training_loop()

    def _init_critic_networks(self, value_network_cls=VNetwork):
        critic_networks = []
        for _ in range(self.train_config.num_agents):
            # Get critic network vision
            if self.train_config.critic_vision != 0:
                if self.env_config.width != 1:
                    critic_input_dim = self.train_config.critic_vision ** 2 + 1
                else:
                    critic_input_dim = self.train_config.critic_vision + 1
            else:
                critic_input_dim = self.env_config.length * self.env_config.width + 1
            critic_networks.append(value_network_cls(critic_input_dim, 1, self.train_config.alpha, self.train_config.discount, self.train_config.hidden_dimensions, self.train_config.num_layers))
        return critic_networks

    def _init_actor_networks(self, actor_network_cls=ActorNetwork):
        actor_networks = []
        for _ in range(self.train_config.num_agents):
            # Get actor network vision
            if self.train_config.actor_vision != 0:
                if self.env_config.width != 1:
                    actor_input_dim = self.train_config.actor_vision ** 2 + 1
                else:
                    actor_input_dim = self.train_config.actor_vision + 1
            else:
                actor_input_dim = self.env_config.length * self.env_config.width + 1
            actor_networks.append(actor_network_cls(actor_input_dim, 5 if self.env_config.width > 1 else 3, self.train_config.actor_alpha, self.train_config.discount, self.train_config.hidden_dimensions_actor, self.train_config.num_layers_actor))
        return actor_networks
