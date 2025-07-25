import random
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch
import numpy as np

from configs.config import ExperimentConfig
from .controllers import AgentControllerCentralized, \
    AgentControllerDecentralized, ViewController
from main import run_environment_1d
from models.value_function import VNetwork
from agents.simple_agent import SimpleAgent
from agents.communicating_agent import CommAgent
from policies.random_policy import random_policy
from algorithm import Algorithm


class ValueFunction(Algorithm, ABC):
    """Base class for value function implementations."""

    def __init__(self, config: ExperimentConfig, name):
        """Initialize the value function algorithm."""
        super().__init__(config, name)

    @abstractmethod
    def _update_network_lr(self, lr: float) -> None:
        """Update learning rate for network(s)."""
        pass

    def update_lr(self, step: int) -> None:
        """Update a learning rate based on training progress."""
        # for threshold, lr in self.train_config.lr_schedule.items():
        #     if step == (threshold * self.train_config.timesteps):
        #         try:
        #             self._update_network_lr(lr)
        #             self.logger.info(f"Learning rate updated to {lr}")
        #         except Exception as e:
        #             self.logger.error(f"Failed to update learning rate: {e}")

    def agent_get_action(self, agent_id: int) -> int:
        if self.agents_list[agent_id].policy == "value_function":
            if random.random() < self.train_config.epsilon:
                action = random_policy(self.env.available_actions)
                if self.train_config.test:
                    self.count_random_actions += 1
            else:
                with torch.no_grad():
                    action = self.agent_controller.get_best_action(self.env.get_state(),
                                                                   agent_id,
                                                                   self.env.available_actions)
        elif self.agents_list[agent_id].policy is random_policy:
            action = self.agents_list[agent_id].policy(
                self.env.available_actions)
        else:
            action = self.agents_list[agent_id].policy(
                self.env.get_state(), self.agents_list[agent_id].position)
        return action


class CentralizedValueFunction(ValueFunction):
    """Centralized implementation of a value function."""
    def __init__(self, config: ExperimentConfig):
        """Initialize the value function algorithm."""
        super().__init__(config, f"Centralized-<{config.train_config.num_agents}>_agents-_length-<{config.env_config.length}>_width-<{config.env_config.width}>_s_target-<{config.env_config.s_target}>-apple_mean_lifetime-<{config.env_config.apple_mean_lifetime}>-<{config.train_config.alpha}>-discount-<{config.train_config.discount}>-hidden_dimensions-<{config.train_config.hidden_dimensions}>-dimensions-<{config.train_config.num_layers}>-vision-<{config.train_config.vision}>")

    def update_actor(self, r_ratio=None) -> None:
        pass

    def update_critic(self) -> None:
        self.agents_list[0].policy_value.train()

    def _format_env_step_return(self, state: dict, new_state: dict,
                                reward: float, agent_id: int,
                                positions: np.ndarray, action: int,
                                old_pos: np.ndarray) -> Tuple[dict, dict, float, int, np.array]:
        return state, new_state, reward, agent_id, old_pos

    def init_agents_for_eval(self) -> List[SimpleAgent]:
        a_list = []
        for ii in range(len(self.agents_list)):
            trained_agent = SimpleAgent(policy="value_function")
            trained_agent.policy_value = self.network_for_eval[0]
            a_list.append(trained_agent)
        return a_list

    def save_networks(self, path: str) -> None:
        torch.save(self.network_for_eval[0].function.state_dict(),
                   path + "/" + self.name + "_cen_" + ".pt")
        print("saved_network", time.time())

    def _update_network_lr(self, lr: float) -> None:
        """Update learning rate for centralized network."""
        for g in self.agents_list[0].policy_value.optimizer.param_groups:
            g['lr'] = lr

    def collect_observation(self, step: int) -> None:
        """Collect observations with vectorized operations where possible."""
        try:
            for tick in range(self.train_config.num_agents):
                s, new_s, r, agent_id, old_pos = self.env_step(step, tick)
                processed_state = self.view_controller.process_state(s, old_pos)
                processed_new_state = self.view_controller.process_state(new_s, self.agents_list[0].position)
                self.agents_list[0].add_experience(processed_state[:self.network_for_eval[0].get_input_dim()], processed_new_state[:self.network_for_eval[0].get_input_dim()], r)

        except Exception as e:
            self.logger.error(f"Error collecting observations: {e}")
            raise

    def run(self) -> Tuple[np.floating, ...]:
        """Run the centralized training process."""
        try:
            self.view_controller = ViewController(self.train_config.vision)
            self.agent_controller = AgentControllerCentralized(self.agents_list, self.view_controller)

            # Initialize network and agents
            if not self.train_config.alt_input:
                network = VNetwork(self.env_config.width * self.env_config.length, self.train_config.alpha, self.train_config.discount, self.train_config.hidden_dimensions, self.train_config.num_layers)
            else:
                if self.env_config.width != 1:
                    network = VNetwork(self.train_config.vision ** 2 + 1, self.train_config.alpha, self.train_config.discount, self.train_config.hidden_dimensions, self.train_config.num_layers)
                else:
                    network = VNetwork(self.train_config.vision + 1, self.train_config.alpha, self.train_config.discount, self.train_config.hidden_dimensions, self.train_config.num_layers)

            for _ in range(self.train_config.num_agents):
                agent = SimpleAgent(policy="value_function")
                agent.policy_value = network
                self.agents_list.append(agent)
            self.network_for_eval = [network]
            return self.train()

        except Exception as e:
            self.logger.error(f"Failed to run centralized training: {e}")
            raise


class DecentralizedValueFunction(ValueFunction):
    """Decentralized implementation of a value function."""
    def __init__(self, config: ExperimentConfig):
        """Initialize the value function algorithm."""
        super().__init__(config, f"Decentralized-<{config.train_config.num_agents}>_agents-_length-<{config.env_config.length}>_width-<{config.env_config.width}>_s_target-<{config.env_config.s_target}>-alpha-<{config.train_config.alpha}>-apple_mean_lifetime-<{config.env_config.apple_mean_lifetime}>-<{config.train_config.hidden_dimensions}>-<{config.train_config.num_layers}>-vision-<{config.train_config.vision}>")

        self.network_list = []

    def _format_env_step_return(self, state: dict, new_state: dict,
                                reward: float, agent_id: int,
                                positions: np.ndarray, action: int,
                                old_pos: np.ndarray) -> Tuple[dict, dict, float, np.ndarray, int]:
        return state, new_state, reward, old_pos, agent_id

    def update_actor(self, t_ratio: Optional[float] = None) -> None:
        return

    def init_agents_for_eval(self) -> List[CommAgent]:
        a_list = []
        for ii in range(len(self.agents_list)):
            trained_agent = CommAgent(policy="value_function")
            trained_agent.policy_value = self.network_list[ii]
            a_list.append(trained_agent)
        return a_list

    def save_networks(self, path: str) -> None:
        for nummer, netwk in enumerate(self.network_list):
            torch.save(netwk.function.state_dict(),
                       path + "/" + self.name + "_decen_" + str(
                           nummer) + ".pt")

    def update_critic(self):
        for agent in self.agents_list:
            agent.policy_value.train()

    def _update_network_lr(self, lr: float) -> None:
        """Update learning rate for all networks."""
        for network in self.network_list:
            for g in network.optimizer.param_groups:
                g['lr'] = lr

    def collect_observation(self, step: int) -> None:
        """Collect observations for decentralized training."""
        try:
            for tick in range(self.train_config.num_agents):
                s, new_s, r, old_pos, agent = self.env_step(step, tick)

                for each_agent in range(len(self.agents_list)):
                    curr_pos = self.agents_list[each_agent].position
                    reward = r if each_agent == agent else 0

                    processed_state = self.view_controller.process_state(s,
                                                                         old_pos if each_agent == agent else curr_pos)
                    processed_new_state = self.view_controller.process_state(new_s, curr_pos)

                    self.agents_list[each_agent].add_experience(
                        processed_state, processed_new_state, reward)

        except Exception as e:
            self.logger.error(f"Error collecting observations: {e}")
            raise

    def run(self) -> Tuple[np.floating, ...]:
        """Run the decentralized training process."""
        try:
            self.view_controller = ViewController(self.train_config.vision)
            self.agent_controller = AgentControllerDecentralized(self.agents_list, self.view_controller)

            # Initialize networks and agents
            for _ in range(self.train_config.num_agents):
                agent = CommAgent(policy="value_function")
                if self.train_config.alt_input:
                    if self.env_config.width != 1:
                        network = VNetwork(self.train_config.vision ** 2 + 1, self.train_config.alpha, self.train_config.discount, self.train_config.hidden_dimensions, self.train_config.num_layers)
                    else:
                        network = VNetwork(self.train_config.vision + 1, self.train_config.alpha, self.train_config.discount, self.train_config.hidden_dimensions, self.train_config.num_layers)
                else:
                    network = VNetwork(self.env_config.length * self.env_config.width + 1, self.train_config.alpha, self.train_config.discount, self.train_config.hidden_dimensions, self.train_config.num_layers)
                agent.policy_value = network
                self.network_list.append(network)
                self.agents_list.append(agent)

            self.network_for_eval = self.network_list
            return self.train()

        except Exception as e:
            self.logger.error(f"Failed to run decentralized training: {e}")
            raise


def evaluate_policy(env_config,
                    num_agents,
                    agent_factory,
                    timesteps=10000,
                    seed=42):
    """
    Runs `run_environment_1d` with agents created by `agent_factory` and
    returns a dict of metrics.

    agent_factory: fn(i: int) -> Agent
    """
    # seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # create agents
    agents = [agent_factory(i) for i in range(num_agents)]

    # run the environment
    total_apples, total_picked, picked_per_agent, per_agent, mean_dist, apples_per_sec = \
        run_environment_1d(
            num_agents,
            env_config.length,
            env_config.width,
            None, None,
            f"Eval-{env_config.length}x{env_config.width}",
            agents_list=agents,
            spawn_algo=env_config.spawn_algo,
            despawn_algo=env_config.despawn_algo,
            timesteps=timesteps,
            s_target=env_config.s_target,
            apple_mean_lifetime=env_config.apple_mean_lifetime
        )

    return {
        "total_apples": int(total_apples),
        "total_picked": int(total_picked),
        "picked_per_agent": float(picked_per_agent),
        "ratio_per_agent": float(per_agent),
        "mean_distance": float(mean_dist),
        "apples_per_sec": float(apples_per_sec)
    }


def make_baseline_factory(policy_name):
    """Returns an agent_factory for SimpleAgent(policy=policy_name)."""
    def factory(i):
        return SimpleAgent(policy=policy_name)
    return factory


def make_network_factory(checkpoint_path, input_dim, discount, alpha, hidden_dim, num_layers):
    """
    Returns an agent_factory that:
      - instantiates a shared VNetwork
      - loads from checkpoint_path
      - assigns it to each SimpleAgent
    """
    # build and load once
    net = VNetwork(input_dim, alpha, discount, hidden_dim, num_layers)
    net.function.load_state_dict(torch.load(checkpoint_path))
    net.function.eval()

    def factory(i):
        agent = SimpleAgent(policy="value_function")
        agent.policy_value = net
        return agent

    return factory
