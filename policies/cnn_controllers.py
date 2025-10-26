# --- START OF FILE helpers/controllers_cnn.py ---

import numpy as np
from typing_extensions import override
import torch
import random
from config import get_config
from helpers.controllers import AgentControllerValue
from orchard.environment import OrchardBasic
from policies.random_policy import random_policy
from helpers.controllers import AgentController


class AgentControllerCentralizedCNN(AgentControllerValue):
    """
    An agent controller for a centralized CNN-based value function.
    It implements a greedy policy by querying the CNN for the value of
    potential next states. It does NOT use a ViewController.
    """

    def __init__(self, agents_list, test=False):
        # We pass None for the view controllers because this class doesn't use them.
        super().__init__(agents_list, critic_view_controller=None, test=test)

    @override
    def get_collective_value(self, processed_state, agent_id):
        # For the centralized case, any agent can query the single network.
        # The 'processed_state' is already a 3D tensor ready for the CNN.
        network = self.agents_list[0].policy_value
        return network.get_value_function(processed_state)

    @override
    def get_best_action(self, env: OrchardBasic, agent_id, **kwargs):
        """
        Finds the best action by simulating each possible move and evaluating
        the resulting state with the centralized CNN.

        **kwargs: Not used here but kept for compatibility.
        """
        best_action_idx = env.available_actions.STAY.idx  # Default action
        max_q_value = -float("inf")

        # Get the shared CNN from the first agent
        network = self.agents_list[0].policy_value

        for action in env.available_actions:
            # 1. Simulate the action to get the immediate reward and next raw state
            immediate_reward, next_agents_map, next_apples_map, _ = (
                env.get_next_state_and_reward(
                    self.agents_list[agent_id].position, action.vector
                )
            )

            next_raw_state = {"agents": next_agents_map, "apples": next_apples_map}

            # 2. Process the raw next state using the CNN's internal method
            processed_next_state = network.raw_state_to_nn_input(next_raw_state)

            # 3. Get the value of the next state from the CNN
            next_state_value = self.get_collective_value(processed_next_state, agent_id)

            # 4. Calculate the total Q-value for this action
            q_value = immediate_reward + get_config()["discount"] * next_state_value

            if q_value > max_q_value:
                max_q_value = q_value
                best_action_idx = action.idx

        return best_action_idx


class AgentControllerDecentralizedCNN(AgentControllerValue):
    """An agent controller for decentralized CNN-based value functions."""

    def __init__(self, agents_list, test=False):
        super().__init__(agents_list, critic_view_controller=None, test=test)

    @override
    def get_collective_value(self, processed_states: list, agent_id):
        """
        Calculates the collective value by summing the individual value predictions
        from each agent's personal network.

        This is needed because our cnn use different state processing than original decentralized.

        Args:
            processed_states: A LIST of processed 3D numpy arrays, one for each agent.
        """
        total_value = 0
        for i, agent in enumerate(self.agents_list):
            # Each agent uses its own network to evaluate its own processed state.
            value = agent.policy_value.get_value_function(processed_states[i])
            total_value += value
        return total_value

    @override
    def get_best_action(self, env: OrchardBasic, agent_id, **kwargs):
        best_action_idx = env.available_actions.STAY.idx
        max_q_value = -float("inf")

        for action in env.available_actions:
            team_reward, next_agents_map, next_apples_map, next_pos_acting_agent = (
                env.get_next_state_and_reward(  # NOTE the ir is 0, or 1 so it is really the team reward.
                    self.agents_list[agent_id].position, action.vector
                )
            )
            next_raw_state = {"agents": next_agents_map, "apples": next_apples_map}

            # --- KEY DECENTRALIZED LOGIC ---
            # We must generate a unique, processed next-state for EACH agent.
            processed_next_states = []
            for i, agent in enumerate(self.agents_list):
                network = agent.policy_value
                # The acting agent is at a new position; others are at their old positions.
                agent_pos_for_state = (
                    next_pos_acting_agent if i == agent_id else agent.position
                )
                processed_next_states.append(
                    network.raw_state_to_nn_input(
                        next_raw_state, agent_pos=agent_pos_for_state
                    )
                )

            next_state_sum_over_all_agents = self.get_collective_value(
                processed_next_states, agent_id
            )
            q_value = team_reward + get_config()["discount"] * next_state_sum_over_all_agents

            if q_value > max_q_value:
                max_q_value = q_value
                best_action_idx = action.idx

        return best_action_idx


class AgentControllerActorCriticCNN(AgentController):
    def __init__(self, agents_list):
        # No ViewController is needed for CNNs
        super().__init__(
            agents_list, critic_view_controller=None, actor_view_controller=None
        )

    @override
    def agent_get_action(self, env, agent_id, epsilon=None):
        """
        Gets an action by processing the state with the agent's personal ActorCNN
        and sampling from the resulting probability distribution.
        """
        agent = self.agents_list[agent_id]

        # The actor network is now a CNN
        actor_network = agent.policy_network

        # 1. Process the raw state using the actor's own method
        processed_state = actor_network.raw_state_to_nn_input(
            env.get_state(), agent_pos=agent.position
        )

        # 2. Get action probabilities from the actor network
        with torch.no_grad():
            action_probs = actor_network.get_action_probabilities(processed_state)

        # 3. Sample an action from the distribution
        num_actions = len(env.available_actions)
        action_idx = np.random.choice(num_actions, p=action_probs)

        return action_idx
