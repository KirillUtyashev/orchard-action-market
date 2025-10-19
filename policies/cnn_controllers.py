# --- START OF FILE helpers/controllers_cnn.py ---

from typing_extensions import override
import torch
import random
from config import get_config
from helpers.controllers import AgentControllerValue
from orchard.environment import OrchardBasic
from policies.random_policy import random_policy


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
            immediate_reward, next_agents_map, next_apples_map, _ = env.calculate_ir(
                self.agents_list[agent_id].position, action.vector
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
