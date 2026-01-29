import numpy as np


class ViewController:
    def __init__(self):
        pass

    def state_to_nn_input(self, state, agent_id=None) -> np.ndarray:
        """Flattens an environment state into a 1D feature vector.

        Feature representation:
        1. Apples matrix
        2. Other-agents matrix with BOTH the current agent and the acting agent masked out
        3. Self-position matrix (1 at current agent position, 0 elsewhere)
        4. Actor-position matrix (1 at acting agent position, 0 elsewhere)
        5. I{Actor=Self} indicator
        6. Mode

        Args:
            state: dict containing 'agents', 'apples', 'mode', 'agent_positions', 'actor_id'
            agent_id: the id of the agent whose perspective we encode

        Returns:
            1D numpy array of features.
        """
        if agent_id is None:
            raise ValueError("agent_id must be provided")

        apples_matrix = state['apples'].copy()
        agents_matrix = state['agents'].copy()
        agent_positions = state['agent_positions']

        # Current (self) agent position
        self_r, self_c = agent_positions[agent_id]

        # Acting agent position
        actor_id = state['actor_id']
        actor_r, actor_c = agent_positions[actor_id]

        # 2. Other agents matrix: mask out BOTH self and actor
        other_agents = agents_matrix.copy()

        # Mask out self
        other_agents[self_r, self_c] = max(0, other_agents[self_r, self_c] - 1)

        # Mask out actor (avoid double-subtract if actor==self)
        if actor_id != agent_id:
            other_agents[actor_r, actor_c] = max(0, other_agents[actor_r, actor_c] - 1)

        # 3. Self position matrix
        self_pos = np.zeros_like(agents_matrix)
        self_pos[self_r, self_c] = 1

        # 4. Actor position matrix
        actor_pos = np.zeros_like(agents_matrix)
        actor_pos[actor_r, actor_c] = 1

        # 5-6. Scalars
        actor_is_self = 1 if actor_id == agent_id else 0
        mode = int(state['mode'])

        features = [
            apples_matrix.flatten(),
            other_agents.flatten(),
            self_pos.flatten(),
            actor_pos.flatten(),
            np.array([actor_is_self, mode], dtype=np.int64),
        ]

        # return np.concatenate(features)

        # Final feature vector
        # final_features = np.append(concatenated, state["actor_id"] == agent_id)
        final_features = np.array([state["actor_id"] == agent_id])
        final_features = np.append(final_features, state["mode"])
        final_features = np.append(final_features, state["apples"][actor_r][actor_c] >= 1)

        return final_features.astype(np.float32)

    def __call__(self, state, agent_id):
        """Make the controller callable for compatibility with existing code."""
        return self.state_to_nn_input(state, agent_id)
