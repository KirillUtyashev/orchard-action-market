import numpy as np


class ViewController:
    def __init__(self, input_dim, k):
        self.input_dim = input_dim
        self.k = k

    def state_to_nn_input(self, state, agent_id=None) -> np.ndarray:
        """
        Design A (self-centered apples) entity encoding (global observability, fixed-size, egocentric):

          Scalars: [actor_is_self, mode]
          Actor block (relative to self): [dx_actor_norm, dy_actor_norm, dist_actor_norm]
          Other agents (relative to self, deterministic order by id): for each j != self
              [dx_norm, dy_norm, dist_norm]
          Apples (top-K nearest to SELF, relative to self): K slots
              [dx_norm, dy_norm, dist_norm, mask]

        Extra rule to keep reward-relevant info recoverable with top-K:
          - If there is an apple at the actor cell, we force-include (actor_r, actor_c)
            among the K apple slots (even if it is far from self), so the network can infer
            "apple under actor" by checking whether any apple-slot (dx,dy) matches
            (dx_actor,dy_actor). [file:1]
        """
        if agent_id is None:
            raise ValueError("agent_id must be provided")

        actor_id = state["actor_id"]
        agent_positions = state["agent_positions"]  # index -> (r,c)
        self_r, self_c = agent_positions[agent_id]
        actor_r, actor_c = agent_positions[actor_id]

        # ---- keep your existing encodings unchanged ----
        if self.input_dim == 326:
            apples_matrix = state["apples"].copy()
            agents_matrix = state["agents"].copy()

            other_agents = agents_matrix.copy()
            other_agents[self_r, self_c] = max(0, other_agents[self_r, self_c] - 1)
            if actor_id != agent_id:
                other_agents[actor_r, actor_c] = max(0, other_agents[actor_r, actor_c] - 1)

            self_pos = np.zeros_like(agents_matrix)
            self_pos[self_r, self_c] = 1

            actor_pos = np.zeros_like(agents_matrix)
            actor_pos[actor_r, actor_c] = 1

            actor_is_self = 1 if actor_id == agent_id else 0
            mode = int(state["mode"])

            features = [
                apples_matrix.flatten(),
                other_agents.flatten(),
                self_pos.flatten(),
                actor_pos.flatten(),
                np.array([actor_is_self, mode], dtype=np.int64),
            ]
            return np.concatenate(features)

        elif self.input_dim == 3:
            actor_is_self = float(actor_id == agent_id)
            mode = float(state["mode"])
            apple_under_actor = float(state["apples"][actor_r, actor_c] >= 1)
            return np.array([actor_is_self, mode, apple_under_actor], dtype=np.float32)

        # ---- Design A entity encoding (self-centered apples) ----
        if not hasattr(self, "k") or self.k is None:
            raise ValueError("For entity encoding, set self.k (top-K apples).")

        apples_matrix = state["apples"]
        H, W = apples_matrix.shape
        denom_x = max(W - 1, 1)
        denom_y = max(H - 1, 1)
        dmax = float(np.sqrt((W - 1) ** 2 + (H - 1) ** 2))
        if dmax <= 0:
            dmax = 1.0

        def rel_norm(r_from, c_from, r_to, c_to):
            """Return (dx_norm, dy_norm, dist_norm) from (from)->(to), using fixed map bounds."""
            dx = c_to - c_from
            dy = r_to - r_from
            dxn = dx / denom_x
            dyn = dy / denom_y
            distn = float(np.sqrt(dx * dx + dy * dy)) / dmax
            return float(dxn), float(dyn), float(distn)

        actor_is_self = 1.0 if actor_id == agent_id else 0.0
        mode = float(int(state["mode"]))
        apple_under_actor = 1.0 if apples_matrix[actor_r, actor_c] > 0 else 0.0  # extra feature [file:1]

        feats = []
        # Scalars: add apple_under_actor
        feats.append(np.array([actor_is_self, mode, apple_under_actor], dtype=np.float32))

        # Actor block: actor position relative to self
        dxn_a, dyn_a, distn_a = rel_norm(self_r, self_c, actor_r, actor_c)
        feats.append(np.array([dxn_a, dyn_a, distn_a], dtype=np.float32))

        # Other agents: relative to self (includes actor too; redundancy is OK)
        for j, (rj, cj) in enumerate(agent_positions):
            if j == agent_id:
                continue
            dxn, dyn, distn = rel_norm(self_r, self_c, rj, cj)
            feats.append(np.array([dxn, dyn, distn], dtype=np.float32))

        # Apples: top-K nearest to SELF, encoded relative to self with mask padding
        apple_rc = np.argwhere(apples_matrix > 0)  # rows are [r, c]

        if apple_rc.size == 0:
            topk = np.empty((0, 2), dtype=np.int64)
        else:
            rs = apple_rc[:, 0]
            cs = apple_rc[:, 1]
            dx = cs - self_c
            dy = rs - self_r
            d2 = dx * dx + dy * dy
            # Deterministic: sort by distance^2, then dx, then dy
            order = np.lexsort((dy, dx, d2))
            topk = apple_rc[order[: self.k]]

        for idx in range(self.k):
            if idx < len(topk):
                r, c = int(topk[idx, 0]), int(topk[idx, 1])
                dxn, dyn, distn = rel_norm(self_r, self_c, r, c)  # relative to SELF
                feats.append(np.array([dxn, dyn, distn, 1.0], dtype=np.float32))
            else:
                feats.append(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

        out = np.concatenate(feats).astype(np.float32)
        return out

    def __call__(self, state, agent_id):
            """Make the controller callable for compatibility with existing code."""
            return self.state_to_nn_input(state, agent_id)
