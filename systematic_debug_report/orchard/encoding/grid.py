"""Grid-based encoders for the orchard environment.

Active encoders:
  - CentralizedTaskGridEncoder: centralized, T+N+1 channels, N scalars
  - BlindTaskGridEncoder: dec O(1), 4 channels, 3 scalars
  - FilteredTaskGridEncoder: dec O(1), 6 channels, 3 scalars
"""

from __future__ import annotations

import torch

from orchard.encoding.base import GridEncoder
from orchard.datatypes import EncoderOutput, Grid, State

class CentralizedTaskGridEncoder(GridEncoder):
    """Centralized task-specialization encoding.

    Channels (T + N + 1 total):
      0..T-1       — task type channels (same as dec)
      T..T+N-1     — per-agent position: channel T+j has 1.0 at agent j's cell
      T+N          — actor position (1.0 at actor's cell)
    Scalar: one-hot actor identity, length N.
    """

    def __init__(self, env_cfg, use_vec_encode: bool = True) -> None:
        super().__init__(env_cfg)
        self._n_types = env_cfg.n_task_types
        self._n_agents = env_cfg.n_agents
        self._use_vec = use_vec_encode

    def grid_channels(self) -> int:
        return self._n_types + self._n_agents + 1

    def scalar_dim(self) -> int:
        return self._n_agents + 1

    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        T = self._n_types
        N = self._n_agents
        h, w = self.env_cfg.height, self.env_cfg.width
        C = T + N + 1
        grid = torch.zeros(C, h, w, dtype=torch.float32)

        if self._use_vec:
            # Task type channels (vectorized)
            if state.task_positions and state.task_types is not None:
                types_t = torch.tensor(state.task_types, dtype=torch.long)
                rows_t = torch.tensor([p.row for p in state.task_positions], dtype=torch.long)
                cols_t = torch.tensor([p.col for p in state.task_positions], dtype=torch.long)
                grid[types_t, rows_t, cols_t] = 1.0
        else:
            if state.task_types is not None:
                for pos, tau in zip(state.task_positions, state.task_types):
                    grid[tau, pos.row, pos.col] = 1.0

        # Per-agent position channels (same for both paths — always loop, small N)
        for j, pos in enumerate(state.agent_positions):
            grid[T + j, pos.row, pos.col] = 1.0

        # Actor position channel
        actor_pos = state.agent_positions[state.actor]
        grid[T + N, actor_pos.row, actor_pos.col] = 1.0

        # Scalar: one-hot actor identity + pick_phase
        scalar = torch.zeros(N + 1, dtype=torch.float32)
        scalar[state.actor] = 1.0
        if state.pick_phase:
            scalar[N] = 1.0

        return EncoderOutput(grid=grid, scalar=scalar)

    def encode_batch_for_actions(self, state: State, agent_idx: int, after_states: list[State]) -> EncoderOutput:
        T = self._n_types
        N = self._n_agents
        h, w = self.env_cfg.height, self.env_cfg.width
        C = T + N + 1
        n = len(after_states)
        actor = state.actor

        if self._use_vec:
            # Base grid: tasks + non-actor agent channels
            base = torch.zeros(C, h, w, dtype=torch.float32)

            if state.task_positions and state.task_types is not None:
                types_t = torch.tensor(state.task_types, dtype=torch.long)
                rows_t = torch.tensor([p.row for p in state.task_positions], dtype=torch.long)
                cols_t = torch.tensor([p.col for p in state.task_positions], dtype=torch.long)
                base[types_t, rows_t, cols_t] = 1.0

            for j, pos in enumerate(state.agent_positions):
                if j != actor:
                    base[T + j, pos.row, pos.col] = 1.0

            grids = torch.zeros(n, C, h, w, dtype=torch.float32)
            for k, s_after in enumerate(after_states):
                grids[k] = base.clone()
                actor_pos = s_after.agent_positions[actor]
                grids[k, T + actor, actor_pos.row, actor_pos.col] = 1.0
                grids[k, T + N, actor_pos.row, actor_pos.col] = 1.0

                if s_after.task_positions != state.task_positions:
                    grids[k, :T] = 0.0
                    if s_after.task_positions and s_after.task_types is not None:
                        at = torch.tensor(s_after.task_types, dtype=torch.long)
                        ar = torch.tensor([p.row for p in s_after.task_positions], dtype=torch.long)
                        ac = torch.tensor([p.col for p in s_after.task_positions], dtype=torch.long)
                        grids[k][at, ar, ac] = 1.0
        else:
            # Loop-based fallback
            grids = torch.zeros(n, C, h, w, dtype=torch.float32)
            for k, s_after in enumerate(after_states):
                if s_after.task_types is not None:
                    for pos, tau in zip(s_after.task_positions, s_after.task_types):
                        grids[k, tau, pos.row, pos.col] = 1.0

                for j, pos in enumerate(s_after.agent_positions):
                    grids[k, T + j, pos.row, pos.col] = 1.0

                actor_pos = s_after.agent_positions[actor]
                grids[k, T + N, actor_pos.row, actor_pos.col] = 1.0

        # Scalar: one-hot + pick_phase (may vary across actions)
        scalar_base = torch.zeros(N + 1, dtype=torch.float32)
        scalar_base[actor] = 1.0
        scalars = scalar_base.unsqueeze(0).expand(n, -1).clone()
        for k, s_after in enumerate(after_states):
            if s_after.pick_phase:
                scalars[k, N] = 1.0

        return EncoderOutput(grid=grids, scalar=scalars)
    
    def encode_all_agents(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        """Wrap the single-agent encode with an N=1 dimension for GpuTrainer."""
        out = self.encode(state, agent_idx=0)
        assert out.grid is not None and out.scalar is not None
        return out.grid.unsqueeze(0), out.scalar.unsqueeze(0)

    def encode_all_agents_for_actions(
        self, state: State, after_states: list[State],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wrap the batch encode with an N=1 dimension for GpuTrainer."""
        out = self.encode_batch_for_actions(state, agent_idx=0, after_states=after_states)
        assert out.grid is not None and out.scalar is not None
        return out.grid.unsqueeze(0), out.scalar.unsqueeze(0)
    
    
    
    
class BlindTaskGridEncoder(GridEncoder):
    """O(1) Blind decentralized encoding.

    Agent sees ONLY entities that affect its reward or movement.
    Strangers and their tasks are invisible.

    Grid (4 channels, fixed regardless of N or T):
      0 — My Tasks: 1.0 where any task of type τ ∈ G_i exists
      1 — Self Position: 1.0 at agent i's cell
      2 — Teammate Positions: count of teammates at each cell
      3 — Actor Position: 1.0 at actor's cell
    Scalars (3):
      0 — is_self_actor
      1 — is_teammate_actor
      2 — is_pick_phase
    """

    def __init__(self, env_cfg, use_vec_encode: bool = True) -> None:
        super().__init__(env_cfg)
        self._n_types = env_cfg.n_task_types
        self._n_agents = env_cfg.n_agents
        self._use_vec = use_vec_encode

        # Precompute my_types[i] = set of task types assigned to agent i
        if env_cfg.task_assignments is not None:
            self._my_types = tuple(
                frozenset(env_cfg.task_assignments[i])
                for i in range(env_cfg.n_agents)
            )
        else:
            self._my_types = tuple(
                frozenset([0]) for _ in range(env_cfg.n_agents)
            )

        # Precompute teammate sets (precompute teammates)
        if env_cfg.task_assignments is not None:
            teammates = []
            for i in range(env_cfg.n_agents):
                my_types = set(env_cfg.task_assignments[i])
                t_set = set()
                for j in range(env_cfg.n_agents):
                    if j != i and my_types & set(env_cfg.task_assignments[j]):
                        t_set.add(j)
                teammates.append(frozenset(t_set))
            self._teammates = tuple(teammates)
        else:
            self._teammates = tuple(
                frozenset(j for j in range(env_cfg.n_agents) if j != i)
                for i in range(env_cfg.n_agents)
            )

        N = env_cfg.n_agents
        self._teammate_matrix = torch.zeros(N, N, dtype=torch.float32)
        for i in range(N):
            for j in self._teammates[i]:
                self._teammate_matrix[i, j] = 1.0

    def grid_channels(self) -> int:
        return 4

    def scalar_dim(self) -> int:
        return 3

    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        h, w = self.env_cfg.height, self.env_cfg.width
        grid = torch.zeros(4, h, w, dtype=torch.float32)
        my_types = self._my_types[agent_idx]
        teammates_of_i = self._teammates[agent_idx]

        # Ch0: My Tasks only
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                if tau in my_types:
                    grid[0, pos.row, pos.col] = 1.0

        # Ch1: Self
        r, c = state.agent_positions[agent_idx]
        grid[1, r, c] = 1.0

        # Ch2: Teammates only (strangers invisible)
        for j, pos in enumerate(state.agent_positions):
            if j != agent_idx and j in teammates_of_i:
                grid[2, pos.row, pos.col] += 1.0

        # Ch3: Actor
        actor_pos = state.agent_positions[state.actor]
        grid[3, actor_pos.row, actor_pos.col] = 1.0

        # Scalars
        is_actor = 1.0 if agent_idx == state.actor else 0.0
        is_teammate_actor = 1.0 if state.actor in teammates_of_i else 0.0
        # If agent IS the actor, is_teammate_actor = 0 (actor is not its own teammate)
        if agent_idx == state.actor:
            is_teammate_actor = 0.0
        p2 = 1.0 if state.pick_phase else 0.0
        scalar = torch.tensor([is_actor, is_teammate_actor, p2], dtype=torch.float32)

        return EncoderOutput(grid=grid, scalar=scalar)

    def encode_batch_for_actions(self, state: State, agent_idx: int, after_states: list[State]) -> EncoderOutput:
        h, w = self.env_cfg.height, self.env_cfg.width
        n = len(after_states)
        actor = state.actor
        is_actor = (agent_idx == actor)
        teammates_of_i = self._teammates[agent_idx]
        is_actor_teammate = actor in teammates_of_i and not is_actor
        my_types = self._my_types[agent_idx]

        base = torch.zeros(4, h, w, dtype=torch.float32)

        # Ch0: My Tasks
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                if tau in my_types:
                    base[0, pos.row, pos.col] = 1.0

        if is_actor:
            # Ch1 (self) varies per action → leave blank
            # Ch2: teammates (all non-self, non-actor teammates are constant; actor=self so skip)
            for j, pos in enumerate(state.agent_positions):
                if j != agent_idx and j in teammates_of_i:
                    base[2, pos.row, pos.col] += 1.0
        else:
            # Ch1: self is constant
            r, c = state.agent_positions[agent_idx]
            base[1, r, c] = 1.0
            # Ch2: teammates excluding actor (constant part)
            for j, pos in enumerate(state.agent_positions):
                if j != agent_idx and j != actor and j in teammates_of_i:
                    base[2, pos.row, pos.col] += 1.0
            # Actor's teammate contribution added per-action below

        grids = torch.zeros(n, 4, h, w, dtype=torch.float32)
        for k, s_after in enumerate(after_states):
            grids[k] = base.clone()
            actor_pos = s_after.agent_positions[actor]

            if is_actor:
                grids[k, 1, actor_pos.row, actor_pos.col] = 1.0
            else:
                if is_actor_teammate:
                    grids[k, 2, actor_pos.row, actor_pos.col] += 1.0
                # If actor is a stranger, they're invisible — nothing to add
            grids[k, 3, actor_pos.row, actor_pos.col] = 1.0

            # Handle pick after-states where tasks changed
            if s_after.task_positions != state.task_positions:
                grids[k, 0] = 0.0
                if s_after.task_positions and s_after.task_types is not None:
                    for pos, tau in zip(s_after.task_positions, s_after.task_types):
                        if tau in my_types:
                            grids[k, 0, pos.row, pos.col] = 1.0

        # Scalars
        is_actor_val = 1.0 if is_actor else 0.0
        is_tm_actor_val = 1.0 if is_actor_teammate else 0.0
        scalars = torch.zeros(n, 3, dtype=torch.float32)
        scalars[:, 0] = is_actor_val
        scalars[:, 1] = is_tm_actor_val
        for k, s_after in enumerate(after_states):
            if s_after.pick_phase:
                scalars[k, 2] = 1.0

        return EncoderOutput(grid=grids, scalar=scalars)

    def encode_all_agents(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        N = self._n_agents
        h, w = self.env_cfg.height, self.env_cfg.width
        grids = torch.zeros(N, 4, h, w, dtype=torch.float32)

        # Ch0: My Tasks — per-agent (each agent sees only its own types)
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                for i in range(N):
                    if tau in self._my_types[i]:
                        grids[i, 0, pos.row, pos.col] = 1.0

        # Ch1: Self — one-hot per agent
        self_grids = torch.zeros(N, h, w, dtype=torch.float32)
        for i, pos in enumerate(state.agent_positions):
            self_grids[i, pos.row, pos.col] = 1.0
        grids[:, 1] = self_grids

        # Ch2: Teammates
        teammate_grids = torch.matmul(
            self._teammate_matrix, self_grids.view(N, -1)
        ).view(N, h, w)
        grids[:, 2] = teammate_grids

        # Ch3: Actor
        actor_pos = state.agent_positions[state.actor]
        grids[:, 3, actor_pos.row, actor_pos.col] = 1.0

        # Scalars (N, 3): [is_self_actor, is_teammate_actor, is_pick_phase]
        scalars = torch.zeros(N, 3, dtype=torch.float32)
        scalars[state.actor, 0] = 1.0
        # is_teammate_actor: for each agent i, is the actor a teammate of i?
        scalars[:, 1] = self._teammate_matrix[:, state.actor]
        # Actor itself should have is_teammate_actor = 0
        scalars[state.actor, 1] = 0.0
        if state.pick_phase:
            scalars[:, 2] = 1.0

        return grids, scalars

    def encode_all_agents_for_actions(
        self, state: State, after_states: list[State],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = self._n_agents
        B = len(after_states)
        h, w = self.env_cfg.height, self.env_cfg.width
        actor = state.actor

        grids = torch.zeros(N, B, 4, h, w, dtype=torch.float32)

        # Ch0: My Tasks — broadcast over B (will fix for pick after-states)
        my_task_grids = torch.zeros(N, h, w, dtype=torch.float32)
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                for i in range(N):
                    if tau in self._my_types[i]:
                        my_task_grids[i, pos.row, pos.col] = 1.0
        grids[:, :, 0] = my_task_grids.unsqueeze(1)

        # Ch1: Self — original positions, broadcast over B
        self_orig = torch.zeros(N, h, w, dtype=torch.float32)
        for i, pos in enumerate(state.agent_positions):
            self_orig[i, pos.row, pos.col] = 1.0
        grids[:, :, 1] = self_orig.unsqueeze(1)

        # Ch2: Teammates at original positions
        teammate_orig = torch.matmul(
            self._teammate_matrix, self_orig.view(N, -1)
        ).view(N, h, w)
        grids[:, :, 2] = teammate_orig.unsqueeze(1)

        # Per-action adjustments for actor movement
        old_actor_pos = state.agent_positions[actor]
        old_ar, old_ac = old_actor_pos.row, old_actor_pos.col
        actor_is_teammate = self._teammate_matrix[:, actor]  # (N,)

        for k, s_after in enumerate(after_states):
            actor_pos = s_after.agent_positions[actor]
            new_ar, new_ac = actor_pos.row, actor_pos.col

            # Fix actor's self channel
            grids[actor, k, 1] = 0
            grids[actor, k, 1, new_ar, new_ac] = 1.0

            # Fix teammate channels for non-actor agents
            if old_ar != new_ar or old_ac != new_ac:
                for i in range(N):
                    if i == actor:
                        continue
                    if actor_is_teammate[i] > 0.5:
                        grids[i, k, 2, old_ar, old_ac] -= 1.0
                        grids[i, k, 2, new_ar, new_ac] += 1.0
                    # If actor is stranger to i, no adjustment needed (invisible)

            # Ch3: Actor position
            grids[:, k, 3, new_ar, new_ac] = 1.0

            # Handle pick after-states
            if s_after.task_positions != state.task_positions:
                grids[:, k, 0] = 0
                if s_after.task_positions and s_after.task_types is not None:
                    for pos, tau in zip(s_after.task_positions, s_after.task_types):
                        for i in range(N):
                            if tau in self._my_types[i]:
                                grids[i, k, 0, pos.row, pos.col] = 1.0

        # Scalars (N, B, 3)
        scalars = torch.zeros(N, B, 3, dtype=torch.float32)
        scalars[actor, :, 0] = 1.0
        scalars[:, :, 1] = actor_is_teammate.unsqueeze(1)
        scalars[actor, :, 1] = 0.0
        for k, s_after in enumerate(after_states):
            if s_after.pick_phase:
                scalars[:, k, 2] = 1.0

        return grids, scalars


class FilteredTaskGridEncoder(GridEncoder):
    """O(1) Filtered decentralized encoding.

    Agent sees the full crowd but categorized into binary groups.

    Grid (6 channels, fixed regardless of N or T):
      0 — My Tasks: 1.0 where any task of type τ ∈ G_i exists
      1 — Irrelevant Tasks: 1.0 where any task of type τ ∉ G_i exists
      2 — Self Position: 1.0 at agent i's cell
      3 — Teammate Positions: count of teammates at each cell
      4 — Stranger Positions: count of non-teammates at each cell
      5 — Actor Position: 1.0 at actor's cell
    Scalars (3):
      0 — is_self_actor
      1 — is_teammate_actor
      2 — is_pick_phase
    """

    def __init__(self, env_cfg, use_vec_encode: bool = True) -> None:
        super().__init__(env_cfg)
        self._n_types = env_cfg.n_task_types
        self._n_agents = env_cfg.n_agents
        self._use_vec = use_vec_encode

        if env_cfg.task_assignments is not None:
            self._my_types = tuple(
                frozenset(env_cfg.task_assignments[i])
                for i in range(env_cfg.n_agents)
            )
        else:
            self._my_types = tuple(
                frozenset([0]) for _ in range(env_cfg.n_agents)
            )

        if env_cfg.task_assignments is not None:
            teammates = []
            for i in range(env_cfg.n_agents):
                my_types = set(env_cfg.task_assignments[i])
                t_set = set()
                for j in range(env_cfg.n_agents):
                    if j != i and my_types & set(env_cfg.task_assignments[j]):
                        t_set.add(j)
                teammates.append(frozenset(t_set))
            self._teammates = tuple(teammates)
        else:
            self._teammates = tuple(
                frozenset(j for j in range(env_cfg.n_agents) if j != i)
                for i in range(env_cfg.n_agents)
            )

        N = env_cfg.n_agents
        self._teammate_matrix = torch.zeros(N, N, dtype=torch.float32)
        for i in range(N):
            for j in self._teammates[i]:
                self._teammate_matrix[i, j] = 1.0

    def grid_channels(self) -> int:
        return 6

    def scalar_dim(self) -> int:
        return 3

    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        h, w = self.env_cfg.height, self.env_cfg.width
        grid = torch.zeros(6, h, w, dtype=torch.float32)
        my_types = self._my_types[agent_idx]
        teammates_of_i = self._teammates[agent_idx]

        # Ch0: My Tasks, Ch1: Irrelevant Tasks
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                if tau in my_types:
                    grid[0, pos.row, pos.col] = 1.0
                else:
                    grid[1, pos.row, pos.col] = 1.0

        # Ch2: Self
        r, c = state.agent_positions[agent_idx]
        grid[2, r, c] = 1.0

        # Ch3: Teammates, Ch4: Strangers
        for j, pos in enumerate(state.agent_positions):
            if j == agent_idx:
                continue
            if j in teammates_of_i:
                grid[3, pos.row, pos.col] += 1.0
            else:
                grid[4, pos.row, pos.col] += 1.0

        # Ch5: Actor
        actor_pos = state.agent_positions[state.actor]
        grid[5, actor_pos.row, actor_pos.col] = 1.0

        # Scalars
        is_actor = 1.0 if agent_idx == state.actor else 0.0
        is_teammate_actor = 1.0 if (state.actor in teammates_of_i and agent_idx != state.actor) else 0.0
        p2 = 1.0 if state.pick_phase else 0.0
        scalar = torch.tensor([is_actor, is_teammate_actor, p2], dtype=torch.float32)

        return EncoderOutput(grid=grid, scalar=scalar)

    def encode_batch_for_actions(self, state: State, agent_idx: int, after_states: list[State]) -> EncoderOutput:
        h, w = self.env_cfg.height, self.env_cfg.width
        n = len(after_states)
        actor = state.actor
        is_actor = (agent_idx == actor)
        teammates_of_i = self._teammates[agent_idx]
        is_actor_teammate = actor in teammates_of_i and not is_actor
        my_types = self._my_types[agent_idx]

        base = torch.zeros(6, h, w, dtype=torch.float32)

        # Ch0: My Tasks, Ch1: Irrelevant Tasks
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                if tau in my_types:
                    base[0, pos.row, pos.col] = 1.0
                else:
                    base[1, pos.row, pos.col] = 1.0

        if is_actor:
            # Ch2 (self) varies
            # Ch3, Ch4: all non-self agents at fixed positions
            for j, pos in enumerate(state.agent_positions):
                if j == agent_idx:
                    continue
                if j in teammates_of_i:
                    base[3, pos.row, pos.col] += 1.0
                else:
                    base[4, pos.row, pos.col] += 1.0
        else:
            # Ch2: self constant
            r, c = state.agent_positions[agent_idx]
            base[2, r, c] = 1.0
            # Ch3, Ch4: all except self and actor
            for j, pos in enumerate(state.agent_positions):
                if j == agent_idx or j == actor:
                    continue
                if j in teammates_of_i:
                    base[3, pos.row, pos.col] += 1.0
                else:
                    base[4, pos.row, pos.col] += 1.0

        grids = torch.zeros(n, 6, h, w, dtype=torch.float32)
        for k, s_after in enumerate(after_states):
            grids[k] = base.clone()
            actor_pos = s_after.agent_positions[actor]

            if is_actor:
                grids[k, 2, actor_pos.row, actor_pos.col] = 1.0
            else:
                if is_actor_teammate:
                    grids[k, 3, actor_pos.row, actor_pos.col] += 1.0
                else:
                    grids[k, 4, actor_pos.row, actor_pos.col] += 1.0
            grids[k, 5, actor_pos.row, actor_pos.col] = 1.0

            if s_after.task_positions != state.task_positions:
                grids[k, 0] = 0.0
                grids[k, 1] = 0.0
                if s_after.task_positions and s_after.task_types is not None:
                    for pos, tau in zip(s_after.task_positions, s_after.task_types):
                        if tau in my_types:
                            grids[k, 0, pos.row, pos.col] = 1.0
                        else:
                            grids[k, 1, pos.row, pos.col] = 1.0

        is_actor_val = 1.0 if is_actor else 0.0
        is_tm_actor_val = 1.0 if is_actor_teammate else 0.0
        scalars = torch.zeros(n, 3, dtype=torch.float32)
        scalars[:, 0] = is_actor_val
        scalars[:, 1] = is_tm_actor_val
        for k, s_after in enumerate(after_states):
            if s_after.pick_phase:
                scalars[k, 2] = 1.0

        return EncoderOutput(grid=grids, scalar=scalars)

    def encode_all_agents(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        N = self._n_agents
        h, w = self.env_cfg.height, self.env_cfg.width
        grids = torch.zeros(N, 6, h, w, dtype=torch.float32)

        # Ch0: My Tasks, Ch1: Irrelevant Tasks (per-agent)
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                for i in range(N):
                    if tau in self._my_types[i]:
                        grids[i, 0, pos.row, pos.col] = 1.0
                    else:
                        grids[i, 1, pos.row, pos.col] = 1.0

        # Ch2: Self
        self_grids = torch.zeros(N, h, w, dtype=torch.float32)
        for i, pos in enumerate(state.agent_positions):
            self_grids[i, pos.row, pos.col] = 1.0
        grids[:, 2] = self_grids

        # Ch3: Teammates
        teammate_grids = torch.matmul(
            self._teammate_matrix, self_grids.view(N, -1)
        ).view(N, h, w)
        grids[:, 3] = teammate_grids

        # Ch4: Strangers = total - self - teammates
        total_agents = self_grids.sum(dim=0)
        grids[:, 4] = total_agents.unsqueeze(0) - self_grids - teammate_grids

        # Ch5: Actor
        actor_pos = state.agent_positions[state.actor]
        grids[:, 5, actor_pos.row, actor_pos.col] = 1.0

        # Scalars (N, 3)
        scalars = torch.zeros(N, 3, dtype=torch.float32)
        scalars[state.actor, 0] = 1.0
        scalars[:, 1] = self._teammate_matrix[:, state.actor]
        scalars[state.actor, 1] = 0.0
        if state.pick_phase:
            scalars[:, 2] = 1.0

        return grids, scalars

    def encode_all_agents_for_actions(
        self, state: State, after_states: list[State],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = self._n_agents
        B = len(after_states)
        h, w = self.env_cfg.height, self.env_cfg.width
        actor = state.actor

        grids = torch.zeros(N, B, 6, h, w, dtype=torch.float32)

        # Ch0, Ch1: task grids per agent, broadcast over B
        my_task_grids = torch.zeros(N, h, w, dtype=torch.float32)
        other_task_grids = torch.zeros(N, h, w, dtype=torch.float32)
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                for i in range(N):
                    if tau in self._my_types[i]:
                        my_task_grids[i, pos.row, pos.col] = 1.0
                    else:
                        other_task_grids[i, pos.row, pos.col] = 1.0
        grids[:, :, 0] = my_task_grids.unsqueeze(1)
        grids[:, :, 1] = other_task_grids.unsqueeze(1)

        # Ch2: Self
        self_orig = torch.zeros(N, h, w, dtype=torch.float32)
        for i, pos in enumerate(state.agent_positions):
            self_orig[i, pos.row, pos.col] = 1.0
        grids[:, :, 2] = self_orig.unsqueeze(1)

        # Ch3: Teammates
        teammate_orig = torch.matmul(
            self._teammate_matrix, self_orig.view(N, -1)
        ).view(N, h, w)
        grids[:, :, 3] = teammate_orig.unsqueeze(1)

        # Ch4: Strangers
        total_orig = self_orig.sum(dim=0)
        strangers_orig = total_orig.unsqueeze(0) - self_orig - teammate_orig
        grids[:, :, 4] = strangers_orig.unsqueeze(1)

        # Per-action adjustments
        old_actor_pos = state.agent_positions[actor]
        old_ar, old_ac = old_actor_pos.row, old_actor_pos.col
        actor_is_teammate = self._teammate_matrix[:, actor]  # (N,)

        for k, s_after in enumerate(after_states):
            actor_pos = s_after.agent_positions[actor]
            new_ar, new_ac = actor_pos.row, actor_pos.col

            # Fix actor's self channel
            grids[actor, k, 2] = 0
            grids[actor, k, 2, new_ar, new_ac] = 1.0

            if old_ar != new_ar or old_ac != new_ac:
                for i in range(N):
                    if i == actor:
                        continue
                    if actor_is_teammate[i] > 0.5:
                        grids[i, k, 3, old_ar, old_ac] -= 1.0
                        grids[i, k, 3, new_ar, new_ac] += 1.0
                    else:
                        grids[i, k, 4, old_ar, old_ac] -= 1.0
                        grids[i, k, 4, new_ar, new_ac] += 1.0

            # Ch5: Actor position
            grids[:, k, 5, new_ar, new_ac] = 1.0

            # Handle pick after-states
            if s_after.task_positions != state.task_positions:
                grids[:, k, 0] = 0
                grids[:, k, 1] = 0
                if s_after.task_positions and s_after.task_types is not None:
                    for pos, tau in zip(s_after.task_positions, s_after.task_types):
                        for i in range(N):
                            if tau in self._my_types[i]:
                                grids[i, k, 0, pos.row, pos.col] = 1.0
                            else:
                                grids[i, k, 1, pos.row, pos.col] = 1.0

        # Scalars (N, B, 3)
        scalars = torch.zeros(N, B, 3, dtype=torch.float32)
        scalars[actor, :, 0] = 1.0
        scalars[:, :, 1] = actor_is_teammate.unsqueeze(1)
        scalars[actor, :, 1] = 0.0
        for k, s_after in enumerate(after_states):
            if s_after.pick_phase:
                scalars[:, k, 2] = 1.0
        return grids, scalars


class PositionAwareTaskGridEncoder(GridEncoder):
    """O(1) Position-aware decentralized encoding.

    Like Blind, but agents can see where strangers are (without knowing
    their task assignments).  Models realistic partial observability:
    agents physically observe each other but lack task-queue info.

    Grid (5 channels, fixed regardless of N or T):
      0 — My Tasks: 1.0 where any task of type τ ∈ G_i exists
      1 — Self Position: 1.0 at agent i's cell
      2 — Teammate Positions: count of teammates at each cell
      3 — Stranger Positions: count of non-teammates at each cell
      4 — Actor Position: 1.0 at actor's cell
    Scalars (3):
      0 — is_self_actor
      1 — is_teammate_actor
      2 — is_pick_phase
    """

    def __init__(self, env_cfg, use_vec_encode: bool = True) -> None:
        super().__init__(env_cfg)
        self._n_types = env_cfg.n_task_types
        self._n_agents = env_cfg.n_agents
        self._use_vec = use_vec_encode

        if env_cfg.task_assignments is not None:
            self._my_types = tuple(
                frozenset(env_cfg.task_assignments[i])
                for i in range(env_cfg.n_agents)
            )
        else:
            self._my_types = tuple(
                frozenset([0]) for _ in range(env_cfg.n_agents)
            )

        if env_cfg.task_assignments is not None:
            teammates = []
            for i in range(env_cfg.n_agents):
                my_types = set(env_cfg.task_assignments[i])
                t_set = set()
                for j in range(env_cfg.n_agents):
                    if j != i and my_types & set(env_cfg.task_assignments[j]):
                        t_set.add(j)
                teammates.append(frozenset(t_set))
            self._teammates = tuple(teammates)
        else:
            self._teammates = tuple(
                frozenset(j for j in range(env_cfg.n_agents) if j != i)
                for i in range(env_cfg.n_agents)
            )

        N = env_cfg.n_agents
        self._teammate_matrix = torch.zeros(N, N, dtype=torch.float32)
        for i in range(N):
            for j in self._teammates[i]:
                self._teammate_matrix[i, j] = 1.0

    def grid_channels(self) -> int:
        return 5

    def scalar_dim(self) -> int:
        return 3

    def encode(self, state: State, agent_idx: int) -> EncoderOutput:
        h, w = self.env_cfg.height, self.env_cfg.width
        grid = torch.zeros(5, h, w, dtype=torch.float32)
        my_types = self._my_types[agent_idx]
        teammates_of_i = self._teammates[agent_idx]

        # Ch0: My Tasks only
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                if tau in my_types:
                    grid[0, pos.row, pos.col] = 1.0

        # Ch1: Self
        r, c = state.agent_positions[agent_idx]
        grid[1, r, c] = 1.0

        # Ch2: Teammates, Ch3: Strangers
        for j, pos in enumerate(state.agent_positions):
            if j == agent_idx:
                continue
            if j in teammates_of_i:
                grid[2, pos.row, pos.col] += 1.0
            else:
                grid[3, pos.row, pos.col] += 1.0

        # Ch4: Actor
        actor_pos = state.agent_positions[state.actor]
        grid[4, actor_pos.row, actor_pos.col] = 1.0

        # Scalars
        is_actor = 1.0 if agent_idx == state.actor else 0.0
        is_teammate_actor = 1.0 if (state.actor in teammates_of_i and agent_idx != state.actor) else 0.0
        p2 = 1.0 if state.pick_phase else 0.0
        scalar = torch.tensor([is_actor, is_teammate_actor, p2], dtype=torch.float32)

        return EncoderOutput(grid=grid, scalar=scalar)

    def encode_batch_for_actions(self, state: State, agent_idx: int, after_states: list[State]) -> EncoderOutput:
        h, w = self.env_cfg.height, self.env_cfg.width
        n = len(after_states)
        actor = state.actor
        is_actor = (agent_idx == actor)
        teammates_of_i = self._teammates[agent_idx]
        is_actor_teammate = actor in teammates_of_i and not is_actor
        my_types = self._my_types[agent_idx]

        base = torch.zeros(5, h, w, dtype=torch.float32)

        # Ch0: My Tasks
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                if tau in my_types:
                    base[0, pos.row, pos.col] = 1.0

        if is_actor:
            # Ch1 (self) varies per action → leave blank
            # Ch2: teammates, Ch3: strangers (all non-self are constant)
            for j, pos in enumerate(state.agent_positions):
                if j == agent_idx:
                    continue
                if j in teammates_of_i:
                    base[2, pos.row, pos.col] += 1.0
                else:
                    base[3, pos.row, pos.col] += 1.0
        else:
            # Ch1: self is constant
            r, c = state.agent_positions[agent_idx]
            base[1, r, c] = 1.0
            # Ch2, Ch3: everyone except self and actor (constant part)
            for j, pos in enumerate(state.agent_positions):
                if j == agent_idx or j == actor:
                    continue
                if j in teammates_of_i:
                    base[2, pos.row, pos.col] += 1.0
                else:
                    base[3, pos.row, pos.col] += 1.0

        grids = torch.zeros(n, 5, h, w, dtype=torch.float32)
        for k, s_after in enumerate(after_states):
            grids[k] = base.clone()
            actor_pos = s_after.agent_positions[actor]

            if is_actor:
                grids[k, 1, actor_pos.row, actor_pos.col] = 1.0
            else:
                if is_actor_teammate:
                    grids[k, 2, actor_pos.row, actor_pos.col] += 1.0
                else:
                    grids[k, 3, actor_pos.row, actor_pos.col] += 1.0
            grids[k, 4, actor_pos.row, actor_pos.col] = 1.0

            # Handle pick after-states
            if s_after.task_positions != state.task_positions:
                grids[k, 0] = 0.0
                if s_after.task_positions and s_after.task_types is not None:
                    for pos, tau in zip(s_after.task_positions, s_after.task_types):
                        if tau in my_types:
                            grids[k, 0, pos.row, pos.col] = 1.0

        # Scalars
        is_actor_val = 1.0 if is_actor else 0.0
        is_tm_actor_val = 1.0 if is_actor_teammate else 0.0
        scalars = torch.zeros(n, 3, dtype=torch.float32)
        scalars[:, 0] = is_actor_val
        scalars[:, 1] = is_tm_actor_val
        for k, s_after in enumerate(after_states):
            if s_after.pick_phase:
                scalars[k, 2] = 1.0

        return EncoderOutput(grid=grids, scalar=scalars)

    def encode_all_agents(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        N = self._n_agents
        h, w = self.env_cfg.height, self.env_cfg.width
        grids = torch.zeros(N, 5, h, w, dtype=torch.float32)

        # Ch0: My Tasks
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                for i in range(N):
                    if tau in self._my_types[i]:
                        grids[i, 0, pos.row, pos.col] = 1.0

        # Ch1: Self
        self_grids = torch.zeros(N, h, w, dtype=torch.float32)
        for i, pos in enumerate(state.agent_positions):
            self_grids[i, pos.row, pos.col] = 1.0
        grids[:, 1] = self_grids

        # Ch2: Teammates
        teammate_grids = torch.matmul(
            self._teammate_matrix, self_grids.view(N, -1)
        ).view(N, h, w)
        grids[:, 2] = teammate_grids

        # Ch3: Strangers = total - self - teammates
        total_agents = self_grids.sum(dim=0)
        grids[:, 3] = total_agents.unsqueeze(0) - self_grids - teammate_grids

        # Ch4: Actor
        actor_pos = state.agent_positions[state.actor]
        grids[:, 4, actor_pos.row, actor_pos.col] = 1.0

        # Scalars (N, 3)
        scalars = torch.zeros(N, 3, dtype=torch.float32)
        scalars[state.actor, 0] = 1.0
        scalars[:, 1] = self._teammate_matrix[:, state.actor]
        scalars[state.actor, 1] = 0.0
        if state.pick_phase:
            scalars[:, 2] = 1.0

        return grids, scalars

    def encode_all_agents_for_actions(
        self, state: State, after_states: list[State],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = self._n_agents
        B = len(after_states)
        h, w = self.env_cfg.height, self.env_cfg.width
        actor = state.actor

        grids = torch.zeros(N, B, 5, h, w, dtype=torch.float32)

        # Ch0: My Tasks, broadcast over B
        my_task_grids = torch.zeros(N, h, w, dtype=torch.float32)
        if state.task_positions and state.task_types is not None:
            for pos, tau in zip(state.task_positions, state.task_types):
                for i in range(N):
                    if tau in self._my_types[i]:
                        my_task_grids[i, pos.row, pos.col] = 1.0
        grids[:, :, 0] = my_task_grids.unsqueeze(1)

        # Ch1: Self
        self_orig = torch.zeros(N, h, w, dtype=torch.float32)
        for i, pos in enumerate(state.agent_positions):
            self_orig[i, pos.row, pos.col] = 1.0
        grids[:, :, 1] = self_orig.unsqueeze(1)

        # Ch2: Teammates
        teammate_orig = torch.matmul(
            self._teammate_matrix, self_orig.view(N, -1)
        ).view(N, h, w)
        grids[:, :, 2] = teammate_orig.unsqueeze(1)

        # Ch3: Strangers
        total_orig = self_orig.sum(dim=0)
        strangers_orig = total_orig.unsqueeze(0) - self_orig - teammate_orig
        grids[:, :, 3] = strangers_orig.unsqueeze(1)

        # Per-action adjustments
        old_actor_pos = state.agent_positions[actor]
        old_ar, old_ac = old_actor_pos.row, old_actor_pos.col
        actor_is_teammate = self._teammate_matrix[:, actor]  # (N,)

        for k, s_after in enumerate(after_states):
            actor_pos = s_after.agent_positions[actor]
            new_ar, new_ac = actor_pos.row, actor_pos.col

            # Fix actor's self channel
            grids[actor, k, 1] = 0
            grids[actor, k, 1, new_ar, new_ac] = 1.0

            if old_ar != new_ar or old_ac != new_ac:
                for i in range(N):
                    if i == actor:
                        continue
                    if actor_is_teammate[i] > 0.5:
                        grids[i, k, 2, old_ar, old_ac] -= 1.0
                        grids[i, k, 2, new_ar, new_ac] += 1.0
                    else:
                        grids[i, k, 3, old_ar, old_ac] -= 1.0
                        grids[i, k, 3, new_ar, new_ac] += 1.0

            # Ch4: Actor position
            grids[:, k, 4, new_ar, new_ac] = 1.0

            # Handle pick after-states
            if s_after.task_positions != state.task_positions:
                grids[:, k, 0] = 0
                if s_after.task_positions and s_after.task_types is not None:
                    for pos, tau in zip(s_after.task_positions, s_after.task_types):
                        for i in range(N):
                            if tau in self._my_types[i]:
                                grids[i, k, 0, pos.row, pos.col] = 1.0

        # Scalars (N, B, 3)
        scalars = torch.zeros(N, B, 3, dtype=torch.float32)
        scalars[actor, :, 0] = 1.0
        scalars[:, :, 1] = actor_is_teammate.unsqueeze(1)
        scalars[actor, :, 1] = 0.0
        for k, s_after in enumerate(after_states):
            if s_after.pick_phase:
                scalars[:, k, 2] = 1.0

        return grids, scalars