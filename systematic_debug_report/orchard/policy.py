"""Policy functions: Q_team, greedy, nearest_task, nearest_correct_task, epsilon_greedy."""

from __future__ import annotations

from dataclasses import replace

import torch

import orchard.encoding as encoding
from orchard.enums import (
    NUM_MOVE_ACTIONS, Action, ACTION_PRIORITY, Heuristic, PickMode,
    make_pick_action,
)
from orchard.env.base import BaseEnv
from orchard.model import ValueNetwork
from orchard.seed import rng
from orchard.datatypes import EnvConfig, State


# ---------------------------------------------------------------------------
# Action space helpers
# ---------------------------------------------------------------------------
def get_all_actions(env_cfg: EnvConfig, state: "State | None" = None) -> list[Action]:
    """Phase-1 actions: always the 5 move actions.

    state parameter kept for backward compatibility but ignored —
    phase 1 is always a move action regardless of position or pick mode.
    """
    return list(ACTION_PRIORITY)


def get_phase2_actions(state: State, env_cfg: EnvConfig) -> list[Action]:
    """Phase-2 actions after a move landed on a task cell.

    Returns {STAY, pick(τ) for each τ present at actor's cell}.
    Returns empty list if actor is not on any task cell (phase 2 skipped).

    In FORCED mode the caller auto-picks without consulting this — but it is
    still correct to call it for introspection / unified code paths.
    """
    actor_pos = state.agent_positions[state.actor]
    tasks_here = state.tasks_at(actor_pos)
    if not tasks_here:
        return []
    types_here = sorted({tau for _, tau in tasks_here})
    actions: list[Action] = [Action.STAY]
    for tau in types_here:
        actions.append(make_pick_action(tau))
    return actions


# ---------------------------------------------------------------------------
# Heuristic policies
# ---------------------------------------------------------------------------
def nearest_task_action(state: State, env_cfg: EnvConfig, phase2: bool = False) -> Action:
    """Move actor toward nearest task (Manhattan distance).

    Phase 1: move toward nearest task.
    Phase 2 (CHOICE only): pick the first task present; STAY if none (shouldn't happen).
    """
    actor = state.actor

    if phase2:
        tasks_here = state.tasks_at(state.agent_positions[actor])
        if tasks_here:
            _, tau = tasks_here[0]
            return make_pick_action(tau)
        return Action.STAY

    if not state.task_positions:
        return Action.STAY

    ar, ac = state.agent_positions[actor]
    best_dist = float("inf")
    best_action = Action.STAY

    for action in ACTION_PRIORITY:
        dr, dc = action.delta
        nr = max(0, min(env_cfg.height - 1, ar + dr))
        nc = max(0, min(env_cfg.width - 1, ac + dc))
        min_d = min(
            abs(nr - tp.row) + abs(nc - tp.col)
            for tp in state.task_positions
        )
        if min_d < best_dist:
            best_dist = min_d
            best_action = action

    return best_action


# Backward compat alias
nearest_apple_action = nearest_task_action


def nearest_correct_task_action(state: State, env_cfg: EnvConfig, phase2: bool = False) -> Action:
    """Move toward nearest correct-type task (τ ∈ G_actor).

    Phase 1: move toward nearest correct task.
    Phase 2: always pick — picks whatever task is present (correct or wrong).
             In FORCED mode this is exactly what happens. In CHOICE mode this
             heuristic behaves identically to forced pick.
    """
    actor = state.actor
    good_types = set(env_cfg.task_assignments[actor]) if env_cfg.task_assignments else set()

    if phase2:
        # Pick whatever is here — mirrors forced-pick behavior
        tasks_here = state.tasks_at(state.agent_positions[actor])
        if tasks_here:
            _, tau = tasks_here[0]
            return make_pick_action(tau)
        return Action.STAY

    # Phase 1: find correct-type tasks
    if state.task_types is not None:
        good_tasks = [
            (pos, typ) for pos, typ in zip(state.task_positions, state.task_types)
            if typ in good_types
        ]
    else:
        good_tasks = [(pos, 0) for pos in state.task_positions]

    if not good_tasks:
        return Action.STAY

    ar, ac = state.agent_positions[actor]
    best_dist = float("inf")
    best_action = Action.STAY

    for action in ACTION_PRIORITY:
        dr, dc = action.delta
        nr = max(0, min(env_cfg.height - 1, ar + dr))
        nc = max(0, min(env_cfg.width - 1, ac + dc))
        min_d = min(
            abs(nr - pos.row) + abs(nc - pos.col)
            for pos, _ in good_tasks
        )
        if min_d < best_dist:
            best_dist = min_d
            best_action = action

    return best_action


def nearest_correct_task_stay_wrong_action(
    state: State, env_cfg: EnvConfig, phase2: bool = False
) -> Action:
    """Move toward nearest correct-type task; phase 2 picks correct type, STAYs on wrong.

    Phase 1: identical to nearest_correct_task_action.
    Phase 2: if a correct-type task is present at current cell → pick(τ).
             If only wrong-type tasks → STAY (decline to pick).
    """
    actor = state.actor
    good_types = set(env_cfg.task_assignments[actor]) if env_cfg.task_assignments else set()

    if phase2:
        tasks_here = state.tasks_at(state.agent_positions[actor])
        for _, tau in tasks_here:
            if tau in good_types:
                return make_pick_action(tau)
        return Action.STAY  # wrong-type task(s) only — decline

    # Phase 1: same as nearest_correct_task
    return nearest_correct_task_action(state, env_cfg, phase2=False)


def heuristic_action(
    state: State, env_cfg: EnvConfig, heuristic: Heuristic, phase2: bool = False
) -> Action:
    """Dispatch to the configured heuristic policy."""
    if heuristic == Heuristic.NEAREST_TASK:
        return nearest_task_action(state, env_cfg, phase2=phase2)
    elif heuristic == Heuristic.NEAREST_CORRECT_TASK:
        return nearest_correct_task_action(state, env_cfg, phase2=phase2)
    elif heuristic == Heuristic.NEAREST_CORRECT_TASK_STAY_WRONG:
        return nearest_correct_task_stay_wrong_action(state, env_cfg, phase2=phase2)
    else:
        raise ValueError(f"Unknown heuristic: {heuristic}")


# ---------------------------------------------------------------------------
# Q-value and greedy policies
# ---------------------------------------------------------------------------
def _after_state_for_action(
    state: State, action: Action, env: BaseEnv, phase2: bool = False,
) -> State:
    """Compute after-state for a move (phase 1) or pick/stay (phase 2) action.

    Sets phase2_pending on the returned state:
      - Phase 1 move that lands on task: True (phase 2 will follow)
      - Everything else: False
    """
    if action.is_pick():
        return env.resolve_pick(state, pick_type=action.pick_type())[0]
    s = env.apply_action(state, action)
    if not phase2 and s.is_agent_on_task(s.actor):
        return replace(s, phase2_pending=True)
    return s


def _after_state_and_team_reward(
    state: State, action: Action, env: BaseEnv, phase2: bool = False,
) -> tuple[State, float]:
    """Compute after-state AND immediate team reward for an action.

    For pick actions: reward from resolve_pick.
    For move/stay: reward is 0.
    Sets phase2_pending on returned state (same logic as _after_state_for_action).
    """
    if action.is_pick():
        s_after, rewards = env.resolve_pick(state, pick_type=action.pick_type())
        return s_after, sum(rewards)
    s = env.apply_action(state, action)
    if not phase2 and s.is_agent_on_task(s.actor):
        return replace(s, phase2_pending=True), 0.0
    return s, 0.0


def Q_team(
    state: State,
    action: Action,
    networks: list[ValueNetwork],
    env: BaseEnv,
    phase2: bool = False,
) -> float:
    """Team Q-value: immediate reward + sum of all agents' after-state values.

    phase2=True adds immediate reward from pick actions (critical for CHOICE mode).
    """
    if phase2:
        s_after, immediate_r = _after_state_and_team_reward(state, action, env, phase2=True)
    else:
        s_after = _after_state_for_action(state, action, env, phase2=False)
        immediate_r = 0.0
    total: float = immediate_r
    with torch.no_grad():
        for i, network in enumerate(networks):
            total += network(encoding.encode(s_after, i)).item()
    return total


def argmax_a_Q_team(
    state: State,
    networks: list[ValueNetwork],
    env: BaseEnv,
    phase2: bool = False,
) -> Action:
    """Greedy action for phase 1 (move) or phase 2 (pick/stay). Tie-break via action order."""
    all_actions = get_phase2_actions(state, env.cfg) if phase2 else get_all_actions(env.cfg)
    best_value: float | None = None
    best_action: Action = all_actions[0]

    for action in all_actions:
        q = Q_team(state, action, networks, env, phase2=phase2)
        if best_value is None or q > best_value:
            best_value = q
            best_action = action

    return best_action


def argmax_a_Q_team_batched(
    state: State,
    networks: list[ValueNetwork],
    env: BaseEnv,
    phase2: bool = False,
) -> Action:
    """Greedy action with batched encoding + forward passes."""
    all_actions = get_phase2_actions(state, env.cfg) if phase2 else get_all_actions(env.cfg)

    after_states: list[State] = []
    immediate_rewards: list[float] = []
    for a in all_actions:
        if phase2:
            s_after, r = _after_state_and_team_reward(state, a, env, phase2=True)
            after_states.append(s_after)
            immediate_rewards.append(r)
        else:
            after_states.append(_after_state_for_action(state, a, env, phase2=False))
            immediate_rewards.append(0.0)

    team_values = list(immediate_rewards)
    with torch.no_grad():
        for i, net in enumerate(networks):
            batch_enc = encoding.encode_batch_for_actions(state, i, after_states)
            vals = net(batch_enc)
            for k in range(len(all_actions)):
                team_values[k] += vals[k].item()

    best_idx = 0
    for k in range(1, len(all_actions)):
        if team_values[k] > team_values[best_idx]:
            best_idx = k
    return all_actions[best_idx]


# ---------------------------------------------------------------------------
# Vmap-batched action selection
# ---------------------------------------------------------------------------
class _VmapForwardWrapper(torch.nn.Module):
    """Thin wrapper so functional_call gets forward(grid, scalar) signature."""
    def __init__(self, base_net: ValueNetwork) -> None:
        super().__init__()
        self.conv = base_net.conv
        self.flatten = base_net.flatten
        self.net = base_net.net

    def forward(self, grid: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        x = grid
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.flatten(self.conv(x))
        if scalar.numel() > 0:
            s = scalar
            if s.dim() == 1:
                s = s.unsqueeze(0)
            x = torch.cat([x, s], dim=-1)
        out = self.net(x).squeeze(-1)
        if out.dim() == 1 and out.size(0) == 1:
            return out.squeeze(0)
        return out


class VmapHelper:
    """Wraps N decentralized networks for vmap-batched forward passes.
    
    Caches stacked parameters. Call refresh() after any training step
    that modifies network weights.
    """
    
    def __init__(self, networks: list[ValueNetwork]) -> None:
        from torch.func import stack_module_state
        self.networks = networks
        self.n = len(networks)
        # Create wrapper modules with same param structure
        self._wrappers = [_VmapForwardWrapper(net) for net in networks]
        self._base = self._wrappers[0]
        self._params: dict = {}
        self._buffers: dict = {}
        self.refresh()
    
    def refresh(self) -> None:
        """Re-stack parameters from all networks. Call after training steps."""
        from torch.func import stack_module_state
        # Copy current params from ValueNetworks to wrappers
        for wrapper, net in zip(self._wrappers, self.networks):
            wrapper.conv.load_state_dict(net.conv.state_dict())
            wrapper.net.load_state_dict(net.net.state_dict())
        self._params, self._buffers = stack_module_state(self._wrappers)
    
    def forward_batched(self, grids: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """Run all N networks on their respective inputs in one vmap call.
        
        Args:
            grids:   (N, B, C, H, W) — N networks, B after-states each
            scalars: (N, B, S) — matching scalars
        Returns:
            values:  (N, B) — value predictions
        """
        from torch.func import functional_call, vmap
        
        def single_call(params, buffers, grid, scalar):
            return functional_call(self._base, (params, buffers), (grid, scalar))

        return vmap(single_call, in_dims=(0, 0, 0, 0))(
            self._params, self._buffers, grids, scalars
        )


_vmap_helper: VmapHelper | None = None


def init_vmap(networks: list[ValueNetwork]) -> None:
    """Initialize vmap helper. Call once after creating networks."""
    global _vmap_helper
    _vmap_helper = VmapHelper(networks)


def refresh_vmap() -> None:
    """Refresh vmap cached parameters. Call after training steps."""
    if _vmap_helper is not None:
        _vmap_helper.refresh()


def argmax_a_Q_team_vmap(
    state: State,
    networks: list[ValueNetwork],
    env: BaseEnv,
    phase2: bool = False,
) -> Action:
    """Greedy action using vmap-batched forward passes across all N networks."""
    assert _vmap_helper is not None, "Call init_vmap() first"

    all_actions = get_phase2_actions(state, env.cfg) if phase2 else get_all_actions(env.cfg)
    n_nets = len(networks)

    after_states: list[State] = []
    immediate_rewards: list[float] = []
    for a in all_actions:
        if phase2:
            s_after, r = _after_state_and_team_reward(state, a, env, phase2=True)
            after_states.append(s_after)
            immediate_rewards.append(r)
        else:
            after_states.append(_after_state_for_action(state, a, env, phase2=False))
            immediate_rewards.append(0.0)

    all_grids = []
    all_scalars = []
    for i in range(n_nets):
        batch_enc = encoding.encode_batch_for_actions(state, i, after_states)
        all_grids.append(batch_enc.grid)
        if batch_enc.scalar is not None:
            all_scalars.append(batch_enc.scalar)

    grids_stacked = torch.stack(all_grids)
    scalars_stacked = torch.stack(all_scalars) if all_scalars else None

    with torch.no_grad():
        values = _vmap_helper.forward_batched(grids_stacked, scalars_stacked)
        team_values = values.sum(dim=0)

    for k in range(len(all_actions)):
        team_values[k] += immediate_rewards[k]

    best_idx = team_values.argmax().item()
    return all_actions[best_idx]


# ---------------------------------------------------------------------------
# Epsilon-greedy
# ---------------------------------------------------------------------------
def epsilon_greedy(
    state: State,
    networks: list[ValueNetwork],
    env: BaseEnv,
    epsilon: float,
    phase2: bool = False,
) -> Action:
    """With probability epsilon choose random, else greedy. Works for phase 1 and 2."""
    actions = get_phase2_actions(state, env.cfg) if phase2 else get_all_actions(env.cfg)
    if rng.random() < epsilon:
        return actions[rng.randint(0, len(actions) - 1)]
    return argmax_a_Q_team(state, networks, env, phase2=phase2)


def epsilon_greedy_batched(
    state: State,
    networks: list[ValueNetwork],
    env: BaseEnv,
    epsilon: float,
    use_vmap: bool = False,
    phase2: bool = False,
) -> Action:
    """Batched epsilon-greedy. Works for phase 1 and 2."""
    actions = get_phase2_actions(state, env.cfg) if phase2 else get_all_actions(env.cfg)
    if rng.random() < epsilon:
        return actions[rng.randint(0, len(actions) - 1)]
    if use_vmap and _vmap_helper is not None:
        return argmax_a_Q_team_vmap(state, networks, env, phase2=phase2)
    return argmax_a_Q_team_batched(state, networks, env, phase2=phase2)


# ---------------------------------------------------------------------------
# GPU-batched action selection (uses BatchedTrainer directly, no sync needed)
# ---------------------------------------------------------------------------
def argmax_a_Q_team_gpu(
    state: State,
    batched_trainer: object,
    env: BaseEnv,
    phase2: bool = False,
) -> Action:
    """Greedy action using BatchedTrainer's GPU params directly.

    Uses encode_all_agents_for_actions to build all N×B grids at once — O(N+B).
    No sync_to_networks or refresh_vmap needed.
    """
    all_actions = get_phase2_actions(state, env.cfg) if phase2 else get_all_actions(env.cfg)

    after_states: list[State] = []
    immediate_rewards: list[float] = []
    for a in all_actions:
        if phase2:
            s_after, r = _after_state_and_team_reward(state, a, env, phase2=True)
            after_states.append(s_after)
            immediate_rewards.append(r)
        else:
            after_states.append(_after_state_for_action(state, a, env, phase2=False))
            immediate_rewards.append(0.0)

    # Build all N×B grids at once — O(N+B), no O(N²)
    grids, scalars = encoding.encode_all_agents_for_actions(state, after_states)
    # grids: (N, B, C, H, W), scalars: (N, B, S)

    values = batched_trainer.forward_batched(grids, scalars)  # (N, B)
    team_values = values.sum(dim=0)  # (B,)

    for k in range(len(all_actions)):
        team_values[k] += immediate_rewards[k]

    best_idx = team_values.argmax().item()
    return all_actions[best_idx]


def epsilon_greedy_gpu(
    state: State,
    batched_trainer: object,
    env: BaseEnv,
    epsilon: float,
    phase2: bool = False,
) -> Action:
    """GPU-batched epsilon-greedy. No sync needed."""
    actions = get_phase2_actions(state, env.cfg) if phase2 else get_all_actions(env.cfg)
    if rng.random() < epsilon:
        return actions[rng.randint(0, len(actions) - 1)]
    return argmax_a_Q_team_gpu(state, batched_trainer, env, phase2=phase2)
