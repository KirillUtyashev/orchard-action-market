"""Fixed-seed evaluation on canonical test scenarios.

Notebook usage:
    from orchard.fixed_eval import evaluate_checkpoint

    results = evaluate_checkpoint(run_dir, scenario="edge_zones_center_agents")
    # {"greedy_team_rps": 0.42, "nearest_correct_task_stay_wrong_team_rps": 0.61, ...}

    # Or iterate a whole experiment:
    for config_dir in sorted(runs_dir.iterdir()):
        run_dirs = [d for d in config_dir.iterdir() if d.is_dir()]
        if run_dirs:
            results = evaluate_checkpoint(run_dirs[0], scenario="edge_zones_center_agents")
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from orchard.datatypes import (
    EnvConfig,
    EvalConfig,
    ExperimentConfig,
    Grid,
    LoggingConfig,
    State,
    StochasticConfig,
)
from orchard.env import create_env
from orchard.trainer import create_trainer


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestScenario:
    """A named evaluation scenario that modifies the training env config for testing."""
    name: str
    description: str
    make_stochastic: Callable[[StochasticConfig, EnvConfig], StochasticConfig]
    # Optional: returns fixed spawn zone top-left (row, col) per type, passed to set_eval_mode
    get_fixed_spawn_zones: Callable[[StochasticConfig, EnvConfig], tuple[tuple[int, int], ...]] | None = None
    # Optional: patches the State returned by env.init_state() (e.g. fixed agent positions)
    override_init_state: Callable[[State, EnvConfig], State] | None = None
    eval_seed: int = 42
    eval_steps: int = 2000
    n_test_states: int = 20


def _frozen_zones(stoch: StochasticConfig, env: EnvConfig) -> StochasticConfig:
    return dataclasses.replace(stoch, eval_spawn_zone_move_interval=0)


def _edge_zone_positions(
    height: int, width: int, size: int, n_zones: int
) -> tuple[tuple[int, int], ...]:
    """Compute n_zones spawn zone top-left corners that hug the grid edges,
    spread as evenly as possible around the perimeter of valid corner positions.

    Valid corner (r, c) means a size×size zone fits: r ∈ [0, height-size],
    c ∈ [0, width-size].  We trace the outermost ring of these corners clockwise
    and pick n_zones evenly-spaced points along that ring.

    Examples (9×9 grid, size=3 → max_r=max_c=6):
      n=1 → (0, 0)  top-left
      n=2 → (0, 0), (6, 6)  diagonal corners
      n=4 → (0,0), (0,6), (6,6), (6,0)  all four corners
      n=8 → corners + edge midpoints
    """
    max_r = height - size
    max_c = width - size

    # Clockwise perimeter of valid corner positions starting from (0, 0)
    perimeter: list[tuple[int, int]] = []
    for c in range(0, max_c + 1):           # top edge: left → right
        perimeter.append((0, c))
    for r in range(1, max_r + 1):           # right edge: top → bottom
        perimeter.append((r, max_c))
    for c in range(max_c - 1, -1, -1):      # bottom edge: right → left
        perimeter.append((max_r, c))
    for r in range(max_r - 1, 0, -1):       # left edge: bottom → top
        perimeter.append((r, 0))

    n = len(perimeter)
    return tuple(perimeter[round(i * n / n_zones) % n] for i in range(n_zones))


def _edge_zones_center_agents(stoch: StochasticConfig, env: EnvConfig) -> StochasticConfig:
    return dataclasses.replace(stoch, eval_spawn_zone_move_interval=0)


def _edge_zones_center_agents_spawn_zones(
    stoch: StochasticConfig, env: EnvConfig
) -> tuple[tuple[int, int], ...]:
    """Return spawn zone top-left corners for the edge_zones_center_agents scenario."""
    size = stoch.spawn_area_size
    if size is None:
        raise ValueError(
            "edge_zones_center_agents requires spawn_area_size to be set in the training config"
        )
    return _edge_zone_positions(env.height, env.width, size, env.n_task_types)


def _edge_zones_center_agents_init(state: State, env: EnvConfig) -> State:
    """Move all agents to grid center."""
    center = Grid(env.height // 2, env.width // 2)
    return dataclasses.replace(
        state,
        agent_positions=tuple(center for _ in range(env.n_agents)),
    )


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, TestScenario] = {
    "frozen_zones": TestScenario(
        name="frozen_zones",
        description=(
            "Freeze spawn zones (eval_spawn_zone_move_interval=0), fixed seed. "
            "Tests learned policy on stable zones without zone drift."
        ),
        make_stochastic=_frozen_zones,
    ),
    "edge_zones_center_agents": TestScenario(
        name="edge_zones_center_agents",
        description=(
            "Spawn zones placed at grid edges (maximally spread around perimeter), "
            "all agents start at grid center. Exposes centralized failure: agents "
            "without per-type zone awareness cannot identify which edge is theirs."
        ),
        make_stochastic=_edge_zones_center_agents,
        get_fixed_spawn_zones=_edge_zones_center_agents_spawn_zones,
        override_init_state=_edge_zones_center_agents_init,
    ),
}


def list_scenarios() -> list[str]:
    """Return names of all registered test scenarios."""
    return list(SCENARIOS)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _find_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints/ directory in {run_dir}")

    if checkpoint == "final":
        p = ckpt_dir / "final.pt"
        if p.exists():
            return p
        # Fall back to the highest-numbered step checkpoint
        steps = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if steps:
            return steps[-1]
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    if checkpoint == "latest":
        steps = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if steps:
            return steps[-1]
        raise FileNotFoundError(f"No step_*.pt checkpoint found in {ckpt_dir}")

    # Treat as explicit filename ("step_300000", "step_300000.pt", or full path)
    p = Path(checkpoint)
    if p.is_absolute() and p.exists():
        return p
    p = ckpt_dir / checkpoint
    if not p.suffix:
        p = p.with_suffix(".pt")
    if p.exists():
        return p
    raise FileNotFoundError(f"Checkpoint not found: {p}")


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    run_dir: str | Path,
    checkpoint: str = "final",
    scenario: str | TestScenario = "frozen_zones",
    device: str = "cpu",
    use_cache: bool = True,
) -> dict[str, float]:
    """Evaluate a trained checkpoint on a fixed test scenario.

    Args:
        run_dir:    Path to the timestamped run directory (contains metadata.yaml
                    and checkpoints/).  E.g. .../runs/dec_cnn16_lr4e4/2026-04-28_.../
        checkpoint: Which checkpoint to load.  "final" (default), "latest", a step
                    name like "step_300000", or an absolute path.
        scenario:   Name from SCENARIOS (see list_scenarios()) or a TestScenario.
        device:     "cpu" (default, safe for notebooks) or "cuda".
        use_cache:  Cache result to <run_dir>/fixed_eval/<scenario>.json.
                    Subsequent calls return the cached value instantly.

    Returns:
        Dict with keys: greedy_team_rps, greedy_rps, greedy_correct_pps,
        greedy_wrong_pps, {heuristic_name}_team_rps, {heuristic_name}_rps, ...
    """
    run_dir = Path(run_dir)

    if isinstance(scenario, str):
        if scenario not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario {scenario!r}. Available: {list(SCENARIOS)}"
            )
        scenario = SCENARIOS[scenario]

    # Cache check
    cache_path = run_dir / "fixed_eval" / f"{scenario.name}.json"
    if use_cache and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    # Load config embedded in metadata.yaml
    meta_path = run_dir / "metadata.yaml"
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    raw = meta["config"]

    from orchard.config import _parse_env, _parse_model, _parse_train, _parse_eval

    env_cfg_base = _parse_env(raw["env"])

    # Apply scenario modifications to stochastic config
    stoch_modified = scenario.make_stochastic(env_cfg_base.stochastic, env_cfg_base)
    env_cfg = dataclasses.replace(env_cfg_base, stochastic=stoch_modified)

    model_cfg = _parse_model(raw["model"])
    actor_model_raw = raw.get("actor_model")
    actor_model_cfg = _parse_model(actor_model_raw) if actor_model_raw else None
    train_cfg = _parse_train(raw["train"], n_task_types=env_cfg.n_task_types)
    train_cfg = dataclasses.replace(train_cfg, use_gpu=(device == "cuda"))

    eval_cfg = EvalConfig(
        eval_steps=scenario.eval_steps,
        n_test_states=scenario.n_test_states,
        checkpoint_freq=0,
        eval_seed=scenario.eval_seed,
    )

    cfg = ExperimentConfig(
        env=env_cfg,
        model=model_cfg,
        actor_model=actor_model_cfg,
        train=train_cfg,
        eval=eval_cfg,
        logging=LoggingConfig(),
    )

    # Build env + trainer, load weights
    from orchard import encoding
    from orchard.eval import evaluate_policy_metrics
    from orchard.policy import heuristic_action
    encoding.init_encoder(model_cfg.encoder, env_cfg)
    env = create_env(env_cfg)
    trainer = create_trainer(cfg, env)

    ckpt_path = _find_checkpoint(run_dir, checkpoint)
    trainer.load_checkpoint(ckpt_path)
    trainer.sync_to_cpu()

    # Run fixed eval with scenario overrides applied to set_eval_mode and init_state
    fixed_zones = (
        scenario.get_fixed_spawn_zones(stoch_modified, env_cfg)
        if scenario.get_fixed_spawn_zones is not None else None
    )
    env.set_eval_mode(True, seed=eval_cfg.eval_seed, fixed_spawn_zones=fixed_zones)
    try:
        eval_start = env.init_state()
        if scenario.override_init_state is not None:
            eval_start = scenario.override_init_state(eval_start, env_cfg)

        greedy_policy = lambda s: trainer._greedy_action(s)
        baseline_policy = lambda s: heuristic_action(s, env.cfg, trainer._heuristic)
        heuristic_name = trainer._heuristic.name.lower()

        greedy_m = evaluate_policy_metrics(eval_start, greedy_policy, env, eval_cfg.eval_steps)
        baseline_m = evaluate_policy_metrics(eval_start, baseline_policy, env, eval_cfg.eval_steps)
    finally:
        env.set_eval_mode(False)

    metrics = {
        "greedy_rps": greedy_m["rps"],
        "greedy_team_rps": greedy_m["team_rps"],
        "greedy_correct_pps": greedy_m["correct_pps"],
        "greedy_wrong_pps": greedy_m["wrong_pps"],
        f"{heuristic_name}_rps": baseline_m["rps"],
        f"{heuristic_name}_team_rps": baseline_m["team_rps"],
        f"{heuristic_name}_correct_pps": baseline_m["correct_pps"],
        f"{heuristic_name}_wrong_pps": baseline_m["wrong_pps"],
    }

    # Write cache
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics
