"""CLI entry point: python -m orchard.viz CONFIG_YAML [options]"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

import numpy as np

import orchard.encoding as encoding
from orchard.actor_critic import (
    PolicyNetwork,
    build_phase1_legal_mask,
    build_phase2_legal_mask,
    policy_index_to_action,
)
from orchard.config import load_config
from orchard.env import create_env
from orchard.model import ValueNetwork, create_networks, create_actor_networks
from orchard.policy import (
    heuristic_action, get_all_actions, get_phase2_actions,
)
from orchard.seed import set_all_seeds, rng
from orchard.datatypes import State
from orchard.enums import Action, Heuristic, PickMode, num_actions

from orchard.viz.export import write_summary_json, write_trajectory_csv
from orchard.viz.frame import Frame
from orchard.viz.html_builder import build_html
from orchard.viz.renderer import render_frame_svg
from orchard.viz.rollout import generate_frames


_HEURISTIC_MAP = {
    "nearest_task": Heuristic.NEAREST_TASK,
    "nearest_correct_task": Heuristic.NEAREST_CORRECT_TASK,
    "nearest_correct_task_stay_wrong": Heuristic.NEAREST_CORRECT_TASK_STAY_WRONG,
    "nearest": Heuristic.NEAREST_TASK,  # backward compat alias
}

_ALL_POLICIES = list(_HEURISTIC_MAP.keys()) + ["random", "learned"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Orchard trajectory visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("config", type=str, help="Path to YAML config file")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    p.add_argument("--policy", choices=_ALL_POLICIES,
                   default=None,
                   help="Policy to visualize (default: learned if --checkpoint, else auto-detect heuristic from config)")
    p.add_argument("--compare", nargs="?", const="auto", default=None,
                   metavar="POLICY",
                   help="Compare against another policy. Optionally specify which: "
                        + ", ".join(_ALL_POLICIES) + ". "
                        "Default: auto-select heuristic based on config.")
    p.add_argument("--show-after-states", action="store_true", help="Show s_t and s_{t+1} per transition")
    p.add_argument("--steps", type=int, default=200, help="Number of agent decisions (default: 200)")
    p.add_argument("--seed", type=int, default=None, help="Override config seed")
    p.add_argument("--fps", type=int, default=3, help="Autoplay FPS (default: 3)")
    p.add_argument("--output-dir", type=str, default="./viz_output", help="Output directory")
    p.add_argument("--decisions", action="store_true", help="Show Q-values for all actions (requires --checkpoint)")
    p.add_argument("--values", action="store_true", help="Show per-agent V_i(s) (requires --checkpoint)")
    p.add_argument("--dpi", type=int, default=120, help="PNG render DPI (default: 120)")
    p.add_argument("--no-html", action="store_true",
                   help="Skip rendering and HTML — just print stats and write CSV/JSON (fast sanity check)")
    return p.parse_args()


def load_checkpoint(
    checkpoint_path: str,
    networks: list[ValueNetwork],
) -> tuple[int, dict]:
    """Load critic weights into networks. Returns (training step, raw ckpt dict)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "networks" in ckpt:
        state_dicts = ckpt["networks"]
    elif "critics" in ckpt:
        state_dicts = ckpt["critics"]
    else:
        raise KeyError(f"Checkpoint has neither 'networks' nor 'critics' key. Keys: {list(ckpt.keys())}")
    if len(state_dicts) != len(networks):
        raise ValueError(
            f"Checkpoint has {len(state_dicts)} networks but config specifies "
            f"{len(networks)} agents"
        )
    for net, sd in zip(networks, state_dicts):
        net.load_state_dict(sd, strict=True)
        net.eval()
    return ckpt.get("step", 0), ckpt


def load_actor_networks(
    ckpt: dict,
    actor_networks: list[PolicyNetwork],
) -> None:
    """Load actor weights from a checkpoint dict into actor networks."""
    if "actors" not in ckpt:
        raise KeyError(f"Checkpoint has no 'actors' key. Keys: {list(ckpt.keys())}")
    state_dicts = ckpt["actors"]
    if len(state_dicts) != len(actor_networks):
        raise ValueError(
            f"Checkpoint has {len(state_dicts)} actors but config specifies "
            f"{len(actor_networks)} agents"
        )
    for net, sd in zip(actor_networks, state_dicts):
        net.load_state_dict(sd, strict=True)
        net.eval()


def _greedy_actor_action(
    state: State,
    actor_networks: list[PolicyNetwork],
    env,
) -> Action:
    """Greedy action via actor network argmax (no critic involved)."""
    actor_id = state.actor
    enc = encoding.encode(state, actor_id)
    legal_mask = build_phase2_legal_mask(state, env.cfg) if state.pick_phase else build_phase1_legal_mask(state, env.cfg)
    with torch.no_grad():
        probs = actor_networks[actor_id].get_action_probabilities(enc, legal_mask)
    return policy_index_to_action(int(np.argmax(probs)))


def _greedy_action_batched(
    state: State,
    networks: list[ValueNetwork],
    env,
) -> Action:
    """Standalone greedy argmax over Q_team = r_team(s,a) + sum_j V_j(after_state)."""
    phase2 = state.pick_phase
    all_actions = get_phase2_actions(state, env.cfg) if phase2 else get_all_actions(env.cfg)

    after_states: list[State] = []
    immediate_rewards: list[float] = []
    for a in all_actions:
        if phase2 and a.is_pick():
            s_after, rewards = env.resolve_pick(state, pick_type=a.pick_type())
            after_states.append(s_after)
            immediate_rewards.append(sum(rewards))
        elif phase2:
            after_states.append(state)
            immediate_rewards.append(0.0)
        else:
            s = env.apply_action(state, a)
            if s.is_agent_on_task(s.actor):
                after_states.append(s.with_pick_phase())
            else:
                after_states.append(s)
            immediate_rewards.append(0.0)

    n_actions = len(after_states)

    # Use encode_all_agents_for_actions to match GPU trainer encoding exactly.
    # encode_batch_for_actions omits Ch5 (actor position) for the actor's own
    # encoding, causing wrong critic inputs vs what was seen during training.
    grids, scalars = encoding.encode_all_agents_for_actions(state, after_states)
    # grids: (N, B, C, H, W), scalars: (N, B, S)

    team_values = [0.0] * n_actions
    with torch.no_grad():
        for i, net in enumerate(networks):
            vals = net.forward_raw(grids[i], scalars[i])  # (B,)
            for k in range(n_actions):
                team_values[k] += vals[k].item()

    best_idx = 0
    best_val = team_values[0] + immediate_rewards[0]
    for k in range(1, n_actions):
        val = team_values[k] + immediate_rewards[k]
        if val > best_val:
            best_val = val
            best_idx = k
    return all_actions[best_idx]


def make_policy_fn(
    policy_name: str,
    networks: list[ValueNetwork] | None,
    env,
    actor_networks: list[PolicyNetwork] | None = None,
):
    """Return a policy function: State -> Action.

    Phase is determined by state.pick_phase. In FORCED mode, rollout_trajectory
    handles the pick automatically and never calls the policy in pick phase.

    If actor_networks is provided, 'learned' uses actor argmax (actor_critic mode).
    Otherwise, 'learned' uses critic argmax (value mode).
    """
    if policy_name == "learned":
        if actor_networks is not None:
            def policy(s: State) -> Action:
                return _greedy_actor_action(s, actor_networks, env)
            return policy
        if networks is None:
            raise ValueError("--policy learned requires --checkpoint")
        def policy(s: State) -> Action:
            return _greedy_action_batched(s, networks, env)
        return policy
    elif policy_name in _HEURISTIC_MAP:
        h = _HEURISTIC_MAP[policy_name]
        def policy(s: State) -> Action:
            return heuristic_action(s, env.cfg, h)
        return policy
    elif policy_name == "random":
        n_act = num_actions(env.cfg.pick_mode, env.cfg.n_task_types)
        def policy(s: State) -> Action:
            return Action(rng.randint(0, n_act - 1))
        return policy
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


def render_all_frames(
    frames: list[Frame],
    show_after_states: bool,
    n_task_types: int = 1,
    task_assignments: tuple[tuple[int, ...], ...] | None = None,
) -> list[str]:
    """Render all frames to inline SVG strings."""
    svgs: list[str] = []
    total = len(frames)
    for i, frame in enumerate(frames):
        if (i + 1) % 50 == 0 or i == 0 or i == total - 1:
            print(f"  Rendering frame {i + 1}/{total}...", end="\r")

        picked_cell = None
        picked_correct = None
        if frame.picked and frame.picked_task_type is not None:
            pos = frame.state.agent_positions[frame.actor]
            picked_cell = (pos.row, pos.col)
            picked_correct = frame.picked_correct

        svg = render_frame_svg(
            state=frame.state,
            state_after=frame.state_after if show_after_states else None,
            height=frame.height,
            width=frame.width,
            state_index=frame.state_index,
            actor=frame.actor,
            action=frame.action,
            rewards=frame.rewards,
            discount=frame.discount,
            show_after_state=show_after_states,
            n_task_types=n_task_types,
            task_assignments=task_assignments,
            picked_cell=picked_cell,
            picked_correct=picked_correct,
        )
        svgs.append(svg)
    print()
    return svgs


def _load_config_or_metadata(path: str):
    """Load config from either a raw config YAML or a run metadata.yaml."""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)

    if "config" in raw and "env" not in raw:
        raw = raw["config"]

    if "env" in raw and "env_type" in raw["env"] and "type" not in raw["env"]:
        raw["env"]["type"] = raw["env"].pop("env_type")

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(raw, tmp)
        tmp_path = tmp.name

    try:
        return load_config(tmp_path)
    finally:
        import os
        os.unlink(tmp_path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load config ---
    print(f"Loading config: {args.config}")
    cfg = _load_config_or_metadata(args.config)

    seed = args.seed if args.seed is not None else cfg.train.seed
    set_all_seeds(seed)

    n_task_types = cfg.env.n_task_types
    task_assignments = cfg.env.task_assignments
    pick_mode = cfg.env.pick_mode

    # --- Create env and encoder ---
    env = create_env(cfg.env)
    encoding.init_encoder(cfg.model.encoder, cfg.env)

    # --- Load checkpoint if provided ---
    from orchard.enums import LearningType, AlgorithmName
    centralized = cfg.train.learning_type == LearningType.CENTRALIZED
    n_networks = 1 if centralized else cfg.env.n_agents
    use_actor_argmax = cfg.train.algorithm.name == AlgorithmName.ACTOR_CRITIC

    networks: list[ValueNetwork] | None = None
    actor_networks: list[PolicyNetwork] | None = None
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        if centralized:
            print(f"  Centralized mode: creating 1 shared network for {cfg.env.n_agents} agents")
        networks = create_networks(cfg.model, cfg.env, cfg.train)
        ckpt_step, ckpt = load_checkpoint(args.checkpoint, networks)
        print(f"  Loaded checkpoint at training step {ckpt_step}")
        if use_actor_argmax:
            print(f"  Algorithm=actor_critic: loading actor networks for argmax")
            actor_networks = create_actor_networks(cfg.actor_model, cfg.env, cfg.train)
            load_actor_networks(ckpt, actor_networks)

    # --- Determine policy ---
    if args.policy is not None:
        policy_name = args.policy
    elif args.checkpoint:
        policy_name = "learned"
    elif n_task_types > 1:
        policy_name = "nearest_correct_task"
    else:
        policy_name = "nearest_task"

    if policy_name == "learned" and networks is None:
        print("ERROR: --policy learned requires --checkpoint", file=sys.stderr)
        sys.exit(1)

    if (args.decisions or args.values) and networks is None:
        print("ERROR: --decisions and --values require --checkpoint", file=sys.stderr)
        sys.exit(1)

    if args.compare is not None and networks is None and args.compare in ("auto", "learned"):
        # Only error if the *primary* policy needs a checkpoint; compare=learned also needs it
        pass  # Validated later when we resolve the compare policy name

    # Print config info (always)
    print(f"  T={n_task_types}, N={cfg.env.n_agents}, grid={cfg.env.height}x{cfg.env.width}")
    print(f"  Pick mode: {pick_mode.name}, r_picker={cfg.env.r_picker}, r_low={cfg.env.r_low}")
    if task_assignments is not None:
        print(f"  Assignments: {task_assignments}")

    # --- Generate initial state ---
    init_state = env.init_state()

    # --- Generate primary rollout ---
    print(f"Rolling out {args.steps} decisions with policy: {policy_name}")
    t0 = time.time()

    policy_fn = make_policy_fn(policy_name, networks, env,
                               actor_networks=actor_networks)
    frames = generate_frames(
        start_state=init_state,
        policy_fn=policy_fn,
        env=env,
        n_steps=args.steps,
        policy_name=policy_name,
        networks=networks,
        include_decisions=args.decisions,
        include_values=args.values,
    )
    print(f"  Generated {len(frames)} transitions ({args.steps} decisions) in {time.time() - t0:.1f}s")

    # --- Generate compare rollout (if requested) ---
    compare_frames: list[Frame] | None = None
    if args.compare is not None:
        # Determine compare policy name
        if args.compare == "auto":
            compare_name = "nearest_correct_task" if n_task_types > 1 else "nearest_task"
        else:
            compare_name = args.compare

        # Validate compare policy
        if compare_name not in _ALL_POLICIES:
            print(f"ERROR: unknown compare policy '{compare_name}'. "
                  f"Choose from: {', '.join(_ALL_POLICIES)}", file=sys.stderr)
            sys.exit(1)

        compare_networks = networks if compare_name == "learned" else None
        if compare_name == "learned" and networks is None:
            print("ERROR: --compare learned requires --checkpoint", file=sys.stderr)
            sys.exit(1)

        print(f"Rolling out {args.steps} decisions with policy: {compare_name} (compare)")
        env_compare = create_env(cfg.env)
        compare_actor_networks = actor_networks if compare_name == "learned" else None
        compare_fn = make_policy_fn(compare_name, compare_networks, env_compare,
                                    actor_networks=compare_actor_networks)
        compare_frames = generate_frames(
            start_state=init_state,
            policy_fn=compare_fn,
            env=env_compare,
            n_steps=args.steps,
            policy_name=compare_name,
            networks=compare_networks,
            include_decisions=False,
            include_values=False,
        )
        print(f"  Generated {len(compare_frames)} compare transitions")

    # --- Stats summary (always printed) ---
    def _team_rps(frms: list[Frame]) -> float:
        return frms[-1].total_team_reward / frms[-1].total_decisions if frms[-1].total_decisions > 0 else 0.0

    print()
    rps_primary = _team_rps(frames)
    td = frames[-1].total_decisions
    print(f"  {policy_name} Team RPS: {rps_primary:.4f}  (total team reward {frames[-1].total_team_reward:.1f} / {td} decisions)")
    if td > 0:
        print(f"    Correct: {frames[-1].total_correct_picks} ({frames[-1].total_correct_picks/td:.4f}/step)  "
              f"Wrong: {frames[-1].total_wrong_picks} ({frames[-1].total_wrong_picks/td:.4f}/step)")

    if compare_frames is not None:
        cname = compare_frames[0].policy_name
        rps_compare = _team_rps(compare_frames)
        td = compare_frames[-1].total_decisions
        print(f"  {cname} Team RPS: {rps_compare:.4f}  (total team reward {compare_frames[-1].total_team_reward:.1f} / {td} decisions)")
        if td > 0:
            print(f"    Correct: {compare_frames[-1].total_correct_picks} ({compare_frames[-1].total_correct_picks/td:.4f}/step)  "
                  f"Wrong: {compare_frames[-1].total_wrong_picks} ({compare_frames[-1].total_wrong_picks/td:.4f}/step)")
        print(f"  Δ Team RPS (learned − {cname}): {_team_rps(frames) - rps_compare:+.4f}")
    print()

    # --- Write CSV and summary ---
    csv_path = out_dir / "trajectory.csv"
    write_trajectory_csv(frames, csv_path)
    print(f"Wrote {csv_path}")

    summary_path = out_dir / "summary.json"
    write_summary_json(
        frames, summary_path,
        config_path=args.config,
        checkpoint_path=args.checkpoint or "",
        seed=seed,
    )
    print(f"Wrote {summary_path}")

    if compare_frames is not None:
        csv_compare_path = out_dir / "trajectory_compare.csv"
        write_trajectory_csv(compare_frames, csv_compare_path)
        print(f"Wrote {csv_compare_path}")

        summary_compare_path = out_dir / "summary_compare.json"
        write_summary_json(
            compare_frames, summary_compare_path,
            config_path=args.config,
            checkpoint_path="",
            seed=seed,
        )
        print(f"Wrote {summary_compare_path}")

    if args.no_html:
        print("Done (--no-html: skipped rendering and HTML).")
        return

    # --- Render SVGs ---
    print("Rendering primary frames...")
    t0 = time.time()
    frame_svgs = render_all_frames(frames, args.show_after_states,
                                   n_task_types=n_task_types, task_assignments=task_assignments)
    print(f"  Rendered in {time.time() - t0:.1f}s")

    compare_svgs: list[str] | None = None
    if compare_frames is not None:
        print("Rendering compare frames...")
        t0 = time.time()
        compare_svgs = render_all_frames(compare_frames, args.show_after_states,
                                         n_task_types=n_task_types, task_assignments=task_assignments)
        print(f"  Rendered in {time.time() - t0:.1f}s")

    # --- Build HTML ---
    html_path = out_dir / "trajectory.html"
    print("Building HTML viewer...")
    t0 = time.time()
    build_html(
        frames=frames,
        frame_svgs=frame_svgs,
        output_path=html_path,
        fps=args.fps,
        compare_frames=compare_frames,
        compare_svgs=compare_svgs,
        n_task_types=n_task_types,
        task_assignments=task_assignments,
        pick_mode=pick_mode,
    )
    print(f"Wrote {html_path} ({html_path.stat().st_size / 1024 / 1024:.1f} MB) in {time.time() - t0:.1f}s")
    print(f"\nDone! Open {html_path} in a browser to view.")


if __name__ == "__main__":
    main()
