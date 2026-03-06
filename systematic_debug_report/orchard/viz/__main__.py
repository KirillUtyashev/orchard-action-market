"""CLI entry point: python -m orchard.viz CONFIG_YAML [options]"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

import orchard.encoding as encoding
from orchard.config import load_config
from orchard.env import create_env
from orchard.model import ValueNetwork, create_networks
from orchard.policy import argmax_a_Q_team, nearest_apple_action
from orchard.seed import set_all_seeds, rng
from orchard.datatypes import State
from orchard.enums import Action

from orchard.viz.export import write_summary_json, write_trajectory_csv
from orchard.viz.frame import Frame
from orchard.viz.html_builder import build_html
from orchard.viz.renderer import render_frame_png
from orchard.viz.rollout import generate_frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Orchard trajectory visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("config", type=str, help="Path to YAML config file")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    p.add_argument("--policy", choices=["nearest", "random", "learned"], default=None,
                   help="Policy to visualize (default: learned if --checkpoint, else nearest)")
    p.add_argument("--compare", action="store_true", help="Vertical stack: learned (top) vs nearest (bottom)")
    p.add_argument("--show-after-states", action="store_true", help="Show s_t and s_{t+1} per transition")
    p.add_argument("--steps", type=int, default=200, help="Number of agent decisions (default: 200)")
    p.add_argument("--seed", type=int, default=None, help="Override config seed")
    p.add_argument("--fps", type=int, default=3, help="Autoplay FPS (default: 3)")
    p.add_argument("--output-dir", type=str, default="./viz_output", help="Output directory")
    p.add_argument("--decisions", action="store_true", help="Show Q-values for all actions (requires --checkpoint)")
    p.add_argument("--values", action="store_true", help="Show per-agent V_i(s) (requires --checkpoint)")
    p.add_argument("--dpi", type=int, default=120, help="PNG render DPI (default: 120)")
    return p.parse_args()


def load_checkpoint(
    checkpoint_path: str,
    networks: list[ValueNetwork],
) -> int:
    """Load checkpoint into networks. Returns the training step."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dicts = ckpt["networks"]
    if len(state_dicts) != len(networks):
        raise ValueError(
            f"Checkpoint has {len(state_dicts)} networks but config specifies "
            f"{len(networks)} agents"
        )
    for net, sd in zip(networks, state_dicts):
        net.load_state_dict(sd, strict=True)
        net.eval()
    return ckpt.get("step", 0)


def make_policy_fn(
    policy_name: str,
    networks: list[ValueNetwork] | None,
    env,
):
    """Return a policy function: State -> Action."""
    if policy_name == "learned":
        if networks is None:
            raise ValueError("--policy learned requires --checkpoint")
        def policy(s: State) -> Action:
            return argmax_a_Q_team(s, networks, env)
        return policy
    elif policy_name == "nearest":
        def policy(s: State) -> Action:
            return nearest_apple_action(s, env.cfg)
        return policy
    elif policy_name == "random":
        def policy(s: State) -> Action:
            return Action(rng.randint(0, len(Action) - 2))  # exclude PICK
        return policy
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


def render_all_frames(
    frames: list[Frame],
    show_after_states: bool,
    dpi: int,
) -> list[bytes]:
    """Render all frames to PNG bytes."""
    pngs: list[bytes] = []
    total = len(frames)
    for i, frame in enumerate(frames):
        if (i + 1) % 50 == 0 or i == 0 or i == total - 1:
            print(f"  Rendering frame {i + 1}/{total}...", end="\r")
        png = render_frame_png(
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
            dpi=dpi,
        )
        pngs.append(png)
    print()
    return pngs


def _load_config_or_metadata(path: str):
    """Load config from either a raw config YAML or a run metadata.yaml.

    Metadata files have the config nested under a 'config:' key and use
    'env_type' instead of 'type'. This handles both formats transparently.
    """
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)

    # If top-level has 'config' key, unwrap it (metadata format)
    if "config" in raw and "env" not in raw:
        raw = raw["config"]

    # Normalize: metadata uses 'env_type' but load_config expects 'type'
    if "env" in raw and "env_type" in raw["env"] and "type" not in raw["env"]:
        raw["env"]["type"] = raw["env"].pop("env_type")

    # Write to a temp file and use load_config for full validation
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

    # --- Create env and encoder ---
    env = create_env(cfg.env)
    encoding.init_encoder(cfg.model.input_type, cfg.env, cfg.model.k_nearest)

    # --- Load checkpoint if provided ---
    from orchard.enums import LearningType
    centralized = cfg.train.learning_type == LearningType.CENTRALIZED
    n_networks = 1 if centralized else cfg.env.n_agents

    networks: list[ValueNetwork] | None = None
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        if centralized:
            print(f"  Centralized mode: creating 1 shared network for {cfg.env.n_agents} agents")
        networks = create_networks(
            cfg.model, cfg.env, cfg.train.lr, cfg.train.total_steps,
            nstep=cfg.train.nstep, td_lambda=cfg.train.td_lambda,
            train_method=cfg.train.train_method, n_networks=n_networks,
        )
        ckpt_step = load_checkpoint(args.checkpoint, networks)
        print(f"  Loaded checkpoint at training step {ckpt_step}")

    # --- Determine policy ---
    if args.policy is not None:
        policy_name = args.policy
    else:
        policy_name = "learned" if args.checkpoint else "nearest"

    if policy_name == "learned" and networks is None:
        print("ERROR: --policy learned requires --checkpoint", file=sys.stderr)
        sys.exit(1)

    if (args.decisions or args.values) and networks is None:
        print("ERROR: --decisions and --values require --checkpoint", file=sys.stderr)
        sys.exit(1)

    if args.compare and networks is None:
        print("ERROR: --compare requires --checkpoint", file=sys.stderr)
        sys.exit(1)

    # --- Generate initial state ---
    init_state = env.init_state()

    # --- Generate primary rollout ---
    print(f"Rolling out {args.steps} decisions with policy: {policy_name}")
    t0 = time.time()

    policy_fn = make_policy_fn(policy_name, networks, env)
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
    if args.compare:
        print(f"Rolling out {args.steps} decisions with policy: nearest (compare)")
        env_compare = create_env(cfg.env)
        nearest_fn = make_policy_fn("nearest", None, env_compare)
        compare_frames = generate_frames(
            start_state=init_state,
            policy_fn=nearest_fn,
            env=env_compare,
            n_steps=args.steps,
            policy_name="nearest",
            networks=None,
            include_decisions=False,
            include_values=False,
        )
        print(f"  Generated {len(compare_frames)} compare transitions")

    # --- Render PNGs ---
    print("Rendering primary frames...")
    t0 = time.time()
    frame_pngs = render_all_frames(frames, args.show_after_states, args.dpi)
    print(f"  Rendered in {time.time() - t0:.1f}s")

    compare_pngs: list[bytes] | None = None
    if compare_frames is not None:
        print("Rendering compare frames...")
        t0 = time.time()
        compare_pngs = render_all_frames(compare_frames, args.show_after_states, args.dpi)
        print(f"  Rendered in {time.time() - t0:.1f}s")

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

    # --- Build HTML ---
    html_path = out_dir / "trajectory.html"
    print("Building HTML viewer...")
    t0 = time.time()
    build_html(
        frames=frames,
        frame_pngs=frame_pngs,
        output_path=html_path,
        fps=args.fps,
        compare_frames=compare_frames,
        compare_pngs=compare_pngs,
    )
    print(f"Wrote {html_path} ({html_path.stat().st_size / 1024 / 1024:.1f} MB) in {time.time() - t0:.1f}s")
    print(f"\nDone! Open {html_path} in a browser to view.")


if __name__ == "__main__":
    main()
