"""Load a training run directory into config, networks, and encoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from orchard.config import load_config
from orchard.datatypes import ExperimentConfig
from orchard.encoding.base import BaseEncoder
from orchard.encoding import _create_encoder
import orchard.encoding as encoding
from orchard.enums import LearningType
from orchard.model import ValueNetwork, create_networks


@dataclass
class LoadedRun:
    label: str
    cfg: ExperimentConfig
    networks: list[ValueNetwork]
    encoder: BaseEncoder
    is_centralized: bool
    checkpoint_step: int
    run_dir: Path


def _load_config_from_metadata(metadata_path: Path) -> ExperimentConfig:
    """Load ExperimentConfig from a run's metadata.yaml."""
    import tempfile
    import os

    with open(metadata_path) as f:
        raw = yaml.safe_load(f)

    if "config" not in raw:
        raise ValueError(f"No 'config' key in {metadata_path}")

    cfg_dict = raw["config"]

    # metadata uses 'env_type' but load_config expects 'type'
    if "env" in cfg_dict and "env_type" in cfg_dict["env"] and "type" not in cfg_dict["env"]:
        cfg_dict["env"]["type"] = cfg_dict["env"].pop("env_type")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(cfg_dict, tmp)
        tmp_path = tmp.name

    try:
        return load_config(tmp_path)
    finally:
        os.unlink(tmp_path)


def load_run(run_dir: Path, checkpoint_name: str = "final.pt") -> LoadedRun:
    """Load a run directory: parse metadata, create networks + encoder, load weights."""
    run_dir = Path(run_dir)
    metadata_path = run_dir / "metadata.yaml"
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name

    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.yaml in {run_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = _load_config_from_metadata(metadata_path)

    is_centralized = cfg.train.learning_type == LearningType.CENTRALIZED
    n_networks = 1 if is_centralized else cfg.env.n_agents

    # Create encoder (independent instance for value computation)
    encoder = _create_encoder(cfg.model.input_type, cfg.env, cfg.model.k_nearest)

    # Must init global encoder before create_networks, because ValueNetwork.__init__
    # reads encoding.get_scalar_dim() / get_grid_channels() from the singleton.
    encoding.init_encoder(cfg.model.input_type, cfg.env, cfg.model.k_nearest)

    # Create networks with matching architecture
    networks = create_networks(
        cfg.model, cfg.env, cfg.train.lr, cfg.train.total_steps,
        nstep=cfg.train.nstep, td_lambda=cfg.train.td_lambda,
        train_method=cfg.train.train_method, n_networks=n_networks,
    )

    # Load checkpoint weights
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dicts = ckpt["networks"]
    if len(state_dicts) != len(networks):
        raise ValueError(
            f"Checkpoint has {len(state_dicts)} networks but config specifies "
            f"{n_networks} ({'centralized' if is_centralized else 'decentralized'})"
        )
    for net, sd in zip(networks, state_dicts):
        net.load_state_dict(sd, strict=True)
        net.eval()

    checkpoint_step = ckpt.get("step", 0)
    lt = "centralized" if is_centralized else "decentralized"
    label = f"{lt}, step {checkpoint_step}"

    return LoadedRun(
        label=label,
        cfg=cfg,
        networks=networks,
        encoder=encoder,
        is_centralized=is_centralized,
        checkpoint_step=checkpoint_step,
        run_dir=run_dir,
    )
