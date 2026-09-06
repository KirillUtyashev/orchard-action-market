"""Per-input-channel weight norm analysis from checkpoints.

Usage in notebooks:
    from orchard.analysis import norms_over_time, plot_channel_norms

    # 3-panel figure: task channels | position channels | scalar channels
    # dec: shows critic_{agent_idx};  cen: shows critic_0
    fig = plot_channel_norms(run_dir, agent_idx=0)
    plt.show()

    # Raw DataFrame: step, network, channel, kind, norm
    df = norms_over_time(run_dir)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import torch
import yaml


# ---------------------------------------------------------------------------
# Channel name definitions per encoder type
# ---------------------------------------------------------------------------

def _everything_channels(T: int, N: int) -> tuple[list[str], list[str]]:
    grid = [f"task_presence_{k}" for k in range(T)] + [
        f"agent_{j}_pos" for j in range(N)
    ] + ["actor_pos"]
    scalar = [f"actor_is_{j}" for j in range(N)] + ["pick_phase"]
    return grid, scalar


def _filtered_dec_channels(KR: int, KW: int) -> tuple[list[str], list[str]]:
    grid = [f"task_R_{t}" for t in range(KR)] + [
        f"agent_W_{s}" for s in range(KW)
    ] + ["actor_pos"]
    scalar = [f"actor_local_{s}" for s in range(KW)] + ["pick_phase"]
    return grid, scalar


_CHANNEL_FNS: dict[str, object] = {
    "everything_cnn_grid": _everything_channels,
}

# Channel groups for plotting: (group_label, [channel_names])
def _everything_groups(T: int, N: int) -> list[tuple[str, list[str]]]:
    return [
        ("task channels",     [f"task_presence_{k}" for k in range(T)]),
        ("position channels", [f"agent_{j}_pos" for j in range(N)] + ["actor_pos"]),
        ("scalar channels",   [f"actor_is_{j}" for j in range(N)] + ["pick_phase"]),
    ]

def _filtered_dec_groups(KR: int, KW: int) -> list[tuple[str, list[str]]]:
    return [
        ("task channels",     [f"task_R_{t}" for t in range(KR)]),
        ("position channels", [f"agent_W_{s}" for s in range(KW)] + ["actor_pos"]),
        ("scalar channels",   [f"actor_local_{s}" for s in range(KW)] + ["pick_phase"]),
    ]

_GROUP_FNS = {
    "everything_cnn_grid":  _everything_groups,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ChannelNorm:
    network: str   # e.g. "critic_0", "actor_1"
    channel: str   # e.g. "task_type_0", "is_actor"
    kind: str      # "grid" or "scalar"
    norm: float


def channel_weight_norms(pt_path: str | Path) -> list[ChannelNorm]:
    """Per-input-channel weight norms from a single checkpoint.

    Grid channel i  → conv.0.weight[:, i, :, :].norm()
    Scalar channel j → net.0.weight[:, flat_conv_dim + j].norm()

    Reads metadata.yaml from the run directory (parent of checkpoints/).
    """
    pt_path = Path(pt_path)
    meta_path = pt_path.parent.parent / "metadata.yaml"
    grid_names, scalar_names = _channel_names_from_meta_path(meta_path)
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
    return _extract_norms(ckpt, grid_names, scalar_names)


def norms_over_time(run_dir: str | Path) -> "pd.DataFrame":
    """Load all checkpoints in run_dir/checkpoints/ and return a DataFrame.

    Columns: step, network, channel, kind, norm
    Sorted by step ascending; final.pt is included with its stored step value.
    """
    import pandas as pd

    run_dir = Path(run_dir)
    grid_names, scalar_names = _channel_names_from_meta_path(run_dir / "metadata.yaml")

    ckpt_dir = run_dir / "checkpoints"
    pt_files = sorted(ckpt_dir.glob("*.pt"), key=lambda p: _sort_key(p.stem))

    rows = []
    for pt_file in pt_files:
        ckpt = torch.load(pt_file, map_location="cpu", weights_only=True)
        step = int(ckpt.get("step", _sort_key(pt_file.stem)))
        for cn in _extract_norms(ckpt, grid_names, scalar_names):
            rows.append({"step": step, **asdict(cn)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _channel_names_from_meta_path(meta_path: Path) -> tuple[list[str], list[str]]:
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    cfg = meta["config"]
    encoder = cfg["model"]["encoder"]
    T = cfg["env"]["n_task_types"]
    N = cfg["env"]["n_agents"]
    if encoder == "filtered_dec_cnn_grid":
        RR = cfg["env"].get("relatedness_width", 0)
        PR = cfg["env"].get("proficiency_width", 0)
        KR = min(N, 2 * RR + 1)
        KW = min(N, 2 * (RR + PR) + 1)
        return _filtered_dec_channels(KR, KW)
    fn = _CHANNEL_FNS.get(encoder)
    if fn is None:
        raise ValueError(f"Unknown encoder type: {encoder!r}")
    return fn(T, N)  # type: ignore[call-arg]


def _extract_norms(
    ckpt: dict,
    grid_names: Sequence[str],
    scalar_names: Sequence[str],
) -> list[ChannelNorm]:
    scalar_dim = len(scalar_names)
    results: list[ChannelNorm] = []

    groups: list[tuple[str, list[dict]]] = []
    if "critics" in ckpt:
        groups.append(("critic", ckpt["critics"]))
    if "actors" in ckpt:
        groups.append(("actor", ckpt["actors"]))

    for role, state_dicts in groups:
        for idx, sd in enumerate(state_dicts):
            net_name = f"{role}_{idx}"
            conv0 = sd.get("conv.0.weight")  # (out_C, C_grid, kH, kW)
            net0 = sd.get("net.0.weight")    # (hidden_or_1, flat_conv + scalar_dim)

            if conv0 is not None:
                for i, name in enumerate(grid_names):
                    norm = conv0[:, i, :, :].norm().item()
                    results.append(ChannelNorm(net_name, name, "grid", norm))

            if net0 is not None:
                flat_conv_dim = net0.shape[1] - scalar_dim
                for j, name in enumerate(scalar_names):
                    norm = net0[:, flat_conv_dim + j].norm().item()
                    results.append(ChannelNorm(net_name, name, "scalar", norm))

    return results


def _sort_key(stem: str) -> int:
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else 10**18
