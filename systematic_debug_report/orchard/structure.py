"""Builders for environment reward/proficiency structure matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from orchard.enums import StructureType

if TYPE_CHECKING:
    from orchard.datatypes import EnvConfig


def build_structure(cfg: EnvConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (phi, relatedness) matrices for the configured structure."""
    if cfg.structure == StructureType.ID_DISTANCE:
        return build_id_distance_structure(cfg)
    if cfg.structure == StructureType.DISJOINT_GROUPS:
        return build_disjoint_group_structure(cfg)
    raise ValueError(f"Unknown structure type: {cfg.structure}")


def build_id_distance_structure(cfg: EnvConfig) -> tuple[np.ndarray, np.ndarray]:
    """Current structure: eligibility/sharing from distances between numeric IDs."""
    N = cfg.n_agents
    T = cfg.n_task_types
    C = cfg.clustering
    S = cfg.specialization

    phi = np.array(
        [[1.0 if abs(i - kappa) <= S else 0.0 for kappa in range(T)] for i in range(N)],
        dtype=np.float32,
    )
    relatedness = np.array(
        [[1.0 if abs(i - j) <= C else 0.0 for j in range(N)] for i in range(N)],
        dtype=np.float32,
    )
    return phi, relatedness


def build_disjoint_group_structure(cfg: EnvConfig) -> tuple[np.ndarray, np.ndarray]:
    """Block-diagonal groups for both relatedness and task proficiency.

    Agents share rewards only within their non-overlapping agent group. Agents are
    proficient only at task types in the matching task-type group.
    """
    N = cfg.n_agents
    T = cfg.n_task_types
    group_size = cfg.structure_group_size
    if group_size is None:
        raise ValueError("env.structure_group_size is required for structure=disjoint_groups")
    if group_size <= 0:
        raise ValueError("env.structure_group_size must be positive")
    n_tasks_per_group = cfg.n_tasks_per_group if cfg.n_tasks_per_group is not None else group_size
    if n_tasks_per_group <= 0:
        raise ValueError("env.n_tasks_per_group must be positive")

    agent_groups = [i // group_size for i in range(N)]
    task_groups = [kappa // n_tasks_per_group for kappa in range(T)]

    phi = np.array(
        [
            [1.0 if agent_groups[i] == task_groups[kappa] else 0.0 for kappa in range(T)]
            for i in range(N)
        ],
        dtype=np.float32,
    )
    relatedness = np.array(
        [[1.0 if agent_groups[i] == agent_groups[j] else 0.0 for j in range(N)] for i in range(N)],
        dtype=np.float32,
    )
    return phi, relatedness
