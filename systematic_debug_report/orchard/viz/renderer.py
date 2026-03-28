"""Matplotlib grid renderer: Frame → PNG bytes.

Each grid cell is a square. Agents are colored circles with ID labels.
Tasks are colored by type with distinct colors. Actor gets a gold ring.
Pick events: correct picks get green border, wrong picks get red border.
"""

from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from orchard.datatypes import Grid, State
from orchard.enums import Action

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
_TAB10 = plt.cm.tab10.colors  # type: ignore[attr-defined]
_AGENT_COLORS = list(_TAB10) + [
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
]

# Colorblind-friendly palette for task types (up to 12)
_TASK_TYPE_COLORS = [
    (0.85, 0.15, 0.15),    # 0: red
    (0.15, 0.55, 0.85),    # 1: blue
    (0.20, 0.70, 0.20),    # 2: green
    (0.80, 0.60, 0.10),    # 3: gold
    (0.60, 0.30, 0.70),    # 4: purple
    (0.95, 0.50, 0.15),    # 5: orange
    (0.40, 0.75, 0.75),    # 6: teal
    (0.85, 0.45, 0.65),    # 7: pink
    (0.55, 0.55, 0.55),    # 8: gray
    (0.65, 0.85, 0.30),    # 9: lime
    (0.30, 0.30, 0.70),    # 10: navy
    (0.85, 0.75, 0.55),    # 11: tan
]

_LEGACY_TASK_COLOR = (0.85, 0.15, 0.15)  # single red for n_task_types=1

_EMPTY_BG = (0.95, 0.95, 0.95)
_ACTOR_RING_COLOR = (0.85, 0.65, 0.0)
_GRID_LINE_COLOR = (0.3, 0.3, 0.3)
_CORRECT_PICK_COLOR = (0.1, 0.7, 0.1)
_WRONG_PICK_COLOR = (0.8, 0.1, 0.1)

_ACTION_ARROWS: dict[int, str] = {
    0: "\u2191",  # UP
    1: "\u2193",  # DOWN
    2: "\u2190",  # LEFT
    3: "\u2192",  # RIGHT
    4: "\u00b7",  # STAY
    5: "\u2605",  # PICK
}

_CELL_SIZE = 1.0


def _task_color(task_type: int, n_task_types: int) -> tuple[float, ...]:
    if n_task_types <= 1:
        return _LEGACY_TASK_COLOR
    return _TASK_TYPE_COLORS[task_type % len(_TASK_TYPE_COLORS)]


def _task_bg_tint(task_type: int, n_task_types: int) -> tuple[float, float, float]:
    if n_task_types <= 1:
        return (0.93, 0.97, 0.93)
    tc = _task_color(task_type, n_task_types)
    # Light tint: blend toward white
    return (0.9 + 0.1 * tc[0], 0.9 + 0.1 * tc[1], 0.9 + 0.1 * tc[2])


def _agent_color(idx: int) -> tuple[float, ...]:
    return _AGENT_COLORS[idx % len(_AGENT_COLORS)]


# ---------------------------------------------------------------------------
# Draw one grid
# ---------------------------------------------------------------------------
def _draw_grid(
    ax: plt.Axes,
    state: State,
    height: int,
    width: int,
    actor: int | None = None,
    title: str = "",
    n_task_types: int = 1,
    task_assignments: tuple[tuple[int, ...], ...] | None = None,
    picked_cell: tuple[int, int] | None = None,
    picked_correct: bool | None = None,
) -> None:
    """Draw the orchard grid onto a matplotlib Axes."""
    # Build task map: (row, col) -> list of (type,)
    task_map: dict[tuple[int, int], list[int]] = {}
    for i, tp in enumerate(state.task_positions):
        key = (tp.row, tp.col)
        if state.task_types is not None:
            task_map.setdefault(key, []).append(state.task_types[i])
        else:
            task_map.setdefault(key, []).append(0)

    for r in range(height):
        for c in range(width):
            key = (r, c)

            # Cell background — no tint for tasks (choice mode can stack)
            bg = _EMPTY_BG

            # Pick highlight border
            edge_color = _GRID_LINE_COLOR
            edge_width = 1.5
            if picked_cell == key and picked_correct is not None:
                edge_color = _CORRECT_PICK_COLOR if picked_correct else _WRONG_PICK_COLOR
                edge_width = 3.5

            rect = mpatches.FancyBboxPatch(
                (c * _CELL_SIZE, (height - 1 - r) * _CELL_SIZE),
                _CELL_SIZE,
                _CELL_SIZE,
                boxstyle="round,pad=0.02",
                facecolor=bg,
                edgecolor=edge_color,
                linewidth=edge_width,
            )
            ax.add_patch(rect)

    # Tasks (small colored squares in top-right corner)
    for (r, c), types_list in task_map.items():
        for ti, tau in enumerate(types_list):
            tc = _task_color(tau, n_task_types)
            # Offset multiple types to avoid overlap
            sq_size = _CELL_SIZE * 0.18
            offset_x = 0.80 - ti * 0.22 if len(types_list) > 1 else 0.80
            sx = c * _CELL_SIZE + _CELL_SIZE * offset_x - sq_size / 2
            sy = (height - 1 - r) * _CELL_SIZE + _CELL_SIZE * 0.80 - sq_size / 2
            task_sq = mpatches.FancyBboxPatch(
                (sx, sy),
                sq_size, sq_size,
                boxstyle="round,pad=0.01",
                facecolor=tc,
                edgecolor=tuple(max(0, v - 0.3) for v in tc),
                linewidth=1.0,
                zorder=5,
            )
            ax.add_patch(task_sq)
            # Type label inside square
            if n_task_types > 1:
                ax.text(
                    sx + sq_size / 2, sy + sq_size / 2, str(tau),
                    fontsize=6, fontweight="bold", color="white",
                    ha="center", va="center", zorder=6,
                )

    # Agents
    cell_agents: dict[tuple[int, int], list[int]] = {}
    for i, pos in enumerate(state.agent_positions):
        key = (pos.row, pos.col)
        cell_agents.setdefault(key, []).append(i)

    for (r, c), agents in cell_agents.items():
        cell_cx = c * _CELL_SIZE + _CELL_SIZE * 0.5
        cell_cy = (height - 1 - r) * _CELL_SIZE + _CELL_SIZE * 0.45

        if len(agents) <= 3:
            _draw_individual_agents(ax, agents, cell_cx, cell_cy, actor)
        else:
            _draw_collapsed_agents(ax, agents, cell_cx, cell_cy, actor)

    # Row/column labels
    for c in range(width):
        ax.text(
            c * _CELL_SIZE + _CELL_SIZE * 0.5,
            height * _CELL_SIZE + _CELL_SIZE * 0.15,
            str(c),
            ha="center", va="bottom", fontsize=9, color=(0.4, 0.4, 0.4),
        )
    for r in range(height):
        ax.text(
            -_CELL_SIZE * 0.15,
            (height - 1 - r) * _CELL_SIZE + _CELL_SIZE * 0.5,
            str(r),
            ha="right", va="center", fontsize=9, color=(0.4, 0.4, 0.4),
        )

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    ax.set_xlim(-_CELL_SIZE * 0.3, width * _CELL_SIZE + _CELL_SIZE * 0.1)
    ax.set_ylim(-_CELL_SIZE * 0.1, height * _CELL_SIZE + _CELL_SIZE * 0.35)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_individual_agents(
    ax: plt.Axes,
    agents: list[int],
    cx: float,
    cy: float,
    actor: int | None,
) -> None:
    n = len(agents)
    radius = _CELL_SIZE * 0.18 if n == 1 else _CELL_SIZE * 0.14

    if n == 1:
        offsets = [(0.0, 0.0)]
    elif n == 2:
        spread = _CELL_SIZE * 0.18
        offsets = [(-spread, 0.0), (spread, 0.0)]
    else:
        spread = _CELL_SIZE * 0.18
        offsets = [(-spread, -_CELL_SIZE * 0.08), (spread, -_CELL_SIZE * 0.08), (0.0, _CELL_SIZE * 0.12)]

    for (dx, dy), aidx in zip(offsets, agents):
        x, y = cx + dx, cy + dy
        color = _agent_color(aidx)
        is_actor = (actor is not None and aidx == actor)

        if is_actor:
            ring = mpatches.Circle(
                (x, y),
                radius=radius + _CELL_SIZE * 0.04,
                facecolor="none",
                edgecolor=_ACTOR_RING_COLOR,
                linewidth=3.0,
                zorder=9,
            )
            ax.add_patch(ring)

        circle = mpatches.Circle(
            (x, y),
            radius=radius,
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
            zorder=10,
        )
        ax.add_patch(circle)

        ax.text(
            x, y, str(aidx),
            ha="center", va="center",
            fontsize=9 if n <= 2 else 7,
            fontweight="bold", color="white", zorder=11,
        )


def _draw_collapsed_agents(
    ax: plt.Axes,
    agents: list[int],
    cx: float,
    cy: float,
    actor: int | None,
) -> None:
    radius = _CELL_SIZE * 0.22
    actor_here = actor if (actor is not None and actor in agents) else None
    color = _agent_color(actor_here) if actor_here is not None else (0.5, 0.5, 0.5)

    if actor_here is not None:
        ring = mpatches.Circle(
            (cx, cy),
            radius=radius + _CELL_SIZE * 0.04,
            facecolor="none",
            edgecolor=_ACTOR_RING_COLOR,
            linewidth=3.0,
            zorder=9,
        )
        ax.add_patch(ring)

    circle = mpatches.Circle(
        (cx, cy),
        radius=radius,
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
        zorder=10,
    )
    ax.add_patch(circle)

    ax.text(
        cx, cy, f"\u00d7{len(agents)}",
        ha="center", va="center",
        fontsize=9, fontweight="bold", color="white", zorder=11,
    )


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
def _draw_legend(
    ax: plt.Axes,
    n_task_types: int,
    n_agents: int,
    task_assignments: tuple[tuple[int, ...], ...] | None = None,
) -> None:
    """Draw task type and agent assignment legend below the grid."""
    if n_task_types <= 1:
        return

    handles = []
    # Task type colors
    for tau in range(n_task_types):
        tc = _task_color(tau, n_task_types)
        handles.append(mpatches.Patch(facecolor=tc, edgecolor='gray', label=f'Type {tau}'))

    # Agent assignments
    if task_assignments is not None:
        for i in range(min(n_agents, len(task_assignments))):
            ac = _agent_color(i)
            g = task_assignments[i]
            handles.append(mpatches.Patch(facecolor=ac, edgecolor='white',
                                          label=f'A{i}\u2192{set(g)}'))

    ax.legend(handles=handles, loc='upper left', fontsize=7, ncol=min(6, len(handles)),
              framealpha=0.8, handlelength=1.2, handletextpad=0.4, columnspacing=0.8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def render_state_png(
    state: State,
    height: int,
    width: int,
    actor: int | None = None,
    title: str = "",
    dpi: int = 120,
    n_task_types: int = 1,
    task_assignments: tuple[tuple[int, ...], ...] | None = None,
    picked_cell: tuple[int, int] | None = None,
    picked_correct: bool | None = None,
) -> bytes:
    """Render a single grid state to PNG bytes."""
    fig_w = max(width * 1.3, 2.5)
    fig_h = max(height * 1.3, 2.5)
    if n_task_types > 1:
        fig_h += 0.6  # space for legend
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    _draw_grid(ax, state, height, width, actor=actor, title=title,
               n_task_types=n_task_types, task_assignments=task_assignments,
               picked_cell=picked_cell, picked_correct=picked_correct)
    if n_task_types > 1:
        _draw_legend(ax, n_task_types, len(state.agent_positions), task_assignments)
    fig.tight_layout(pad=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_frame_png(
    state: State,
    state_after: State | None,
    height: int,
    width: int,
    state_index: int = 0,
    actor: int | None = None,
    action: Action | None = None,
    rewards: tuple[float, ...] | None = None,
    discount: float = 1.0,
    show_after_state: bool = False,
    dpi: int = 120,
    n_task_types: int = 1,
    task_assignments: tuple[tuple[int, ...], ...] | None = None,
    picked_cell: tuple[int, int] | None = None,
    picked_correct: bool | None = None,
) -> bytes:
    """Render one or two grids (s_t and optionally s_{t+1}) to PNG bytes."""
    t = state_index

    if not show_after_state or state_after is None:
        return render_state_png(
            state, height, width, actor=actor,
            title=f"$s_{{{t}}}$", dpi=dpi,
            n_task_types=n_task_types, task_assignments=task_assignments,
            picked_cell=picked_cell, picked_correct=picked_correct,
        )

    # Side-by-side: s_t | s_t^a
    fig_w = max(width * 1.3, 2.5) * 2 + 0.5
    fig_h = max(height * 1.3, 2.5)
    if n_task_types > 1:
        fig_h += 0.6

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=dpi)

    _draw_grid(ax_left, state, height, width, actor=actor, title=f"$s_{{{t}}}$",
               n_task_types=n_task_types, task_assignments=task_assignments)
    _draw_grid(ax_right, state_after, height, width, actor=actor, title=f"$s_{{{t}}}^a$",
               n_task_types=n_task_types, task_assignments=task_assignments,
               picked_cell=picked_cell, picked_correct=picked_correct)

    if n_task_types > 1:
        _draw_legend(ax_left, n_task_types, len(state.agent_positions), task_assignments)

    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
