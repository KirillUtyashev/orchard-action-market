"""Matplotlib grid renderer: Frame → PNG bytes.

Each grid cell is a square. Agents are colored circles with ID labels.
Apples are small red circles in the top-right corner. Actor gets a gold ring.
Threshold: 1-3 agents rendered individually, 4+ collapsed to count badge.
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

_APPLE_COLOR = (0.85, 0.15, 0.15)
_APPLE_BG_TINT = (0.93, 0.97, 0.93)
_EMPTY_BG = (0.95, 0.95, 0.95)
_ACTOR_RING_COLOR = (0.85, 0.65, 0.0)
_GRID_LINE_COLOR = (0.3, 0.3, 0.3)

_ACTION_ARROWS: dict[Action, str] = {
    Action.UP: "\u2191",
    Action.DOWN: "\u2193",
    Action.LEFT: "\u2190",
    Action.RIGHT: "\u2192",
    Action.STAY: "\u00b7",
    Action.PICK: "\u2605",
}

_CELL_SIZE = 1.0


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
) -> None:
    """Draw the orchard grid onto a matplotlib Axes."""
    apple_set: dict[tuple[int, int], int] = {}
    for ap in state.apple_positions:
        key = (ap.row, ap.col)
        apple_set[key] = apple_set.get(key, 0) + 1

    for r in range(height):
        for c in range(width):
            has_apple = (r, c) in apple_set
            bg = _APPLE_BG_TINT if has_apple else _EMPTY_BG
            rect = mpatches.FancyBboxPatch(
                (c * _CELL_SIZE, (height - 1 - r) * _CELL_SIZE),
                _CELL_SIZE,
                _CELL_SIZE,
                boxstyle="round,pad=0.02",
                facecolor=bg,
                edgecolor=_GRID_LINE_COLOR,
                linewidth=1.5,
            )
            ax.add_patch(rect)

    # Apples (top-right corner)
    for (r, c), count in apple_set.items():
        cx = c * _CELL_SIZE + _CELL_SIZE * 0.82
        cy = (height - 1 - r) * _CELL_SIZE + _CELL_SIZE * 0.82
        apple_circle = mpatches.Circle(
            (cx, cy),
            radius=_CELL_SIZE * 0.12,
            facecolor=_APPLE_COLOR,
            edgecolor=(0.6, 0.0, 0.0),
            linewidth=1.0,
            zorder=5,
        )
        ax.add_patch(apple_circle)
        if count > 1:
            ax.text(
                cx + _CELL_SIZE * 0.08,
                cy + _CELL_SIZE * 0.08,
                f"\u00d7{count}",
                fontsize=7,
                fontweight="bold",
                color="white",
                ha="left",
                va="bottom",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.1", facecolor=_APPLE_COLOR, alpha=0.9),
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
# Public API
# ---------------------------------------------------------------------------
def render_state_png(
    state: State,
    height: int,
    width: int,
    actor: int | None = None,
    title: str = "",
    dpi: int = 120,
) -> bytes:
    """Render a single grid state to PNG bytes."""
    fig_w = max(width * 1.3, 2.5)
    fig_h = max(height * 1.3, 2.5)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    _draw_grid(ax, state, height, width, actor=actor, title=title)
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
) -> bytes:
    """Render one or two grids (s_t and optionally s_{t+1}) to PNG bytes.

    Uses math notation: s_t, r_{t+1}, γ_{t+1}.
    """
    t = state_index

    if not show_after_state or state_after is None:
        return render_state_png(
            state, height, width, actor=actor,
            title=f"$s_{{{t}}}$",
            dpi=dpi,
        )

    # Side-by-side: s_t | s_t^a
    fig_w = max(width * 1.3, 2.5) * 2 + 0.5
    fig_h = max(height * 1.3, 2.5)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=dpi)

    _draw_grid(ax_left, state, height, width, actor=actor, title=f"$s_{{{t}}}$")
    _draw_grid(ax_right, state_after, height, width, actor=actor, title=f"$s_{{{t}}}^a$")

    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
