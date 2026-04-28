"""Grid renderer: PNG (matplotlib) and SVG (inline, clickable).

SVG renderer is the primary path for the HTML viewer. It supports:
  - Task collapse: >3 tasks per cell → count badge (clickable popup)
  - Agent collapse: >3 agents per cell → count badge (clickable popup)
  - Type-colored agents: if every agent has exactly one task assignment,
    agents are colored by their task type instead of by agent index.

PNG renderer (render_frame_png) is kept for backward compatibility.
"""

from __future__ import annotations

import html as _html
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from orchard.datatypes import State
from orchard.enums import Action

# ---------------------------------------------------------------------------
# Shared color palettes (RGB for matplotlib, hex for SVG)
# ---------------------------------------------------------------------------
_TAB10 = plt.cm.tab10.colors  # type: ignore[attr-defined]
_AGENT_COLORS_RGB = list(_TAB10) + [(0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]

_TASK_TYPE_COLORS_RGB = [
    (0.85, 0.15, 0.15), (0.15, 0.55, 0.85), (0.20, 0.70, 0.20),
    (0.80, 0.60, 0.10), (0.60, 0.30, 0.70), (0.95, 0.50, 0.15),
    (0.40, 0.75, 0.75), (0.85, 0.45, 0.65), (0.55, 0.55, 0.55),
    (0.65, 0.85, 0.30), (0.30, 0.30, 0.70), (0.85, 0.75, 0.55),
]

TASK_TYPE_HEX: list[str] = [
    "#d92525", "#267fd9", "#33b233", "#cc9900",
    "#9952b3", "#f27f24", "#66c0c0", "#d973a4",
    "#8c8c8c", "#a6d94d", "#4d4db3", "#d9c48c",
]

AGENT_HEX: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#800080", "#008080",
]

# ---------------------------------------------------------------------------
# SVG layout constants
# ---------------------------------------------------------------------------
CELL = 60          # pixels per grid cell
LABEL_PAD = 22     # pixels for row/col number labels
TITLE_H = 20       # pixels for the grid title bar
TASK_COLLAPSE = 3  # collapse to badge when more than this many tasks per cell
AGENT_COLLAPSE = 3 # collapse to badge when more than this many agents per cell

_ACTOR_RING_HEX   = "#d9a600"
_CORRECT_BORD_HEX = "#1ab21a"
_WRONG_BORD_HEX   = "#cc1a1a"
_EMPTY_BG_HEX     = "#f2f2f2"
_GRID_STROKE_HEX  = "#4d4d4d"

# ---------------------------------------------------------------------------
# Matplotlib constants (legacy PNG path)
# ---------------------------------------------------------------------------
_LEGACY_TASK_COLOR  = (0.85, 0.15, 0.15)
_EMPTY_BG           = (0.95, 0.95, 0.95)
_ACTOR_RING_COLOR   = (0.85, 0.65, 0.0)
_GRID_LINE_COLOR    = (0.3, 0.3, 0.3)
_CORRECT_PICK_COLOR = (0.1, 0.7, 0.1)
_WRONG_PICK_COLOR   = (0.8, 0.1, 0.1)
_CELL_SIZE          = 1.0

_ACTION_ARROWS: dict[int, str] = {
    0: "↑", 1: "↓", 2: "←",
    3: "→", 4: "·", 5: "★",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def use_type_colors_for_agents(
    task_assignments: tuple[tuple[int, ...], ...] | None,
    n_task_types: int,
) -> bool:
    """True when every agent has exactly one task-type assignment."""
    if task_assignments is None or n_task_types <= 1:
        return False
    return all(len(g) == 1 for g in task_assignments)


def _task_hex(task_type: int, n_task_types: int) -> str:
    if n_task_types <= 1:
        return TASK_TYPE_HEX[0]
    return TASK_TYPE_HEX[task_type % len(TASK_TYPE_HEX)]


def _agent_hex(
    idx: int,
    task_assignments: tuple[tuple[int, ...], ...] | None,
    n_task_types: int,
    use_type_colors: bool,
) -> str:
    if (use_type_colors and task_assignments is not None
            and idx < len(task_assignments)
            and len(task_assignments[idx]) == 1):
        return _task_hex(task_assignments[idx][0], n_task_types)
    return AGENT_HEX[idx % len(AGENT_HEX)]


def _darken_hex(hex_color: str, amount: float = 0.35) -> str:
    h = hex_color.lstrip("#")
    r = max(0, int(int(h[0:2], 16) * (1 - amount)))
    g = max(0, int(int(h[2:4], 16) * (1 - amount)))
    b = max(0, int(int(h[4:6], 16) * (1 - amount)))
    return f"#{r:02x}{g:02x}{b:02x}"


def _attr_json(data: object) -> str:
    """JSON-encode data and HTML-escape it for safe use in an SVG attribute."""
    return _html.escape(json.dumps(data), quote=True)


# ---------------------------------------------------------------------------
# SVG grid element builder
# ---------------------------------------------------------------------------
def _render_grid_svg_elements(
    state: State,
    height: int,
    width: int,
    actor: int | None,
    n_task_types: int,
    task_assignments: tuple[tuple[int, ...], ...] | None,
    picked_cell: tuple[int, int] | None,
    picked_correct: bool | None,
    use_type_colors: bool,
    x0: float,
    y0: float,
) -> list[str]:
    """Return SVG element strings for a single grid (no outer <svg> tag)."""
    parts: list[str] = []

    # Build maps
    task_map: dict[tuple[int, int], list[int]] = {}
    for i, tp in enumerate(state.task_positions):
        key = (tp.row, tp.col)
        t = state.task_types[i] if state.task_types is not None else 0
        task_map.setdefault(key, []).append(t)

    agent_map: dict[tuple[int, int], list[int]] = {}
    for i, pos in enumerate(state.agent_positions):
        agent_map.setdefault((pos.row, pos.col), []).append(i)

    # --- Cells ---
    for r in range(height):
        for c in range(width):
            cx = x0 + c * CELL
            cy = y0 + r * CELL
            if picked_cell == (r, c) and picked_correct is not None:
                stroke = _CORRECT_BORD_HEX if picked_correct else _WRONG_BORD_HEX
                sw = 3.5
            else:
                stroke = _GRID_STROKE_HEX
                sw = 1.5
            parts.append(
                f'<rect x="{cx:.0f}" y="{cy:.0f}" width="{CELL}" height="{CELL}" '
                f'rx="3" fill="{_EMPTY_BG_HEX}" stroke="{stroke}" stroke-width="{sw}"/>'
            )

    # --- Tasks ---
    sq = CELL * 0.22
    for (r, c), types_list in task_map.items():
        cx = x0 + c * CELL
        cy = y0 + r * CELL

        if len(types_list) <= TASK_COLLAPSE:
            for ti, tau in enumerate(types_list):
                tc = _task_hex(tau, n_task_types)
                dark = _darken_hex(tc)
                sq_x = cx + CELL - 4 - (ti + 1) * (sq + 2) + 2
                sq_y = cy + 4
                popup = _attr_json({"types": [tau]})
                parts.append(
                    f'<g data-popup-type="tasks" data-popup-data="{popup}">'
                    f'<rect x="{sq_x:.1f}" y="{sq_y:.1f}" width="{sq:.1f}" height="{sq:.1f}" '
                    f'rx="2" fill="{tc}" stroke="{dark}" stroke-width="1"/>'
                )
                if n_task_types > 1:
                    parts.append(
                        f'<text x="{sq_x + sq/2:.1f}" y="{sq_y + sq/2:.1f}" '
                        f'text-anchor="middle" dominant-baseline="central" '
                        f'font-size="8" font-weight="bold" fill="white" pointer-events="none">'
                        f'{tau}</text>'
                    )
                parts.append("</g>")
        else:
            # Count badge
            br = sq * 0.9
            bx = cx + CELL - br - 4
            by = cy + br + 4
            popup = _attr_json({"types": types_list})
            parts.append(
                f'<g data-popup-type="tasks" data-popup-data="{popup}" style="cursor:pointer">'
                f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="{br:.1f}" '
                f'fill="#555" stroke="white" stroke-width="1"/>'
                f'<text x="{bx:.1f}" y="{by:.1f}" text-anchor="middle" dominant-baseline="central" '
                f'font-size="9" font-weight="bold" fill="white" pointer-events="none">'
                f'{len(types_list)}</text>'
                f'</g>'
            )

    # --- Agents ---
    for (r, c), agents in agent_map.items():
        cx = x0 + c * CELL
        cy = y0 + r * CELL
        ccx = cx + CELL * 0.5
        ccy = cy + CELL * 0.55

        if len(agents) <= AGENT_COLLAPSE:
            n = len(agents)
            radius = CELL * 0.18 if n == 1 else CELL * 0.14
            if n == 1:
                offsets = [(0.0, 0.0)]
            elif n == 2:
                sp = CELL * 0.18
                offsets = [(-sp, 0.0), (sp, 0.0)]
            else:
                sp = CELL * 0.18
                offsets = [(-sp, CELL * 0.05), (sp, CELL * 0.05), (0.0, -CELL * 0.12)]

            for (dx, dy), aidx in zip(offsets, agents):
                ax = ccx + dx
                ay = ccy + dy
                color = _agent_hex(aidx, task_assignments, n_task_types, use_type_colors)
                is_actor = (actor is not None and aidx == actor)
                popup = _attr_json({"agents": [aidx]})
                fs = 9 if n <= 2 else 7

                parts.append(
                    f'<g data-popup-type="agents" data-popup-data="{popup}" style="cursor:pointer">'
                )
                if is_actor:
                    parts.append(
                        f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="{radius + CELL*0.04:.1f}" '
                        f'fill="none" stroke="{_ACTOR_RING_HEX}" stroke-width="3"/>'
                    )
                parts.append(
                    f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="{radius:.1f}" '
                    f'fill="{color}" stroke="white" stroke-width="1.5"/>'
                    f'<text x="{ax:.1f}" y="{ay:.1f}" text-anchor="middle" dominant-baseline="central" '
                    f'font-size="{fs}" font-weight="bold" fill="white" pointer-events="none">'
                    f'{aidx}</text>'
                    f'</g>'
                )
        else:
            cr = CELL * 0.22
            actor_here = actor if (actor is not None and actor in agents) else None
            color = (
                _agent_hex(actor_here, task_assignments, n_task_types, use_type_colors)
                if actor_here is not None else "#888888"
            )
            popup = _attr_json({"agents": agents})
            parts.append(
                f'<g data-popup-type="agents" data-popup-data="{popup}" style="cursor:pointer">'
            )
            if actor_here is not None:
                parts.append(
                    f'<circle cx="{ccx:.1f}" cy="{ccy:.1f}" r="{cr + CELL*0.04:.1f}" '
                    f'fill="none" stroke="{_ACTOR_RING_HEX}" stroke-width="3"/>'
                )
            parts.append(
                f'<circle cx="{ccx:.1f}" cy="{ccy:.1f}" r="{cr:.1f}" '
                f'fill="{color}" stroke="white" stroke-width="1.5"/>'
                f'<text x="{ccx:.1f}" y="{ccy:.1f}" text-anchor="middle" dominant-baseline="central" '
                f'font-size="9" font-weight="bold" fill="white" pointer-events="none">'
                f'×{len(agents)}</text>'
                f'</g>'
            )

    # --- Row/col labels ---
    for r in range(height):
        parts.append(
            f'<text x="{x0 - 4:.0f}" y="{y0 + r*CELL + CELL//2:.0f}" '
            f'text-anchor="end" dominant-baseline="central" font-size="10" fill="#666">{r}</text>'
        )
    for c in range(width):
        parts.append(
            f'<text x="{x0 + c*CELL + CELL//2:.0f}" y="{y0 - 4:.0f}" '
            f'text-anchor="middle" font-size="10" fill="#666">{c}</text>'
        )

    return parts


# ---------------------------------------------------------------------------
# Public SVG API
# ---------------------------------------------------------------------------
def render_frame_svg(
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
    n_task_types: int = 1,
    task_assignments: tuple[tuple[int, ...], ...] | None = None,
    picked_cell: tuple[int, int] | None = None,
    picked_correct: bool | None = None,
) -> str:
    """Render one or two grids as an inline SVG string."""
    use_type_colors = use_type_colors_for_agents(task_assignments, n_task_types)
    t = state_index

    grid_w = LABEL_PAD + width * CELL
    grid_h = LABEL_PAD + height * CELL

    if not show_after_state or state_after is None:
        total_w = grid_w + 8
        total_h = grid_h + TITLE_H + 8
        x0 = LABEL_PAD + 4
        y0 = float(TITLE_H + 4)

        parts = [
            f'<svg width="{total_w}" height="{total_h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="font-family:sans-serif;background:#f8f8f8;border-radius:4px;display:block">'
        ]
        parts.append(
            f'<text x="{total_w // 2}" y="{TITLE_H - 5}" text-anchor="middle" '
            f'font-size="13" font-weight="bold" fill="#333">s[{t}]</text>'
        )
        parts.extend(_render_grid_svg_elements(
            state, height, width, actor, n_task_types, task_assignments,
            picked_cell, picked_correct, use_type_colors, x0, y0,
        ))
        parts.append("</svg>")
        return "\n".join(parts)

    # Side-by-side
    gap = 20
    total_w = 2 * grid_w + gap + 16
    total_h = grid_h + TITLE_H + 8
    x0_l = LABEL_PAD + 4
    x0_r = float(LABEL_PAD + 4 + grid_w + gap)
    y0 = float(TITLE_H + 4)

    mid_l = x0_l + (grid_w - LABEL_PAD) // 2
    mid_r = x0_r + (grid_w - LABEL_PAD) // 2

    parts = [
        f'<svg width="{total_w}" height="{total_h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family:sans-serif;background:#f8f8f8;border-radius:4px;display:block">'
    ]
    parts.append(
        f'<text x="{mid_l:.0f}" y="{TITLE_H - 5}" text-anchor="middle" '
        f'font-size="13" font-weight="bold" fill="#333">s[{t}]</text>'
    )
    parts.append(
        f'<text x="{mid_r:.0f}" y="{TITLE_H - 5}" text-anchor="middle" '
        f'font-size="13" font-weight="bold" fill="#555">s[{t}] after action</text>'
    )
    parts.extend(_render_grid_svg_elements(
        state, height, width, actor, n_task_types, task_assignments,
        None, None, use_type_colors, x0_l, y0,
    ))
    parts.extend(_render_grid_svg_elements(
        state_after, height, width, actor, n_task_types, task_assignments,
        picked_cell, picked_correct, use_type_colors, x0_r, y0,
    ))
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Legacy matplotlib PNG renderer (unchanged, kept for backward compat)
# ---------------------------------------------------------------------------
def _task_color(task_type: int, n_task_types: int) -> tuple[float, ...]:
    if n_task_types <= 1:
        return _LEGACY_TASK_COLOR
    return _TASK_TYPE_COLORS_RGB[task_type % len(_TASK_TYPE_COLORS_RGB)]


def _task_bg_tint(task_type: int, n_task_types: int) -> tuple[float, float, float]:
    if n_task_types <= 1:
        return (0.93, 0.97, 0.93)
    tc = _task_color(task_type, n_task_types)
    return (0.9 + 0.1 * tc[0], 0.9 + 0.1 * tc[1], 0.9 + 0.1 * tc[2])


def _agent_color(idx: int) -> tuple[float, ...]:
    return _AGENT_COLORS_RGB[idx % len(_AGENT_COLORS_RGB)]


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
            bg = _EMPTY_BG
            edge_color = _GRID_LINE_COLOR
            edge_width = 1.5
            if picked_cell == key and picked_correct is not None:
                edge_color = _CORRECT_PICK_COLOR if picked_correct else _WRONG_PICK_COLOR
                edge_width = 3.5
            rect = mpatches.FancyBboxPatch(
                (c * _CELL_SIZE, (height - 1 - r) * _CELL_SIZE),
                _CELL_SIZE, _CELL_SIZE,
                boxstyle="round,pad=0.02",
                facecolor=bg, edgecolor=edge_color, linewidth=edge_width,
            )
            ax.add_patch(rect)

    for (r, c), types_list in task_map.items():
        for ti, tau in enumerate(types_list):
            tc = _task_color(tau, n_task_types)
            sq_size = _CELL_SIZE * 0.18
            offset_x = 0.80 - ti * 0.22 if len(types_list) > 1 else 0.80
            sx = c * _CELL_SIZE + _CELL_SIZE * offset_x - sq_size / 2
            sy = (height - 1 - r) * _CELL_SIZE + _CELL_SIZE * 0.80 - sq_size / 2
            task_sq = mpatches.FancyBboxPatch(
                (sx, sy), sq_size, sq_size,
                boxstyle="round,pad=0.01",
                facecolor=tc,
                edgecolor=tuple(max(0, v - 0.3) for v in tc),
                linewidth=1.0, zorder=5,
            )
            ax.add_patch(task_sq)
            if n_task_types > 1:
                ax.text(
                    sx + sq_size / 2, sy + sq_size / 2, str(tau),
                    fontsize=6, fontweight="bold", color="white",
                    ha="center", va="center", zorder=6,
                )

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

    for c in range(width):
        ax.text(
            c * _CELL_SIZE + _CELL_SIZE * 0.5, height * _CELL_SIZE + _CELL_SIZE * 0.15,
            str(c), ha="center", va="bottom", fontsize=9, color=(0.4, 0.4, 0.4),
        )
    for r in range(height):
        ax.text(
            -_CELL_SIZE * 0.15, (height - 1 - r) * _CELL_SIZE + _CELL_SIZE * 0.5,
            str(r), ha="right", va="center", fontsize=9, color=(0.4, 0.4, 0.4),
        )

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    ax.set_xlim(-_CELL_SIZE * 0.3, width * _CELL_SIZE + _CELL_SIZE * 0.1)
    ax.set_ylim(-_CELL_SIZE * 0.1, height * _CELL_SIZE + _CELL_SIZE * 0.35)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_individual_agents(ax, agents, cx, cy, actor):
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
            ax.add_patch(mpatches.Circle(
                (x, y), radius=radius + _CELL_SIZE * 0.04,
                facecolor="none", edgecolor=_ACTOR_RING_COLOR, linewidth=3.0, zorder=9,
            ))
        ax.add_patch(mpatches.Circle(
            (x, y), radius=radius, facecolor=color, edgecolor="white", linewidth=1.5, zorder=10,
        ))
        ax.text(x, y, str(aidx), ha="center", va="center",
                fontsize=9 if n <= 2 else 7, fontweight="bold", color="white", zorder=11)


def _draw_collapsed_agents(ax, agents, cx, cy, actor):
    radius = _CELL_SIZE * 0.22
    actor_here = actor if (actor is not None and actor in agents) else None
    color = _agent_color(actor_here) if actor_here is not None else (0.5, 0.5, 0.5)
    if actor_here is not None:
        ax.add_patch(mpatches.Circle(
            (cx, cy), radius=radius + _CELL_SIZE * 0.04,
            facecolor="none", edgecolor=_ACTOR_RING_COLOR, linewidth=3.0, zorder=9,
        ))
    ax.add_patch(mpatches.Circle(
        (cx, cy), radius=radius, facecolor=color, edgecolor="white", linewidth=1.5, zorder=10,
    ))
    ax.text(cx, cy, f"×{len(agents)}", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=11)


def _draw_legend(ax, n_task_types, n_agents, task_assignments=None):
    if n_task_types <= 1:
        return
    handles = []
    for tau in range(n_task_types):
        tc = _task_color(tau, n_task_types)
        handles.append(mpatches.Patch(facecolor=tc, edgecolor="gray", label=f"Type {tau}"))
    if task_assignments is not None:
        for i in range(min(n_agents, len(task_assignments))):
            ac = _agent_color(i)
            g = task_assignments[i]
            handles.append(mpatches.Patch(facecolor=ac, edgecolor="white",
                                          label=f"A{i}→{set(g)}"))
    ax.legend(handles=handles, loc="upper left", fontsize=7, ncol=min(6, len(handles)),
              framealpha=0.8, handlelength=1.2, handletextpad=0.4, columnspacing=0.8)


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
    fig_w = max(width * 1.3, 2.5)
    fig_h = max(height * 1.3, 2.5)
    if n_task_types > 1:
        fig_h += 0.6
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
    t = state_index
    if not show_after_state or state_after is None:
        return render_state_png(
            state, height, width, actor=actor, title=f"$s_{{{t}}}$", dpi=dpi,
            n_task_types=n_task_types, task_assignments=task_assignments,
            picked_cell=picked_cell, picked_correct=picked_correct,
        )
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
