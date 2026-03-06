"""Build a self-contained HTML report from comparison results."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from orchard.compare_values.compare import StateComparison
from orchard.compare_values.loader import LoadedRun
from orchard.datatypes import State
from orchard.viz.renderer import render_frame_png


# ---------------------------------------------------------------------------
# Grid rendering helpers
# ---------------------------------------------------------------------------

def _render_state_png(state: State, height: int, width: int, dpi: int = 100) -> bytes:
    """Render a state as a PNG using the viz renderer."""
    return render_frame_png(
        state=state,
        state_after=None,
        height=height,
        width=width,
        state_index=0,
        actor=state.actor,
        action=None,
        rewards=None,
        show_after_state=False,
        dpi=dpi,
    )


def _png_to_data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def _fig_to_data_uri(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return _png_to_data_uri(buf.read())


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def _build_scatter_plot(
    comparisons: list[StateComparison],
    label_a: str,
    label_b: str,
) -> str:
    """V_team_A vs V_team_B scatter with diagonal."""
    fig, ax = plt.subplots(figsize=(7, 7))

    xs = [c.team_value_a for c in comparisons]
    ys = [c.team_value_b for c in comparisons]
    abs_diffs = [c.team_abs_diff for c in comparisons]

    sc = ax.scatter(xs, ys, c=abs_diffs, cmap="RdYlGn_r", s=20, alpha=0.8, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="|V_A - V_B|")

    all_vals = xs + ys
    lo = min(all_vals) - 0.1
    hi = max(all_vals) + 0.1
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"V_team — {label_a}")
    ax.set_ylabel(f"V_team — {label_b}")
    ax.set_title("Team Value: A vs B")
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def _build_diff_histogram(comparisons: list[StateComparison]) -> str:
    """Histogram of (V_team_A - V_team_B)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    diffs = [c.team_diff for c in comparisons]
    ax.hist(diffs, bins=min(50, max(10, len(diffs) // 5)), color="#4A90D9", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    mean_diff = np.mean(diffs)
    ax.axvline(mean_diff, color="red", linewidth=1.5, linestyle="-", alpha=0.7)
    ax.set_xlabel("V_team_A - V_team_B")
    ax.set_ylabel("Count")
    ax.set_title(f"Difference Distribution (mean = {mean_diff:.4f})")
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def _build_sorted_diff_plot(comparisons: list[StateComparison]) -> str:
    """States sorted by team_diff."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sorted_comps = sorted(comparisons, key=lambda c: c.team_diff)
    diffs = [c.team_diff for c in sorted_comps]
    colors = ["#D9534F" if d > 0 else "#5CB85C" for d in diffs]
    ax.bar(range(len(diffs)), diffs, color=colors, width=1.0, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("States (sorted by diff)")
    ax.set_ylabel("V_team_A - V_team_B")
    ax.set_title("Per-State Difference (sorted)")

    red_patch = mpatches.Patch(color="#D9534F", label="A > B")
    green_patch = mpatches.Patch(color="#5CB85C", label="B > A")
    ax.legend(handles=[red_patch, green_patch], loc="upper left")
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def _build_per_agent_scatter(
    comparisons: list[StateComparison],
    label_a: str,
    label_b: str,
    n_agents: int,
    both_decentralized: bool,
) -> str | None:
    """Per-agent V_i scatter plots. Only if both runs are decentralized."""
    if not both_decentralized:
        return None

    fig, axes = plt.subplots(1, n_agents, figsize=(6 * n_agents, 5), squeeze=False)

    for i in range(n_agents):
        ax = axes[0, i]
        xs = [c.agent_values_a[i] for c in comparisons]
        ys = [c.agent_values_b[i] for c in comparisons]
        ax.scatter(xs, ys, s=15, alpha=0.6, color=f"C{i}")

        all_v = xs + ys
        lo = min(all_v) - 0.1
        hi = max(all_v) + 0.1
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(label_a)
        ax.set_ylabel(label_b)
        ax.set_title(f"Agent {i}")

    fig.suptitle("Per-Agent Values: A vs B", fontsize=14)
    fig.tight_layout()
    return _fig_to_data_uri(fig)


# ---------------------------------------------------------------------------
# Extreme state renderings
# ---------------------------------------------------------------------------

def _render_extreme_states(
    comparisons: list[StateComparison],
    height: int,
    width: int,
    k: int = 5,
    dpi: int = 100,
) -> tuple[list[tuple[int, str, StateComparison]], list[tuple[int, str, StateComparison]]]:
    """Render top-K worst and best agreement states.

    Returns: (worst_list, best_list) where each entry is
        (state_index, data_uri, comparison).
    """
    sorted_by_abs = sorted(comparisons, key=lambda c: c.team_abs_diff, reverse=True)

    worst = []
    for c in sorted_by_abs[:k]:
        png = _render_state_png(c.state, height, width, dpi)
        worst.append((c.state_index, _png_to_data_uri(png), c))

    best = []
    for c in sorted_by_abs[-k:]:
        png = _render_state_png(c.state, height, width, dpi)
        best.append((c.state_index, _png_to_data_uri(png), c))

    return worst, best


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_report(
    comparisons: list[StateComparison],
    run_a: LoadedRun,
    run_b: LoadedRun,
    output_path: Path,
    dpi: int = 100,
) -> None:
    """Build self-contained HTML comparison report."""
    n_agents = run_a.cfg.env.n_agents
    height = run_a.cfg.env.height
    width = run_a.cfg.env.width
    both_dec = (not run_a.is_centralized) and (not run_b.is_centralized)

    # --- Aggregate stats ---
    diffs = [c.team_diff for c in comparisons]
    abs_diffs = [c.team_abs_diff for c in comparisons]
    mean_diff = float(np.mean(diffs))
    mean_abs_diff = float(np.mean(abs_diffs))
    max_abs_diff = float(np.max(abs_diffs))
    std_diff = float(np.std(diffs))
    mean_mag_a = float(np.mean([abs(c.team_value_a) for c in comparisons]))
    mean_mag_b = float(np.mean([abs(c.team_value_b) for c in comparisons]))
    mean_mag_avg = (mean_mag_a + mean_mag_b) / 2
    pct_error = (mean_abs_diff / mean_mag_avg * 100) if mean_mag_avg > 1e-8 else float("inf")

    # --- Plots ---
    scatter_uri = _build_scatter_plot(comparisons, run_a.label, run_b.label)
    hist_uri = _build_diff_histogram(comparisons)
    sorted_uri = _build_sorted_diff_plot(comparisons)
    agent_scatter_uri = _build_per_agent_scatter(
        comparisons, run_a.label, run_b.label, n_agents, both_dec
    )

    # --- Extreme states ---
    k = min(5, len(comparisons))
    worst, best = _render_extreme_states(comparisons, height, width, k, dpi)

    # --- Build table rows ---
    table_rows = []
    for c in comparisons:
        av_a_str = ", ".join(f"{i}: {v:.4f}" for i, v in sorted(c.agent_values_a.items()))
        av_b_str = ", ".join(f"{i}: {v:.4f}" for i, v in sorted(c.agent_values_b.items()))
        agents_str = str(c.state.agent_positions)
        apples_str = str(c.state.apple_positions)
        table_rows.append(
            f"<tr>"
            f"<td>{c.state_index}</td>"
            f"<td>{c.team_value_a:.4f}</td>"
            f"<td>{c.team_value_b:.4f}</td>"
            f"<td class=\"{'neg' if c.team_diff < 0 else 'pos'}\">{c.team_diff:+.4f}</td>"
            f"<td>{c.team_abs_diff:.4f}</td>"
            f"<td class='detail'>{av_a_str}</td>"
            f"<td class='detail'>{av_b_str}</td>"
            f"<td class='detail'>{agents_str}</td>"
            f"<td class='detail'>{apples_str}</td>"
            f"<td>{c.state.actor}</td>"
            f"</tr>"
        )
    table_html = "\n".join(table_rows)

    # --- Extreme state HTML ---
    def _extreme_html(items: list[tuple[int, str, StateComparison]], title: str) -> str:
        if not items:
            return ""
        cards = []
        for idx, uri, c in items:
            cards.append(
                f'<div class="state-card">'
                f'<img src="{uri}" />'
                f'<div class="card-info">'
                f'<b>State {idx}</b><br>'
                f'V_A = {c.team_value_a:.4f}<br>'
                f'V_B = {c.team_value_b:.4f}<br>'
                f'Diff = {c.team_diff:+.4f}'
                f'</div></div>'
            )
        return f"<h2>{title}</h2><div class='state-gallery'>{''.join(cards)}</div>"

    worst_html = _extreme_html(worst, f"Top {k} Largest Disagreements")
    best_html = _extreme_html(best, f"Top {k} Closest Agreements")

    # --- Per-agent scatter section ---
    agent_section = ""
    if agent_scatter_uri:
        agent_section = f"""
        <h2>Per-Agent Values</h2>
        <img src="{agent_scatter_uri}" class="plot" />
        """

    # --- Assemble HTML ---
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Value Comparison: {run_a.label} vs {run_b.label}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
         background: #f5f5f5; color: #333; padding: 20px; max-width: 1400px; margin: 0 auto; }}
  h1 {{ margin-bottom: 10px; font-size: 1.5em; }}
  h2 {{ margin: 30px 0 15px; font-size: 1.2em; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
  .summary {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .summary table {{ border-collapse: collapse; }}
  .summary td {{ padding: 4px 16px 4px 0; }}
  .summary .label {{ font-weight: bold; color: #666; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px; margin: 15px 0; }}
  .stat-box {{ background: #e8f0fe; border-radius: 6px; padding: 12px; text-align: center; }}
  .stat-box .val {{ font-size: 1.4em; font-weight: bold; color: #1a73e8; }}
  .stat-box .lbl {{ font-size: 0.85em; color: #666; margin-top: 4px; }}
  .plot {{ max-width: 100%; border-radius: 6px; margin: 10px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .state-gallery {{ display: flex; flex-wrap: wrap; gap: 15px; }}
  .state-card {{ background: white; border-radius: 6px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                 text-align: center; }}
  .state-card img {{ max-width: 200px; border-radius: 4px; }}
  .card-info {{ font-size: 0.85em; margin-top: 8px; line-height: 1.5; }}
  /* Table */
  .table-wrap {{ overflow-x: auto; margin: 15px 0; }}
  table.data {{ border-collapse: collapse; width: 100%; background: white;
                border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                font-size: 0.85em; }}
  table.data th {{ background: #f0f0f0; padding: 8px 10px; text-align: left; cursor: pointer;
                   user-select: none; white-space: nowrap; border-bottom: 2px solid #ccc; }}
  table.data th:hover {{ background: #e0e0e0; }}
  table.data td {{ padding: 6px 10px; border-bottom: 1px solid #eee; white-space: nowrap; }}
  table.data tr:hover {{ background: #f8f8f8; }}
  td.pos {{ color: #c0392b; }}
  td.neg {{ color: #27ae60; }}
  td.detail {{ font-size: 0.8em; color: #666; }}
  .sort-arrow {{ font-size: 0.7em; margin-left: 4px; }}
</style>
</head>
<body>

<h1>Value Comparison Report</h1>

<div class="summary">
  <table>
    <tr><td class="label">Run A:</td><td>{run_a.label}</td></tr>
    <tr><td class="label">Run B:</td><td>{run_b.label}</td></tr>
    <tr><td class="label">Env:</td><td>{height}&times;{width}, {n_agents} agents, &gamma;={run_a.cfg.env.gamma}</td></tr>
    <tr><td class="label">TD target:</td><td>{run_a.cfg.train.td_target.name.lower()}</td></tr>
    <tr><td class="label">States compared:</td><td>{len(comparisons)}</td></tr>
  </table>

  <div class="stats">
    <div class="stat-box"><div class="val">{mean_mag_a:.4f}</div><div class="lbl">Mean |V_team| A</div></div>
    <div class="stat-box"><div class="val">{mean_mag_b:.4f}</div><div class="lbl">Mean |V_team| B</div></div>
    <div class="stat-box"><div class="val">{mean_abs_diff:.4f}</div><div class="lbl">Mean |Diff|</div></div>
    <div class="stat-box"><div class="val">{pct_error:.1f}%</div><div class="lbl">Mean |Diff| / Mean |V|</div></div>
    <div class="stat-box"><div class="val">{mean_diff:+.4f}</div><div class="lbl">Mean Diff (A&minus;B)</div></div>
    <div class="stat-box"><div class="val">{max_abs_diff:.4f}</div><div class="lbl">Max |Diff|</div></div>
    <div class="stat-box"><div class="val">{std_diff:.4f}</div><div class="lbl">Std Dev of Diff</div></div>
  </div>
</div>

<h2>Team Value: A vs B</h2>
<img src="{scatter_uri}" class="plot" />

<h2>Difference Distribution</h2>
<img src="{hist_uri}" class="plot" />

<h2>Per-State Differences (sorted)</h2>
<img src="{sorted_uri}" class="plot" />

{agent_section}

{worst_html}
{best_html}

<h2>All States</h2>
<div class="table-wrap">
<table class="data" id="stateTable">
<thead><tr>
  <th data-col="0" data-type="num">State #</th>
  <th data-col="1" data-type="num">V_team A</th>
  <th data-col="2" data-type="num">V_team B</th>
  <th data-col="3" data-type="num">Diff (A&minus;B)</th>
  <th data-col="4" data-type="num">|Diff|</th>
  <th data-col="5" data-type="str">Agent Values A</th>
  <th data-col="6" data-type="str">Agent Values B</th>
  <th data-col="7" data-type="str">Agent Pos</th>
  <th data-col="8" data-type="str">Apple Pos</th>
  <th data-col="9" data-type="num">Actor</th>
</tr></thead>
<tbody>
{table_html}
</tbody>
</table>
</div>

<script>
// Sortable table
(function() {{
  const table = document.getElementById('stateTable');
  const headers = table.querySelectorAll('th');
  let sortCol = -1, sortAsc = true;

  headers.forEach(th => {{
    th.addEventListener('click', () => {{
      const col = parseInt(th.dataset.col);
      const isNum = th.dataset.type === 'num';
      if (sortCol === col) {{ sortAsc = !sortAsc; }} else {{ sortCol = col; sortAsc = true; }}

      // Update arrows
      headers.forEach(h => {{
        let arrow = h.querySelector('.sort-arrow');
        if (arrow) arrow.remove();
      }});
      const arrow = document.createElement('span');
      arrow.className = 'sort-arrow';
      arrow.textContent = sortAsc ? '▲' : '▼';
      th.appendChild(arrow);

      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort((a, b) => {{
        let va = a.children[col].textContent;
        let vb = b.children[col].textContent;
        if (isNum) {{ va = parseFloat(va); vb = parseFloat(vb); }}
        if (va < vb) return sortAsc ? -1 : 1;
        if (va > vb) return sortAsc ? 1 : -1;
        return 0;
      }});
      rows.forEach(r => tbody.appendChild(r));
    }});
  }});
}})();
</script>

</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
