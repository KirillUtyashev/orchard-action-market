"""Build a self-contained HTML report comparing N runs."""

from __future__ import annotations

import base64
import io
from itertools import combinations
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
# Image helpers
# ---------------------------------------------------------------------------

def _render_state_png(state: State, height: int, width: int, dpi: int = 100) -> bytes:
    return render_frame_png(
        state=state, state_after=None, height=height, width=width,
        state_index=0, actor=state.actor, action=None, rewards=None,
        show_after_state=False, dpi=dpi,
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
# Plots
# ---------------------------------------------------------------------------

def _build_agreement_matrix(
    comparisons: list[StateComparison],
    labels: list[str],
) -> str:
    """NxN heatmap of mean |V_i - V_j| between each pair of runs."""
    n = len(labels)
    matrix = np.zeros((n, n))

    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if i == j:
                continue
            diffs = [abs(c.team_values[li] - c.team_values[lj]) for c in comparisons]
            matrix[i, j] = np.mean(diffs)

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Mean |V_team_i − V_team_j|", shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=7, color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")

    ax.set_title("Pairwise Mean |Diff|")
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def _build_pairwise_scatters(
    comparisons: list[StateComparison],
    labels: list[str],
) -> str:
    """Pairwise scatter grid of V_team for all run pairs."""
    pairs = list(combinations(range(len(labels)), 2))
    n_pairs = len(pairs)
    ncols = min(3, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx // ncols, idx % ncols]
        li, lj = labels[i], labels[j]
        xs = [c.team_values[li] for c in comparisons]
        ys = [c.team_values[lj] for c in comparisons]
        abs_diffs = [abs(x - y) for x, y in zip(xs, ys)]

        sc = ax.scatter(xs, ys, c=abs_diffs, cmap="RdYlGn_r", s=15, alpha=0.7, edgecolors="none")

        all_vals = xs + ys
        lo = min(all_vals) - 0.1
        hi = max(all_vals) + 0.1
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(li, fontsize=7)
        ax.set_ylabel(lj, fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for idx in range(n_pairs, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("Pairwise V_team Scatter", fontsize=13)
    fig.tight_layout()
    return _fig_to_data_uri(fig)


def _build_variance_histogram(comparisons: list[StateComparison], labels: list[str]) -> str:
    """Histogram of per-state cross-model std dev."""
    stds = []
    for c in comparisons:
        vals = [c.team_values[l] for l in labels]
        stds.append(np.std(vals))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(stds, bins=min(50, max(10, len(stds) // 5)),
            color="#4A90D9", edgecolor="white", alpha=0.85)
    mean_std = np.mean(stds)
    ax.axvline(mean_std, color="red", linewidth=1.5, linestyle="-", alpha=0.7,
               label=f"mean = {mean_std:.4f}")
    ax.set_xlabel("Cross-model Std Dev of V_team")
    ax.set_ylabel("Count")
    ax.set_title("Per-State Disagreement Distribution")
    ax.legend()
    fig.tight_layout()
    return _fig_to_data_uri(fig)


# ---------------------------------------------------------------------------
# Extreme states
# ---------------------------------------------------------------------------

def _render_extreme_states(
    comparisons: list[StateComparison],
    labels: list[str],
    height: int,
    width: int,
    k: int = 5,
    dpi: int = 100,
) -> tuple[list[tuple[int, str, StateComparison]], list[tuple[int, str, StateComparison]]]:
    """Top-K highest and lowest cross-model variance states."""
    scored = []
    for c in comparisons:
        vals = [c.team_values[l] for l in labels]
        scored.append((np.std(vals), c))

    scored.sort(key=lambda x: x[0], reverse=True)

    worst = []
    for _, c in scored[:k]:
        png = _render_state_png(c.state, height, width, dpi)
        worst.append((c.state_index, _png_to_data_uri(png), c))

    best = []
    for _, c in scored[-k:]:
        png = _render_state_png(c.state, height, width, dpi)
        best.append((c.state_index, _png_to_data_uri(png), c))

    return worst, best


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_report(
    comparisons: list[StateComparison],
    runs: list[LoadedRun],
    output_path: Path,
    dpi: int = 100,
) -> None:
    """Build self-contained HTML report comparing N runs."""
    labels = [r.label for r in runs]
    n_runs = len(runs)
    n_agents = runs[0].cfg.env.n_agents
    height = runs[0].cfg.env.height
    width = runs[0].cfg.env.width

    # --- Aggregate stats ---
    # Pairwise mean |diff| summary
    pair_stats = []
    for i, j in combinations(range(n_runs), 2):
        li, lj = labels[i], labels[j]
        diffs = [abs(c.team_values[li] - c.team_values[lj]) for c in comparisons]
        pair_stats.append((li, lj, float(np.mean(diffs)), float(np.max(diffs))))

    # Per-model mean |V_team|
    model_means = {}
    for l in labels:
        model_means[l] = float(np.mean([abs(c.team_values[l]) for c in comparisons]))

    # Cross-model variance per state
    per_state_stds = []
    for c in comparisons:
        vals = [c.team_values[l] for l in labels]
        per_state_stds.append(np.std(vals))
    mean_cross_std = float(np.mean(per_state_stds))

    # --- Plots ---
    matrix_uri = _build_agreement_matrix(comparisons, labels)
    scatter_uri = _build_pairwise_scatters(comparisons, labels)
    variance_uri = _build_variance_histogram(comparisons, labels)

    # --- Extreme states ---
    k = min(5, len(comparisons))
    worst, best = _render_extreme_states(comparisons, labels, height, width, k, dpi)

    # --- Run info rows ---
    run_info_rows = "".join(
        f"<tr><td class='label'>Run {i}:</td><td>{l}</td></tr>"
        for i, l in enumerate(labels)
    )

    # --- Stat boxes ---
    stat_boxes = ""
    for l in labels:
        stat_boxes += (
            f'<div class="stat-box"><div class="val">{model_means[l]:.4f}</div>'
            f'<div class="lbl">Mean |V| {l}</div></div>\n'
        )
    stat_boxes += (
        f'<div class="stat-box"><div class="val">{mean_cross_std:.4f}</div>'
        f'<div class="lbl">Mean cross-model σ</div></div>\n'
    )

    # --- Table ---
    header_cols = "".join(
        f'<th data-col="{i + 1}" data-type="num">V_team {l}</th>' for i, l in enumerate(labels)
    )
    table_rows = []
    for c in comparisons:
        vals = [c.team_values[l] for l in labels]
        std = np.std(vals)
        val_cells = "".join(f"<td>{v:.4f}</td>" for v in vals)
        agents_str = str(c.state.agent_positions)
        tasks_str = str(c.state.task_positions)
        table_rows.append(
            f"<tr>"
            f"<td>{c.state_index}</td>"
            f"{val_cells}"
            f"<td>{std:.4f}</td>"
            f"<td>{c.state.actor}</td>"
            f"<td class='detail'>{agents_str}</td>"
            f"<td class='detail'>{tasks_str}</td>"
            f"</tr>"
        )
    table_html = "\n".join(table_rows)

    n_data_cols = 1 + n_runs + 3  # index + N values + std + actor + positions

    # --- Extreme state HTML ---
    def _extreme_html(items: list[tuple[int, str, StateComparison]], title: str) -> str:
        if not items:
            return ""
        cards = []
        for idx, uri, c in items:
            vals_str = "<br>".join(f"{l}: {c.team_values[l]:.4f}" for l in labels)
            vals = [c.team_values[l] for l in labels]
            cards.append(
                f'<div class="state-card">'
                f'<img src="{uri}" />'
                f'<div class="card-info">'
                f'<b>State {idx}</b> (σ={np.std(vals):.4f})<br>'
                f'{vals_str}'
                f'</div></div>'
            )
        return f"<h2>{title}</h2><div class='state-gallery'>{''.join(cards)}</div>"

    worst_html = _extreme_html(worst, f"Top {k} Largest Disagreements (by σ)")
    best_html = _extreme_html(best, f"Top {k} Closest Agreements (by σ)")

    # --- Pairwise stats table ---
    pair_rows = "".join(
        f"<tr><td>{a}</td><td>{b}</td><td>{mean_d:.4f}</td><td>{max_d:.4f}</td></tr>"
        for a, b, mean_d, max_d in pair_stats
    )

    # --- Assemble HTML ---
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Value Comparison: {n_runs} runs</title>
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
  .state-card {{ background: white; border-radius: 6px; padding: 10px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
  .state-card img {{ max-width: 200px; border-radius: 4px; }}
  .card-info {{ font-size: 0.85em; margin-top: 8px; line-height: 1.5; }}
  .table-wrap {{ overflow-x: auto; margin: 15px 0; }}
  table.data {{ border-collapse: collapse; width: 100%; background: white;
                border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                font-size: 0.85em; }}
  table.data th {{ background: #f0f0f0; padding: 8px 10px; text-align: left; cursor: pointer;
                   user-select: none; white-space: nowrap; border-bottom: 2px solid #ccc; }}
  table.data th:hover {{ background: #e0e0e0; }}
  table.data td {{ padding: 6px 10px; border-bottom: 1px solid #eee; white-space: nowrap; }}
  table.data tr:hover {{ background: #f8f8f8; }}
  td.detail {{ font-size: 0.8em; color: #666; }}
  .sort-arrow {{ font-size: 0.7em; margin-left: 4px; }}
  table.pair {{ border-collapse: collapse; margin: 10px 0; }}
  table.pair th, table.pair td {{ padding: 6px 14px; border: 1px solid #ddd; }}
  table.pair th {{ background: #f0f0f0; }}
</style>
</head>
<body>

<h1>Value Comparison Report ({n_runs} runs)</h1>

<div class="summary">
  <table>
    {run_info_rows}
    <tr><td class="label">Env:</td><td>{height}&times;{width}, {n_agents} agents, &gamma;={runs[0].cfg.env.gamma}</td></tr>
    <tr><td class="label">TD target:</td><td>{runs[0].cfg.train.td_target.name.lower()}</td></tr>
    <tr><td class="label">States compared:</td><td>{len(comparisons)}</td></tr>
  </table>

  <div class="stats">
    {stat_boxes}
  </div>
</div>

<h2>Agreement Matrix</h2>
<img src="{matrix_uri}" class="plot" />

<h2>Pairwise Statistics</h2>
<table class="pair">
<tr><th>Run A</th><th>Run B</th><th>Mean |Diff|</th><th>Max |Diff|</th></tr>
{pair_rows}
</table>

<h2>Pairwise Scatter Plots</h2>
<img src="{scatter_uri}" class="plot" />

<h2>Per-State Disagreement</h2>
<img src="{variance_uri}" class="plot" />

{worst_html}
{best_html}

<h2>All States</h2>
<div class="table-wrap">
<table class="data" id="stateTable">
<thead><tr>
  <th data-col="0" data-type="num">State #</th>
  {header_cols}
  <th data-col="{n_runs + 1}" data-type="num">&sigma;</th>
  <th data-col="{n_runs + 2}" data-type="num">Actor</th>
  <th data-col="{n_runs + 3}" data-type="str">Agent Pos</th>
  <th data-col="{n_runs + 4}" data-type="str">Apple Pos</th>
</tr></thead>
<tbody>
{table_html}
</tbody>
</table>
</div>

<script>
(function() {{
  const table = document.getElementById('stateTable');
  const headers = table.querySelectorAll('th');
  let sortCol = -1, sortAsc = true;

  headers.forEach(th => {{
    th.addEventListener('click', () => {{
      const col = parseInt(th.dataset.col);
      const isNum = th.dataset.type === 'num';
      if (sortCol === col) {{ sortAsc = !sortAsc; }} else {{ sortCol = col; sortAsc = true; }}

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
