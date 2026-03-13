"""Build a self-contained HTML report comparing greedy policies."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from orchard.compare_policies.compare import PolicyComparison
from orchard.compare_values.loader import LoadedRun
from orchard.datatypes import State
from orchard.enums import Action, ACTION_PRIORITY
from orchard.viz.renderer import render_state_png


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ACTION_SYMBOLS: dict[Action, str] = {
    Action.UP: "\u2191",
    Action.DOWN: "\u2193",
    Action.LEFT: "\u2190",
    Action.RIGHT: "\u2192",
    Action.STAY: "\u00b7",
}


def _png_to_data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def _render_state(state: State, height: int, width: int, dpi: int = 100) -> str:
    png = render_state_png(state, height, width, actor=state.actor, dpi=dpi)
    return _png_to_data_uri(png)


# ---------------------------------------------------------------------------
# Q-value table for one state
# ---------------------------------------------------------------------------

def _q_table_html(comp: PolicyComparison, labels: list[str]) -> str:
    """Build an HTML table: rows = actions, columns = models."""
    rows = []

    # Header
    header = "<tr><th>Action</th>"
    for label in labels:
        header += f"<th>{label}</th>"
    header += "</tr>"
    rows.append(header)

    for action in ACTION_PRIORITY:
        sym = _ACTION_SYMBOLS.get(action, action.name)
        row = f"<tr><td><b>{sym} {action.name}</b></td>"
        for i, label in enumerate(labels):
            q = comp.q_values[i][action]
            chosen = comp.actions[i] == action
            cls = "chosen" if chosen else ""
            row += f'<td class="{cls}">{q:.6f}</td>'
        row += "</tr>"
        rows.append(row)

    return f'<table class="qtable">\n' + "\n".join(rows) + "\n</table>"


# ---------------------------------------------------------------------------
# State card
# ---------------------------------------------------------------------------

def _state_card_html(
    comp: PolicyComparison,
    labels: list[str],
    state_uri: str,
) -> str:
    """One state's card: grid image + action summary + Q table."""
    # Action summary line
    action_parts = []
    for i, label in enumerate(labels):
        sym = _ACTION_SYMBOLS.get(comp.actions[i], comp.actions[i].name)
        action_parts.append(f"<b>{label}</b>: {sym} {comp.actions[i].name}")
    action_summary = " &nbsp;|&nbsp; ".join(action_parts)

    # Q gap info
    gaps = []
    for qv in comp.q_values:
        sorted_q = sorted(qv.values(), reverse=True)
        if len(sorted_q) >= 2:
            gaps.append(sorted_q[0] - sorted_q[1])
    gap_strs = [f"{g:.6f}" for g in gaps]
    gap_line = "Q gap (best \u2212 2nd): " + ", ".join(gap_strs)

    # Agent/apple info
    agents_str = ", ".join(f"A{i}=({p.row},{p.col})" for i, p in enumerate(comp.state.agent_positions))
    apples_str = f"{len(comp.state.apple_positions)} apples"
    actor_str = f"actor={comp.state.actor}"

    qtable = _q_table_html(comp, labels)

    return f"""
    <div class="state-card {'agree' if comp.agrees else 'disagree'}">
      <div class="card-header">
        <span class="state-num">State {comp.state_index}</span>
        <span class="badge {'badge-agree' if comp.agrees else 'badge-disagree'}">
          {'AGREE' if comp.agrees else 'DISAGREE'}
        </span>
      </div>
      <div class="card-body">
        <div class="grid-col">
          <img src="{state_uri}" />
          <div class="meta">{agents_str}<br>{apples_str}, {actor_str}</div>
        </div>
        <div class="info-col">
          <div class="actions">{action_summary}</div>
          <div class="gap">{gap_line}</div>
          {qtable}
        </div>
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def build_report(
    comparisons: list[PolicyComparison],
    runs: list[LoadedRun],
    labels: list[str],
    output_path: Path,
    dpi: int = 100,
) -> None:
    """Build self-contained HTML policy comparison report."""
    height = runs[0].cfg.env.height
    width = runs[0].cfg.env.width

    # Partition
    disagree = [c for c in comparisons if not c.agrees]
    agree = [c for c in comparisons if c.agrees]

    # Sort disagreements: largest max Q-gap first (most confident = most interesting)
    disagree.sort(key=lambda c: c.q_gap, reverse=True)

    # Summary stats
    n_total = len(comparisons)
    n_disagree = len(disagree)
    n_agree = len(agree)
    pct_agree = n_agree / n_total * 100 if n_total > 0 else 0

    # Action distribution per model
    action_counts: list[dict[Action, int]] = [{a: 0 for a in ACTION_PRIORITY} for _ in runs]
    for c in comparisons:
        for i, a in enumerate(c.actions):
            action_counts[i][a] += 1

    # Render disagreement state images
    disagree_cards = []
    for c in disagree:
        uri = _render_state(c.state, height, width, dpi)
        disagree_cards.append(_state_card_html(c, labels, uri))

    # For agreements: render a small sample (first 10)
    agree_sample = agree[:10]
    agree_cards = []
    for c in agree_sample:
        uri = _render_state(c.state, height, width, dpi)
        agree_cards.append(_state_card_html(c, labels, uri))

    # Action distribution table
    dist_header = "<tr><th>Action</th>" + "".join(f"<th>{l}</th>" for l in labels) + "</tr>"
    dist_rows = [dist_header]
    for action in ACTION_PRIORITY:
        sym = _ACTION_SYMBOLS.get(action, action.name)
        row = f"<tr><td><b>{sym} {action.name}</b></td>"
        for i in range(len(runs)):
            cnt = action_counts[i][action]
            pct = cnt / n_total * 100 if n_total > 0 else 0
            row += f"<td>{cnt} ({pct:.1f}%)</td>"
        row += "</tr>"
        dist_rows.append(row)
    dist_table = "<table class='qtable'>\n" + "\n".join(dist_rows) + "\n</table>"

    # Run info rows
    run_info = ""
    for i, (run, label) in enumerate(zip(runs, labels)):
        lt = "centralized" if run.is_centralized else "decentralized"
        run_info += f"<tr><td class='label'>{label}:</td><td>{lt}, step {run.checkpoint_step}, {run.run_dir}</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Policy Comparison</title>
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
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px; margin: 15px 0; }}
  .stat-box {{ background: #e8f0fe; border-radius: 6px; padding: 12px; text-align: center; }}
  .stat-box .val {{ font-size: 1.4em; font-weight: bold; color: #1a73e8; }}
  .stat-box .lbl {{ font-size: 0.85em; color: #666; margin-top: 4px; }}

  .state-card {{ background: white; border-radius: 8px; margin: 15px 0; padding: 15px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .state-card.disagree {{ border-left: 4px solid #e74c3c; }}
  .state-card.agree {{ border-left: 4px solid #27ae60; }}
  .card-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }}
  .state-num {{ font-weight: bold; font-size: 1.1em; }}
  .badge {{ font-size: 0.75em; padding: 2px 8px; border-radius: 4px; font-weight: bold; }}
  .badge-disagree {{ background: #fde8e8; color: #c0392b; }}
  .badge-agree {{ background: #e8fde8; color: #27ae60; }}
  .card-body {{ display: flex; gap: 20px; align-items: flex-start; }}
  .grid-col {{ flex-shrink: 0; }}
  .grid-col img {{ max-width: 220px; border-radius: 4px; }}
  .meta {{ font-size: 0.8em; color: #888; margin-top: 6px; }}
  .info-col {{ flex-grow: 1; }}
  .actions {{ font-size: 0.95em; margin-bottom: 8px; }}
  .gap {{ font-size: 0.85em; color: #888; margin-bottom: 10px; }}

  .qtable {{ border-collapse: collapse; font-size: 0.85em; }}
  .qtable th {{ background: #f0f0f0; padding: 5px 10px; text-align: left; border-bottom: 2px solid #ccc; }}
  .qtable td {{ padding: 5px 10px; border-bottom: 1px solid #eee; font-family: monospace; }}
  .qtable td.chosen {{ background: #fff3cd; font-weight: bold; }}

  .collapse-btn {{ background: #eee; border: 1px solid #ccc; border-radius: 4px;
                   padding: 6px 14px; cursor: pointer; font-size: 0.9em; margin: 10px 0; }}
  .collapse-btn:hover {{ background: #ddd; }}
  .collapsed {{ display: none; }}
</style>
</head>
<body>

<h1>Policy Comparison Report</h1>

<div class="summary">
  <table>
    {run_info}
    <tr><td class="label">Env:</td><td>{height}&times;{width}, {runs[0].cfg.env.n_agents} agents, &gamma;={runs[0].cfg.env.gamma}</td></tr>
    <tr><td class="label">States compared:</td><td>{n_total}</td></tr>
  </table>

  <div class="stats">
    <div class="stat-box"><div class="val">{n_agree}</div><div class="lbl">Agree</div></div>
    <div class="stat-box"><div class="val">{n_disagree}</div><div class="lbl">Disagree</div></div>
    <div class="stat-box"><div class="val">{pct_agree:.1f}%</div><div class="lbl">Agreement Rate</div></div>
  </div>
</div>

<h2>Action Distribution (all {n_total} states)</h2>
{dist_table}

<h2>Disagreements ({n_disagree} states)</h2>
<p style="color:#888; margin-bottom:10px;">Sorted by Q-gap (largest first = most confident disagreement).</p>
{''.join(disagree_cards) if disagree_cards else '<p>All states agree!</p>'}

<h2>Agreements (sample of {len(agree_sample)}/{n_agree})</h2>
<button class="collapse-btn" onclick="
  var el = document.getElementById('agree-section');
  el.classList.toggle('collapsed');
  this.textContent = el.classList.contains('collapsed') ? 'Show agreements' : 'Hide agreements';
">Show agreements</button>
<div id="agree-section" class="collapsed">
{''.join(agree_cards)}
</div>

</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
