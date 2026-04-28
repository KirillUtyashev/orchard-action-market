"""HTML builder: assemble self-contained HTML trajectory viewer.

Frames are rendered as inline SVGs (not PNGs) so individual cells,
agent groups, and task groups are clickable.
"""

from __future__ import annotations

import json
from pathlib import Path

from orchard.enums import Action, PickMode
from orchard.viz.frame import Frame
from orchard.viz.renderer import TASK_TYPE_HEX, AGENT_HEX, use_type_colors_for_agents

_ACTION_ARROWS: dict[int, str] = {
    0: "↑", 1: "↓", 2: "←",
    3: "→", 4: "·", 5: "★",
}


def _action_symbol(action: Action) -> str:
    if action.value in _ACTION_ARROWS:
        return _ACTION_ARROWS[action.value]
    if action.is_pick():
        return f"★{action.pick_type()}"
    return action.name


def _agent_css_color(
    idx: int,
    task_assignments: tuple[tuple[int, ...], ...] | None = None,
    n_task_types: int = 1,
    use_type_colors: bool = False,
) -> str:
    if (use_type_colors and task_assignments is not None
            and idx < len(task_assignments)
            and len(task_assignments[idx]) == 1):
        t = task_assignments[idx][0]
        return TASK_TYPE_HEX[t % len(TASK_TYPE_HEX)]
    return AGENT_HEX[idx % len(AGENT_HEX)]


def _build_frame_info_html(
    frame: Frame,
    n_task_types: int = 1,
    task_assignments: tuple[tuple[int, ...], ...] | None = None,
    pick_mode: PickMode = PickMode.CHOICE,
    use_type_colors: bool = False,
) -> str:
    parts: list[str] = []
    t = frame.state_index
    t1 = t + 1
    is_pick = frame.action.is_pick()

    parts.append(
        f'<b>s<sub>{t}</sub> → s<sub>{t}</sub><sup>a</sup></b>'
        f' &nbsp; <span style="color:#888">(decision {frame.step}, transition {frame.transition_index})</span>'
    )

    ac = _agent_css_color(frame.actor, task_assignments, n_task_types, use_type_colors)
    arrow = _action_symbol(frame.action)
    parts.append(
        f'<span style="color:{ac};font-weight:bold">A{frame.actor}</span> '
        f'→ {frame.action.name} {arrow}'
    )

    if task_assignments is not None:
        g = task_assignments[frame.actor]
        parts.append(f'<span style="color:#888">G<sub>{frame.actor}</sub> = {set(g)}</span>')

    if any(r != 0.0 for r in frame.rewards):
        r_parts = []
        for i, r in enumerate(frame.rewards):
            c = _agent_css_color(i, task_assignments, n_task_types, use_type_colors)
            sign = "+" if r > 0 else ""
            r_parts.append(f'<span style="color:{c}">{sign}{r:.2f}</span>')
        parts.append(f'r<sub>{t1}</sub> = [{", ".join(r_parts)}]')
        if frame.picked_task_type is not None:
            if frame.picked_correct:
                parts.append(
                    f'<span style="color:#2ca02c;font-weight:bold">✓ Correct pick (type {frame.picked_task_type})</span>'
                )
            elif frame.picked_correct is False:
                parts.append(
                    f'<span style="color:#d62728;font-weight:bold">✗ Wrong pick (type {frame.picked_task_type})</span>'
                )
    else:
        parts.append(f'r<sub>{t1}</sub> = 0')

    parts.append(f'γ<sub>{t1}</sub> = {frame.discount}')

    agent_on_task_after = frame.state_after.is_agent_on_task(frame.actor)
    if not is_pick and agent_on_task_after:
        if pick_mode == PickMode.FORCED:
            parts.append('<span style="color:#e377c2">→ Next: forced pick (γ=1)</span>')
        else:
            parts.append('<span style="color:#e377c2">→ Next: pick decision (γ=1)</span>')
    else:
        parts.append('<span style="color:#888">→ Next: env responds (spawn/despawn + advance actor)</span>')

    stats_parts = [
        f'Tasks: <span style="color:#d62728">{frame.tasks_on_grid}</span>',
        f'Team RPS: <span style="color:#2ca02c">{frame.team_reward_per_step:.4f}</span>',
        f'Correct: <span style="color:#2ca02c">{frame.total_correct_picks}</span>'
        f' Wrong: <span style="color:#d62728">{frame.total_wrong_picks}</span>',
    ]
    parts.append(" | ".join(stats_parts))

    if frame.state.task_types is not None:
        type_counts: dict[int, int] = {}
        for tt in frame.state.task_types:
            type_counts[tt] = type_counts.get(tt, 0) + 1
        type_str = " ".join(f'τ{k}:{v}' for k, v in sorted(type_counts.items()))
        parts.append(f'Per-type: {type_str}')

    if frame.agent_picks:
        n_agents = len(frame.rewards)
        pick_parts = []
        for i in range(n_agents):
            c = _agent_css_color(i, task_assignments, n_task_types, use_type_colors)
            cnt = frame.agent_picks.get(i, 0)
            pps = frame.agent_picks_per_step(i)
            pick_parts.append(f'<span style="color:{c}">A{i}: {cnt} ({pps:.4f}/step)</span>')
        parts.append(f'Agent picks: {" &nbsp; ".join(pick_parts)}')

    pos_parts = []
    for i, p in enumerate(frame.state.agent_positions):
        c = _agent_css_color(i, task_assignments, n_task_types, use_type_colors)
        label = f"A{i}({p.row},{p.col})"
        if i == frame.actor:
            label += " [actor]"
        pos_parts.append(
            f'<span style="color:{c};font-weight:{"bold" if i == frame.actor else "normal"}">{label}</span>'
        )
    parts.append(f'Positions: {" &nbsp; ".join(pos_parts)}')

    if frame.decisions:
        has_agent_breakdown = any(d.agent_q_values is not None for d in frame.decisions)
        dec_lines = ['<div style="margin-top:6px"><b>Q-values:</b>']
        sorted_decs = sorted(frame.decisions, key=lambda d: d.q_value, reverse=True)
        for d in sorted_decs:
            sym = _action_symbol(d.action)
            chosen = " ◄" if d.is_chosen else ""
            style = "font-weight:bold;color:#2ca02c" if d.is_chosen else "color:#555"
            dec_lines.append(
                f'<div style="{style}">&nbsp;&nbsp;{sym} {d.action.name:<5} Q={d.q_value:+.4f}{chosen}</div>'
            )
            if has_agent_breakdown and d.agent_q_values:
                for i, v in sorted(d.agent_q_values.items()):
                    c = _agent_css_color(i, task_assignments, n_task_types, use_type_colors)
                    dec_lines.append(
                        f'<div style="color:{c};font-size:11px">'
                        f'&nbsp;&nbsp;&nbsp;&nbsp;V<sub>{i}</sub>(s<sup>a</sup>) = {v:+.4f}</div>'
                    )
        dec_lines.append("</div>")
        parts.append("\n".join(dec_lines))

    if frame.agent_values:
        val_lines = ['<div style="margin-top:6px"><b>V(s):</b>']
        for i, v in sorted(frame.agent_values.items()):
            c = _agent_css_color(i, task_assignments, n_task_types, use_type_colors)
            val_lines.append(
                f'<div style="color:{c}">&nbsp;&nbsp;V<sub>{i}</sub>(s<sub>{t}</sub>) = {v:+.4f}</div>'
            )
        val_lines.append("</div>")
        parts.append("\n".join(val_lines))

    return "<br>".join(parts)


def _build_legend_html(
    n_task_types: int,
    n_agents: int,
    task_assignments: tuple[tuple[int, ...], ...] | None,
) -> str:
    if n_task_types <= 1:
        return ""

    # Group agents by task type (only when single-assignment)
    single = (
        task_assignments is not None
        and all(len(g) == 1 for g in task_assignments)
    )
    type_to_agents: dict[int, list[int]] = {}
    if single and task_assignments:
        for i, g in enumerate(task_assignments[:n_agents]):
            type_to_agents.setdefault(g[0], []).append(i)

    rows: list[str] = []
    for tau in range(n_task_types):
        hex_c = TASK_TYPE_HEX[tau % len(TASK_TYPE_HEX)]
        swatch = (
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'background:{hex_c};border-radius:3px;flex-shrink:0"></span>'
        )
        label = f"Type {tau}"
        if tau in type_to_agents:
            ag = type_to_agents[tau]
            if len(ag) > 4:
                label += f" — A{ag[0]}–A{ag[-1]}"
            else:
                label += " — " + ", ".join(f"A{a}" for a in ag)
        rows.append(
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'font-size:11px;color:#ccc;margin:2px 4px">{swatch}{label}</span>'
        )

    items = "".join(rows)
    return (
        f'<div style="background:#22223a;border:1px solid #333;border-radius:8px;'
        f'padding:10px 14px;margin-top:12px;max-width:90vw">'
        f'<div style="font-size:11px;color:#888;font-weight:bold;margin-bottom:6px">Task Types</div>'
        f'<div style="display:flex;flex-wrap:wrap">{items}</div>'
        f'</div>'
    )


def build_html(
    frames: list[Frame],
    frame_svgs: list[str],
    output_path: Path,
    fps: int = 3,
    compare_frames: list[Frame] | None = None,
    compare_svgs: list[str] | None = None,
    n_task_types: int = 1,
    task_assignments: tuple[tuple[int, ...], ...] | None = None,
    pick_mode: PickMode = PickMode.CHOICE,
) -> None:
    """Write a self-contained HTML file with embedded SVG frames and slider."""
    n = len(frames)
    is_compare = compare_frames is not None and compare_svgs is not None

    use_type_colors = use_type_colors_for_agents(task_assignments, n_task_types)

    info_htmls = [
        _build_frame_info_html(f, n_task_types, task_assignments, pick_mode, use_type_colors)
        for f in frames
    ]
    compare_info_htmls: list[str] = []
    if is_compare and compare_frames:
        compare_info_htmls = [
            _build_frame_info_html(f, n_task_types, task_assignments, pick_mode, use_type_colors)
            for f in compare_frames
        ]

    task_counts = [f.tasks_on_grid for f in frames]
    compare_task_counts = [f.tasks_on_grid for f in compare_frames] if is_compare else []

    svgs_json = json.dumps(frame_svgs)
    compare_svgs_json = json.dumps(compare_svgs) if is_compare else "[]"
    info_json = json.dumps(info_htmls)
    compare_info_json = json.dumps(compare_info_htmls) if is_compare else "[]"

    policy_name = frames[0].policy_name
    compare_policy_name = compare_frames[0].policy_name if is_compare and compare_frames else ""

    n_compare = len(compare_frames) if is_compare and compare_frames else 0
    max_slider = n - 1

    legend_html = _build_legend_html(n_task_types, len(frames[0].rewards), task_assignments)

    # Pass color data to JS for popup rendering
    task_type_colors_js = json.dumps(TASK_TYPE_HEX)
    agent_colors_js = json.dumps(AGENT_HEX)
    assignments_js = json.dumps(
        [list(g) for g in task_assignments] if task_assignments else None
    )
    use_type_colors_js = "true" if use_type_colors else "false"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Orchard Trajectory Viewer</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Menlo', 'Consolas', 'DejaVu Sans Mono', monospace;
    background: #1a1a2e;
    color: #e0e0e0;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
    min-height: 100vh;
  }}
  h1 {{ font-size: 18px; margin-bottom: 10px; color: #aaa; }}
  .controls {{
    display: flex; align-items: center; gap: 12px; margin: 12px 0;
    flex-wrap: wrap; justify-content: center;
  }}
  .controls button {{
    background: #2a2a4a; color: #e0e0e0; border: 1px solid #444;
    padding: 6px 14px; border-radius: 4px; cursor: pointer;
    font-family: inherit; font-size: 13px;
  }}
  .controls button:hover {{ background: #3a3a5a; }}
  #slider {{ width: 500px; max-width: 80vw; accent-color: #7a7aaa; }}
  .step-label {{ font-size: 14px; min-width: 180px; text-align: center; }}
  .viewer {{ display: flex; flex-direction: column; align-items: center; gap: 8px; margin-top: 8px; width: 100%; }}
  .frame-section {{
    background: #22223a; border-radius: 8px; padding: 12px; border: 1px solid #333;
    max-width: 95vw;
  }}
  .frame-section h2 {{ font-size: 13px; color: #888; margin-bottom: 8px; }}
  .frame-svg {{ overflow-x: auto; }}
  .frame-svg svg {{ display: block; max-width: 100%; height: auto; }}
  .info-panel {{
    font-size: 12px; line-height: 1.7; margin-top: 8px;
    padding: 8px; background: #1a1a2e; border-radius: 4px;
  }}
  .sparkline-container {{
    width: 520px; max-width: 85vw; margin-top: 16px;
    background: #22223a; border-radius: 8px; padding: 12px; border: 1px solid #333;
  }}
  .sparkline-container h3 {{ font-size: 12px; color: #888; margin-bottom: 6px; }}
  canvas {{ width: 100%; height: 60px; display: block; }}
  .keyboard-hint {{ font-size: 11px; color: #555; margin-top: 12px; }}
  #clickPopup {{
    position: fixed;
    display: none;
    background: #2a2a4a;
    color: #e0e0e0;
    border: 1px solid #666;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 12px;
    line-height: 1.7;
    max-width: 300px;
    z-index: 1000;
    pointer-events: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5);
  }}
</style>
</head>
<body>

<h1>Orchard Trajectory Viewer</h1>

<div class="controls">
  <button id="btnPrev" title="Previous (←)">◀ Prev</button>
  <button id="btnPlay" title="Play/Pause (Space)">▶ Play</button>
  <button id="btnNext" title="Next (→)">Next ▶</button>
  <input type="range" id="slider" min="0" max="{max_slider}" value="0">
  <span class="step-label" id="stepLabel">Transition 0 / {max_slider}</span>
  <label style="font-size:12px;color:#888;">
    FPS: <input type="number" id="fpsInput" value="{fps}" min="1" max="30"
         style="width:40px;background:#2a2a4a;color:#e0e0e0;border:1px solid #444;border-radius:3px;text-align:center;">
  </label>
</div>

<div class="viewer">
  <div class="frame-section">
    <h2 id="primaryLabel">{policy_name}</h2>
    <div id="primarySvg" class="frame-svg"></div>
    <div class="info-panel" id="primaryInfo"></div>
  </div>
  {"" if not is_compare else f'''
  <div class="frame-section">
    <h2 id="compareLabel">{compare_policy_name}</h2>
    <div id="compareSvg" class="frame-svg"></div>
    <div class="info-panel" id="compareInfo"></div>
  </div>
  '''}
</div>

{legend_html}

<div class="sparkline-container">
  <h3>Task count over trajectory</h3>
  <canvas id="sparkline"></canvas>
</div>

<div class="keyboard-hint">← → step &nbsp; | &nbsp; Space play/pause &nbsp; | &nbsp; click agents/tasks for details</div>

<div id="clickPopup"></div>

<script>
const N = {n};
const N_COMPARE = {n_compare};
const FPS_DEFAULT = {fps};
const IS_COMPARE = {'true' if is_compare else 'false'};

const svgFrames = {svgs_json};
const compareSvgFrames = {compare_svgs_json};

const infoData = {info_json};
const compareInfoData = {compare_info_json};

const taskCounts = {json.dumps(task_counts)};
const compareTaskCounts = {json.dumps(compare_task_counts)};

// Color data for popups
const TASK_TYPE_COLORS = {task_type_colors_js};
const AGENT_COLORS = {agent_colors_js};
const AGENT_ASSIGNMENTS = {assignments_js};
const USE_TYPE_COLORS = {use_type_colors_js};

function agentHexColor(a) {{
  if (USE_TYPE_COLORS && AGENT_ASSIGNMENTS && AGENT_ASSIGNMENTS[a] && AGENT_ASSIGNMENTS[a].length === 1) {{
    const t = AGENT_ASSIGNMENTS[a][0];
    return TASK_TYPE_COLORS[t % TASK_TYPE_COLORS.length];
  }}
  return AGENT_COLORS[a % AGENT_COLORS.length];
}}

// --- Popup ---
const popup = document.getElementById('clickPopup');

function showPopup(event, type, data) {{
  let html = '';
  if (type === 'tasks') {{
    const count = data.types.length;
    html = `<b>${{count > 1 ? count + ' tasks' : 'Task'}} on cell:</b><br>`;
    html += data.types.map(t => {{
      const c = TASK_TYPE_COLORS[t % TASK_TYPE_COLORS.length];
      return `<span style="display:inline-flex;align-items:center;gap:5px;margin:1px 0">` +
             `<span style="width:12px;height:12px;background:${{c}};border-radius:2px;display:inline-block;flex-shrink:0"></span>` +
             `Type ${{t}}</span>`;
    }}).join('<br>');
  }} else if (type === 'agents') {{
    const count = data.agents.length;
    html = `<b>${{count > 1 ? count + ' agents' : 'Agent'}} on cell:</b><br>`;
    html += data.agents.map(a => {{
      const c = agentHexColor(a);
      const g = AGENT_ASSIGNMENTS ? AGENT_ASSIGNMENTS[a] : null;
      const gStr = g ? ` → {{${{g.join(', ')}}}}` : '';
      return `<span style="display:inline-flex;align-items:center;gap:5px;margin:1px 0">` +
             `<span style="width:12px;height:12px;background:${{c}};border-radius:50%;display:inline-block;flex-shrink:0"></span>` +
             `A${{a}}${{gStr}}</span>`;
    }}).join('<br>');
  }}
  popup.innerHTML = html;
  popup.style.display = 'block';
  // Keep popup inside viewport
  const pw = 220, ph = 30 + data[type === 'tasks' ? 'types' : 'agents'].length * 22;
  let px = event.clientX + 14;
  let py = event.clientY + 14;
  if (px + pw > window.innerWidth) px = event.clientX - pw - 8;
  if (py + ph > window.innerHeight) py = event.clientY - ph - 8;
  popup.style.left = px + 'px';
  popup.style.top = py + 'px';
}}

function hidePopup() {{
  popup.style.display = 'none';
}}

function handleSvgClick(event) {{
  const group = event.target.closest('[data-popup-type]');
  if (!group) {{ hidePopup(); return; }}
  event.stopPropagation();
  const type = group.dataset.popupType;
  const data = JSON.parse(group.dataset.popupData);
  showPopup(event, type, data);
}}

document.addEventListener('click', hidePopup);

// Attach popup handler to SVG containers (survives innerHTML replacement)
document.getElementById('primarySvg').addEventListener('click', handleSvgClick);
{"document.getElementById('compareSvg') && document.getElementById('compareSvg').addEventListener('click', handleSvgClick);" if is_compare else ""}

// --- Playback ---
const slider = document.getElementById('slider');
const stepLabel = document.getElementById('stepLabel');
const primarySvg = document.getElementById('primarySvg');
const primaryInfo = document.getElementById('primaryInfo');
const compareSvgEl = document.getElementById('compareSvg');
const compareInfo = document.getElementById('compareInfo');
const btnPlay = document.getElementById('btnPlay');
const btnPrev = document.getElementById('btnPrev');
const btnNext = document.getElementById('btnNext');
const fpsInput = document.getElementById('fpsInput');
const canvas = document.getElementById('sparkline');

let currentStep = 0;
let playing = false;
let playInterval = null;

function showStep(step) {{
  currentStep = Math.max(0, Math.min(N - 1, step));
  slider.value = currentStep;
  stepLabel.textContent = `Transition ${{currentStep}} / ${{N - 1}}`;
  primarySvg.innerHTML = svgFrames[currentStep];
  primaryInfo.innerHTML = infoData[currentStep];
  if (IS_COMPARE && compareSvgEl) {{
    const ci = Math.min(currentStep, N_COMPARE - 1);
    compareSvgEl.innerHTML = compareSvgFrames[ci];
    compareInfo.innerHTML = compareInfoData[ci];
  }}
  drawSparkline();
}}

function togglePlay() {{
  playing = !playing;
  btnPlay.textContent = playing ? '⏸ Pause' : '▶ Play';
  if (playing) {{
    const fps = parseInt(fpsInput.value) || FPS_DEFAULT;
    playInterval = setInterval(() => {{
      if (currentStep >= N - 1) {{ togglePlay(); return; }}
      showStep(currentStep + 1);
    }}, 1000 / fps);
  }} else {{
    clearInterval(playInterval);
    playInterval = null;
  }}
}}

function drawSparkline() {{
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.offsetWidth * 2;
  const H = canvas.height = canvas.offsetHeight * 2;
  ctx.clearRect(0, 0, W, H);

  const allVals = [...taskCounts, ...compareTaskCounts];
  const maxVal = Math.max(...allVals, 1);
  const minVal = Math.min(...allVals, 0);
  const range = maxVal - minVal || 1;
  const pad = 4;

  function drawLine(data, color, total) {{
    if (data.length === 0) return;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    for (let i = 0; i < data.length; i++) {{
      const x = pad + (i / Math.max(total - 1, 1)) * (W - 2 * pad);
      const y = H - pad - ((data[i] - minVal) / range) * (H - 2 * pad);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }}
    ctx.stroke();
  }}

  drawLine(taskCounts, '#d62728', N);
  if (IS_COMPARE) drawLine(compareTaskCounts, '#1f77b4', N_COMPARE);

  const markerX = pad + (currentStep / Math.max(N - 1, 1)) * (W - 2 * pad);
  ctx.beginPath();
  ctx.strokeStyle = '#ffffff88';
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.moveTo(markerX, 0);
  ctx.lineTo(markerX, H);
  ctx.stroke();
  ctx.setLineDash([]);
}}

slider.addEventListener('input', (e) => showStep(parseInt(e.target.value)));
btnPlay.addEventListener('click', togglePlay);
btnPrev.addEventListener('click', () => showStep(currentStep - 1));
btnNext.addEventListener('click', () => showStep(currentStep + 1));
fpsInput.addEventListener('change', () => {{
  if (playing) {{ togglePlay(); togglePlay(); }}
}});

document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') {{ e.preventDefault(); showStep(currentStep - 1); }}
  else if (e.key === 'ArrowRight') {{ e.preventDefault(); showStep(currentStep + 1); }}
  else if (e.key === ' ') {{ e.preventDefault(); togglePlay(); }}
}});

showStep(0);
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
