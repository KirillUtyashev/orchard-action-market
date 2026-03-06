"""HTML builder: assemble self-contained HTML trajectory viewer."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from orchard.enums import Action
from orchard.viz.frame import Frame

_ACTION_ARROWS: dict[Action, str] = {
    Action.UP: "\u2191",
    Action.DOWN: "\u2193",
    Action.LEFT: "\u2190",
    Action.RIGHT: "\u2192",
    Action.STAY: "\u00b7",
    Action.PICK: "\u2605",
}


def _agent_css_color(idx: int) -> str:
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#800080", "#008080",
    ]
    return palette[idx % len(palette)]


def _build_frame_info_html(frame: Frame) -> str:
    """Build the HTML info panel for one frame with math notation."""
    parts: list[str] = []
    t = frame.state_index
    t1 = t + 1
    is_pick = (frame.action == Action.PICK)

    # Transition header
    parts.append(
        f'<b>s<sub>{t}</sub> \u2192 s<sub>{t}</sub><sup>a</sup></b>'
        f' &nbsp; <span style="color:#888">(decision {frame.step}, transition {frame.transition_index})</span>'
    )

    # Actor and action
    ac = _agent_css_color(frame.actor)
    arrow = _ACTION_ARROWS.get(frame.action, "?")
    parts.append(
        f'<span style="color:{ac};font-weight:bold">A{frame.actor}</span> '
        f'\u2192 {frame.action.name} {arrow}'
    )

    # Rewards: r_{t+1}
    if any(r != 0.0 for r in frame.rewards):
        r_parts = []
        for i, r in enumerate(frame.rewards):
            c = _agent_css_color(i)
            sign = "+" if r > 0 else ""
            r_parts.append(f'<span style="color:{c}">{sign}{r:.2f}</span>')
        parts.append(f'r<sub>{t1}</sub> = [{", ".join(r_parts)}]')
    else:
        parts.append(f'r<sub>{t1}</sub> = 0')

    # Discount: γ_{t+1}
    parts.append(f'\u03b3<sub>{t1}</sub> = {frame.discount}')

    # What happens on "next": env response or not
    agent_on_apple_after = frame.state_after.is_agent_on_apple(frame.actor)
    if not is_pick and agent_on_apple_after:
        # MOVE landed on apple — forced PICK follows with no env response
        parts.append(
            '<span style="color:#e377c2">'
            '\u2192 Next: forced PICK (no env response, \u03b3=1)'
            '</span>'
        )
    else:
        # After this transition, env responds before next frame
        parts.append(
            '<span style="color:#888">'
            '\u2192 Next: env responds (spawn/despawn + advance actor)'
            '</span>'
        )

    # Stats
    stats_line = (
        f'Apples: <span style="color:#d62728">{frame.apples_on_grid}</span> '
        f'| Picks/step: <span style="color:#2ca02c">{frame.picks_per_step:.4f}</span>'
    )
    parts.append(stats_line)

    # Per-agent picks
    if frame.agent_picks:
        n_agents = len(frame.rewards)
        pick_parts = []
        for i in range(n_agents):
            c = _agent_css_color(i)
            cnt = frame.agent_picks.get(i, 0)
            pps = frame.agent_picks_per_step(i)
            pick_parts.append(
                f'<span style="color:{c}">A{i}: {cnt} ({pps:.4f}/step)</span>'
            )
        parts.append(f'Agent picks: {" &nbsp; ".join(pick_parts)}')

    # Position table
    pos_parts = []
    for i, p in enumerate(frame.state.agent_positions):
        c = _agent_css_color(i)
        label = f"A{i}({p.row},{p.col})"
        if i == frame.actor:
            label += " [actor]"
        pos_parts.append(
            f'<span style="color:{c};font-weight:{"bold" if i == frame.actor else "normal"}">{label}</span>'
        )
    parts.append(f'Positions: {" &nbsp; ".join(pos_parts)}')

    # Decisions (Q-values)
    if frame.decisions:
        has_agent_breakdown = any(d.agent_q_values is not None for d in frame.decisions)
        dec_lines = ['<div style="margin-top:6px"><b>Q-values:</b>']
        sorted_decs = sorted(frame.decisions, key=lambda d: d.q_value, reverse=True)
        for d in sorted_decs:
            sym = _ACTION_ARROWS.get(d.action, "?")
            chosen = " \u25c4" if d.is_chosen else ""
            style = "font-weight:bold;color:#2ca02c" if d.is_chosen else "color:#555"
            dec_lines.append(
                f'<div style="{style}">&nbsp;&nbsp;{sym} {d.action.name:<5} Q={d.q_value:+.4f}{chosen}</div>'
            )
            # Per-agent breakdown (decentralized only)
            if has_agent_breakdown and d.agent_q_values:
                for i, v in sorted(d.agent_q_values.items()):
                    c = _agent_css_color(i)
                    dec_lines.append(
                        f'<div style="color:{c};font-size:11px">'
                        f'&nbsp;&nbsp;&nbsp;&nbsp;V<sub>{i}</sub>(s<sup>a</sup>) = {v:+.4f}</div>'
                    )
        dec_lines.append("</div>")
        parts.append("\n".join(dec_lines))

    # Agent values
    if frame.agent_values:
        val_lines = ['<div style="margin-top:6px"><b>V(s):</b>']
        for i, v in sorted(frame.agent_values.items()):
            c = _agent_css_color(i)
            val_lines.append(f'<div style="color:{c}">&nbsp;&nbsp;V<sub>{i}</sub>(s<sub>{t}</sub>) = {v:+.4f}</div>')
        val_lines.append("</div>")
        parts.append("\n".join(val_lines))

    return "<br>".join(parts)


def build_html(
    frames: list[Frame],
    frame_pngs: list[bytes],
    output_path: Path,
    fps: int = 3,
    compare_frames: list[Frame] | None = None,
    compare_pngs: list[bytes] | None = None,
) -> None:
    """Write a self-contained HTML file with embedded frames and slider."""
    n = len(frames)
    is_compare = compare_frames is not None and compare_pngs is not None

    b64_images = [base64.b64encode(png).decode("ascii") for png in frame_pngs]
    b64_compare = []
    if is_compare:
        b64_compare = [base64.b64encode(png).decode("ascii") for png in compare_pngs]

    info_htmls = [_build_frame_info_html(f) for f in frames]
    compare_info_htmls = []
    if is_compare:
        compare_info_htmls = [_build_frame_info_html(f) for f in compare_frames]

    apple_counts = [f.apples_on_grid for f in frames]
    compare_apple_counts = [f.apples_on_grid for f in compare_frames] if is_compare else []

    info_json = json.dumps(info_htmls)
    compare_info_json = json.dumps(compare_info_htmls) if is_compare else "[]"

    policy_name = frames[0].policy_name
    compare_policy_name = compare_frames[0].policy_name if is_compare else ""

    # Determine max slider for compare mode (may have different lengths due to picks)
    n_compare = len(compare_frames) if is_compare else 0
    max_slider = n - 1

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
  h1 {{
    font-size: 18px;
    margin-bottom: 10px;
    color: #aaa;
  }}
  .controls {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 12px 0;
    flex-wrap: wrap;
    justify-content: center;
  }}
  .controls button {{
    background: #2a2a4a;
    color: #e0e0e0;
    border: 1px solid #444;
    padding: 6px 14px;
    border-radius: 4px;
    cursor: pointer;
    font-family: inherit;
    font-size: 13px;
  }}
  .controls button:hover {{ background: #3a3a5a; }}
  #slider {{
    width: 500px;
    max-width: 80vw;
    accent-color: #7a7aaa;
  }}
  .step-label {{
    font-size: 14px;
    min-width: 180px;
    text-align: center;
  }}
  .viewer {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
  }}
  .frame-section {{
    background: #22223a;
    border-radius: 8px;
    padding: 12px;
    border: 1px solid #333;
  }}
  .frame-section h2 {{
    font-size: 13px;
    color: #888;
    margin-bottom: 8px;
  }}
  .frame-img {{
    max-width: 90vw;
    border-radius: 4px;
  }}
  .info-panel {{
    font-size: 12px;
    line-height: 1.7;
    margin-top: 8px;
    padding: 8px;
    background: #1a1a2e;
    border-radius: 4px;
  }}
  .sparkline-container {{
    width: 520px;
    max-width: 85vw;
    margin-top: 16px;
    background: #22223a;
    border-radius: 8px;
    padding: 12px;
    border: 1px solid #333;
  }}
  .sparkline-container h3 {{
    font-size: 12px;
    color: #888;
    margin-bottom: 6px;
  }}
  canvas {{
    width: 100%;
    height: 60px;
    display: block;
  }}
  .keyboard-hint {{
    font-size: 11px;
    color: #555;
    margin-top: 12px;
  }}
</style>
</head>
<body>

<h1>Orchard Trajectory Viewer</h1>

<div class="controls">
  <button id="btnPrev" title="Previous (\u2190)">\u25c0 Prev</button>
  <button id="btnPlay" title="Play/Pause (Space)">\u25b6 Play</button>
  <button id="btnNext" title="Next (\u2192)">Next \u25b6</button>
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
    <img id="primaryImg" class="frame-img" src="" alt="frame">
    <div class="info-panel" id="primaryInfo"></div>
  </div>
  {"" if not is_compare else f'''
  <div class="frame-section">
    <h2 id="compareLabel">{compare_policy_name}</h2>
    <img id="compareImg" class="frame-img" src="" alt="compare frame">
    <div class="info-panel" id="compareInfo"></div>
  </div>
  '''}
</div>

<div class="sparkline-container">
  <h3>Apple count over trajectory</h3>
  <canvas id="sparkline"></canvas>
</div>

<div class="keyboard-hint">\u2190 \u2192 step &nbsp; | &nbsp; Space play/pause</div>

<script>
const N = {n};
const N_COMPARE = {n_compare};
const FPS_DEFAULT = {fps};
const IS_COMPARE = {'true' if is_compare else 'false'};

const images = [{','.join(f'"data:image/png;base64,{b}"' for b in b64_images)}];
{"const compareImages = [" + ','.join(f'"data:image/png;base64,{b}"' for b in b64_compare) + "];" if is_compare else "const compareImages = [];"}

const infoData = {info_json};
const compareInfoData = {compare_info_json};

const appleCounts = {json.dumps(apple_counts)};
const compareAppleCounts = {json.dumps(compare_apple_counts)};

const slider = document.getElementById('slider');
const stepLabel = document.getElementById('stepLabel');
const primaryImg = document.getElementById('primaryImg');
const primaryInfo = document.getElementById('primaryInfo');
const compareImg = document.getElementById('compareImg');
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
  primaryImg.src = images[currentStep];
  primaryInfo.innerHTML = infoData[currentStep];
  if (IS_COMPARE && compareImg) {{
    const ci = Math.min(currentStep, N_COMPARE - 1);
    compareImg.src = compareImages[ci];
    compareInfo.innerHTML = compareInfoData[ci];
  }}
  drawSparkline();
}}

function togglePlay() {{
  playing = !playing;
  btnPlay.textContent = playing ? '\u23f8 Pause' : '\u25b6 Play';
  if (playing) {{
    const fps = parseInt(fpsInput.value) || FPS_DEFAULT;
    playInterval = setInterval(() => {{
      if (currentStep >= N - 1) {{
        togglePlay();
        return;
      }}
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

  const allVals = [...appleCounts, ...compareAppleCounts];
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

  drawLine(appleCounts, '#d62728', N);
  if (IS_COMPARE) drawLine(compareAppleCounts, '#1f77b4', N_COMPARE);

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
