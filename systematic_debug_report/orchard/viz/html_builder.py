"""HTML builder: assemble self-contained HTML trajectory viewer.

Frames are rendered as inline SVGs. Task squares are clickable and show
a per-agent reward breakdown popup: r_j = φ(actor,κ)·r'^(κ)_j, where the
task reward vector r'^(κ) already carries the C^(κ) mask (r'^(κ)_j = 0 for
agents that do not care about task κ). Two dropdowns in the popup: select
task type κ and select actor agent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orchard.viz.frame import Frame
from orchard.viz.renderer import TASK_TYPE_HEX, AGENT_HEX

_ACTION_ARROWS: dict[int, str] = {
    0: "↑", 1: "↓", 2: "←",
    3: "→", 4: "·",
}


def _action_symbol(action) -> str:
    if action.value in _ACTION_ARROWS:
        return _ACTION_ARROWS[action.value]
    if action.is_pick():
        return f"★κ{action.pick_type()}"
    return action.name


def _agent_css_color(idx: int) -> str:
    return AGENT_HEX[idx % len(AGENT_HEX)]


def _build_frame_info_html(frame: Frame, n_task_types: int = 1) -> str:
    parts: list[str] = []
    t = frame.state_index
    t1 = t + 1
    is_pick = frame.action.is_pick()

    parts.append(
        f'<b>s<sub>{t}</sub> → s<sub>{t}</sub><sup>a</sup></b>'
        f' &nbsp; <span style="color:#888">(decision {frame.step}, transition {frame.transition_index})</span>'
    )

    ac = _agent_css_color(frame.actor)
    arrow = _action_symbol(frame.action)
    parts.append(
        f'<span style="color:{ac};font-weight:bold">A{frame.actor}</span> '
        f'→ {frame.action.name} {arrow}'
    )

    team_r = sum(frame.rewards)
    if any(r != 0.0 for r in frame.rewards):
        r_parts = []
        for i, r in enumerate(frame.rewards):
            c = _agent_css_color(i)
            sign = "+" if r >= 0 else ""
            r_parts.append(f'<span style="color:{c}">{sign}{r:.3f}</span>')
        team_sign = "+" if team_r >= 0 else ""
        team_color = "#2ca02c" if team_r > 0 else ("#d62728" if team_r < 0 else "#888")
        parts.append(
            f'<details style="display:inline">'
            f'<summary style="cursor:pointer;display:inline">'
            f'r<sub>{t1}</sub> = <span style="color:{team_color}">{team_sign}{team_r:.3f}</span>'
            f'</summary>'
            f'<div style="padding-left:12px;margin-top:2px">[{", ".join(r_parts)}]</div>'
            f'</details>'
        )
        if frame.picked_task_type is not None:
            phi_pos = frame.picked_correct
            if phi_pos:
                parts.append(
                    f'<span style="color:#2ca02c;font-weight:bold">✓ Pick κ={frame.picked_task_type} (φ>0)</span>'
                )
            elif phi_pos is False:
                parts.append(
                    f'<span style="color:#d62728;font-weight:bold">✗ Pick κ={frame.picked_task_type} (φ=0)</span>'
                )
    else:
        parts.append(f'r<sub>{t1}</sub> = 0')

    parts.append(f'γ<sub>{t1}</sub> = {frame.discount}')

    agent_on_task_after = frame.state_after.is_agent_on_task(frame.actor)
    if not is_pick and agent_on_task_after:
        parts.append('<span style="color:#e377c2">→ Next: pick decision (γ=1)</span>')
    else:
        parts.append('<span style="color:#888">→ Next: env responds</span>')

    per_agent_detail = ""
    if frame.agent_picks:
        n_agents = len(frame.rewards)
        pick_parts = []
        for i in range(n_agents):
            c = _agent_css_color(i)
            cnt = frame.agent_picks.get(i, 0)
            pps = frame.agent_picks_per_step(i)
            pick_parts.append(f'<span style="color:{c}">A{i}: {cnt} ({pps:.3f}/step)</span>')
        per_agent_detail = (
            f'<div style="margin-top:4px;padding-left:8px;line-height:1.8">'
            + " &nbsp; ".join(pick_parts)
            + "</div>"
        )

    rps_html = (
        f'<details style="display:inline">'
        f'<summary style="cursor:pointer;display:inline;color:#2ca02c">'
        f'Team RPS: {frame.team_reward_per_step:.4f}</summary>'
        f'{per_agent_detail}'
        f'</details>'
    )

    stats_parts = [
        f'Tasks: <span style="color:#d62728">{frame.tasks_on_grid}</span>',
        rps_html,
    ]
    parts.append(" | ".join(stats_parts))

    if frame.state.task_types is not None:
        type_counts: dict[int, int] = {}
        for tt in frame.state.task_types:
            type_counts[tt] = type_counts.get(tt, 0) + 1
        type_str = " ".join(f'κ{k}:{v}' for k, v in sorted(type_counts.items()))
        parts.append(f'<span style="color:#888">On grid: {type_str}</span>')

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
                    c = _agent_css_color(i)
                    dec_lines.append(
                        f'<div style="color:{c};font-size:11px">'
                        f'&nbsp;&nbsp;&nbsp;&nbsp;V<sub>{i}</sub>(s<sup>a</sup>) = {v:+.4f}</div>'
                    )
        dec_lines.append("</div>")
        parts.append("\n".join(dec_lines))

    if frame.agent_values:
        val_lines = ['<div style="margin-top:6px"><b>V(s):</b>']
        for i, v in sorted(frame.agent_values.items()):
            c = _agent_css_color(i)
            val_lines.append(
                f'<div style="color:{c}">&nbsp;&nbsp;V<sub>{i}</sub>(s<sub>{t}</sub>) = {v:+.4f}</div>'
            )
        val_lines.append("</div>")
        parts.append("\n".join(val_lines))

    return "<br>".join(parts)


def _build_legend_html(
    n_task_types: int,
    n_agents: int,
    proficiency: np.ndarray,
    relatedness_width: int,
    proficiency_width: int,
) -> str:
    """Legend showing task types and proficiency/relatedness structure."""
    pills: list[str] = []
    max_show = 16  # show at most 16 type pills before truncating

    for tau in range(min(n_task_types, max_show)):
        hex_c = TASK_TYPE_HEX[tau % len(TASK_TYPE_HEX)]
        swatch = (
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{hex_c};border-radius:2px;flex-shrink:0;vertical-align:middle"></span>'
        )
        # Show which agents are proficient in this type (proficiency > 0)
        specialists = [i for i in range(n_agents) if proficiency[i, tau] > 0]
        if len(specialists) > 4:
            ag_str = f"A{specialists[0]}–A{specialists[-1]}"
        elif specialists:
            ag_str = ", ".join(f"A{a}" for a in specialists)
        else:
            ag_str = "none"

        pill_content = f"{swatch} κ{tau}"
        detail = (
            f'<div style="position:absolute;z-index:10;font-size:10px;color:#ccc;'
            f'padding:5px 8px;background:#1a1a2e;border:1px solid #555;'
            f'border-radius:4px;margin-top:2px;white-space:nowrap">{ag_str}</div>'
        )
        summary = (
            f'<summary style="cursor:pointer;list-style:none;display:inline-flex;'
            f'align-items:center;gap:4px;font-size:11px;color:#ccc;'
            f'padding:3px 7px;background:#2a2a4a;border:1px solid #444;'
            f'border-radius:4px;user-select:none">{pill_content}</summary>'
        )
        pills.append(
            f'<details style="display:inline-block;position:relative;margin:2px 3px">'
            f'{summary}{detail}</details>'
        )

    if n_task_types > max_show:
        pills.append(
            f'<span style="font-size:11px;color:#888;padding:3px 7px">…+{n_task_types-max_show} more</span>'
        )

    items = "".join(pills)
    params_str = f"w_R={relatedness_width}&nbsp;&nbsp;&nbsp;w_P={proficiency_width}"
    return (
        f'<div style="background:#22223a;border:1px solid #333;border-radius:8px;'
        f'padding:8px 12px;margin-top:12px;width:min(90vw,800px);overflow-x:hidden">'
        f'<div style="font-size:10px;color:#888;font-weight:bold;margin-bottom:4px">'
        f'Task types — click to see specialist agents &nbsp;|&nbsp; {params_str}</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:3px">{items}</div>'
        f'</div>'
    )


def _enc_labels(encoder_type, T: int, N: int,
                relatedness_width: int = 0, proficiency_width: int = 0,
                ) -> tuple[list[str], list[str]]:
    """Return (channel_labels, scalar_labels) for the given encoder type."""
    from orchard.enums import EncoderType
    if encoder_type == EncoderType.FILTERED_DEC_CNN_GRID:
        # FilteredDecEncoder masks to R_i (KR task channels) and W_i (KW agent
        # channels). The real task type / agent id behind each channel depends on
        # the selected agent i, so the viewer relabels per-agent in JS
        # (encLabelsForAgent). These generic labels are only a non-dec fallback.
        KR = min(N, 2 * relatedness_width + 1)
        KW = min(N, 2 * (relatedness_width + proficiency_width) + 1)
        ch = [f"task (R_i) ch{t}" for t in range(KR)] + [f"agent (W_i) ch{s}" for s in range(KW)] + ["actor pos"]
        sc = [f"actor-local ch{s}" for s in range(KW)] + ["pick_phase"]
    else:  # EVERYTHING_CNN_GRID
        ch = [f"task κ{k} present" for k in range(T)] + [f"agent {j} pos" for j in range(N)] + ["actor pos"]
        sc = [f"actor={j}" for j in range(N)] + ["pick_phase"]
    return ch, sc


def build_html(
    frames: list[Frame],
    frame_svgs: list[str],
    output_path: Path,
    fps: int = 3,
    compare_frames: list[Frame] | None = None,
    compare_svgs: list[str] | None = None,
    n_task_types: int = 1,
    task_assignments: tuple | None = None,   # ignored, kept for compat
    pick_mode=None,                          # ignored, kept for compat
    proficiency: np.ndarray | None = None,
    relatedness: np.ndarray | None = None,
    category_rewards: np.ndarray | None = None,
    relatedness_width: int = 0,
    proficiency_width: int = 0,
    encoder_type=None,
    n_agents: int = 1,
) -> None:
    """Write a self-contained HTML trajectory viewer with interactive reward popup."""
    n = len(frames)
    is_compare = compare_frames is not None and compare_svgs is not None

    n_agents = len(frames[0].rewards)

    info_htmls = [_build_frame_info_html(f, n_task_types) for f in frames]
    compare_info_htmls: list[str] = []
    if is_compare and compare_frames:
        compare_info_htmls = [_build_frame_info_html(f, n_task_types) for f in compare_frames]

    task_counts = [f.tasks_on_grid for f in frames]
    compare_task_counts = [f.tasks_on_grid for f in compare_frames] if is_compare else []
    actor_per_frame = [f.actor for f in frames]

    svgs_json = json.dumps(frame_svgs)
    compare_svgs_json = json.dumps(compare_svgs) if is_compare else "[]"
    info_json = json.dumps(info_htmls)
    compare_info_json = json.dumps(compare_info_htmls) if is_compare else "[]"

    policy_name = frames[0].policy_name
    compare_policy_name = compare_frames[0].policy_name if is_compare and compare_frames else ""
    n_compare = len(compare_frames) if is_compare and compare_frames else 0
    max_slider = n - 1

    # Embed proficiency/rel/cr as JS arrays (round to 4 decimal places to keep file size down)
    def _mat_to_js(m: np.ndarray | None, default_val: float = 0.0) -> str:
        if m is None:
            return "null"
        return json.dumps([[round(float(v), 4) for v in row] for row in m])

    proficiency_js = _mat_to_js(proficiency)
    cr_js = _mat_to_js(category_rewards)  # shape (T, N); already C^(κ)-masked

    legend_html = _build_legend_html(
        n_task_types, n_agents,
        proficiency if proficiency is not None else np.zeros((n_agents, n_task_types)),
        relatedness_width, proficiency_width,
    )

    task_type_colors_js = json.dumps(TASK_TYPE_HEX)
    agent_colors_js = json.dumps(AGENT_HEX)
    actor_per_frame_js = json.dumps(actor_per_frame)

    # --- Encoding data (only if --show-encoding was used) ---
    enc_data_js = "null"
    enc_scalars_js = "null"
    enc_channel_labels_js = "null"
    enc_scalar_labels_js = "null"
    is_dec_js = "false"
    # FilteredDecEncoder channels/scalars index into agent i's windows R_i / W_i, so
    # the real task type / agent id each channel shows depends on which agent is
    # selected. These let the JS relabel per-agent; for EverythingEncoder they stay 0.
    enc_is_filtered_js = "false"
    enc_kr_js, enc_kw_js = "0", "0"
    enc_rel_w_js, enc_prof_w_js = "0", "0"
    show_encoding = encoder_type is not None and frames[0].encoding_grids is not None
    if show_encoding:
        from orchard.enums import EncoderType
        # IS_DEC = multiple per-agent encodings were stored (n_networks > 1),
        # regardless of encoder type. EverythingEncoder with dec training also
        # stores N separate grids (one per network), so the dropdown is shown.
        is_dec = len(frames[0].encoding_grids) > 1
        is_dec_js = "true" if is_dec else "false"
        ch_labels, sc_labels = _enc_labels(
            encoder_type, n_task_types, n_agents,
            relatedness_width, proficiency_width,
        )
        enc_channel_labels_js = json.dumps(ch_labels)
        enc_scalar_labels_js = json.dumps(sc_labels)
        if encoder_type == EncoderType.FILTERED_DEC_CNN_GRID:
            enc_is_filtered_js = "true"
            enc_kr_js = str(min(n_agents, 2 * relatedness_width + 1))
            enc_kw_js = str(min(n_agents, 2 * (relatedness_width + proficiency_width) + 1))
            enc_rel_w_js = str(relatedness_width)
            enc_prof_w_js = str(proficiency_width)

        def _round_enc_frame(frame: Frame):
            if frame.encoding_grids is None:
                return None, None
            grids_out = [
                [[[round(float(v), 4) for v in row] for row in ch] for ch in net_g]
                for net_g in frame.encoding_grids
            ]
            scalars_out = [
                [round(float(v), 4) for v in net_s]
                for net_s in frame.encoding_scalars
            ]
            return grids_out, scalars_out

        all_g, all_s = [], []
        for f in frames:
            g, s = _round_enc_frame(f)
            all_g.append(g)
            all_s.append(s)
        enc_data_js = json.dumps(all_g)
        enc_scalars_js = json.dumps(all_s)

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
  .frame-svg svg {{ display: block; max-width: 100%; height: auto; cursor: default; }}
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

  /* Task reward popup — interactive, not pointer-events:none */
  #taskPopup {{
    position: fixed;
    display: none;
    background: #1e1e3a;
    color: #e0e0e0;
    border: 1px solid #666;
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 12px;
    line-height: 1.6;
    width: 320px;
    max-width: 90vw;
    max-height: 60vh;
    overflow-y: auto;
    z-index: 1000;
    box-shadow: 0 6px 24px rgba(0,0,0,0.7);
  }}
  #taskPopup h3 {{ font-size: 13px; margin-bottom: 8px; color: #aaa; }}
  #taskPopup .popup-controls {{ display: flex; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }}
  #taskPopup label {{ font-size: 11px; color: #888; }}
  #taskPopup select {{
    background: #2a2a4a; color: #e0e0e0; border: 1px solid #555;
    border-radius: 4px; padding: 2px 6px; font-size: 11px; font-family: inherit;
  }}
  #taskPopup .reward-table {{ width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 6px; }}
  #taskPopup .reward-table th {{
    text-align: left; color: #888; padding: 2px 6px;
    border-bottom: 1px solid #444;
  }}
  #taskPopup .reward-table td {{ padding: 2px 6px; }}
  #taskPopup .team-total {{ margin-top: 6px; font-weight: bold; font-size: 12px; }}
  #taskPopup .phi-info {{ margin-top: 4px; color: #aaa; font-size: 11px; }}
  #taskPopup .close-btn {{
    float: right; cursor: pointer; color: #888; font-size: 14px;
    margin: -4px -4px 0 0; line-height: 1;
  }}
  #taskPopup .close-btn:hover {{ color: #e0e0e0; }}

  /* General element popup (agents) */
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
    max-width: 260px;
    z-index: 999;
    pointer-events: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5);
  }}

  /* Encoding panel */
  #encPanel {{
    background: #22223a; border: 1px solid #333; border-radius: 8px;
    padding: 10px 12px; margin-top: 8px; width: min(90vw, 820px);
  }}
  #encPanel .enc-controls {{
    display: flex; gap: 14px; align-items: center; flex-wrap: wrap; margin-bottom: 8px;
  }}
  #encPanel label {{ font-size: 11px; color: #888; }}
  #encPanel select {{
    background: #2a2a4a; color: #e0e0e0; border: 1px solid #555;
    border-radius: 4px; padding: 2px 6px; font-size: 11px; font-family: inherit;
  }}
  #encPanelContent {{ font-size: 11px; }}
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

{'<div id="encPanel"><div style="font-size:12px;font-weight:bold;color:#aaa;margin-bottom:6px">Encoding</div><div class="enc-controls"><div><label>Channel: </label><select id="encChannelSel" onchange="updateEncoding(currentStep)"></select></div></div><div id="encPanelContent"><span style="color:#555;font-size:11px">Navigate to a frame to see encoding.</span></div></div>' if show_encoding else ''}

<div class="sparkline-container">
  <h3>Task count over trajectory</h3>
  <canvas id="sparkline"></canvas>
</div>

<div class="keyboard-hint">← → step &nbsp;|&nbsp; Space play/pause &nbsp;|&nbsp; click task squares for reward breakdown &nbsp;|&nbsp; click agents for info</div>

<!-- Interactive task reward popup -->
<div id="taskPopup">
  <span class="close-btn" id="taskPopupClose">✕</span>
  <h3 id="taskPopupTitle">Task at (r, c)</h3>
  <div class="popup-controls">
    <div>
      <label>Type κ: </label>
      <select id="popupTypeSelect" onchange="updateTaskPopupTable()"></select>
    </div>
    <div>
      <label>Actor: </label>
      <select id="popupActorSelect" onchange="updateTaskPopupTable()"></select>
    </div>
  </div>
  <div class="phi-info" id="taskPopupPhiInfo"></div>
  <table class="reward-table" id="taskPopupTable">
    <thead>
      <tr>
        <th>Agent j</th>
        <th>cares? (j∈C<sup>κ</sup>)</th>
        <th>r&#39;[κ,j]</th>
        <th>reward r_j</th>
      </tr>
    </thead>
    <tbody id="taskPopupBody"></tbody>
  </table>
  <div class="team-total" id="taskPopupTeam"></div>
</div>

<!-- Agent info popup (hover-like, non-interactive) -->
<div id="clickPopup"></div>

<script>
const N = {n};
const N_COMPARE = {n_compare};
const N_AGENTS = {n_agents};
const N_TASK_TYPES = {n_task_types};
const FPS_DEFAULT = {fps};
const IS_COMPARE = {'true' if is_compare else 'false'};

const svgFrames = {svgs_json};
const compareSvgFrames = {compare_svgs_json};
const infoData = {info_json};
const compareInfoData = {compare_info_json};
const taskCounts = {json.dumps(task_counts)};
const compareTaskCounts = {json.dumps(compare_task_counts)};
const actorPerFrame = {actor_per_frame_js};

const TASK_TYPE_COLORS = {task_type_colors_js};
const AGENT_COLORS = {agent_colors_js};

// proficiency[i][kappa], cr[kappa][j] (cr already carries the C^(κ) mask)
const PHI = {proficiency_js};   // (N_AGENTS x N_TASK_TYPES) or null
const CR  = {cr_js};    // (N_TASK_TYPES x N_AGENTS) or null

// ---- Task reward popup ----
const taskPopup = document.getElementById('taskPopup');
const clickPopup = document.getElementById('clickPopup');
let currentTaskData = null;

document.getElementById('taskPopupClose').addEventListener('click', () => {{
  taskPopup.style.display = 'none';
}});

function showTaskPopup(event, data) {{
  // data: {{row, col, types, focus?}}
  currentTaskData = data;
  document.getElementById('taskPopupTitle').textContent =
    `Tasks at (${{data.row}}, ${{data.col}})`;

  // Populate type dropdown
  const typeEl = document.getElementById('popupTypeSelect');
  typeEl.innerHTML = '';
  data.types.forEach(tau => {{
    const opt = document.createElement('option');
    opt.value = tau;
    opt.textContent = `κ=${{tau}}`;
    if (data.focus !== undefined && tau === data.focus) opt.selected = true;
    typeEl.appendChild(opt);
  }});

  // Populate actor dropdown — default to current frame actor
  const actorEl = document.getElementById('popupActorSelect');
  actorEl.innerHTML = '';
  const defaultActor = actorPerFrame[currentStep];
  for (let a = 0; a < N_AGENTS; a++) {{
    const opt = document.createElement('option');
    opt.value = a;
    opt.textContent = `A${{a}}`;
    if (a === defaultActor) opt.selected = true;
    actorEl.appendChild(opt);
  }}

  updateTaskPopupTable();

  // Position popup near click, keep in viewport
  taskPopup.style.display = 'block';
  const pw = taskPopup.offsetWidth || 320;
  const ph = taskPopup.offsetHeight || 300;
  let px = event.clientX + 16;
  let py = event.clientY + 8;
  if (px + pw > window.innerWidth - 8) px = event.clientX - pw - 8;
  if (py + ph > window.innerHeight - 8) py = event.clientY - ph - 8;
  if (py < 8) py = 8;
  taskPopup.style.left = px + 'px';
  taskPopup.style.top = py + 'px';
}}

function updateTaskPopupTable() {{
  if (!currentTaskData) return;
  const tau = parseInt(document.getElementById('popupTypeSelect').value);
  const actor = parseInt(document.getElementById('popupActorSelect').value);

  const phi_val = PHI ? PHI[actor][tau] : 1.0;
  document.getElementById('taskPopupPhiInfo').textContent =
    `φ(A${{actor}}, κ${{tau}}) = ${{phi_val.toFixed(3)}}`;

  const tc = TASK_TYPE_COLORS[tau % TASK_TYPE_COLORS.length];
  const tbody = document.getElementById('taskPopupBody');
  tbody.innerHTML = '';
  let team_total = 0;

  for (let j = 0; j < N_AGENTS; j++) {{
    // r'^(κ)_j already carries the C^(κ) mask: it is 0 iff agent j does not care
    // about task κ. The pick reward is simply φ(actor,κ)·r'^(κ)_j (no relatedness
    // multiply — that masking lives inside r').
    const rp_val = CR ? CR[tau][j] : (1.0 / N_AGENTS);
    const cares = Math.abs(rp_val) > 1e-9;
    const reward = phi_val * rp_val;
    team_total += reward;

    const agent_color = AGENT_COLORS[j % AGENT_COLORS.length];
    const zero = Math.abs(reward) < 1e-6;
    const reward_color = zero ? '#555' : (reward > 0 ? '#2ca02c' : '#d62728');
    const row_style = zero ? 'color:#555' : '';

    const tr = document.createElement('tr');
    tr.style.cssText = row_style;
    tr.innerHTML =
      `<td style="color:${{zero ? '#555' : agent_color}};padding:2px 6px">A${{j}}</td>` +
      `<td style="padding:2px 6px">${{cares ? '✓' : '·'}}</td>` +
      `<td style="padding:2px 6px">${{rp_val.toFixed(4)}}</td>` +
      `<td style="color:${{reward_color}};padding:2px 6px;font-weight:${{zero?'normal':'bold'}}">` +
      `${{reward >= 0 ? '+' : ''}}${{reward.toFixed(4)}}</td>`;
    tbody.appendChild(tr);
  }}

  const team_color = team_total > 0.001 ? '#2ca02c' : (team_total < -0.001 ? '#d62728' : '#888');
  document.getElementById('taskPopupTeam').innerHTML =
    `Team total: <span style="color:${{team_color}}">${{team_total >= 0 ? '+' : ''}}${{team_total.toFixed(4)}}</span>`;
}}

// ---- Agent hover popup ----
function showAgentPopup(event, data) {{
  const count = data.agents.length;
  let html = `<b>${{count > 1 ? count + ' agents' : 'Agent'}} on cell:</b><br>`;
  html += data.agents.map(a => {{
    const c = AGENT_COLORS[a % AGENT_COLORS.length];
    return `<span style="display:inline-flex;align-items:center;gap:5px;margin:1px 0">` +
           `<span style="width:12px;height:12px;background:${{c}};border-radius:50%;display:inline-block;flex-shrink:0"></span>` +
           `A${{a}}</span>`;
  }}).join('<br>');
  clickPopup.innerHTML = html;
  clickPopup.style.display = 'block';
  const pw = 200, ph = 30 + data.agents.length * 22;
  let px = event.clientX + 14;
  let py = event.clientY + 14;
  if (px + pw > window.innerWidth) px = event.clientX - pw - 8;
  if (py + ph > window.innerHeight) py = event.clientY - ph - 8;
  clickPopup.style.left = px + 'px';
  clickPopup.style.top = py + 'px';
}}

function handleSvgClick(event) {{
  const group = event.target.closest('[data-popup-type]');
  if (!group) {{
    clickPopup.style.display = 'none';
    return;
  }}
  event.stopPropagation();
  const type = group.dataset.popupType;
  const data = JSON.parse(group.dataset.popupData);
  if (type === 'tasks') {{
    showTaskPopup(event, data);
  }} else if (type === 'agents') {{
    showAgentPopup(event, data);
  }}
}}

// Close agent popup on any click outside
document.addEventListener('click', (e) => {{
  if (!e.target.closest('[data-popup-type]') && !e.target.closest('#taskPopup')) {{
    clickPopup.style.display = 'none';
  }}
}});

// Attach to SVG containers
document.getElementById('primarySvg').addEventListener('click', handleSvgClick);
{'document.getElementById("compareSvg") && document.getElementById("compareSvg").addEventListener("click", handleSvgClick);' if is_compare else ''}

// ---- Playback ----
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
  // Re-attach click handler after innerHTML replacement
  primarySvg.addEventListener('click', handleSvgClick);
  drawSparkline();
  if (ENC_DATA) updateEncoding(currentStep);
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
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft') {{ e.preventDefault(); showStep(currentStep - 1); }}
  else if (e.key === 'ArrowRight') {{ e.preventDefault(); showStep(currentStep + 1); }}
  else if (e.key === ' ') {{ e.preventDefault(); togglePlay(); }}
}});

// ---- Encoding panel ----
const ENC_DATA = {enc_data_js};
const ENC_SCALARS = {enc_scalars_js};
const ENC_CHANNEL_LABELS = {enc_channel_labels_js};
const ENC_SCALAR_LABELS = {enc_scalar_labels_js};
const IS_DEC = {is_dec_js};
// FilteredDecEncoder: channel c maps to a real task type / agent id that depends on
// the selected agent i (circular windows). When ENC_IS_FILTERED we relabel per-agent.
const ENC_IS_FILTERED = {enc_is_filtered_js};
const ENC_KR = {enc_kr_js};
const ENC_KW = {enc_kw_js};
const ENC_REL_W = {enc_rel_w_js};
const ENC_PROF_W = {enc_prof_w_js};

// Real labels for agent i's encoding (filtered: ids resolved through R_i / W_i;
// everything: the static global labels).
function encLabelsForAgent(agent) {{
  if (!ENC_IS_FILTERED) return ENC_CHANNEL_LABELS;
  const N = N_AGENTS;
  const ch = [];
  for (let c = 0; c < ENC_KR; c++) {{
    const tau = (((agent - ENC_REL_W + c) % N) + N) % N;
    ch.push(`task κ${{tau}} (R_i ch${{c}})`);
  }}
  for (let c = 0; c < ENC_KW; c++) {{
    const j = (((agent - (ENC_REL_W + ENC_PROF_W) + c) % N) + N) % N;
    ch.push(`agent ${{j}} pos (W_i ch${{c}})`);
  }}
  ch.push('actor pos');
  return ch;
}}

function encScalarLabelsForAgent(agent) {{
  if (!ENC_IS_FILTERED) return ENC_SCALAR_LABELS;
  const N = N_AGENTS;
  const sc = [];
  for (let c = 0; c < ENC_KW; c++) {{
    const j = (((agent - (ENC_REL_W + ENC_PROF_W) + c) % N) + N) % N;
    sc.push(`actor=${{j}}? (W_i ch${{c}})`);
  }}
  sc.push('pick_phase');
  return sc;
}}

// (Re)populate the channel dropdown with labels for the given agent, preserving
// the current selection. For the filtered encoder the labels resolve to real
// task type / agent ids through agent i's windows, so they change per agent.
function populateChannelDropdown(agent) {{
  const chSel = document.getElementById('encChannelSel');
  if (!chSel) return;
  const prev = chSel.value || 'all';
  const labels = encLabelsForAgent(agent);
  chSel.innerHTML = '';
  const allOpt = document.createElement('option');
  allOpt.value = 'all';
  allOpt.textContent = 'All channels';
  chSel.appendChild(allOpt);
  labels.forEach((label, i) => {{
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = `${{i}}: ${{label}}`;
    chSel.appendChild(opt);
  }});
  chSel.value = prev;
  if (chSel.selectedIndex < 0) chSel.value = 'all';
}}

(function initEncPanel() {{
  const chSel = document.getElementById('encChannelSel');
  if (!chSel || !ENC_CHANNEL_LABELS) return;

  populateChannelDropdown(0);

  // Agent dropdown (only for dec)
  if (IS_DEC) {{
    const ctrl = chSel.closest('.enc-controls');
    const agDiv = document.createElement('div');
    agDiv.innerHTML = '<label>Agent: </label>';
    const agSel = document.createElement('select');
    agSel.id = 'encAgentSel';
    agSel.onchange = () => {{ populateChannelDropdown(parseInt(agSel.value)); updateEncoding(currentStep); }};
    for (let i = 0; i < N_AGENTS; i++) {{
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = `A${{i}}`;
      agSel.appendChild(opt);
    }}
    agDiv.appendChild(agSel);
    ctrl.insertBefore(agDiv, ctrl.firstChild);
  }}
}})();

function lerpColor(a, b, t) {{
  const ah = parseInt(a.slice(1), 16);
  const bh = parseInt(b.slice(1), 16);
  const ar = (ah >> 16) & 0xff, ag = (ah >> 8) & 0xff, ab = ah & 0xff;
  const br = (bh >> 16) & 0xff, bg = (bh >> 8) & 0xff, bb = bh & 0xff;
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const b2 = Math.round(ab + (bb - ab) * t);
  return '#' + [r, g, b2].map(x => x.toString(16).padStart(2, '0')).join('');
}}

function renderChannelSmall(grid2d, label) {{
  const H = grid2d.length, W = grid2d[0].length;
  const SZ = Math.max(8, Math.min(14, Math.floor(110 / W)));
  const flat = grid2d.flat();
  const maxVal = Math.max(...flat, 1e-9);
  let cells = '';
  for (let r = 0; r < H; r++) {{
    for (let c = 0; c < W; c++) {{
      const v = grid2d[r][c] / maxVal;
      const col = lerpColor('#ffffff', '#e06010', v);
      cells += `<rect x="${{c*SZ}}" y="${{r*SZ}}" width="${{SZ}}" height="${{SZ}}" fill="${{col}}" stroke="#999" stroke-width="0.3"/>`;
      if (grid2d[r][c] > 0.001 && SZ >= 12) {{
        cells += `<text x="${{c*SZ+SZ/2}}" y="${{r*SZ+SZ*0.72}}" font-size="7" text-anchor="middle" fill="#333">${{grid2d[r][c].toFixed(2)}}</text>`;
      }}
    }}
  }}
  return `<div style="display:inline-block;margin:3px;vertical-align:top;text-align:center">` +
    `<div style="font-size:9px;color:#888;margin-bottom:2px;max-width:${{W*SZ}}px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${{label}}">${{label}}</div>` +
    `<svg width="${{W*SZ}}" height="${{H*SZ}}">${{cells}}</svg></div>`;
}}

function renderChannelLarge(grid2d, label) {{
  const H = grid2d.length, W = grid2d[0].length;
  const SZ = 20;
  const flat = grid2d.flat();
  const maxVal = Math.max(...flat, 1e-9);
  let cells = '';
  for (let r = 0; r < H; r++) {{
    for (let c = 0; c < W; c++) {{
      const v = grid2d[r][c] / maxVal;
      const col = lerpColor('#ffffff', '#e06010', v);
      cells += `<rect x="${{c*SZ}}" y="${{r*SZ}}" width="${{SZ}}" height="${{SZ}}" fill="${{col}}" stroke="#999" stroke-width="0.5"/>`;
      if (grid2d[r][c] !== 0) {{
        cells += `<text x="${{c*SZ+SZ/2}}" y="${{r*SZ+SZ*0.67}}" font-size="8" text-anchor="middle" fill="#111">${{grid2d[r][c].toFixed(3)}}</text>`;
      }}
    }}
  }}
  return `<div style="text-align:center;margin:4px 0">` +
    `<div style="font-size:11px;color:#aaa;margin-bottom:4px">${{label}}</div>` +
    `<svg width="${{W*SZ}}" height="${{H*SZ}}">${{cells}}</svg></div>`;
}}

function updateEncoding(frameIdx) {{
  if (!ENC_DATA || !ENC_DATA[frameIdx]) return;
  const agentSel = document.getElementById('encAgentSel');
  const agent = (IS_DEC && agentSel) ? parseInt(agentSel.value) : 0;
  const chSel = document.getElementById('encChannelSel');
  const chVal = chSel ? chSel.value : 'all';
  const grids = ENC_DATA[frameIdx][agent];
  const scalars = ENC_SCALARS[frameIdx][agent];
  const chLabels = encLabelsForAgent(agent);
  const scLabels = encScalarLabelsForAgent(agent);

  let html = '';
  if (chVal === 'all') {{
    html += '<div style="display:flex;flex-wrap:wrap;gap:4px;max-height:320px;overflow-y:auto;padding:4px;background:#111128;border-radius:4px">';
    for (let ch = 0; ch < grids.length; ch++) {{
      html += renderChannelSmall(grids[ch], chLabels[ch]);
    }}
    html += '</div>';
  }} else {{
    html += renderChannelLarge(grids[parseInt(chVal)], chLabels[parseInt(chVal)]);
  }}

  html += '<div style="margin-top:8px"><span style="font-size:11px;font-weight:bold;color:#aaa">Scalars</span>';
  html += '<table style="font-size:11px;margin-top:3px;border-collapse:collapse"><tbody>';
  for (let s = 0; s < scalars.length; s++) {{
    const v = scalars[s];
    const style = Math.abs(v) > 0.01 ? 'font-weight:bold;color:#e0c060' : 'color:#888';
    html += `<tr><td style="color:#666;padding:1px 8px 1px 0">${{scLabels[s]}}</td>` +
            `<td style="${{style}}">${{v.toFixed(4)}}</td></tr>`;
  }}
  html += '</tbody></table></div>';

  document.getElementById('encPanelContent').innerHTML = html;
}}

showStep(0);
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
