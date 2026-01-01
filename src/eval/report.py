from __future__ import annotations

import html
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _escape(s: Any) -> str:
    return html.escape(str(s), quote=False)


def _jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    return x


def _bar(before: Optional[float], after: Optional[float], *, max_value: float) -> str:
    """A tiny HTML bar pair for before/after."""
    if max_value <= 0:
        max_value = 1.0
    b = (float(before) / max_value * 100.0) if before is not None else 0.0
    a = (float(after) / max_value * 100.0) if after is not None else 0.0
    return (
        f"<div class='barwrap'>"
        f"<div class='bar before' style='width:{b:.1f}%'></div>"
        f"<div class='bar after' style='width:{a:.1f}%'></div>"
        f"</div>"
    )


def render_benchmark_report(results: Dict[str, Any]) -> str:
    """Render a single self-contained HTML report."""
    modes: List[str] = list(results.get('modes') or [])
    tasks: List[Dict[str, Any]] = list(results.get('tasks') or [])
    runs: Dict[str, Any] = dict(results.get('runs') or {})

    # Choose a stable list of metrics to visualize.
    # We'll scale bars per-metric across the entire report.
    metric_values: Dict[str, float] = {}

    def upd(name: str, v: Optional[float]) -> None:
        if v is None:
            return
        metric_values[name] = max(metric_values.get(name, 0.0), float(v))

    for mode in modes:
        for t in tasks:
            r = (runs.get(mode) or {}).get(t['name']) or {}
            pre_m = (r.get('pre_focus_method') or {})
            post_m = (r.get('post_focus_method') or {})
            for k in ('loc', 'loc_non_empty', 'cyclomatic'):
                upd(f"focus_{k}", pre_m.get(k))
                upd(f"focus_{k}", post_m.get(k))
            pre_pr = (r.get('pre_change_risk') or {})
            post_pr = (r.get('post_change_risk') or {})
            for k in ('fan_in', 'fan_out', 'upstream_depth_max', 'downstream_depth_max'):
                upd(f"risk_{k}", pre_pr.get(k))
                upd(f"risk_{k}", post_pr.get(k))

    css = """
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; margin:0; background:#fafafa; color:#111;}
    header{padding:16px 20px; background:#111; color:#fff;}
    h1{margin:0; font-size:18px;}
    .sub{opacity:.85; font-size:12px; margin-top:4px;}
    .wrap{padding:16px 20px;}
    .card{background:#fff; border:1px solid #e6e6e6; border-radius:10px; padding:14px 14px; margin-bottom:14px;}
    table{width:100%; border-collapse:collapse; font-size:13px;}
    th, td{border-bottom:1px solid #eee; padding:8px 6px; vertical-align:top;}
    th{text-align:left; background:#fcfcfc;}
    code{background:#f2f2f2; padding:1px 4px; border-radius:4px;}
    .pill{display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:#f2f2f2;}
    .pill.ok{background:#e7f7ee;}
    .pill.fail{background:#fde8e8;}
    .barwrap{position:relative; height:10px; background:#f3f3f3; border-radius:999px; overflow:hidden; width:140px;}
    .bar{position:absolute; top:0; bottom:0;}
    .bar.before{left:0; opacity:0.5;}
    .bar.after{left:0; opacity:0.9;}
    .muted{color:#666;}
    details summary{cursor:pointer;}
    """

    def task_title(t: Dict[str, Any]) -> str:
        desc = t.get('description') or ''
        return f"<b>{_escape(t.get('name'))}</b> <span class='muted'>{_escape(desc)}</span>"

    out: List[str] = []
    out.append("<html><head><meta charset='utf-8'/>")
    out.append("<meta name='viewport' content='width=device-width, initial-scale=1'/>")
    out.append(f"<style>{css}</style></head><body>")
    out.append("<header>")
    out.append("<h1>Refactoring Benchmark Report</h1>")
    out.append(
        f"<div class='sub'>Project: <code>{_escape(results.get('project_root'))}</code> &nbsp; Generated: {_escape(results.get('generated_at'))}</div>"
    )
    out.append("</header>")
    out.append("<div class='wrap'>")

    # One card per task (rows = modes)
    for t in tasks:
        out.append("<div class='card'>")
        out.append(f"<div style='margin-bottom:8px'>{task_title(t)}</div>")
        out.append("<table>")
        out.append(
            "<tr><th style='width:120px'>Mode</th><th>Status</th><th>Context coverage</th><th>Focus maintainability</th><th>Change risk</th><th>Changed files</th></tr>"
        )

        for mode in modes:
            r = (runs.get(mode) or {}).get(t['name']) or {}
            status = str(r.get('status') or '')
            pill = "pill ok" if status == 'success' else "pill fail"
            attempts = r.get('attempts')
            cov = (r.get('context_coverage') or {}).get('coverage')
            cov_s = f"{float(cov)*100:.0f}%" if cov is not None else "n/a"

            pre_m = r.get('pre_focus_method') or {}
            post_m = r.get('post_focus_method') or {}
            pre_cc = pre_m.get('cyclomatic')
            post_cc = post_m.get('cyclomatic')
            pre_loc = pre_m.get('loc')
            post_loc = post_m.get('loc')

            pre_r = r.get('pre_change_risk') or {}
            post_r = r.get('post_change_risk') or {}

            cc_bar = _bar(pre_cc, post_cc, max_value=metric_values.get('focus_cyclomatic', 1.0))
            loc_bar = _bar(pre_loc, post_loc, max_value=metric_values.get('focus_loc', 1.0))

            risk_bar = _bar(
                pre_r.get('fan_in'),
                post_r.get('fan_in'),
                max_value=metric_values.get('risk_fan_in', 1.0),
            )

            changed = r.get('modified_files') or []
            changed_s = "<br/>".join(f"<code>{_escape(x)}</code>" for x in changed[:8])
            if len(changed) > 8:
                changed_s += f"<div class='muted'>... +{len(changed)-8} more</div>"

            out.append("<tr>")
            out.append(f"<td><code>{_escape(mode)}</code></td>")
            out.append(
                f"<td><span class='{pill}'>{_escape(status)}</span> <span class='muted'>(attempts={_escape(attempts)})</span></td>"
            )
            out.append(f"<td>{_escape(cov_s)}</td>")
            out.append(
                "<td>"
                f"<div class='muted'>CC: { _escape(pre_cc) } → { _escape(post_cc) }</div>{cc_bar}"
                f"<div class='muted' style='margin-top:6px'>LOC: { _escape(pre_loc) } → { _escape(post_loc) }</div>{loc_bar}"
                "</td>"
            )
            out.append(
                "<td>"
                f"<div class='muted'>fan-in: { _escape(pre_r.get('fan_in')) } → { _escape(post_r.get('fan_in')) }</div>{risk_bar}"
                f"<div class='muted' style='margin-top:6px'>depth(up): { _escape(pre_r.get('upstream_depth_max')) } → { _escape(post_r.get('upstream_depth_max')) }</div>"
                f"<div class='muted'>depth(down): { _escape(pre_r.get('downstream_depth_max')) } → { _escape(post_r.get('downstream_depth_max')) }</div>"
                "</td>"
            )
            out.append(f"<td>{changed_s if changed_s else '<span class=muted>n/a</span>'}</td>")
            out.append("</tr>")

        out.append("</table>")

        # Raw JSON details for reproducibility
        out.append("<details style='margin-top:10px'><summary>Raw JSON (for reproducibility)</summary>")
        raw = {
            'task': t,
            'runs': {m: (runs.get(m) or {}).get(t['name']) for m in modes},
        }
        out.append(f"<pre style='white-space:pre-wrap'>{_escape(json.dumps(raw, ensure_ascii=False, indent=2, default=_jsonable))}</pre>")
        out.append("</details>")

        out.append("</div>")

    out.append("</div></body></html>")
    return "".join(out)


def write_benchmark_report(results_path: Path, *, out_html: Path) -> Path:
    results = json.loads(results_path.read_text(encoding='utf-8'))
    html_text = render_benchmark_report(results)
    out_html.write_text(html_text, encoding='utf-8')
    return out_html
