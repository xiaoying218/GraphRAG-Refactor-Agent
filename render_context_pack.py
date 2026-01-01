#!/usr/bin/env python3
"""Render context_pack.json into a simple IDE-like HTML preview.

Usage:
  python render_context_pack.py /path/to/context_pack.json -o context_preview.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _escape(s: str) -> str:
    return html.escape(s, quote=False)


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _node_map(pack: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for n in pack.get("nodes", []):
        nid = n.get("node_id") or n.get("id")
        if nid:
            m[nid] = n
    return m


def _format_why(why: Any) -> str:
    items = _as_list(why)
    if not items:
        return ""
    lis = "\n".join(f"<li>{_escape(str(it))}</li>" for it in items)
    return f"<ul class='why'>{lis}</ul>"


def _code_with_line_numbers(snippet: str, start_line: int, highlight: Optional[Dict[str, int]]) -> str:
    lines = snippet.splitlines()
    hl_start = hl_end = None
    if isinstance(highlight, dict):
        hl_start = highlight.get("start_line")
        hl_end = highlight.get("end_line")
    out = []
    for i, line in enumerate(lines):
        lineno = start_line + i
        cls = "code-line"
        if hl_start is not None and hl_end is not None and hl_start <= lineno <= hl_end:
            cls += " hl"
        out.append(
            f"<div class='{cls}'><span class='ln'>{lineno:>4}</span><span class='src'>{_escape(line)}</span></div>"
        )
    return "<div class='code'>" + "\n".join(out) + "</div>"


def _node_card(n: Dict[str, Any]) -> str:
    nid = n.get("node_id") or ""
    kind = n.get("type") or n.get("kind") or ""
    role = n.get("role") or ""
    roles = n.get("roles") or []
    fp = n.get("file_path") or ""
    sig = n.get("signature") or ""
    doc = n.get("docstring") or ""
    snippet = n.get("snippet") or ""
    start_line = int(n.get("start_line") or 0)
    highlight = n.get("highlight_span")

    meta_bits = []
    if fp:
        meta_bits.append(f"<span class='meta'><b>File:</b> {_escape(fp)}:{start_line}-{n.get('end_line')}</span>")
    if sig:
        meta_bits.append(f"<span class='meta'><b>Sig:</b> {_escape(sig)}</span>")
    if role:
        meta_bits.append(f"<span class='meta'><b>Role:</b> {_escape(role)}</span>")
    if roles:
        meta_bits.append(f"<span class='meta'><b>Roles:</b> {_escape(', '.join(map(str, roles)))}</span>")

    meta_html = "<div class='meta-row'>" + " | ".join(meta_bits) + "</div>" if meta_bits else ""
    doc_html = f"<pre class='doc'>{_escape(doc)}</pre>" if doc else ""
    code_html = _code_with_line_numbers(snippet, start_line if start_line else 1, highlight) if snippet else ""
    why_html = _format_why(n.get("why"))

    return f"""
    <div class=\"card\" id=\"{_escape(nid)}\">
      <div class=\"card-head\">
        <div class=\"title\"><span class=\"badge kind\">{_escape(kind)}</span> <b>{_escape(nid)}</b></div>
        {meta_html}
      </div>
      {doc_html}
      {code_html}
      {why_html}
    </div>
    """


def _mini_node_link(nid: str, nmap: Dict[str, Dict[str, Any]]) -> str:
    n = nmap.get(nid, {})
    kind = n.get("type") or n.get("kind") or "Node"
    role = n.get("role") or ""
    return f'<a class="node-link" href="#{_escape(nid)}"><span class="badge kind">{_escape(kind)}</span> {_escape(nid)} <span class="badge role">{_escape(role)}</span></a>'


def _section_list(title: str, items: List[Any], nmap: Dict[str, Dict[str, Any]], key: str = "node_id") -> str:
    if not items:
        body = "<div class='empty'>None</div>"
    else:
        rows = []
        for it in items:
            if isinstance(it, dict):
                nid = it.get(key) or it.get("node_id")
                extra = []
                if "depth" in it:
                    extra.append(f"depth={it['depth']}")
                if extra:
                    rows.append(f"<li>{_mini_node_link(nid, nmap)} <span class='small'>({', '.join(extra)})</span></li>")
                else:
                    rows.append(f"<li>{_mini_node_link(nid, nmap)}</li>")
            else:
                rows.append(f"<li>{_mini_node_link(str(it), nmap)}</li>")
        body = "<ul class='list'>" + "\n".join(rows) + "</ul>"
    return f"<details open class='section'><summary>{_escape(title)}</summary>{body}</details>"


def render(pack: Dict[str, Any]) -> str:
    nmap = _node_map(pack)

    query = pack.get("query", "")
    focus = (pack.get("focus") or {}).get("node_id") or pack.get("focus_node") or ""
    seeds = pack.get("seed_nodes", [])
    seed_lines = []
    for s in seeds:
        nid = s.get("node_id")
        score = s.get("score")
        seed_lines.append(f"<li>{_mini_node_link(nid, nmap)} <span class='small'>(score={float(score):.3f})</span></li>")
    seed_html = "<ul class='list'>" + "\n".join(seed_lines) + "</ul>" if seed_lines else "<div class='empty'>None</div>"

    cg = pack.get("call_graph", {}) or {}
    df = pack.get("data_flow", {}) or {}

    left = []
    left.append(f"<div class='kv'><b>Query:</b> {_escape(str(query))}</div>")
    left.append(f"<div class='kv'><b>Focus:</b> {(_mini_node_link(focus, nmap) if focus else '<span class=empty>None</span>')}</div>")
    left.append("<details open class='section'><summary>Seeds</summary>" + seed_html + "</details>")
    left.append(_section_list("Call Graph · Callers", cg.get("callers", []), nmap))
    left.append(_section_list("Call Graph · Callees", cg.get("callees", []), nmap))
    left.append(_section_list("Data Flow · Read fields", df.get("read_fields", []), nmap, key="node_id"))
    left.append(_section_list("Data Flow · Written fields", df.get("written_fields", []), nmap, key="node_id"))
    left.append(_section_list("Same Class", pack.get("same_class", []), nmap, key="node_id"))
    left.append(_section_list("Same File", pack.get("same_file", []), nmap, key="node_id"))

    role_order = {
        "focus": 0, "seed": 1, "caller": 2, "callee": 3,
        "shared_field": 4, "shared_field_reader": 5, "shared_field_writer": 6,
        "same_class_member": 7, "same_file_helper": 8
    }
    nodes = pack.get("nodes", [])
    def sort_key(n):
        r = n.get("role") or ""
        return (role_order.get(r, 99), n.get("type",""), n.get("node_id",""))

    nodes_sorted = sorted(nodes, key=sort_key)
    main_cards = "\n".join(_node_card(n) for n in nodes_sorted)

    css = (Path(__file__).with_name("style.css")).read_text(encoding="utf-8")

    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\"/>
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
<title>Graph-RAG Context Pack Preview</title>
<style>
{css}
</style>
</head>
<body>
<header>
  <div class=\"brand\">Graph-RAG · Context Pack Preview</div>
  <div class=\"hint\">Click items in the left panel to jump to code cards. Highlighted lines indicate focus spans.</div>
</header>
<div class=\"layout\">
  <aside class=\"sidebar\">
    {''.join(left)}
  </aside>
  <main class=\"main\">
    {main_cards}
  </main>
</div>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("context_pack", type=str, help="path to context_pack.json")
    ap.add_argument("-o", "--out", type=str, default="context_preview.html")
    args = ap.parse_args()

    pack = json.loads(Path(args.context_pack).read_text(encoding="utf-8"))
    html_out = render(pack)
    Path(args.out).write_text(html_out, encoding="utf-8")
    print(f"✅ Wrote: {args.out}")


if __name__ == "__main__":
    main()
