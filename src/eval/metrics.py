from __future__ import annotations

import os
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx

from ..parser import JavaCodeParser
from ..graph_builder import CodeGraphBuilder


def iter_java_files(project_root: str) -> Iterable[str]:
    ignore = {
        ".git",
        ".refactor_agent_runs",
        "target",
        "build",
        ".gradle",
        "node_modules",
        "__MACOSX",
        "__pycache__",
    }
    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in ignore]
        for fn in files:
            if fn.endswith(".java"):
                yield os.path.join(root, fn)


_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//.*?$", re.MULTILINE)


def strip_java_comments(text: str) -> str:
    # Best-effort: not a full lexer, but good enough for metrics.
    text = _BLOCK_COMMENT_RE.sub("", text)
    text = _LINE_COMMENT_RE.sub("", text)
    return text


def normalize_line(line: str) -> str:
    # Normalize for duplication detection
    s = strip_java_comments(line)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def cyclomatic_complexity_approx(code: str) -> int:
    """Very lightweight cyclomatic complexity approximation.

    CC = 1 + (#decision points)

    Decision points counted:
      - if / for / while / case / catch
      - && / ||
      - ?: ternary

    This is intentionally simple and reproducible (no external tooling).
    """
    if not code:
        return 1

    code = strip_java_comments(code)
    # Remove string literals to avoid false positives.
    code = re.sub(r'"(?:\\.|[^"\\])*"', '""', code)
    code = re.sub(r"'(?:\\.|[^'\\])*'", "''", code)

    kw = re.findall(r"\b(if|for|while|case|catch)\b", code)
    and_or = code.count("&&") + code.count("||")
    ternary = code.count("?")
    return 1 + len(kw) + and_or + ternary


@dataclass
class MethodMetrics:
    method_id: str
    file_path: str
    start_line: int
    end_line: int
    loc: int
    loc_non_empty: int
    cyclomatic: int


def compute_method_metrics(project_root: str, *, prefer_tree_sitter: bool = True) -> Dict[str, MethodMetrics]:
    parser = JavaCodeParser(prefer_tree_sitter=prefer_tree_sitter)
    data = parser.parse_project(project_root)
    out: Dict[str, MethodMetrics] = {}

    # Cache file lines
    file_lines: Dict[str, List[str]] = {}

    for m in data.get("methods", []) or []:
        mid = str(m.get("id") or "")
        fp = str(m.get("file_path") or "")
        sl = int(m.get("start_line") or 0)
        el = int(m.get("end_line") or 0)
        if not mid or not fp or sl <= 0 or el <= 0 or el < sl:
            continue

        if fp not in file_lines:
            try:
                file_lines[fp] = Path(fp).read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                file_lines[fp] = []
        lines = file_lines.get(fp, [])
        snippet_lines = lines[sl - 1 : el]
        snippet = "\n".join(snippet_lines)

        loc = el - sl + 1
        loc_non_empty = sum(1 for ln in snippet_lines if normalize_line(ln))
        cc = cyclomatic_complexity_approx(snippet)

        out[mid] = MethodMetrics(
            method_id=mid,
            file_path=fp,
            start_line=sl,
            end_line=el,
            loc=loc,
            loc_non_empty=loc_non_empty,
            cyclomatic=cc,
        )

    return out


@dataclass
class DuplicationMetrics:
    total_lines: int
    duplicate_lines: int
    duplicate_line_ratio: float
    distinct_lines: int


def compute_duplication_metrics(project_root: str) -> DuplicationMetrics:
    """Line-level duplication (simple + reproducible).

    We treat a line as *duplicate* if the same normalized line occurs in
    2+ places across the project (comments/whitespace ignored).
    """
    norm_lines: List[str] = []
    for fp in iter_java_files(project_root):
        try:
            raw = Path(fp).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for ln in raw.splitlines():
            n = normalize_line(ln)
            if not n:
                continue
            norm_lines.append(n)

    total = len(norm_lines)
    if total == 0:
        return DuplicationMetrics(total_lines=0, duplicate_lines=0, duplicate_line_ratio=0.0, distinct_lines=0)

    freq = Counter(norm_lines)
    dup = sum(c for c in freq.values() if c > 1)
    distinct = len(freq)
    return DuplicationMetrics(
        total_lines=total,
        duplicate_lines=dup,
        duplicate_line_ratio=float(dup) / float(total) if total else 0.0,
        distinct_lines=distinct,
    )


@dataclass
class ProjectMaintainabilityMetrics:
    methods: int
    avg_loc: float
    avg_cyclomatic: float
    p95_loc: float
    p95_cyclomatic: float
    max_loc: int
    max_cyclomatic: int
    duplication: DuplicationMetrics


def _percentile(values: List[int], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return float(values[f])
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return float(d0 + d1)


def compute_project_maintainability(project_root: str, *, prefer_tree_sitter: bool = True) -> ProjectMaintainabilityMetrics:
    mm = compute_method_metrics(project_root, prefer_tree_sitter=prefer_tree_sitter)
    locs = [m.loc for m in mm.values()]
    ccs = [m.cyclomatic for m in mm.values()]

    methods = len(mm)
    avg_loc = float(sum(locs)) / float(methods) if methods else 0.0
    avg_cc = float(sum(ccs)) / float(methods) if methods else 0.0

    dup = compute_duplication_metrics(project_root)
    return ProjectMaintainabilityMetrics(
        methods=methods,
        avg_loc=avg_loc,
        avg_cyclomatic=avg_cc,
        p95_loc=_percentile(locs, 0.95),
        p95_cyclomatic=_percentile(ccs, 0.95),
        max_loc=max(locs) if locs else 0,
        max_cyclomatic=max(ccs) if ccs else 0,
        duplication=dup,
    )


def build_code_graph(project_root: str, *, prefer_tree_sitter: bool = True) -> nx.DiGraph:
    parser = JavaCodeParser(prefer_tree_sitter=prefer_tree_sitter)
    data = parser.parse_project(project_root)
    builder = CodeGraphBuilder()
    builder.build_from_parsed_data(data)
    return builder.get_graph()


@dataclass
class ChangeRiskMetrics:
    focus_node: str
    fan_in: int
    fan_out: int
    upstream_depth_max: int
    downstream_depth_max: int
    impacted_nodes_upstream: int
    impacted_nodes_downstream: int
    impacted_files: List[str]


def _bfs_depth(
    graph: nx.DiGraph,
    start: str,
    *,
    direction: str,
    relation: str = "CALLS",
    max_hops: Optional[int] = None,
) -> Dict[str, int]:
    if start not in graph:
        return {}
    q: deque[Tuple[str, int]] = deque([(start, 0)])
    visited: Set[str] = {start}
    depth_map: Dict[str, int] = {}
    while q:
        nid, d = q.popleft()
        if max_hops is not None and d >= max_hops:
            continue

        if direction == "in":
            edges = graph.in_edges(nid, data=True)
            for u, _, attrs in edges:
                rel = attrs.get("relation") or attrs.get("type")
                if rel != relation:
                    continue
                if u not in visited:
                    visited.add(u)
                    depth_map[str(u)] = d + 1
                    q.append((str(u), d + 1))
        else:
            edges = graph.out_edges(nid, data=True)
            for _, v, attrs in edges:
                rel = attrs.get("relation") or attrs.get("type")
                if rel != relation:
                    continue
                if v not in visited:
                    visited.add(v)
                    depth_map[str(v)] = d + 1
                    q.append((str(v), d + 1))
    depth_map.pop(start, None)
    return depth_map


def compute_change_risk(graph: nx.DiGraph, focus_node: str, *, max_hops: Optional[int] = None) -> ChangeRiskMetrics:
    """Compute a few graph-native "change risk" metrics.

    Interpretation for a refactoring focus node:
      - fan_in: how many direct callers depend on it
      - fan_out: how many direct callees it depends on
      - upstream_depth_max: max call-chain depth of its callers (impact propagates upward)
      - downstream_depth_max: max call-chain depth of its callees (impact propagates downward)
      - impacted_files: unique files of reachable caller/callee nodes (plus focus file when present)
    """
    if focus_node not in graph:
        return ChangeRiskMetrics(
            focus_node=focus_node,
            fan_in=0,
            fan_out=0,
            upstream_depth_max=0,
            downstream_depth_max=0,
            impacted_nodes_upstream=0,
            impacted_nodes_downstream=0,
            impacted_files=[],
        )

    # Direct fan-in/out for CALLS edges
    fan_in = sum(1 for u, _, a in graph.in_edges(focus_node, data=True) if (a.get("relation") or a.get("type")) == "CALLS")
    fan_out = sum(1 for _, v, a in graph.out_edges(focus_node, data=True) if (a.get("relation") or a.get("type")) == "CALLS")

    upstream = _bfs_depth(graph, focus_node, direction="in", relation="CALLS", max_hops=max_hops)
    downstream = _bfs_depth(graph, focus_node, direction="out", relation="CALLS", max_hops=max_hops)

    upstream_depth_max = max(upstream.values()) if upstream else 0
    downstream_depth_max = max(downstream.values()) if downstream else 0

    impacted_nodes = set([focus_node]) | set(upstream.keys()) | set(downstream.keys())
    files: Set[str] = set()
    for nid in impacted_nodes:
        fp = graph.nodes.get(nid, {}).get("file_path")
        if fp:
            files.add(str(fp))

    return ChangeRiskMetrics(
        focus_node=focus_node,
        fan_in=fan_in,
        fan_out=fan_out,
        upstream_depth_max=upstream_depth_max,
        downstream_depth_max=downstream_depth_max,
        impacted_nodes_upstream=len(upstream),
        impacted_nodes_downstream=len(downstream),
        impacted_files=sorted(files),
    )


def relpath_under(project_root: str, path: str) -> str:
    try:
        return os.path.relpath(path, project_root).replace("\\", "/")
    except Exception:
        return path.replace("\\", "/")


def context_pack_file_coverage(
    context_pack: Dict[str, Any],
    *,
    project_root: str,
    expected_files_rel: Sequence[str],
) -> Dict[str, Any]:
    """How many expected files are present in the context pack (by node.file_path)."""
    expected = {p.replace("\\", "/") for p in expected_files_rel if p}
    found: Set[str] = set()
    for n in context_pack.get("nodes", []) or []:
        fp = n.get("file_path")
        if not fp:
            continue
        found.add(relpath_under(project_root, str(fp)))

    hit = sorted(list(expected & found))
    miss = sorted(list(expected - found))
    coverage = (len(hit) / len(expected)) if expected else 1.0
    return {
        "expected_files": sorted(expected),
        "found_files": sorted(found),
        "hit_files": hit,
        "missing_files": miss,
        "coverage": coverage,
    }
