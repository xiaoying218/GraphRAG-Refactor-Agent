#!/usr/bin/env python3
"""
gen_readme_report.py

Generate a README-friendly benchmark report from one or more benchmark output folders.

It supports:
- Directories that contain `benchmark_results.json` (e.g., bench_out_case_task1/)
- A parent directory that contains multiple such folders (it will auto-discover)
- (Optional) zip inputs if you pass a .zip file (it will extract to a temp folder)

Outputs (by default into current working directory):
- README_REPORT.md
- summary_table.csv

Usage examples:
  python gen_readme_report.py --inputs bench_out_case_task1 bench_out_case_task2
  python gen_readme_report.py --inputs . --out bench_out/README_REPORT.md
  python gen_readme_report.py --inputs results.zip

Notes:
- "pass" is defined as run_record.status == "success"
- "accept" in this report is defined as (pass && accept_ok) to avoid confusing cases where accept_ok is logged true but the run failed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import statistics
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def md_escape(x: Any) -> str:
    s = str(x)
    return s.replace("\n", "<br>")


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    out = []
    out.append("| " + " | ".join(md_escape(h) for h in headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(md_escape(c) for c in r) + " |")
    return "\n".join(out)


def safe_relpath(path_str: str) -> str:
    if not path_str:
        return ""
    s = path_str.replace("\\", "/")
    m = re.search(r"/(src/main/java/.*)$", s)
    if m:
        return m.group(1)
    m = re.search(r"/(src/.*)$", s)
    if m:
        return m.group(1)
    parts = [x for x in s.split("/") if x]
    return "/".join(parts[-2:]) if len(parts) >= 2 else s


def classify_failure(status: str,
                     accept_reasons: List[str],
                     coverage: Optional[float],
                     failed_step: Optional[int]) -> Tuple[str, str]:
    """
    Failure taxonomy requested:
      - retrieve_miss
      - edit_compile_fail
      - test_fail
      - accept_failed
    Plus:
      - success
    """
    if status == "success":
        return ("success", "")

    reasons = [str(x) for x in (accept_reasons or [])]
    reasons_text = "; ".join(reasons)

    # Retrieve miss: explicit signals or (very low coverage + early failure)
    if any("missing_expected_files" in r for r in reasons) or any("post_focus_method_missing" in r for r in reasons):
        return ("retrieve_miss", reasons_text)
    if coverage is not None and coverage <= 0.05 and (failed_step is not None and failed_step <= 1):
        return ("retrieve_miss", f"low_coverage={coverage:.3f}, failed_step={failed_step}")

    # Compile / test (keyword heuristics)
    if re.search(r"compile|javac|mvn|gradle|build failed", reasons_text, re.I):
        return ("edit_compile_fail", reasons_text)
    if re.search(r"test|junit|surefire|fails? tests?", reasons_text, re.I):
        return ("test_fail", reasons_text)

    # Everything else
    if failed_step is not None:
        return ("accept_failed", f"failed_step={failed_step}" + (f"; {reasons_text}" if reasons_text else ""))
    return ("accept_failed", reasons_text or status)


@dataclass
class RunRow:
    bench: str
    task: str
    desc: str
    mode: str
    expected_files_n: int
    coverage: Optional[float]
    status: str
    pass_ok: bool
    accept_ok: bool
    accept_success: bool
    failure_cat: str
    failure_detail: str


def discover_bench_roots(input_path: Path) -> List[Path]:
    """
    Returns a list of directories that contain benchmark_results.json.
    """
    if input_path.is_file() and input_path.name == "benchmark_results.json":
        return [input_path.parent]
    if input_path.is_dir() and (input_path / "benchmark_results.json").exists():
        return [input_path]

    roots = []
    if input_path.is_dir():
        for p in input_path.rglob("benchmark_results.json"):
            roots.append(p.parent)
    return sorted(set(roots))


def maybe_extract_zip(p: Path) -> Optional[Path]:
    if p.is_file() and p.suffix.lower() == ".zip":
        td = Path(tempfile.mkdtemp(prefix="bench_zip_"))
        with zipfile.ZipFile(p) as z:
            z.extractall(td)
        return td
    return None


def load_optional(p: Path) -> Optional[Dict[str, Any]]:
    if p.exists():
        try:
            return load_json(p)
        except Exception:
            return None
    return None


def run_report(inputs: List[Path], out_md: Path, out_csv: Path, case_task: Optional[str]) -> None:
    # Expand zip inputs to temp dirs
    expanded: List[Path] = []
    for p in inputs:
        zdir = maybe_extract_zip(p)
        expanded.append(zdir if zdir else p)

    bench_roots: List[Path] = []
    for p in expanded:
        bench_roots.extend(discover_bench_roots(p))

    if not bench_roots:
        raise SystemExit("No benchmark_results.json found under provided --inputs.")

    rows: List[RunRow] = []

    # Parse all runs
    for root in bench_roots:
        bm = load_json(root / "benchmark_results.json")
        bench_name = root.name
        modes: List[str] = bm.get("modes") or []
        tasks: List[Dict[str, Any]] = bm.get("tasks") or []
        for t in tasks:
            task = t["name"]
            desc = t.get("description", "")
            expected_files_n = len(t.get("expected_files") or [])
            for mode in modes:
                run_p = root / mode / task / "run_record.json"
                run = load_json(run_p) if run_p.exists() else (bm.get("runs", {}).get(mode, {}).get(task, {}) or {})
                status = run.get("status", "")
                accept_ok = bool(run.get("accept_ok", False))
                pass_ok = (status == "success")
                accept_success = pass_ok and accept_ok
                coverage = (run.get("context_coverage") or {}).get("coverage")
                agent_summary = load_optional(root / mode / task / "agent_summary.json")
                failed_step = agent_summary.get("failed_step") if agent_summary else None
                cat, detail = classify_failure(status, run.get("accept_reasons") or [], coverage, failed_step)
                rows.append(RunRow(
                    bench=bench_name,
                    task=task,
                    desc=desc,
                    mode=mode,
                    expected_files_n=expected_files_n,
                    coverage=coverage,
                    status=status,
                    pass_ok=pass_ok,
                    accept_ok=accept_ok,
                    accept_success=accept_success,
                    failure_cat=cat,
                    failure_detail=detail,
                ))

    # Write CSV (row-level)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "bench", "task", "mode", "expected_files_n", "coverage",
            "status", "pass_ok", "accept_ok", "accept_success",
            "failure_cat", "failure_detail"
        ])
        for r in rows:
            w.writerow([
                r.bench, r.task, r.mode, r.expected_files_n,
                "" if r.coverage is None else r.coverage,
                r.status, int(r.pass_ok), int(r.accept_ok), int(r.accept_success),
                r.failure_cat, r.failure_detail
            ])

    # Aggregate per task, per mode
    by_task_mode: Dict[Tuple[str, str], List[RunRow]] = defaultdict(list)
    for r in rows:
        by_task_mode[(r.task, r.mode)].append(r)

    tasks = sorted({r.task for r in rows})
    modes = sorted({r.mode for r in rows})

    def agg(task: str, mode: str) -> Dict[str, Any]:
        rs = by_task_mode.get((task, mode), [])
        if not rs:
            return {"n": 0}
        covs = [x.coverage for x in rs if x.coverage is not None]
        avg_cov = statistics.mean(covs) if covs else None
        pass_rate = sum(1 for x in rs if x.pass_ok) / len(rs)
        accept_rate = sum(1 for x in rs if x.accept_success) / len(rs)
        cats = Counter(x.failure_cat for x in rs)
        top_fail = ", ".join(f"{k}:{v}" for k, v in cats.most_common(3))
        return {
            "n": len(rs),
            "avg_cov": avg_cov,
            "pass_rate": pass_rate,
            "accept_rate": accept_rate,
            "cats": cats,
            "top_fail": top_fail,
        }

    # Failure taxonomy overall per mode
    tax_by_mode: Dict[str, Counter] = {m: Counter() for m in modes}
    for r in rows:
        tax_by_mode[r.mode][r.failure_cat] += 1

    # Case study selection (GraphRAG-only implicit deps)
    def load_context_pack(root: Path, mode: str, task: str) -> Optional[Dict[str, Any]]:
        p = root / mode / task / "context_pack.json"
        return load_json(p) if p.exists() else None

    def has_call_marker(node: Dict[str, Any]) -> bool:
        txt = " ".join((node.get("why") or []) + (node.get("roles") or []))
        return bool(re.search(r"CALLER|CALLEE|caller|callee|call[- ]graph", txt))

    case_info = None
    if case_task is None:
        best = (-1, None, None)  # score, root, task
        for root in bench_roots:
            bm = load_json(root / "benchmark_results.json")
            for t in (bm.get("tasks") or []):
                task = t["name"]
                cg = load_context_pack(root, "graph_rag", task)
                cv = load_context_pack(root, "vector_only", task)
                if not cg or not cv:
                    continue
                setg = {n.get("node_id") for n in (cg.get("nodes") or [])}
                setv = {n.get("node_id") for n in (cv.get("nodes") or [])}
                extra = [n for n in (cg.get("nodes") or []) if n.get("node_id") in (setg - setv)]
                call_extra = [n for n in extra if has_call_marker(n)]
                score = len(call_extra) * 10 + len(extra)
                if score > best[0]:
                    best = (score, root, task)
        if best[1] is not None:
            case_task = best[2]
            case_root = best[1]
        else:
            case_root = bench_roots[0]
            case_task = tasks[0]
    else:
        # Find the first root that contains that task
        case_root = None
        for root in bench_roots:
            bm = load_json(root / "benchmark_results.json")
            if any(t["name"] == case_task for t in (bm.get("tasks") or [])):
                case_root = root
                break
        case_root = case_root or bench_roots[0]

    # Build case study section
    cg = load_context_pack(case_root, "graph_rag", case_task) if case_root else None
    cv = load_context_pack(case_root, "vector_only", case_task) if case_root else None
    case_lines: List[str] = []
    if cg and cv:
        setg = {n.get("node_id") for n in (cg.get("nodes") or [])}
        setv = {n.get("node_id") for n in (cv.get("nodes") or [])}
        extra = [n for n in (cg.get("nodes") or []) if n.get("node_id") in (setg - setv)]

        # Prioritize CALLER/CALLEE nodes
        extra_sorted = sorted(
            extra,
            key=lambda n: (0 if has_call_marker(n) else 1,
                           0 if "src/main/java" in (n.get("file_path") or "") else 1,
                           -(len(n.get("why") or []))),
        )[:10]

        case_lines.append(f"## Case study: `{case_task}` — GraphRAG recovers implicit dependencies\n")
        case_lines.append(f"- Vector-only context nodes: **{len(cv.get('nodes') or [])}**")
        case_lines.append(f"- GraphRAG context nodes: **{len(cg.get('nodes') or [])}**")
        case_lines.append(f"- GraphRAG-only nodes: **{len(extra)}**\n")
        case_lines.append("GraphRAG-only nodes (selected):\n")
        for n in extra_sorted:
            node_id = n.get("node_id", "")
            fp = safe_relpath(n.get("file_path", ""))
            roles = ", ".join(n.get("roles") or [])
            why = "; ".join(n.get("why") or [])
            case_lines.append(f"- `{node_id}`  \n  - file: `{fp}`  \n  - roles: {roles if roles else '—'}  \n  - why: {why if why else '—'}")
        case_lines.append(
            "\n**Takeaway:** GraphRAG expands along repo-level relations (e.g., CALLER/CALLEE). "
            "This tends to surface *implicit* cross-file dependencies (call sites, strategy implementations, engine entry points) "
            "that a vector-only “same-file / semantically similar” retrieval can miss.\n"
        )
    else:
        case_lines.append("## Case study\n\n⚠️ Could not load both GraphRAG and Vector-only context_pack.json for a case study.\n")

    # Build markdown report
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md: List[str] = []
    md.append("# Benchmark report (auto-generated)\n")
    md.append(f"_Generated at {now}_\n")
    md.append("This report aggregates multiple benchmark output folders under the provided `--inputs`.\n")

    # Overall (per mode across tasks)
    md.append("## Overall (across tasks)\n")
    overall_rows = []
    for mode in modes:
        rs = [r for r in rows if r.mode == mode]
        covs = [r.coverage for r in rs if r.coverage is not None]
        avg_cov = statistics.mean(covs) if covs else None
        pass_rate = sum(1 for r in rs if r.pass_ok) / len(rs)
        accept_rate = sum(1 for r in rs if r.accept_success) / len(rs)
        overall_rows.append([
            mode,
            f"{avg_cov*100:.1f}%" if avg_cov is not None else "n/a",
            f"{pass_rate*100:.1f}%",
            f"{accept_rate*100:.1f}%",
            len(rs),
        ])
    md.append(md_table(["mode", "avg context coverage", "pass rate", "accept rate", "runs"], overall_rows))
    md.append("")

    md.append("## Per-task: GraphRAG vs Vector-only\n")
    headers = [
        "task",
        "GraphRAG coverage",
        "GraphRAG pass",
        "GraphRAG accept",
        "GraphRAG failure dist",
        "Vector coverage",
        "Vector pass",
        "Vector accept",
        "Vector failure dist",
    ]
    table_rows = []
    for task in tasks:
        g = agg(task, "graph_rag")
        v = agg(task, "vector_only")
        def pct(x):
            return "n/a" if x is None else f"{x*100:.1f}%"
        table_rows.append([
            f"`{task}`",
            pct(g.get("avg_cov")),
            pct(g.get("pass_rate")),
            pct(g.get("accept_rate")),
            g.get("top_fail", ""),
            pct(v.get("avg_cov")),
            pct(v.get("pass_rate")),
            pct(v.get("accept_rate")),
            v.get("top_fail", ""),
        ])
    md.append(md_table(headers, table_rows))
    md.append("")
    md.append("> **Note on `context coverage`:** this is computed from the produced `context_pack` only. "
              "An agent can still succeed if it reads files directly during execution, even when coverage is low.\n")

    md.append("## Failure taxonomy (counts)\n")
    cats = sorted({c for m in modes for c in tax_by_mode[m].keys()}, key=lambda c: (c != "success", c))
    md.append(md_table(["category"] + modes, [[c] + [tax_by_mode[m][c] for m in modes] for c in cats]))
    md.append("")

    md.extend(case_lines)
    md.append("## Future work\n")
    md.append(
        "- **Replace heuristic acceptance with tests:** define characterization/unit tests as the success criterion "
        "(compile + test suite pass), instead of relying on missing-expected-files / focus-method checks.\n"
        "- **Test-guided repair loops:** during agent iterations (`--max-iters`), run tests after each edit and use "
        "the failing test outputs/stack traces as feedback to guide repair.\n"
        "- **Stronger evaluation:** add refactoring-specific test cases for each task (including negative/edge cases) "
        "and report success rates across multiple seeds.\n"
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md).strip() + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, default=[Path(".")], help="Bench output folders, parent folders, or zip files.")
    ap.add_argument("--out", type=Path, default=Path("README_REPORT.md"), help="Output markdown path.")
    ap.add_argument("--csv", type=Path, default=Path("summary_table.csv"), help="Output CSV path.")
    ap.add_argument("--case-task", type=str, default=None, help="Optional: force a specific task for the case study.")
    args = ap.parse_args()
    run_report(args.inputs, args.out, args.csv, args.case_task)


if __name__ == "__main__":
    main()
