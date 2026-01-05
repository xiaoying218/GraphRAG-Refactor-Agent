#!/usr/bin/env python3
"""
make_readme_report.py

Generate a lightweight, README-friendly benchmark report from `bench_out/`.

Input expected (produced by your demo benchmark runner):
  bench_out/benchmark_results.json
  bench_out/<mode>/<task>/{run_record.json, context_pack.json, agent_summary.json}

Outputs:
  - README_REPORT.md         (Markdown you can link or copy into GitHub README)
  - summary_table.csv        (Machine-readable summary for plotting later)
  - benchmark_results.sanitized.json (optional; strips absolute local paths)
  - context_pack.sanitized.json      (optional; strips absolute local paths)

Usage:
  python scripts/make_readme_report.py --bench_out bench_out
  python scripts/make_readme_report.py --bench_out bench_out --sanitize_paths
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def dump_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_relpath(path_str: str) -> str:
    """Convert absolute paths to stable repo-relative-ish paths for display."""
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


def md_escape(x: Any) -> str:
    return str(x).replace("\n", "<br>")


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    headers2 = [md_escape(h) for h in headers]
    out = []
    out.append("| " + " | ".join(headers2) + " |")
    out.append("| " + " | ".join(["---"] * len(headers2)) + " |")
    for r in rows:
        out.append("| " + " | ".join(md_escape(c) for c in r) + " |")
    return "\n".join(out)


def classify_failure(run: Dict[str, Any], agent_summary: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """
    Taxonomy:
      - success
      - retrieve_miss
      - edit_compile_fail
      - test_fail
      - accept_failed
      - other
    """
    status = run.get("status", "")
    accept_ok = bool(run.get("accept_ok", False))

    if status == "success" and accept_ok:
        return ("success", "")

    reasons = [str(x) for x in (run.get("accept_reasons") or [])]
    failed_step = agent_summary.get("failed_step") if agent_summary else None

    # Retrieval miss signals (common early-stage failure in repo-level editing)
    if any("missing_expected_files" in r for r in reasons) or any("post_focus_method_missing" in r for r in reasons):
        return ("retrieve_miss", "; ".join(reasons))
    if failed_step is not None and isinstance(failed_step, int) and failed_step <= 1:
        return ("retrieve_miss", f"failed_step={failed_step}")

    # Compile/test keywords (only works if you log these into accept_reasons or status)
    if any(re.search(r"compile|javac|mvn", r, re.I) for r in reasons):
        return ("edit_compile_fail", "; ".join(reasons))
    if any(re.search(r"test|junit|surefire", r, re.I) for r in reasons):
        return ("test_fail", "; ".join(reasons))

    if status in ("failed_acceptance", "failed") or not accept_ok:
        return ("accept_failed", "; ".join(reasons) or status)

    return ("other", "; ".join(reasons) or status)


def read_run_record(bench_out: Path, mode: str, task: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    p = bench_out / mode / task / "run_record.json"
    if p.exists():
        try:
            return load_json(p)
        except Exception:
            pass
    return fallback


def read_agent_summary(bench_out: Path, mode: str, task: str) -> Optional[Dict[str, Any]]:
    p = bench_out / mode / task / "agent_summary.json"
    if p.exists():
        try:
            return load_json(p)
        except Exception:
            return None
    return None


def select_case_study(bench_out: Path, tasks: List[str]) -> str:
    """
    Prefer a task where GraphRAG retrieves *project* caller/callee nodes that vector-only misses.
    (Heuristic: extra nodes + caller neighbors + project files.)
    """
    best_task = tasks[0]
    best_score = -1

    for task in tasks:
        try:
            cp_g = load_json(bench_out / "graph_rag" / task / "context_pack.json")
            cp_v = load_json(bench_out / "vector_only" / task / "context_pack.json")
        except Exception:
            continue

        set_g = {n.get("node_id") for n in cp_g.get("nodes", [])}
        set_v = {n.get("node_id") for n in cp_v.get("nodes", [])}
        extra = [n for n in cp_g.get("nodes", []) if n.get("node_id") in (set_g - set_v)]

        proj = [n for n in extra if "src/main/java" in (n.get("file_path") or "")]
        callers = [n for n in extra if any("CALLER" in w for w in (n.get("why") or []))]

        score = 2 * len(proj) + len(callers)
        if score > best_score:
            best_score = score
            best_task = task

    return best_task


def make_case_study_section(bench_out: Path, task: str) -> str:
    cp_g = load_json(bench_out / "graph_rag" / task / "context_pack.json")
    cp_v = load_json(bench_out / "vector_only" / task / "context_pack.json")

    set_g = {n.get("node_id") for n in cp_g.get("nodes", [])}
    set_v = {n.get("node_id") for n in cp_v.get("nodes", [])}
    extra = [n for n in cp_g.get("nodes", []) if n.get("node_id") in (set_g - set_v)]

    def rank(n: Dict[str, Any]) -> Tuple[int, int, int]:
        fp = n.get("file_path") or ""
        is_proj = 0 if "src/main/java" in fp else 1
        is_caller = 0 if any("CALLER" in w for w in (n.get("why") or [])) else 1
        return (is_proj, is_caller, len(n.get("why") or []))

    extra_sorted = sorted(extra, key=rank)[:8]

    lines: List[str] = []
    lines.append(f"## Case study: `{task}` — GraphRAG recovers implicit dependencies\n")
    lines.append(f"**Query:** `{cp_g.get('query')}`  \n**Focus node:** `{cp_g.get('focus_node')}`\n")
    lines.append(
        f"- Vector-only context nodes: **{len(cp_v.get('nodes', []))}**\n"
        f"- GraphRAG context nodes: **{len(cp_g.get('nodes', []))}**\n"
        f"- Extra nodes found by GraphRAG: **{len(extra)}**\n"
    )
    lines.append("GraphRAG-only nodes (selected):\n")

    for n in extra_sorted:
        node_id = n.get("node_id", "")
        fp = safe_relpath(n.get("file_path", ""))
        why = "; ".join(n.get("why") or [])
        sig = n.get("signature", "") or ""
        if len(sig) > 140:
            sig = sig[:137] + "..."

        lines.append(f"- `{node_id}`  \n  - file: `{fp}`  \n  - why: {why if why else '—'}")
        if sig:
            lines.append(f"  - signature: `{sig}`")

    lines.append(
        "\n**Why this matters:** In refactoring, many dependencies are *implicit* "
        "(call sites / helper methods / framework hooks). GraphRAG expands along the "
        "repository call graph (CALLER/CALLEE neighbors), so it tends to surface these "
        "cross-file dependencies earlier. Vector-only retrieval tends to stay “near” the "
        "query symbol (same file/class), and can miss those call sites.\n"
    )
    return "\n".join(lines)


def sanitize_bench_out(bench_out: Path, modes: List[str], tasks: List[str]) -> None:
    """
    Write sanitized copies (does NOT overwrite originals):
      - benchmark_results.sanitized.json
      - <mode>/<task>/context_pack.sanitized.json
      - <mode>/<task>/run_record.sanitized.json
    """
    src_results = load_json(bench_out / "benchmark_results.json")
    sanitized = dict(src_results)
    if "project_root" in sanitized:
        sanitized["project_root"] = "<REDACTED_LOCAL_PATH>"

    # Run records and context packs often contain local absolute paths (work_dir, file_path).
    for mode in modes:
        for task in tasks:
            run_p = bench_out / mode / task / "run_record.json"
            if run_p.exists():
                r = load_json(run_p)
                if "work_dir" in r:
                    r["work_dir"] = "<REDACTED_LOCAL_PATH>"
                (bench_out / mode / task / "run_record.sanitized.json").write_text(
                    json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            cp_p = bench_out / mode / task / "context_pack.json"
            if cp_p.exists():
                cp = load_json(cp_p)
                for node in cp.get("nodes", []):
                    if "file_path" in node:
                        node["file_path"] = safe_relpath(node.get("file_path") or "")
                (bench_out / mode / task / "context_pack.sanitized.json").write_text(
                    json.dumps(cp, ensure_ascii=False, indent=2), encoding="utf-8"
                )

    dump_json(bench_out / "benchmark_results.sanitized.json", sanitized)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench_out", type=Path, default=Path("bench_out"), help="Path to bench_out/")
    ap.add_argument("--out", type=Path, default=None, help="Output markdown path (default: <bench_out>/README_REPORT.md)")
    ap.add_argument("--case_task", type=str, default=None, help="Optional: force a specific case study task name")
    ap.add_argument("--sanitize_paths", action="store_true", help="Write *.sanitized.json copies for GitHub-safe sharing")
    args = ap.parse_args()

    bench_out: Path = args.bench_out
    results_path = bench_out / "benchmark_results.json"
    if not results_path.exists():
        raise SystemExit(f"Missing: {results_path} (run your benchmark first)")

    results = load_json(results_path)
    modes: List[str] = list(results.get("modes") or [])
    tasks: List[str] = [t["name"] for t in (results.get("tasks") or [])]
    task_desc: Dict[str, str] = {t["name"]: t.get("description", "") for t in results.get("tasks") or []}

    # Collect per-task summaries
    per_task: Dict[str, Dict[str, Dict[str, Any]]] = {t: {} for t in tasks}
    csv_rows: List[Dict[str, Any]] = []

    for mode in modes:
        for task in tasks:
            run_fallback = (results.get("runs", {}).get(mode, {}).get(task) or {})
            run = read_run_record(bench_out, mode, task, fallback=run_fallback)
            agent_summary = read_agent_summary(bench_out, mode, task)

            cov = (run.get("context_coverage") or {}).get("coverage")
            pass_ok = (run.get("status") == "success")
            accept_ok = bool(run.get("accept_ok"))
            cat, detail = classify_failure(run, agent_summary)

            per_task[task][mode] = {
                "coverage": cov,
                "pass_ok": pass_ok,
                "accept_ok": accept_ok,
                "category": cat,
                "detail": detail,
                "status": run.get("status", ""),
            }

            csv_rows.append(
                {
                    "task": task,
                    "description": task_desc.get(task, ""),
                    "mode": mode,
                    "coverage": cov if cov is not None else "",
                    "pass": 1 if pass_ok else 0,
                    "accept": 1 if accept_ok else 0,
                    "failure_category": cat,
                    "failure_detail": detail,
                }
            )

    # Overall metrics
    overall: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        coverages = [per_task[t][mode]["coverage"] for t in tasks if per_task[t][mode]["coverage"] is not None]
        overall[mode] = {
            "avg_coverage": statistics.mean(coverages) if coverages else float("nan"),
            "pass_rate": sum(1 for t in tasks if per_task[t][mode]["pass_ok"]) / max(1, len(tasks)),
            "accept_rate": sum(1 for t in tasks if per_task[t][mode]["accept_ok"]) / max(1, len(tasks)),
        }

    # Failure taxonomy (counts)
    tax_by_mode: Dict[str, Counter] = {m: Counter() for m in modes}
    for task in tasks:
        for mode in modes:
            tax_by_mode[mode][per_task[task][mode]["category"]] += 1

    # Write CSV
    out_md = args.out or (bench_out / "README_REPORT.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_md.parent / "summary_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    # Build markdown
    generated_at = results.get("generated_at") or dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md: List[str] = []
    md.append("# Benchmark summary (auto-generated)\n")
    md.append(f"_Generated from `{bench_out.as_posix()}` on {generated_at}_\n")

    md.append("## Metric definitions (current demo)\n")
    md.append(
        "- **context coverage**: fraction of `expected_files` that appear in the produced `context_pack`.\n"
        "- **pass**: `status == success` from the benchmark runner.\n"
        "- **accept**: `accept_ok` from the runner’s acceptance checks.\n\n"
        "> Note: In the current stage, *acceptance* is a **sanity-check** signal (compile/tests if configured + lightweight "
        "heuristics like expected files / focus method). It is **not yet** a full behavior-preservation guarantee (no new "
        "characterization tests).\n"
    )

    md.append("## Overall\n")
    md.append(
        md_table(
            ["mode", "avg context coverage", "pass rate", "accept rate"],
            [
                [
                    mode,
                    f"{overall[mode]['avg_coverage']*100:.1f}%" if not math.isnan(overall[mode]["avg_coverage"]) else "n/a",
                    f"{overall[mode]['pass_rate']*100:.1f}%",
                    f"{overall[mode]['accept_rate']*100:.1f}%",
                ]
                for mode in modes
            ],
        )
    )

    md.append("\n## Per-task comparison\n")
    headers = [
        "task",
        "desc",
        "GraphRAG cov",
        "GraphRAG pass",
        "GraphRAG accept",
        "GraphRAG failure",
        "Vector cov",
        "Vector pass",
        "Vector accept",
        "Vector failure",
    ]
    rows: List[List[Any]] = []
    for task in tasks:
        g = per_task[task].get("graph_rag", {})
        v = per_task[task].get("vector_only", {})

        def pct(x: Any) -> str:
            if x is None:
                return "n/a"
            return f"{float(x)*100:.1f}%"

        rows.append(
            [
                f"`{task}`",
                task_desc.get(task, ""),
                pct(g.get("coverage")),
                "✅" if g.get("pass_ok") else "❌",
                "✅" if g.get("accept_ok") else "❌",
                g.get("category", "") + (f": {g.get('detail')}" if g.get("detail") else ""),
                pct(v.get("coverage")),
                "✅" if v.get("pass_ok") else "❌",
                "✅" if v.get("accept_ok") else "❌",
                v.get("category", "") + (f": {v.get('detail')}" if v.get("detail") else ""),
            ]
        )
    md.append(md_table(headers, rows))

    md.append("\n## Failure taxonomy\n")
    categories = sorted({c for m in modes for c in tax_by_mode[m].keys()}, key=lambda c: (c != "success", c))
    md.append(
        md_table(
            ["category"] + modes,
            [[cat] + [str(tax_by_mode[m][cat]) for m in modes] for cat in categories],
        )
    )

    # Case study
    case_task = args.case_task or select_case_study(bench_out, tasks)
    try:
        md.append("\n" + make_case_study_section(bench_out, case_task))
    except Exception as e:
        md.append(f"\n## Case study\n\n⚠️ Failed to build case study section: {e}\n")

    out_md.write_text("\n".join(md).strip() + "\n", encoding="utf-8")
    print(f"✅ Wrote: {out_md}")
    print(f"✅ Wrote: {csv_path}")

    if args.sanitize_paths:
        sanitize_bench_out(bench_out, modes, tasks)
        print(f"✅ Wrote sanitized JSON copies under: {bench_out}")


if __name__ == "__main__":
    main()
