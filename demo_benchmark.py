#!/usr/bin/env python3
from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Optional

from src.dotenv import auto_load_dotenv
from src.eval.benchmark import load_tasks, run_benchmark
from src.eval.report import write_benchmark_report
from src.trace.trace_utils import TraceLogger, using_tracer


def _infer_run_id_from_out(out_dir: Path) -> Optional[str]:
    parts = list(out_dir.parts)
    for i, part in enumerate(parts):
        if part == "runs" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dotenv",
        default=".env",
        help=(
            "Optional path to a .env file. If omitted, the runner will still try to "
            "auto-load .env from the current directory or parent directories."
        ),
    )
    ap.add_argument("--project", required=True, help="Java project root directory")
    ap.add_argument("--tasks", required=True, help="Path to tasks JSON (see data/bench_tasks.json)")
    ap.add_argument("--out", default="bench_out", help="Output directory")
    ap.add_argument("--modes", default="graph_rag,vector_only", help="Comma-separated: graph_rag,vector_only")
    ap.add_argument(
        "--vector-only-no-search-tools",
        action="store_true",
        help="For vector_only runs, remove search commands (rg/grep/find) from the sandbox whitelist to better isolate retrieval differences.",
    )
    ap.add_argument("--seed-top-k", type=int, default=5)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--max-nodes", type=int, default=30)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--dry-llm", action="store_true")
    ap.add_argument(
        "--accept-mode",
        default="strict",
        choices=["strict", "semantic"],
        help="Acceptance check mode: strict (exact file paths/method ids) or semantic (allow reasonable variations).",
    )
    ap.add_argument(
        "--force-regex-parser",
        action="store_true",
        help="Force the regex fallback parser (useful if you don't want tree-sitter).",
    )
    args = ap.parse_args()

    auto_load_dotenv(args.dotenv)

    project_root = Path(args.project)
    tasks_path = Path(args.tasks)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = _infer_run_id_from_out(out_dir) or uuid.uuid4().hex[:12]
    # 关键：如果 out_dir= runs/<run_id>/bench，trace 就写到 runs/<run_id>/trace.jsonl
    trace_path = (Path("runs") / run_id / "trace.jsonl") if ("runs" in out_dir.parts) else (out_dir / "trace.jsonl")
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    tracer = TraceLogger(
        run_id=run_id,
        trace_path=trace_path,
        base_extra={"project": str(project_root), "out_dir": str(out_dir)},
    )

    tasks = load_tasks(tasks_path)
    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]

    with using_tracer(tracer):
        tracer.log(
            stage="benchmark_start",
            tool="demo_benchmark",
            input_obj={
                "tasks": [t.name for t in tasks],
                "modes": modes,
                "seed_top_k": args.seed_top_k,
                "hops": args.hops,
                "max_nodes": args.max_nodes,
                "max_iters": args.max_iters,
                "dry_llm": bool(args.dry_llm),
                "accept_mode": args.accept_mode,
            },
            output_obj={"ok": True},
        )

        with tracer.span(stage="benchmark_total", tool="run_benchmark") as sp:
            results = run_benchmark(
                project_root=project_root,
                tasks=tasks,
                out_dir=out_dir,
                modes=modes,
                prefer_tree_sitter=not bool(args.force_regex_parser),
                seed_top_k=args.seed_top_k,
                hops=args.hops,
                max_nodes=args.max_nodes,
                max_iters=args.max_iters,
                dry_llm=args.dry_llm,
                accept_mode=args.accept_mode,
                restrict_vector_only_tools=args.vector_only_no_search_tools,
            )
            # 只写一个“轻量 summary”进 trace，避免太大
            sp.set_output(
                {
                    "modes": modes,
                    "tasks": [t.name for t in tasks],
                    "ok": True,
                    "result_keys": sorted(list(results.keys())),
                }
            )

    results_path = out_dir / "benchmark_results.json"
    html_path = out_dir / "benchmark_report.html"
    write_benchmark_report(results_path, out_html=html_path)

    print(f"✅ run_id: {run_id}")
    print(f"✅ Results JSON: {results_path.resolve()}")
    print(f"✅ Report HTML:  {html_path.resolve()}")
    print(f"✅ Trace JSONL:  {trace_path.resolve()}")
    print(f"Modes: {', '.join(modes)}")
    print(f"Tasks: {', '.join(t.name for t in tasks)}")


if __name__ == "__main__":
    main()
