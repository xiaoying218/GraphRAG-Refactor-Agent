#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Optional

from src.dotenv import auto_load_dotenv
from src.eval.benchmark import _build_graph_and_index as build_graph_and_index
from src.context_engine import GraphRAGContextEngine
from src.trace.trace_utils import TraceLogger, using_tracer


def _infer_run_id_from_path(p: Path) -> Optional[str]:
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part == "runs" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _resolve_out_and_trace(out_arg: str) -> tuple[str, Path, Path]:
    out = Path(out_arg)

    # 如果用户给的是目录（或没写后缀），默认写 context_pack.json
    if out.suffix == "" and not str(out).endswith(".json"):
        out = out / "context_pack.json"

    run_id = _infer_run_id_from_path(out)
    if not run_id:
        run_id = uuid.uuid4().hex[:12]
        out = Path("runs") / run_id / out.name

    out.parent.mkdir(parents=True, exist_ok=True)
    trace_path = Path("runs") / run_id / "trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    return run_id, out, trace_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dotenv", default=".env", help="Optional .env path (default: .env)")
    ap.add_argument("--project", required=True, help="Java project root directory")
    ap.add_argument("--query", required=True, help="Natural language query OR exact node id")
    ap.add_argument("--seed_top_k", type=int, default=5)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--max_nodes", type=int, default=30)
    ap.add_argument("--out", default="context_pack.json")
    args = ap.parse_args()

    auto_load_dotenv(args.dotenv)

    run_id, out_path, trace_path = _resolve_out_and_trace(args.out)

    tracer = TraceLogger(
        run_id=run_id,
        trace_path=trace_path,
        base_extra={"project": str(args.project)},
    )

    with using_tracer(tracer):
        # Build graph + index
        with tracer.span(
            stage="build_graph_index",
            tool="_build_graph_and_index",
            input_obj={"project": str(args.project)},
        ) as sp:
            graph, vindex = build_graph_and_index(args.project, prefer_tree_sitter=True)
            sp.set_output({"nodes": int(graph.number_of_nodes()), "edges": int(graph.number_of_edges())})

        engine = GraphRAGContextEngine(graph, vindex)

        # Query (Stage1/2/3 细分 trace 在 src/context_engine.py 里打点)
        with tracer.span(
            stage="context_pack_total",
            tool="GraphRAGContextEngine.query",
            input_obj={
                "query": args.query,
                "seed_top_k": args.seed_top_k,
                "hops": args.hops,
                "max_nodes": args.max_nodes,
            },
        ) as sp:
            pack = engine.query(
                args.query,
                seed_top_k=args.seed_top_k,
                hops=args.hops,
                max_nodes=args.max_nodes,
            )
            sp.set_output(
                {
                    "focus_node": pack.get("focus_node", ""),
                    "selected_nodes": len(pack.get("nodes") or []),
                    "selected_edges": len(pack.get("edges") or []),
                }
            )

        out_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
        tracer.log(
            stage="io",
            tool="write_json",
            input_obj={"path": str(out_path)},
            output_obj={"bytes": out_path.stat().st_size},
        )

    print(f"✅ run_id: {run_id}")
    print(f"✅ context_pack: {out_path.resolve()}")
    print(f"✅ trace: {trace_path.resolve()}")


if __name__ == "__main__":
    main()
