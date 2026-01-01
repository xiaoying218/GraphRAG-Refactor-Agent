#!/usr/bin/env python3
"""
Demo: build Graph-RAG context pack for a Java repository.

Usage:
  python demo_context_pack.py --project /path/to/java/repo --query "refactor duplication in offer decision"

Outputs:
  - Prints a short summary
  - Saves context_pack.json in current directory
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.parser import JavaCodeParser
from src.graph_builder import CodeGraphBuilder
from src.vector_index import NodeVectorIndex
from src.context_engine import GraphRAGContextEngine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Java project root directory")
    ap.add_argument("--query", required=True, help="Natural language query OR exact node id")
    ap.add_argument("--seed_top_k", type=int, default=5)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--max_nodes", type=int, default=30)
    ap.add_argument("--out", default="context_pack.json")
    args = ap.parse_args()

    project_root = args.project
    if not os.path.isdir(project_root):
        raise SystemExit(f"Project path not found: {project_root}")

    print(f"🔎 Parsing project: {project_root}")
    parser = JavaCodeParser()
    data = parser.parse_project(project_root)

    builder = CodeGraphBuilder()
    builder.build_from_parsed_data(data)
    graph = builder.get_graph()

    print("🧠 Building vector index (TF-IDF)...")
    vindex = NodeVectorIndex()
    vindex.build_from_graph(graph)

    engine = GraphRAGContextEngine(graph, vindex)

    print(f"🧩 Query: {args.query}")
    pack = engine.query(
        args.query,
        seed_top_k=args.seed_top_k,
        hops=args.hops,
        max_nodes=args.max_nodes,
    )

    # Print a small human-friendly summary
    print("\n================= CONTEXT PACK SUMMARY =================")
    print(f"Focus: {pack.get('focus_node')}")
    print("Seeds:")
    for s in pack.get("seed_nodes", []):
        print(f"  - {s['node_id']} (score={s['score']:.3f})")
    print(f"Selected nodes: {pack['stats']['selected_nodes']}, edges: {pack['stats']['selected_edges']}")

    # Show top 5 nodes with why
    nodes = pack.get("nodes", [])
    print("\nTop nodes (first 5):")
    for n in nodes[:5]:
        why = "; ".join(n.get("why", [])[:2])
        loc = f"{n.get('file_path')}:{n.get('start_line')}-{n.get('end_line')}" if n.get("file_path") else ""
        print(f"- [{n.get('type')}] {n.get('node_id')}  {loc}")
        if n.get("signature"):
            print(f"    sig: {n.get('signature')[:120]}")
        if why:
            print(f"    why: {why}")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
