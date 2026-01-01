from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx

from ..agent.agent import AgentConfig, RefactoringAgent
from ..agent.llm import DummyEchoLLM, OpenAICompatibleChatClient, OpenAICompatibleConfig
from ..context_engine import GraphRAGContextEngine
from ..vector_index import NodeVectorIndex
from ..graph_builder import CodeGraphBuilder
from ..parser import JavaCodeParser
from .metrics import (
    ChangeRiskMetrics,
    MethodMetrics,
    ProjectMaintainabilityMetrics,
    build_code_graph,
    compute_change_risk,
    compute_method_metrics,
    compute_project_maintainability,
    context_pack_file_coverage,
    relpath_under,
)


@dataclass
class BenchmarkTask:
    name: str
    query: str
    request: str
    expected_files: List[str]
    focus_node: Optional[str] = None
    focus_node_after: Optional[str] = None
    description: str = ""


def load_tasks(path: Path) -> List[BenchmarkTask]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    tasks: List[BenchmarkTask] = []
    for t in obj.get("tasks", []):
        tasks.append(
            BenchmarkTask(
                name=str(t["name"]),
                query=str(t.get("query") or t.get("focus_node") or ""),
                request=str(t["request"]),
                expected_files=[str(x) for x in (t.get("expected_files") or [])],
                focus_node=str(t.get("focus_node") or "") or None,
                focus_node_after=str(t.get("focus_node_after") or "") or None,
                description=str(t.get("description") or ""),
            )
        )
    if not tasks:
        raise ValueError(f"No tasks found in: {path}")
    return tasks


def _build_graph_and_index(project_root: str, *, prefer_tree_sitter: bool = True) -> Tuple[nx.DiGraph, NodeVectorIndex]:
    parser = JavaCodeParser(prefer_tree_sitter=prefer_tree_sitter)
    data = parser.parse_project(project_root)
    builder = CodeGraphBuilder()
    builder.build_from_parsed_data(data)
    graph = builder.get_graph()

    vindex = NodeVectorIndex()
    vindex.build_from_graph(graph)
    return graph, vindex


def build_context_pack(
    project_root: str,
    *,
    query: str,
    mode: str = "graph_rag",
    seed_top_k: int = 5,
    hops: int = 2,
    max_nodes: int = 30,
    prefer_tree_sitter: bool = True,
) -> Dict[str, Any]:
    """Build a context pack in different ablation modes.

    mode:
      - graph_rag: full Graph-RAG expansion (vector seeds + neighborhood expansion)
      - vector_only: vector seeds only (graph expansion disabled)
    """
    graph, vindex = _build_graph_and_index(project_root, prefer_tree_sitter=prefer_tree_sitter)
    engine = GraphRAGContextEngine(graph, vindex)

    if mode == "vector_only":
        return engine.query(
            query,
            seed_top_k=seed_top_k,
            hops=0,
            max_nodes=max_nodes,
            same_class_limit=0,
            same_file_limit=0,
            shared_field_limit=0,
        )

    if mode != "graph_rag":
        raise ValueError(f"Unknown mode: {mode}")

    return engine.query(
        query,
        seed_top_k=seed_top_k,
        hops=hops,
        max_nodes=max_nodes,
    )


def _default_llm_from_env(*, dry_llm: bool) -> Any:
    if dry_llm:
        return DummyEchoLLM()
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    return OpenAICompatibleChatClient(OpenAICompatibleConfig(base_url=base_url, api_key=api_key, model=model))


def run_benchmark(
    *,
    project_root: Path,
    tasks: Sequence[BenchmarkTask],
    out_dir: Path,
    modes: Sequence[str] = ("graph_rag", "vector_only"),
    prefer_tree_sitter: bool = True,
    # Context engine params
    seed_top_k: int = 5,
    hops: int = 2,
    max_nodes: int = 30,
    # Agent params
    max_iters: int = 3,
    use_docker: bool = False,
    docker_image: str = "python:3.10-slim",
    verify_cmds: Optional[List[str]] = None,
    allow_cmds: Optional[List[str]] = None,
    dry_llm: bool = False,
) -> Dict[str, Any]:
    """Run tasks under different retrieval modes and compute before/after metrics.

    This function does NOT modify `project_root` in-place: RefactoringAgent always
    works on a copied sandbox directory.
    """
    project_root = project_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute baseline maintainability for the *original* project.
    pre_project_maint = compute_project_maintainability(str(project_root), prefer_tree_sitter=prefer_tree_sitter)

    llm = _default_llm_from_env(dry_llm=dry_llm)

    results: Dict[str, Any] = {
        "project_root": str(project_root),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modes": list(modes),
        "tasks": [asdict(t) for t in tasks],
        "pre_project_maintainability": asdict(pre_project_maint),
        "runs": {},
    }

    for mode in modes:
        results["runs"][mode] = {}
        for task in tasks:
            task_dir = out_dir / mode / task.name
            task_dir.mkdir(parents=True, exist_ok=True)

            # 1) Build context pack
            pack = build_context_pack(
                str(project_root),
                query=task.query or (task.focus_node or ""),
                mode=mode,
                seed_top_k=seed_top_k,
                hops=hops,
                max_nodes=max_nodes,
                prefer_tree_sitter=prefer_tree_sitter,
            )
            (task_dir / "context_pack.json").write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")

            # 2) Coverage metric (benchmark style)
            cov = context_pack_file_coverage(pack, project_root=str(project_root), expected_files_rel=task.expected_files)
            (task_dir / "context_coverage.json").write_text(json.dumps(cov, ensure_ascii=False, indent=2), encoding="utf-8")

            # 3) Pre metrics (focus-level) on original project
            pre_method_mm = compute_method_metrics(str(project_root), prefer_tree_sitter=prefer_tree_sitter)
            focus_before = task.focus_node or pack.get("focus_node") or ""
            pre_focus_method: Optional[MethodMetrics] = pre_method_mm.get(focus_before)

            pre_graph = build_code_graph(str(project_root), prefer_tree_sitter=prefer_tree_sitter)
            pre_risk = compute_change_risk(pre_graph, focus_before)

            # 4) Run the refactoring agent
            cfg = AgentConfig(
                project_dir=project_root,
                max_iters=max_iters,
                use_docker=use_docker,
                docker_image=docker_image,
                default_verify_cmds=verify_cmds,
                allowed_commands=allow_cmds,
            )
            agent = RefactoringAgent(llm=llm, cfg=cfg)
            summary = agent.run(request=task.request, context_pack=pack)
            (task_dir / "agent_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

            # 5) Post metrics on agent's sandbox workdir (even on failure, work_dir exists)
            work_dir = Path(summary.get("work_dir") or "").resolve()
            post_project_maint = compute_project_maintainability(str(work_dir), prefer_tree_sitter=prefer_tree_sitter)
            post_method_mm = compute_method_metrics(str(work_dir), prefer_tree_sitter=prefer_tree_sitter)

            focus_after = task.focus_node_after or focus_before
            post_focus_method: Optional[MethodMetrics] = post_method_mm.get(focus_after)

            post_graph = build_code_graph(str(work_dir), prefer_tree_sitter=prefer_tree_sitter)
            post_risk = compute_change_risk(post_graph, focus_after)

            # 6) Normalize impacted files relative to project roots for easier reading
            def _rel_files(files: List[str], root: Path) -> List[str]:
                out: List[str] = []
                for f in files:
                    out.append(relpath_under(str(root), f))
                return sorted(list(dict.fromkeys(out)))

            run_record: Dict[str, Any] = {
                "status": summary.get("status"),
                "attempts": summary.get("attempts"),
                "objective": summary.get("objective"),
                "modified_files": summary.get("modified_files", []),
                "work_dir": str(work_dir),
                "context_coverage": cov,
                "focus_before": focus_before,
                "focus_after": focus_after,
                "pre_focus_method": asdict(pre_focus_method) if pre_focus_method else None,
                "post_focus_method": asdict(post_focus_method) if post_focus_method else None,
                "pre_project_maintainability": asdict(pre_project_maint),
                "post_project_maintainability": asdict(post_project_maint),
                "pre_change_risk": asdict(pre_risk),
                "post_change_risk": asdict(post_risk),
                "pre_impacted_files": _rel_files(pre_risk.impacted_files, project_root),
                "post_impacted_files": _rel_files(post_risk.impacted_files, work_dir),
            }

            (task_dir / "run_record.json").write_text(json.dumps(run_record, ensure_ascii=False, indent=2), encoding="utf-8")
            results["runs"][mode][task.name] = run_record

    (out_dir / "benchmark_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results
