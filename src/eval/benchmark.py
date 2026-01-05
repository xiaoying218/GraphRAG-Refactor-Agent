from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx

from ..agent.agent import AgentConfig, RefactoringAgent
from ..agent.llm import DummyEchoLLM, OpenAICompatibleChatClient, OpenAICompatibleConfig
from ..agent.sandbox import SandboxConfig
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

    vindex = NodeVectorIndex(project_root=project_root)
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
        # Vector-only ablation: no graph-hop expansion, but we still allow limited same-file context
        # so the model can see nearby helpers/imports and avoid trivial "missing symbol" failures.
        return engine.query(
            query,
            seed_top_k=seed_top_k,
            hops=0,
            max_nodes=max_nodes,
            same_class_limit=2,
            same_file_limit=5,
            shared_field_limit=1,
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
    restrict_vector_only_tools: bool = False,
    dry_llm: bool = False,
    accept_mode: str = "strict",  # strict|semantic
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
            # Inject benchmark hard requirements into the context pack so the agent can enforce them.
            pack["expected_files"] = list(task.expected_files or [])
            if task.focus_node_after:
                pack["focus_node_after"] = task.focus_node_after
            if task.focus_node:
                pack["focus_node"] = task.focus_node

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
            mode_allow_cmds = allow_cmds
            if restrict_vector_only_tools and mode == "vector_only":
                # Make the ablation between retrieval modes more meaningful by
                # restricting shell-based code search in vector_only runs.
                base_cmds = mode_allow_cmds or SandboxConfig(root_dir=project_root).allowed_commands
                mode_allow_cmds = [c for c in base_cmds if c not in ("rg", "grep", "find")]

            cfg = AgentConfig(
                project_dir=project_root,
                max_iters=max_iters,
                use_docker=use_docker,
                docker_image=docker_image,
                default_verify_cmds=verify_cmds,
                allowed_commands=mode_allow_cmds,
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
                "allowed_commands": mode_allow_cmds,
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

            # 7) Lightweight acceptance checks
            # We compute BOTH:
            # - strict: requires exact expected file paths + exact focus_node_after method id (if provided)
            # - semantic: allows reasonable variations (e.g., file path differs but class exists; method moved but name preserved)
            # The final accept_ok is controlled by accept_mode.
            accept_strict_ok = True
            accept_strict_reasons: List[str] = []
            accept_semantic_ok = True
            accept_semantic_reasons: List[str] = []

            def _looks_like_method_id(s: Optional[str]) -> bool:
                if not s:
                    return False
                s = str(s).strip()
                if not s:
                    return False
                if "," in s or " " in s or "(" in s or ")" in s:
                    return False
                # method ids are typically "Class.method" (exactly one dot)
                if s.count(".") != 1:
                    return False
                left, right = s.split(".", 1)
                return bool(left) and bool(right)

            # ---- Focus method acceptance ----
            if _looks_like_method_id(task.focus_node_after):
                if post_focus_method is None:
                    accept_strict_ok = False
                    accept_strict_reasons.append("post_focus_method_missing")

                    # Semantic fallback: allow the method to have moved to a different class
                    method_name = str(task.focus_node_after).split(".", 1)[1]
                    candidates = [mid for mid in post_method_mm.keys() if mid.endswith("." + method_name)]
                    if candidates:
                        accept_semantic_reasons.append(
                            "post_focus_method_missing_but_found_similar_method: " + ", ".join(candidates[:3])
                        )
                    else:
                        accept_semantic_ok = False
                        accept_semantic_reasons.append("post_focus_method_missing")

            # ---- Expected files acceptance ----
            missing_expected: List[str] = []
            for f in task.expected_files:
                f = str(f or "").strip()
                if not f:
                    continue
                if not (work_dir / f).exists():
                    missing_expected.append(f)

            if missing_expected:
                accept_strict_ok = False
                accept_strict_reasons.append("missing_expected_files: " + ", ".join(missing_expected))

                # Semantic fallback: if the expected file is missing, but the class exists somewhere else, allow it.
                # (This avoids punishing reasonable file/package layout differences while still checking intent.)
                class_index: Dict[str, str] = {}
                try:
                    for dirpath, _, filenames in os.walk(work_dir):
                        for fn in filenames:
                            if not fn.endswith(".java"):
                                continue
                            abs_fp = Path(dirpath) / fn
                            rel_fp = abs_fp.relative_to(work_dir).as_posix()
                            try:
                                txt = abs_fp.read_text(encoding="utf-8", errors="ignore")
                            except Exception:
                                continue
                            for m in re.finditer(r"\b(class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)\b", txt):
                                cname = m.group(2)
                                if cname and cname not in class_index:
                                    class_index[cname] = rel_fp
                except Exception:
                    class_index = {}

                still_missing: List[str] = []
                for f in missing_expected:
                    base = Path(f).name
                    cname = base[:-5] if base.endswith(".java") else base
                    if cname and cname in class_index:
                        accept_semantic_reasons.append(f"expected_file_missing_but_class_found:{cname}@{class_index[cname]}")
                    else:
                        still_missing.append(f)

                if still_missing:
                    accept_semantic_ok = False
                    accept_semantic_reasons.append("missing_expected_files: " + ", ".join(still_missing))

            # Select final acceptance
            mode_sel = (accept_mode or "strict").strip().lower()
            if mode_sel not in {"strict", "semantic"}:
                mode_sel = "strict"
            accept_ok = accept_semantic_ok if mode_sel == "semantic" else accept_strict_ok
            accept_reasons = accept_semantic_reasons if mode_sel == "semantic" else accept_strict_reasons

            run_record["accept_mode"] = mode_sel
            run_record["accept_ok"] = accept_ok
            run_record["accept_reasons"] = accept_reasons
            run_record["accept_strict_ok"] = accept_strict_ok
            run_record["accept_strict_reasons"] = accept_strict_reasons
            run_record["accept_semantic_ok"] = accept_semantic_ok
            run_record["accept_semantic_reasons"] = accept_semantic_reasons

            if run_record.get("status") == "success" and not accept_ok:
                run_record["status"] = "failed_acceptance"
            (task_dir / "run_record.json").write_text(json.dumps(run_record, ensure_ascii=False, indent=2), encoding="utf-8")
            results["runs"][mode][task.name] = run_record

    (out_dir / "benchmark_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results

