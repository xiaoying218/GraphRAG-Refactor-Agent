"""
Graph-RAG Context Engine

Goal:
- Given a natural language query, first do vector search over graph nodes to find seed(s).
- Then expand the graph neighborhood (k-hop + rule-based expansion).
- Finally output a structured "Context Pack" (like IDE refactoring preview).

This is the *bridge* between your graph builder and the future refactoring agent.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from .utils import load_snippet
from .vector_index import NodeVectorIndex, ScoredNode


@dataclass
class ContextNode:
    node_id: str
    name: str
    type: str
    file_path: str
    start_line: int
    end_line: int
    signature: str
    docstring: str
    snippet: str
    role: str
    roles: List[str]
    highlight_span: Optional[Dict[str, int]]
    why: List[str]


class GraphRAGContextEngine:
    def __init__(
        self,
        graph: nx.DiGraph,
        vector_index: NodeVectorIndex,
        *,
        max_snippet_lines: int = 120,
    ) -> None:
        self.graph = graph
        self.vindex = vector_index
        self.max_snippet_lines = max_snippet_lines

        # Precompute helper indexes for rule-based expansion
        self.class_to_members: Dict[str, Set[str]] = defaultdict(set)
        self.file_to_nodes: Dict[str, Set[str]] = defaultdict(set)
        self.field_to_methods: Dict[str, Set[str]] = defaultdict(set)

        self._build_aux_indexes()

    def _build_aux_indexes(self) -> None:
        # class_to_members + file_to_nodes
        for nid, attrs in self.graph.nodes(data=True):
            cls = attrs.get("class") or (nid if attrs.get("type") == "Class" else None)
            if cls:
                self.class_to_members[str(cls)].add(nid)
            fp = attrs.get("file_path")
            if fp:
                self.file_to_nodes[str(fp)].add(nid)

        # field_to_methods via READS/WRITES edges
        for u, v, attrs in self.graph.edges(data=True):
            rel = attrs.get("relation") or attrs.get("type")
            if rel in ("READS", "WRITES"):
                # v should be a field node (or unresolved)
                self.field_to_methods[str(v)].add(str(u))

    # -------------------------
    # Public API
    # -------------------------
    def query(
        self,
        query: str,
        *,
        seed_top_k: int = 5,
        hops: int = 2,
        max_nodes: int = 30,
        same_class_limit: int = 8,
        same_file_limit: int = 8,
        shared_field_limit: int = 10,
    ) -> Dict:
        """
        Return a structured context pack.

        Parameters:
        - query: natural language or a node_id (if exact match)
        """
        seeds = self._select_seeds(query, seed_top_k=seed_top_k)

        selected: Set[str] = set()
        reasons: Dict[str, List[str]] = defaultdict(list)
        roles: Dict[str, Set[str]] = defaultdict(set)

        def add(nid: str, why: str, role: Optional[str] = None) -> None:
            if not nid:
                return
            if nid not in selected:
                selected.add(nid)
            if why and why not in reasons[nid]:
                reasons[nid].append(why)
            if role:
                roles[nid].add(role)

        # # If query is an exact node id, mark it as focus; otherwise take top seed.
        # focus_node_id = query if query in self.graph else (seeds[0].node_id if seeds else "")

        # # Always include focus (even when it is not selected by seeds for some reason).
        # if focus_node_id:
        #     add(focus_node_id, "focus node", role="focus")

        # # 1) add seed nodes
        # for s in seeds:
        #     add(s.node_id, f"seed via vector search (score={s.score:.3f})", role="seed")

        # # 2) call graph expansion: separate callees vs callers (IDE-like preview)
        # callees = self._expand_relation_dir([focus_node_id], relation="CALLS", hops=hops, direction="out")
        # callers = self._expand_relation_dir([focus_node_id], relation="CALLS", hops=hops, direction="in")
        # src/context_engine.py — GraphRAGContextEngine.query

        # focus_node_id = query if query in self.graph else (seeds[0].node_id if seeds else "")
        # related_nodes = {focus_node_id} if focus_node_id else set()

        # # --- NEW: Field -> Method bridge ---
        # focus_is_field = False
        # if focus_node_id:
        #     node_type = (self.graph.nodes[focus_node_id].get("type") or "").lower()
        #     focus_is_field = (node_type == "field") or (focus_node_id in self.field_to_methods)

        # field_usage_methods = []
        # if focus_is_field:
        #     field_usage_methods = sorted(self.field_to_methods.get(focus_node_id, []))
        #     # 把使用该 Field 的 Methods 先拉进 related_nodes（也可用你现有的 add() 给 role/why）
        #     for mid in field_usage_methods:
        #         if mid in self.graph:
        #             related_nodes.add(mid)

        # # 用 methods 作为 CALLS 扩展起点（否则从 Field 出发基本扩不出调用图）
        # call_roots = field_usage_methods if (focus_is_field and field_usage_methods) else [focus_node_id]

        # callees = self._expand_relation_dir(call_roots, relation="CALLS", direction="out", max_depth=hops, max_nodes=max_nodes)
        # callers = self._expand_relation_dir(call_roots, relation="CALLS", direction="in",  max_depth=hops, max_nodes=max_nodes)

        # If query is an exact node id, mark it as focus; otherwise take top seed.
        raw_focus_id = query if query in self.graph else (seeds[0].node_id if seeds else "")
        
        # --- Field -> Method bridge ---
        # If the focus is a Field (e.g., GOLDEN_RATIO), jump back to Methods that READ/WRITE it.
        focus_node_id = raw_focus_id
        focus_field_id: Optional[str] = None
        field_usage_methods: List[str] = []
        if focus_node_id:
            node_type = (self.graph.nodes[focus_node_id].get("type") or "").lower()
            focus_is_field = (node_type == "field") or (focus_node_id in self.field_to_methods)
            if focus_is_field:
                focus_field_id = focus_node_id
                field_usage_methods = sorted([m for m in self.field_to_methods.get(focus_field_id, []) if m in self.graph])
                # Prefer a Method node as the focus for downstream expansions.
                method_candidates = [m for m in field_usage_methods if (self.graph.nodes[m].get("type") or "").lower() == "method"]
                if method_candidates:
                    focus_node_id = method_candidates[0]
                elif field_usage_methods:
                    focus_node_id = field_usage_methods[0]
        
        # Always include focus + seed nodes in the selection set.
        if focus_node_id:
            add(focus_node_id, "focus node (exact match or top seed)", role="focus")
        for s in seeds:
            add(s.node_id, f"seed via vector search (score={s.score:.3f})", role="seed")
        if focus_field_id:
            add(focus_field_id, "focus field (matched by query) – expanded to usage methods", role="seed")
            for mid in field_usage_methods:
                add(mid, f"uses field {focus_field_id}", role="field_user")
        
        # Call graph expansion (IDE-like preview): separate callees vs callers
        call_roots = field_usage_methods if field_usage_methods else ([focus_node_id] if focus_node_id else [])
        callees = self._expand_relation_dir(call_roots, relation="CALLS", hops=hops, direction="out")
        callers = self._expand_relation_dir(call_roots, relation="CALLS", hops=hops, direction="in")


        for nid, depth in callees.items():
            add(nid, f"call-graph neighbor (CALLEE, depth={depth})", role="callee")
        for nid, depth in callers.items():
            add(nid, f"call-graph neighbor (CALLER, depth={depth})", role="caller")

        # 3) same-class members (siblings of focus)
        focus_cls = self.graph.nodes[focus_node_id].get("class") if focus_node_id in self.graph else None
        if focus_cls:
            members = list(self.class_to_members.get(str(focus_cls), set()))
            members = [m for m in members if self.graph.nodes[m].get("type") != "Class"]
            # Keep bounded
            for m in members[:same_class_limit]:
                add(m, f"same class member of {focus_cls}", role="same_class_member")

        # 4) shared field access: focus method -> field -> other methods (READS/WRITES separated)
        read_fields: Set[str] = set()
        written_fields: Set[str] = set()
        if focus_node_id:
            for _, v, attrs in self.graph.out_edges(focus_node_id, data=True):
                rel = attrs.get("relation") or attrs.get("type")
                if rel == "READS":
                    read_fields.add(str(v))
                elif rel == "WRITES":
                    written_fields.add(str(v))

        touched_fields = list(read_fields | written_fields)
        for f in touched_fields[:shared_field_limit]:
            add(f, "shared state (field accessed by focus)", role="shared_field")
            # Identify readers/writers of that field
            for u, _, attrs in self.graph.in_edges(f, data=True):
                rel = attrs.get("relation") or attrs.get("type")
                if rel == "READS":
                    add(u, f"shares field {f} (READS)", role="shared_field_reader")
                elif rel == "WRITES":
                    add(u, f"shares field {f} (WRITES)", role="shared_field_writer")

        # 5) same-file helpers: nodes in same file as focus (bounded)
        fp = self.graph.nodes[focus_node_id].get("file_path") if focus_node_id in self.graph else None
        if fp:
            nodes = list(self.file_to_nodes.get(str(fp), set()))
            nodes.sort(key=lambda nid: 0 if self.graph.nodes[nid].get("type") in ("Method", "Field") else 1)
            for nid in nodes[:same_file_limit]:
                add(nid, f"same file helper ({fp})", role="same_file_helper")

        # Enforce max_nodes budget
        if len(selected) > max_nodes:
            # Keep focus + seeds first, then by number of roles/reasons.
            seed_ids = [s.node_id for s in seeds]
            keep: List[str] = []
            if focus_node_id and focus_node_id in selected:
                keep.append(focus_node_id)
            for nid in seed_ids:
                if nid in selected and nid not in keep:
                    keep.append(nid)
            remaining = [nid for nid in selected if nid not in keep]
            remaining.sort(
                key=lambda nid: (len(roles.get(nid, set())), len(reasons.get(nid, []))),
                reverse=True,
            )
            keep.extend(remaining[: max_nodes - len(keep)])
            selected = set(keep)

        # Build node payloads (stable ordering like an IDE preview)
        nodes_payload = [
            self._build_context_node(
                nid,
                reasons[nid],
                roles=sorted(list(roles.get(nid, set()))),
                focus_id=focus_node_id,
            )
            for nid in selected
        ]

        # Sort nodes by role priority
        role_priority = {
            "focus": 0,
            "seed": 1,
            "field_user": 2,
            "callee": 2,
            "caller": 2,
            "shared_field_writer": 3,
            "shared_field_reader": 4,
            "shared_field": 5,
            "same_class_member": 6,
            "same_file_helper": 7,
        }

        def _node_sort_key(n: Dict) -> Tuple[int, str]:
            rls = n.get("roles") or []
            pri = min([role_priority.get(r, 99) for r in rls] or [99])
            return pri, str(n.get("node_id"))

        nodes_payload = sorted([node.__dict__ for node in nodes_payload], key=_node_sort_key)

        # Collect edges among selected nodes for explainability
        edges_payload = []
        for u, v, attrs in self.graph.edges(data=True):
            if u in selected and v in selected:
                edges_payload.append({
                    "source": u,
                    "target": v,
                    "relation": attrs.get("relation") or attrs.get("type"),
                    "file_path": attrs.get("file_path"),
                    "line": attrs.get("line"),
                })

        # IDE-style grouped views
        call_graph_view = {
            "callees": [{"node_id": nid, "depth": d} for nid, d in sorted(callees.items(), key=lambda kv: kv[1]) if nid in selected],
            "callers": [{"node_id": nid, "depth": d} for nid, d in sorted(callers.items(), key=lambda kv: kv[1]) if nid in selected],
        }

        # For IDE preview, keep fields separated by access type
        data_flow_view = {
            "read_fields": [fid for fid in sorted(read_fields) if fid in selected],
            "written_fields": [fid for fid in sorted(written_fields) if fid in selected],
        }

        same_class_view: List[str] = []
        if focus_cls:
            same_class_view = sorted([nid for nid in list(self.class_to_members.get(str(focus_cls), set())) if nid in selected])

        same_file_view: List[str] = []
        if fp:
            same_file_view = sorted([nid for nid in list(self.file_to_nodes.get(str(fp), set())) if nid in selected])

        return {
            "query": query,
            "focus_node": focus_node_id,
            "seed_nodes": [{"node_id": s.node_id, "score": s.score} for s in seeds],
            "nodes": nodes_payload,
            "edges": edges_payload,
            # New: IDE preview sub-views
            "focus": {"node_id": focus_node_id},
            "call_graph": call_graph_view,
            "data_flow": data_flow_view,
            "same_class": same_class_view,
            "same_file": same_file_view,
            "stats": {
                "seed_top_k": seed_top_k,
                "hops": hops,
                "selected_nodes": len(nodes_payload),
                "selected_edges": len(edges_payload),
            },
        }

    # -------------------------
    # Seed selection
    # -------------------------
    def _select_seeds(self, query: str, seed_top_k: int) -> List[ScoredNode]:
        if query in self.graph:
            return [ScoredNode(node_id=query, score=1.0)]
        return self.vindex.search(query, top_k=seed_top_k, min_score=0.0)

    # -------------------------
    # Graph expansion helpers
    # -------------------------
    def _expand_relation_dir(self, start_nodes: List[str], relation: str, hops: int, direction: str) -> Dict[str, int]:
        """Direction-aware BFS expansion.

        direction:
          - "out": follow out_edges (caller -> callee)
          - "in":  follow in_edges  (callee <- caller)
        Returns: node_id -> min depth (>=1)
        """
        if not start_nodes or not any(bool(s) for s in start_nodes):
            return {}
        depth_map: Dict[str, int] = {}
        q: deque[Tuple[str, int]] = deque()
        visited: Set[str] = set()

        for s in start_nodes:
            if not s:
                continue
            q.append((s, 0))
            visited.add(s)

        while q:
            nid, d = q.popleft()
            if d >= hops:
                continue

            if direction == "out":
                edges = self.graph.out_edges(nid, data=True)
                for _, v, attrs in edges:
                    rel = attrs.get("relation") or attrs.get("type")
                    if rel != relation:
                        continue
                    if v not in visited:
                        visited.add(v)
                        depth_map[str(v)] = min(depth_map.get(str(v), 10**9), d + 1)
                        q.append((v, d + 1))
            else:
                edges = self.graph.in_edges(nid, data=True)
                for u, _, attrs in edges:
                    rel = attrs.get("relation") or attrs.get("type")
                    if rel != relation:
                        continue
                    if u not in visited:
                        visited.add(u)
                        depth_map[str(u)] = min(depth_map.get(str(u), 10**9), d + 1)
                        q.append((u, d + 1))

        for s in start_nodes:
            depth_map.pop(s, None)
        return depth_map

    def _expand_relation(self, seeds: List[str], relation: str, hops: int) -> Dict[str, int]:
        """
        Expand by following edges (both directions) that match `relation` up to `hops`.
        Returns: node_id -> min depth
        """
        depth_map: Dict[str, int] = {}
        q: deque[Tuple[str, int]] = deque()
        visited: Set[str] = set()

        for s in seeds:
            q.append((s, 0))
            visited.add(s)

        while q:
            nid, d = q.popleft()
            if d >= hops:
                continue

            # out edges
            for _, v, attrs in self.graph.out_edges(nid, data=True):
                rel = attrs.get("relation") or attrs.get("type")
                if rel != relation:
                    continue
                if v not in visited:
                    visited.add(v)
                    depth_map[str(v)] = min(depth_map.get(str(v), 10**9), d + 1)
                    q.append((v, d + 1))

            # in edges
            for u, _, attrs in self.graph.in_edges(nid, data=True):
                rel = attrs.get("relation") or attrs.get("type")
                if rel != relation:
                    continue
                if u not in visited:
                    visited.add(u)
                    depth_map[str(u)] = min(depth_map.get(str(u), 10**9), d + 1)
                    q.append((u, d + 1))

        # Remove seeds
        for s in seeds:
            depth_map.pop(s, None)
        return depth_map

    # -------------------------
    # Node payload
    # -------------------------
    def _build_context_node(
        self,
        node_id: str,
        why: List[str],
        *,
        roles: List[str],
        focus_id: str,
    ) -> ContextNode:
        """Build a traceable, IDE-like node payload.

        roles: all roles assigned during selection.
        focus_id: focus node id (used for ...)
        """
        attrs = self.graph.nodes[node_id]
        ntype = str(attrs.get("type", "Unknown"))
        name = str(attrs.get("name") or node_id)
        fp = str(attrs.get("file_path", "") or "")
        start_line = int(attrs.get("start_line", 0) or 0)
        end_line = int(attrs.get("end_line", 0) or 0)
        signature = str(attrs.get("signature", "") or "")
        docstring = str(attrs.get("docstring", "") or "")

        snippet = ""
        if fp and start_line and end_line:
            snippet = load_snippet(fp, start_line, end_line, max_lines=self.max_snippet_lines)

        # Primary role (for UI / sorting) based on a simple priority.
        role_priority = {
            "focus": 0,
            "seed": 1,
            "callee": 2,
            "caller": 2,
            "shared_field_writer": 3,
            "shared_field_reader": 4,
            "shared_field": 5,
            "same_class_member": 6,
            "same_file_helper": 7,
        }
        primary_role = "other"
        if node_id == focus_id and "focus" not in roles:
            roles = ["focus"] + roles
        if roles:
            primary_role = sorted(roles, key=lambda r: role_priority.get(r, 99))[0]

        # Highlight span (approx): find the first occurrence of the symbol name in the snippet.
        highlight_span: Optional[Dict[str, int]] = None
        token = str(attrs.get("name") or "")
        if snippet and token and start_line:
            for idx, line in enumerate(snippet.splitlines()):
                if token in line:
                    highlight_span = {"start_line": start_line + idx, "end_line": start_line + idx}
                    break
        if highlight_span is None and fp and start_line:
            highlight_span = {"start_line": start_line, "end_line": start_line}

        return ContextNode(
            node_id=node_id,
            name=name,
            type=ntype,
            file_path=fp,
            start_line=start_line,
            end_line=end_line,
            signature=signature,
            docstring=docstring,
            snippet=snippet,
            role=primary_role,
            roles=roles,
            highlight_span=highlight_span,
            why=why,
        )

