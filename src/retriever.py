"""
Backward-compatible retriever.

- `retrieve_context(node_id)` keeps your old behavior (in/out edges)
- `retrieve_context_pack(query)` returns the new structured Graph-RAG Context Pack
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import networkx as nx

from .vector_index import NodeVectorIndex
from .context_engine import GraphRAGContextEngine


class GraphRetriever:
    def __init__(self, graph: nx.DiGraph, vector_index: Optional[NodeVectorIndex] = None):
        self.graph = graph
        self.vector_index = vector_index
        self._engine: Optional[GraphRAGContextEngine] = None

        if vector_index is not None:
            self._engine = GraphRAGContextEngine(graph, vector_index)

    def retrieve_context(self, query_node_id: str, hops: int = 1) -> Dict[str, Any]:
        """
        Old-style graph retrieval: dependencies (out edges) + usages (in edges).
        """
        if query_node_id not in self.graph:
            return {"error": f"Node not found: {query_node_id}"}

        dependencies = []
        for neighbor in self.graph.successors(query_node_id):
            edge_data = self.graph.get_edge_data(query_node_id, neighbor) or {}
            dependencies.append(f"  -> {edge_data.get('relation')} -> {neighbor}")

        usages = []
        for neighbor in self.graph.predecessors(query_node_id):
            edge_data = self.graph.get_edge_data(neighbor, query_node_id) or {}
            usages.append(f"  <- {edge_data.get('relation')} <- {neighbor}")

        return {
            "focus_node": query_node_id,
            "dependencies": dependencies,
            "usages": usages,
        }

    def retrieve_context_pack(self, query: str, **kwargs) -> Dict:
        """
        New Graph-RAG retrieval: vector seeds + graph expansion + structured Context Pack.

        kwargs are forwarded to GraphRAGContextEngine.query().
        """
        if not self._engine:
            if not self.vector_index:
                raise RuntimeError("Vector index not provided. Build NodeVectorIndex and pass it into GraphRetriever.")
            self._engine = GraphRAGContextEngine(self.graph, self.vector_index)
        return self._engine.query(query, **kwargs)

