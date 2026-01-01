"""
Vector index for code graph nodes.

For the demo we implement TF-IDF (offline-friendly).
Later you can swap to embedding-based retrieval (OpenAI / local embedding models)
without changing the Graph-RAG interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils import normalize_code_text


@dataclass
class ScoredNode:
    node_id: str
    score: float


class NodeVectorIndex:
    def __init__(self) -> None:
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.node_ids: List[str] = []
        self.matrix = None  # scipy sparse matrix

    @staticmethod
    def _node_to_text(node_id: str, attrs: Dict) -> str:
        parts: List[str] = []
        ntype = attrs.get("type", "Unknown")
        parts.append(f"{ntype} {node_id}")
        if attrs.get("name"):
            parts.append(str(attrs["name"]))
        if attrs.get("signature"):
            parts.append(str(attrs["signature"]))
        if attrs.get("docstring"):
            parts.append(str(attrs["docstring"]))
        # Light metadata can help retrieval
        if attrs.get("class"):
            parts.append(f"class {attrs['class']}")
        if attrs.get("file_path"):
            parts.append(f"file {attrs['file_path']}")
        return normalize_code_text(" ".join(parts))

    def build_from_graph(self, graph) -> None:
        texts: List[str] = []
        node_ids: List[str] = []
        for node_id, attrs in graph.nodes(data=True):
            node_ids.append(node_id)
            texts.append(self._node_to_text(node_id, attrs))

        self.node_ids = node_ids
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[ScoredNode]:
        if not self.vectorizer or self.matrix is None:
            raise RuntimeError("Vector index is not built. Call build_from_graph() first.")

        q = normalize_code_text(query)
        q_vec = self.vectorizer.transform([q])

        # cosine similarity for TF-IDF vectors: dot product is enough because vectors are L2-normalized by default.
        scores = (self.matrix @ q_vec.T).toarray().ravel()
        if scores.size == 0:
            return []

        top_idx = np.argsort(scores)[::-1][: top_k * 2]  # take a bit more then filter
        out: List[ScoredNode] = []
        for idx in top_idx:
            s = float(scores[idx])
            if s <= min_score:
                continue
            out.append(ScoredNode(node_id=self.node_ids[idx], score=s))
            if len(out) >= top_k:
                break
        return out
