"""
Build a NetworkX code knowledge graph with traceable nodes.

The graph is designed for two purposes:
1) Visualization (optional)
2) Retrieval-time neighborhood expansion (Graph-RAG)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx


class CodeGraphBuilder:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def build_from_parsed_data(self, data: Dict) -> None:
        """
        Data format is produced by src.parser.JavaCodeParser.parse_project().
        """
        classes = data.get("classes", [])
        methods = data.get("methods", [])
        fields = data.get("fields", [])
        relationships = data.get("relationships", [])

        print(
            f"📊 [GraphBuilder] classes={len(classes)} methods={len(methods)} fields={len(fields)} rels={len(relationships)}"
        )

        # 1) Add class nodes
        for cls in classes:
            node_id = cls["id"]
            self.graph.add_node(
                node_id,
                **cls,
            )

        # 2) Add field nodes (+ membership edges)
        for fld in fields:
            node_id = fld["id"]
            self.graph.add_node(node_id, **fld)
            class_id = fld.get("class")
            if class_id:
                self.graph.add_edge(class_id, node_id, relation="CONTAINS", type="CONTAINS")

        # 3) Add method nodes (+ membership edges)
        for m in methods:
            node_id = m["id"]
            self.graph.add_node(node_id, **m)
            class_id = m.get("class")
            if class_id:
                self.graph.add_edge(class_id, node_id, relation="CONTAINS", type="CONTAINS")

        # Helper: normalize relationship targets.
        # We preserve explicit namespaces produced by the parser:
        #   - EXTERNAL::...
        #   - UNRESOLVED::...
        #   - UNRESOLVED_FIELD::...
        # so we don't double-prefix them.
        def _normalize_target_id(source_id: str, target_raw: str) -> str:
            if target_raw in self.graph:
                return target_raw
            if target_raw.startswith(("EXTERNAL::", "UNRESOLVED::", "UNRESOLVED_FIELD::")):
                return target_raw
            # If target looks like a fully qualified id and doesn't exist, keep it anyway.
            if "." in target_raw and "::" not in target_raw:
                return target_raw
            return f"UNRESOLVED::{target_raw}"

        # 4) Add relationship edges
        for rel in relationships:
            src = rel["source"]
            tgt_raw = rel["target"]
            rel_type = rel["type"]

            if src not in self.graph:
                # Create minimal node for source if missing
                self.graph.add_node(src, id=src, type="Unknown", name=src)

            tgt = _normalize_target_id(src, tgt_raw)
            if tgt not in self.graph:
                # Create minimal node for unresolved/external targets
                node_type = "Unresolved"
                display_name = tgt_raw
                if tgt.startswith("EXTERNAL::"):
                    node_type = "External"
                    display_name = tgt.split("::", 1)[1]
                elif tgt.startswith("UNRESOLVED_FIELD::"):
                    node_type = "Unresolved"
                    display_name = tgt.split("::", 1)[1]
                elif tgt.startswith("UNRESOLVED::"):
                    node_type = "Unresolved"
                    display_name = tgt.split("::", 1)[1]

                self.graph.add_node(tgt, id=tgt, type=node_type, name=display_name)

            self.graph.add_edge(
                src,
                tgt,
                relation=rel_type,
                type=rel_type,
                file_path=rel.get("file_path"),
                line=rel.get("line"),
            )

    def get_graph(self) -> nx.DiGraph:
        return self.graph

    # Optional: visualize graph (requires pyvis)
    def visualize(self, output_file: str = "code_graph.html") -> None:
        try:
            from pyvis.network import Network  # type: ignore
        except Exception as e:
            raise ImportError(
                "pyvis is required for visualization. Install with: pip install pyvis"
            ) from e

        net = Network(height="900px", width="100%", directed=True)

        # Add nodes with lightweight styling based on type
        for node_id, attrs in self.graph.nodes(data=True):
            ntype = attrs.get("type", "Unknown")
            title_parts = [f"{ntype}: {node_id}"]
            for k in ("file_path", "start_line", "end_line", "signature"):
                if attrs.get(k):
                    title_parts.append(f"{k}: {attrs.get(k)}")
            title = "\n".join(title_parts)

            color = "#CCCCCC"
            shape = "dot"
            if ntype == "Class":
                color = "#FF9999"
                shape = "box"
            elif ntype == "Method":
                color = "#99CCFF"
                shape = "ellipse"
            elif ntype == "Field":
                color = "#90EE90"
                shape = "dot"

            net.add_node(node_id, label=attrs.get("name", node_id), title=title, color=color, shape=shape)

        for u, v, attrs in self.graph.edges(data=True):
            rel = attrs.get("relation", "")
            net.add_edge(u, v, label=rel, title=rel, arrows="to")

        net.show(output_file)
        print(f"🖼️ [Visualizer] saved: {output_file}")
