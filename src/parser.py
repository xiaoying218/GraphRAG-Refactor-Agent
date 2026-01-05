"""
Java repository parser for building a *traceable* code knowledge graph.

Output schema (dict):
{
  "classes": [ {...} ],
  "methods": [ {...} ],
  "fields": [ {...} ],
  "relationships": [ {"source": str, "target": str, "type": str, "file_path": str, "line": int} ]
}

Each method/field/class record includes traceability fields:
- file_path
- start_line, end_line
- signature
- docstring
"""
from __future__ import annotations

import os
from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .utils import extract_comment_spans, find_attached_doc, safe_read_text

# Optional dependency: tree-sitter Java.
#
# In many environments tree-sitter may not be installed. For a demo / benchmark runner,
# we provide a regex-based fallback parser so the repo still runs out-of-the-box.
#
# When tree-sitter is available, we use it (more accurate relationships and spans).
try:
    import tree_sitter_java as tsjava
    from tree_sitter import Language, Parser
    _TREE_SITTER_OK = True
except Exception:  # pragma: no cover
    tsjava = None
    Language = None
    Parser = None
    _TREE_SITTER_OK = False


def _iter_java_files(project_root: str) -> Iterable[str]:
    ignore = {
        ".git",
        ".refactor_agent_runs",
        "target",
        "build",
        ".gradle",
        "node_modules",
        "__MACOSX",
        "__pycache__",
    }
    for root, dirs, files in os.walk(project_root):
        # Prune ignored dirs in-place
        dirs[:] = [d for d in dirs if d not in ignore]
        for fn in files:
            if fn.endswith(".java"):
                yield os.path.join(root, fn)


def _node_text(code_bytes: bytes, node) -> str:
    return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def _node_lines(node) -> Tuple[int, int]:
    # tree-sitter rows are 0-indexed
    return int(node.start_point[0]) + 1, int(node.end_point[0]) + 1


def _compress_ws(s: str) -> str:
    return " ".join(s.replace("\n", " ").replace("\t", " ").split())


def _find_nodes_by_type(node, type_name: str) -> List:
    out = []
    stack = [node]
    while stack:
        n = stack.pop()
        if getattr(n, "type", None) == type_name:
            out.append(n)
        # children is a property in tree-sitter python
        stack.extend(getattr(n, "children", []) or [])
    return out


class JavaCodeParser:
    """
    Parse a Java project into a symbol table + relationships suitable for:
    - building a code knowledge graph (NetworkX)
    - building a vector index (TF-IDF/Embeddings)
    - later: powering an editing/refactoring agent

    Notes:
    - For a demo, we focus on class/method/field extraction and basic CALLS/READS.
    - Cross-class method resolution is intentionally conservative (best-effort).
    """

    def __init__(self, *, prefer_tree_sitter: bool = True) -> None:
        """Create a Java parser.

        prefer_tree_sitter:
          - True (default): use tree-sitter when available; otherwise fallback to regex.
          - False: always use regex fallback (useful for ultra-lightweight demos).
        """
        self.use_tree_sitter = bool(prefer_tree_sitter and _TREE_SITTER_OK)

        # Tree-sitter path (more accurate spans / relationships)
        if self.use_tree_sitter:
            # Build Language object (API differs across versions)
            try:
                self.JAVA_LANGUAGE = Language(tsjava.language())
            except Exception:
                # Some versions may already return a Language-like object
                self.JAVA_LANGUAGE = tsjava.language()

            # Build parser (API differs across versions)
            try:
                self.parser = Parser(self.JAVA_LANGUAGE)
            except Exception:
                self.parser = Parser()
                if hasattr(self.parser, "set_language"):
                    self.parser.set_language(self.JAVA_LANGUAGE)
        else:
            self.JAVA_LANGUAGE = None
            self.parser = None

    def parse_project(self, project_root: str) -> Dict:
        """Parse a Java project directory into symbols and relationships.

        The output format is stable across both implementations:
          - tree-sitter mode (preferred when available)
          - regex fallback mode (best-effort)
        """
        if not self.use_tree_sitter:
            return self._parse_project_regex(project_root)

        classes: List[Dict] = []
        methods: List[Dict] = []
        fields: List[Dict] = []
        relationships: List[Dict] = []

        # Project-wide indexes to support lightweight intra-repo resolution.
        # - global_method_name_map: methodName -> [Type.methodName]
        # - global_type_to_methods: Type -> {methodName -> Type.methodName}
        # - global_method_return_type: Type.methodName -> "ReturnType" (best-effort)
        # - global_types: set of type names (classes + interfaces)
        global_method_name_map: Dict[str, List[str]] = {}
        global_type_to_methods: Dict[str, Dict[str, str]] = {}
        global_method_return_type: Dict[str, str] = {}
        global_types: Set[str] = set()

        # First pass: parse each file, build symbols; also build global indexes.
        per_file_symbols: List[Tuple[str, Dict]] = []
        for file_path in _iter_java_files(project_root):
            symbols = self._parse_file_symbols(file_path)
            per_file_symbols.append((file_path, symbols))

            for t in symbols["classes"]:
                global_types.add(str(t["name"]))

            for m in symbols["methods"]:
                global_method_name_map.setdefault(m["name"], []).append(m["id"])
                global_type_to_methods.setdefault(str(m["class"]), {})[str(m["name"])] = str(m["id"])
                global_method_return_type[str(m["id"])] = str(m.get("return_type", "") or "")

        # Second pass: parse relationships with access to method map.
        for file_path, symbols in per_file_symbols:
            # Add classes/methods/fields
            classes.extend(symbols["classes"])
            methods.extend(symbols["methods"])
            fields.extend(symbols["fields"])

            # Relationship extraction depends on method ids in the same file/class.
            rels = self._parse_file_relationships(
                file_path=file_path,
                class_records=symbols["classes"],
                method_records=symbols["methods"],
                field_records=symbols["fields"],
                global_method_name_map=global_method_name_map,
                global_type_to_methods=global_type_to_methods,
                global_method_return_type=global_method_return_type,
                global_types=global_types,
            )
            relationships.extend(rels)

        return {
            "classes": classes,
            "methods": methods,
            "fields": fields,
            "relationships": relationships,
        }

    # -------------------------
    # File-level parsing
    # -------------------------
    def _parse_file_symbols(self, file_path: str) -> Dict:
        source_text = safe_read_text(file_path)
        comments = extract_comment_spans(source_text)
        code_bytes = source_text.encode("utf-8", errors="ignore")

        tree = self.parser.parse(code_bytes)
        root = tree.root_node

        classes: List[Dict] = []
        methods: List[Dict] = []
        fields: List[Dict] = []

        # Parse both classes and interfaces (important for resolving interface-method calls
        # such as IPromotionStrategy.execute()).
        type_nodes: List[Tuple[str, object]] = []
        for n in _find_nodes_by_type(root, "class_declaration"):
            type_nodes.append(("Class", n))
        for n in _find_nodes_by_type(root, "interface_declaration"):
            type_nodes.append(("Interface", n))

        for type_kind, class_node in type_nodes:
            name_node = class_node.child_by_field_name("name")
            if not name_node:
                continue
            class_name = _node_text(code_bytes, name_node)

            cls_start, cls_end = _node_lines(class_node)
            cls_doc = find_attached_doc(comments, cls_start) or ""

            # Signature: take the first line of class declaration (best-effort)
            # We take bytes from start_byte to the first "{" in the class node text.
            class_text = _node_text(code_bytes, class_node)
            sig = class_text.split("{", 1)[0].strip()
            sig = _compress_ws(sig)

            class_id = class_name  # keep simple for demo

            classes.append({
                "id": class_id,
                "name": class_name,
                "type": type_kind,
                "file_path": file_path,
                "start_line": cls_start,
                "end_line": cls_end,
                "signature": sig,
                "docstring": cls_doc,
            })

            # Extract members *inside this class*
            body_node = class_node.child_by_field_name("body")
            if not body_node:
                continue

            # Field declarations
            for field_decl in _find_nodes_by_type(body_node, "field_declaration"):
                fld_start, fld_end = _node_lines(field_decl)
                fld_doc = find_attached_doc(comments, fld_start) or ""

                # Type node (best-effort)
                type_node = field_decl.child_by_field_name("type")
                fld_type = _compress_ws(_node_text(code_bytes, type_node)) if type_node else ""

                # One declaration can declare multiple variables: int a, b;
                for var_decl in _find_nodes_by_type(field_decl, "variable_declarator"):
                    var_name_node = var_decl.child_by_field_name("name")
                    if not var_name_node:
                        continue
                    fld_name = _node_text(code_bytes, var_name_node)
                    field_id = f"{class_name}.{fld_name}"

                    # Signature: "Type name" + optional initializer excerpt
                    var_text = _node_text(code_bytes, var_decl)
                    signature = f"{fld_type} {var_text}".strip()
                    signature = _compress_ws(signature)

                    fields.append({
                        "id": field_id,
                        "name": fld_name,
                        "class": class_id,
                        "type": "Field",
                        "file_path": file_path,
                        "start_line": fld_start,
                        "end_line": fld_end,
                        "signature": signature,
                        "declared_type": fld_type,
                        "docstring": fld_doc,
                    })

            # Method declarations
            for method_node in _find_nodes_by_type(body_node, "method_declaration"):
                name_node = method_node.child_by_field_name("name")
                if not name_node:
                    continue
                method_name = _node_text(code_bytes, name_node)
                m_start, m_end = _node_lines(method_node)
                m_doc = find_attached_doc(comments, m_start) or ""

                # Return type (best-effort) for later lightweight call-chain inference.
                rt_node = method_node.child_by_field_name("type")
                return_type = _compress_ws(_node_text(code_bytes, rt_node)) if rt_node else ""

                # Signature: from method start to body start (or to end if no body)
                body = method_node.child_by_field_name("body")
                end_byte = body.start_byte if body else method_node.end_byte
                sig_bytes = code_bytes[method_node.start_byte:end_byte]
                signature = _compress_ws(sig_bytes.decode("utf-8", errors="ignore"))

                # ID: keep stable and human-friendly.
                # For demo: Class.methodName
                method_id = f"{class_name}.{method_name}"

                methods.append({
                    "id": method_id,
                    "name": method_name,
                    "class": class_id,
                    "type": "Method",
                    "file_path": file_path,
                    "start_line": m_start,
                    "end_line": m_end,
                    "signature": signature,
                    "return_type": return_type,
                    "docstring": m_doc,
                })

        return {"classes": classes, "methods": methods, "fields": fields}

    def _parse_file_relationships(
        self,
        file_path: str,
        class_records: List[Dict],
        method_records: List[Dict],
        field_records: List[Dict],
        global_method_name_map: Dict[str, List[str]],
        global_type_to_methods: Dict[str, Dict[str, str]],
        global_method_return_type: Dict[str, str],
        global_types: Set[str],
    ) -> List[Dict]:
        """
        Extract intra-file relationships. For the demo we extract:
          - CALLS: method -> method
          - READS: method -> field (best-effort; only field_access nodes)
          - WRITES: method -> field (best-effort; only assignment_expression to field_access)

        Relationship dict includes file_path and an approximate line.
        """
        source_text = safe_read_text(file_path)
        code_bytes = source_text.encode("utf-8", errors="ignore")

        tree = self.parser.parse(code_bytes)
        root = tree.root_node

        # Build quick lookup for same-class members + field types
        class_to_methods: Dict[str, Dict[str, str]] = {}
        for m in method_records:
            class_to_methods.setdefault(str(m["class"]), {})[str(m["name"])] = str(m["id"])

        class_to_fields: Dict[str, Dict[str, str]] = {}
        class_to_field_types: Dict[str, Dict[str, str]] = {}
        for f in field_records:
            cls = str(f.get("class") or "")
            nm = str(f.get("name") or "")
            if not cls or not nm:
                continue
            class_to_fields.setdefault(cls, {})[nm] = str(f["id"])
            class_to_field_types.setdefault(cls, {})[nm] = str(f.get("declared_type") or "")

        def _split_generic_args(s: str) -> List[str]:
            # Split generics by commas at depth=0: "A<B,C<D>>" -> ["B", "C<D>"]
            out: List[str] = []
            buf: List[str] = []
            depth = 0
            for ch in s:
                if ch == "<":
                    depth += 1
                    buf.append(ch)
                elif ch == ">":
                    depth = max(0, depth - 1)
                    buf.append(ch)
                elif ch == "," and depth == 0:
                    part = "".join(buf).strip()
                    if part:
                        out.append(part)
                    buf = []
                else:
                    buf.append(ch)
            tail = "".join(buf).strip()
            if tail:
                out.append(tail)
            return out

        def _parse_type(type_str: str) -> Tuple[str, List[str]]:
            t = (type_str or "").strip()
            if not t:
                return "", []
            # strip array
            t = t.replace("[]", "").strip()
            # remove annotations like @Nullable
            t = re.sub(r"@\w+", "", t).strip()
            # base + generics
            base = t
            gens: List[str] = []
            if "<" in t and ">" in t:
                base = t.split("<", 1)[0].strip()
                inner = t[t.find("<") + 1 : t.rfind(">")].strip()
                gens = [g.strip() for g in _split_generic_args(inner) if g.strip()]
            # simple names (strip package)
            base_simple = base.split(".")[-1].strip()
            gen_simple: List[str] = []
            for g in gens:
                g = g.replace("[]", "").strip()
                gen_simple.append(g.split(".")[-1].strip())
            return base_simple, gen_simple

        COLLECTION_BASES = {"List", "ArrayList", "LinkedList", "Set", "HashSet", "Collection", "Iterable"}

        def _looks_like_external_type(simple_name: str) -> bool:
            if not simple_name:
                return False
            if simple_name in global_types:
                return False
            # java.lang / standard library shorthands
            if simple_name in {"System", "Math", "Objects", "String", "Integer", "Long", "Double", "Float", "Boolean"}:
                return True
            # heuristic: PascalCase but not defined in this repo
            return simple_name[:1].isupper()

        def _resolve_on_type(type_str: str, method_name: str) -> Optional[str]:
            base, gens = _parse_type(type_str)
            if not base:
                return None
            # direct dispatch on known type
            if base in global_type_to_methods and method_name in global_type_to_methods[base]:
                return global_type_to_methods[base][method_name]
            # collection element dispatch (List<T>.get(i).execute())
            if base in COLLECTION_BASES and gens:
                return _resolve_on_type(gens[0], method_name)
            return None

        def _build_local_symbols(method_node) -> Dict[str, str]:
            """Lightweight local symbol table: varName -> declaredType."""
            syms: Dict[str, str] = {}
            # Parameters
            for p in _find_nodes_by_type(method_node, "formal_parameter"):
                tnode = p.child_by_field_name("type")
                nnode = p.child_by_field_name("name")
                if not tnode or not nnode:
                    continue
                ptype = _compress_ws(_node_text(code_bytes, tnode))
                pname = _node_text(code_bytes, nnode)
                if pname:
                    syms[pname] = ptype

            # Local variables
            for lv in _find_nodes_by_type(method_node, "local_variable_declaration"):
                tnode = lv.child_by_field_name("type")
                ltype = _compress_ws(_node_text(code_bytes, tnode)) if tnode else ""
                for vd in _find_nodes_by_type(lv, "variable_declarator"):
                    nnode = vd.child_by_field_name("name")
                    if not nnode:
                        continue
                    vname = _node_text(code_bytes, nnode)
                    if vname and ltype:
                        syms[vname] = ltype
            return syms

        def _is_external_qualifier(obj_text: str, local_syms: Dict[str, str], field_types: Dict[str, str]) -> bool:
            t = (obj_text or "").strip()
            if not t:
                return False
            if t.startswith("java.") or t.startswith("javax."):
                return True
            root = t.split(".")[0]
            root_simple = root.split(".")[-1]
            if root_simple in local_syms or root_simple in field_types:
                return False
            return _looks_like_external_type(root_simple)

        def _infer_return_type_of_invocation(
            current_class_id: str,
            inv_node,
            local_syms: Dict[str, str],
            field_types: Dict[str, str],
            depth: int = 2,
        ) -> str:
            """Infer return type of an invocation (best-effort) to help resolve chained calls."""
            if depth <= 0 or not inv_node or getattr(inv_node, "type", None) != "method_invocation":
                return ""
            name_node = inv_node.child_by_field_name("name")
            if not name_node:
                return ""
            inner_name = _node_text(code_bytes, name_node)
            obj_node = inv_node.child_by_field_name("object")

            # Heuristic: collection.get(i) returns element type
            if inner_name == "get" and obj_node is not None:
                if obj_node.type in ("identifier", "scoped_identifier"):
                    base_var = _node_text(code_bytes, obj_node).split(".")[-1]
                    vtype = local_syms.get(base_var) or field_types.get(base_var) or ""
                    base, gens = _parse_type(vtype)
                    if base in COLLECTION_BASES and gens:
                        return gens[0]

            # Try resolve the inner invocation to an internal method id, then use known return type
            target = _resolve_invocation_target(current_class_id, inner_name, obj_node, local_syms, field_types, depth - 1)
            if target and not target.startswith("UNRESOLVED::") and not target.startswith("EXTERNAL::"):
                return str(global_method_return_type.get(target, "") or "")
            return ""

        def _resolve_invocation_target(
            current_class_id: str,
            callee_name: str,
            object_node,
            local_syms: Dict[str, str],
            field_types: Dict[str, str],
            depth: int = 2,
        ) -> str:
            """Resolve a method invocation target into either an internal id or EXTERNAL/UNRESOLVED."""
            # Unqualified: execute() or this.execute()
            if object_node is None or object_node.type in ("this", "super"):
                # same-class first
                tgt = class_to_methods.get(current_class_id, {}).get(callee_name)
                if tgt:
                    return tgt
                # unique global resolution
                candidates = global_method_name_map.get(callee_name, [])
                if len(candidates) == 1:
                    return candidates[0]
                return f"UNRESOLVED::{callee_name}"

            # Qualified call: obj.execute()
            obj_type: Optional[str] = None
            obj_text = _compress_ws(_node_text(code_bytes, object_node))

            if object_node.type in ("identifier", "scoped_identifier"):
                ident = obj_text.split(".")[-1]
                # variable dispatch
                if ident in local_syms:
                    obj_type = local_syms[ident]
                elif ident in field_types:
                    obj_type = field_types[ident]
                else:
                    # static call: SomeClass.execute()
                    if ident in global_types:
                        tgt = global_type_to_methods.get(ident, {}).get(callee_name)
                        if tgt:
                            return tgt
                    # external type
                    if _looks_like_external_type(ident):
                        return f"EXTERNAL::{ident}.{callee_name}"
                    return f"UNRESOLVED::{callee_name}"

            elif object_node.type == "field_access":
                # e.g., System.out.println() where object is "System.out"
                full_obj = _compress_ws(_node_text(code_bytes, object_node))
                if _is_external_qualifier(full_obj, local_syms, field_types):
                    return f"EXTERNAL::{full_obj}.{callee_name}"
                return f"UNRESOLVED::{callee_name}"

            elif object_node.type == "method_invocation":
                # chained call: strategies.get(i).execute()
                inferred = _infer_return_type_of_invocation(current_class_id, object_node, local_syms, field_types, depth=depth)
                if inferred:
                    obj_type = inferred

            # If we have an object type, try dispatch.
            if obj_type:
                tgt = _resolve_on_type(obj_type, callee_name)
                if tgt:
                    return tgt
                base, _ = _parse_type(obj_type)
                if _looks_like_external_type(base):
                    return f"EXTERNAL::{base}.{callee_name}"

            # fallback
            # if the method name exists in-repo, it's an unresolved internal call; otherwise likely external
            if callee_name in global_method_name_map:
                return f"UNRESOLVED::{callee_name}"
            if _is_external_qualifier(obj_text, local_syms, field_types):
                return f"EXTERNAL::{obj_text}.{callee_name}"
            return f"UNRESOLVED::{callee_name}"

        relationships: List[Dict] = []

        # Find type declarations (class + interface) and scan inside methods
        type_nodes: List[Tuple[str, object]] = []
        for n in _find_nodes_by_type(root, "class_declaration"):
            type_nodes.append(("Class", n))
        for n in _find_nodes_by_type(root, "interface_declaration"):
            type_nodes.append(("Interface", n))

        for _, type_node in type_nodes:
            name_node = type_node.child_by_field_name("name")
            if not name_node:
                continue
            class_name = _node_text(code_bytes, name_node)
            class_id = class_name

            body_node = type_node.child_by_field_name("body")
            if not body_node:
                continue

            for method_node in _find_nodes_by_type(body_node, "method_declaration"):
                mn_node = method_node.child_by_field_name("name")
                if not mn_node:
                    continue
                method_name = _node_text(code_bytes, mn_node)
                method_id = f"{class_name}.{method_name}"

                local_syms = _build_local_symbols(method_node)
                field_types = class_to_field_types.get(class_id, {})

                # 1) CALLS edges: method_invocation
                for inv in _find_nodes_by_type(method_node, "method_invocation"):
                    callee_node = inv.child_by_field_name("name")
                    if not callee_node:
                        continue
                    callee_name = _node_text(code_bytes, callee_node)
                    obj_node = inv.child_by_field_name("object")
                    line, _ = _node_lines(inv)

                    target_id = _resolve_invocation_target(class_id, callee_name, obj_node, local_syms, field_types)

                    relationships.append({
                        "source": method_id,
                        "target": target_id,
                        "type": "CALLS",
                        "file_path": file_path,
                        "line": line,
                    })

                # 2) READS edges: field_access nodes
                for fa in _find_nodes_by_type(method_node, "field_access"):
                    field_node = fa.child_by_field_name("field")
                    obj_node = fa.child_by_field_name("object")
                    if not field_node or not obj_node:
                        continue
                    field_name = _node_text(code_bytes, field_node)
                    obj_text = _compress_ws(_node_text(code_bytes, obj_node))
                    line, _ = _node_lines(fa)

                    if obj_node.type in ("this", "super"):
                        field_id = class_to_fields.get(class_id, {}).get(field_name, f"{class_name}.{field_name}")
                    else:
                        full_field = f"{obj_text}.{field_name}" if obj_text else field_name
                        if _is_external_qualifier(obj_text, local_syms, field_types):
                            field_id = f"EXTERNAL::{full_field}"
                        else:
                            field_id = f"UNRESOLVED_FIELD::{full_field}"

                    relationships.append({
                        "source": method_id,
                        "target": field_id,
                        "type": "READS",
                        "file_path": file_path,
                        "line": line,
                    })

                # 3) WRITES edges: assignment_expression where left is field_access
                for assign in _find_nodes_by_type(method_node, "assignment_expression"):
                    left = assign.child_by_field_name("left")
                    if not left or left.type != "field_access":
                        continue
                    field_node = left.child_by_field_name("field")
                    obj_node = left.child_by_field_name("object")
                    if not field_node or not obj_node:
                        continue
                    field_name = _node_text(code_bytes, field_node)
                    obj_text = _compress_ws(_node_text(code_bytes, obj_node))
                    line, _ = _node_lines(assign)

                    if obj_node.type in ("this", "super"):
                        field_id = class_to_fields.get(class_id, {}).get(field_name, f"{class_name}.{field_name}")
                    else:
                        full_field = f"{obj_text}.{field_name}" if obj_text else field_name
                        if _is_external_qualifier(obj_text, local_syms, field_types):
                            field_id = f"EXTERNAL::{full_field}"
                        else:
                            field_id = f"UNRESOLVED_FIELD::{full_field}"

                    relationships.append({
                        "source": method_id,
                        "target": field_id,
                        "type": "WRITES",
                        "file_path": file_path,
                        "line": line,
                    })

        return relationships


    # ---------------------------------------------------------------------
    # Regex fallback implementation (no tree-sitter)
    # ---------------------------------------------------------------------
    _RE_CLASS_DECL = re.compile(
        r"\b(class|interface)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b(?P<rest>[^\n{]*)\{"
    )
    _RE_EXTENDS = re.compile(r"\bextends\s+([A-Za-z_][A-Za-z0-9_]*)\b")

    # Method declaration (with body). Best-effort; ignores edge cases.
    _RE_METHOD_DECL = re.compile(
        r"^\s*(?:public|protected|private)?\s*(?:static\s+)?(?:final\s+)?(?P<rtype>[A-Za-z_][A-Za-z0-9_<>\[\]]*)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{\s*$"
    )

    # Constructor declaration (no return type). We keep it as a Method node
    # with id=Class.<init> for completeness.
    _RE_CTOR_DECL = re.compile(
        r"^\s*(?:public|protected|private)?\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{\s*$"
    )

    # Interface method (no body)
    _RE_INTERFACE_METHOD = re.compile(
        r"^\s*(?:public|protected|private)?\s*(?:static\s+)?(?:default\s+)?(?P<rtype>[A-Za-z_][A-Za-z0-9_<>\[\]]*)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*;\s*$"
    )

    _RE_FIELD_DECL = re.compile(
        r"^\s*(?:public|protected|private)?\s*(?:static\s+)?(?:final\s+)?(?P<type>[A-Za-z_][A-Za-z0-9_<>\[\]]*)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:=[^;]*)?;\s*$"
    )

    # Call patterns (line-level)
    _RE_CALL_QUAL = re.compile(r"\b(?P<recv>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
    _RE_CALL_UNQUAL = re.compile(r"\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")

    _JAVA_KEYWORDS_CALLLIKE = {
        "if", "for", "while", "switch", "catch", "new", "return", "throw", "synchronized",
        "try", "super", "this", "do", "else", "case", "assert",
    }

    @staticmethod
    def _brace_balance(line: str) -> int:
        # NOTE: best-effort; strings/comments can break this but acceptable for demo.
        return line.count("{") - line.count("}")

    @classmethod
    def _find_block_end(cls, lines: List[str], start_idx: int) -> int:
        """Return the inclusive line index where the '{' block opened at/after start_idx ends."""
        depth = 0
        started = False
        for i in range(start_idx, len(lines)):
            ln = lines[i]
            delta = cls._brace_balance(ln)
            if "{" in ln:
                started = True
            depth += delta
            if started and depth <= 0:
                return i
        return len(lines) - 1

    def _parse_project_regex(self, project_root: str) -> Dict:
        classes: List[Dict] = []
        methods: List[Dict] = []
        fields: List[Dict] = []
        relationships: List[Dict] = []

        global_method_name_map: Dict[str, List[str]] = {}
        global_type_to_methods: Dict[str, Dict[str, str]] = {}
        global_method_return_type: Dict[str, str] = {}
        global_types: Set[str] = set()
        extends_map: Dict[str, str] = {}

        per_file_symbols: List[Tuple[str, Dict]] = []

        # Pass 1: symbols
        for file_path in _iter_java_files(project_root):
            symbols = self._parse_file_symbols_regex(file_path)
            per_file_symbols.append((file_path, symbols))
            for t in symbols["classes"]:
                global_types.add(str(t["name"]))
                if t.get("extends"):
                    extends_map[str(t["name"])] = str(t["extends"])
            for m in symbols["methods"]:
                global_method_name_map.setdefault(str(m["name"]), []).append(str(m["id"]))
                global_type_to_methods.setdefault(str(m["class"]), {})[str(m["name"])] = str(m["id"])
                global_method_return_type[str(m["id"])] = str(m.get("return_type", "") or "")

        # Pass 2: relationships
        for file_path, symbols in per_file_symbols:
            classes.extend(symbols["classes"])
            methods.extend(symbols["methods"])
            fields.extend(symbols["fields"])
            relationships.extend(
                self._parse_file_relationships_regex(
                    file_path=file_path,
                    class_records=symbols["classes"],
                    method_records=symbols["methods"],
                    field_records=symbols["fields"],
                    global_method_name_map=global_method_name_map,
                    global_type_to_methods=global_type_to_methods,
                    global_types=global_types,
                    extends_map=extends_map,
                )
            )

        return {
            "classes": classes,
            "methods": methods,
            "fields": fields,
            "relationships": relationships,
        }

    def _parse_file_symbols_regex(self, file_path: str) -> Dict:
        source_text = safe_read_text(file_path)
        lines = source_text.splitlines()
        comments = extract_comment_spans(source_text)

        classes: List[Dict] = []
        methods: List[Dict] = []
        fields: List[Dict] = []

        # Find top-level class/interface declaration (best-effort)
        m = self._RE_CLASS_DECL.search(source_text)
        if not m:
            return {"classes": classes, "methods": methods, "fields": fields}

        type_kind = "Class" if m.group(1) == "class" else "Interface"
        class_name = m.group("name")
        rest = m.group("rest") or ""
        extends_match = self._RE_EXTENDS.search(rest)
        extends_name = extends_match.group(1) if extends_match else ""

        # Approximate class span by brace matching from the line containing the first '{'
        cls_start_line = source_text.count("\n", 0, m.start()) + 1
        cls_start_idx = max(0, cls_start_line - 1)
        cls_end_idx = self._find_block_end(lines, cls_start_idx)
        cls_end_line = cls_end_idx + 1
        cls_doc = find_attached_doc(comments, cls_start_line) or ""

        # Signature = declaration line (trimmed)
        sig = (lines[cls_start_idx] if cls_start_idx < len(lines) else "").strip()
        sig = _compress_ws(sig)

        classes.append({
            "id": class_name,
            "name": class_name,
            "type": type_kind,
            "file_path": file_path,
            "start_line": cls_start_line,
            "end_line": cls_end_line,
            "signature": sig,
            "docstring": cls_doc,
            "extends": extends_name,
        })

        # Scan members inside class block.
        in_class = False
        class_depth = 0
        in_method = False
        method_depth = 0
        for i in range(cls_start_idx, min(cls_end_idx + 1, len(lines))):
            ln = lines[i]
            if not in_class:
                if "{" in ln:
                    in_class = True
                    class_depth = 1
                continue

            # Track class brace depth to know when we are inside methods.
            class_depth += self._brace_balance(ln)
            if class_depth <= 0:
                break

            # Track method depth to avoid parsing fields inside methods.
            if in_method:
                method_depth += self._brace_balance(ln)
                if method_depth <= 0:
                    in_method = False
                    method_depth = 0
                continue

            # Interface methods (no body)
            mi = self._RE_INTERFACE_METHOD.match(ln)
            if mi and type_kind == "Interface":
                method_name = mi.group("name")
                start_line = i + 1
                m_doc = find_attached_doc(comments, start_line) or ""
                signature = _compress_ws(ln.strip())
                method_id = f"{class_name}.{method_name}"
                methods.append({
                    "id": method_id,
                    "name": method_name,
                    "class": class_name,
                    "type": "Method",
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": start_line,
                    "signature": signature,
                    "return_type": mi.group("rtype") or "",
                    "docstring": m_doc,
                })
                continue

            # Method declarations (with body)
            mm = self._RE_METHOD_DECL.match(ln)
            if mm:
                method_name = mm.group("name")
                start_line = i + 1
                m_doc = find_attached_doc(comments, start_line) or ""
                end_idx = self._find_block_end(lines, i)
                end_line = end_idx + 1
                signature = _compress_ws(ln.strip())
                method_id = f"{class_name}.{method_name}"
                methods.append({
                    "id": method_id,
                    "name": method_name,
                    "class": class_name,
                    "type": "Method",
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "signature": signature,
                    "return_type": mm.group("rtype") or "",
                    "docstring": m_doc,
                })
                # Enter method body tracking
                in_method = True
                method_depth = 1
                continue

            # Constructor declarations (treated as Class.<init>)
            mc = self._RE_CTOR_DECL.match(ln)
            if mc and mc.group("name") == class_name:
                start_line = i + 1
                m_doc = find_attached_doc(comments, start_line) or ""
                end_idx = self._find_block_end(lines, i)
                end_line = end_idx + 1
                signature = _compress_ws(ln.strip())
                method_id = f"{class_name}.<init>"
                methods.append({
                    "id": method_id,
                    "name": "<init>",
                    "class": class_name,
                    "type": "Method",
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "signature": signature,
                    "return_type": "",
                    "docstring": m_doc,
                })
                in_method = True
                method_depth = 1
                continue

            # Field declarations (only when not inside a method)
            mf = self._RE_FIELD_DECL.match(ln)
            if mf and "(" not in ln:
                fld_name = mf.group("name")
                fld_type = mf.group("type")
                start_line = i + 1
                fld_doc = find_attached_doc(comments, start_line) or ""
                field_id = f"{class_name}.{fld_name}"
                signature = _compress_ws(ln.strip())
                fields.append({
                    "id": field_id,
                    "name": fld_name,
                    "class": class_name,
                    "type": "Field",
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": start_line,
                    "signature": signature,
                    "declared_type": fld_type,
                    "docstring": fld_doc,
                })
                continue

        return {"classes": classes, "methods": methods, "fields": fields}

    def _parse_file_relationships_regex(
        self,
        *,
        file_path: str,
        class_records: List[Dict],
        method_records: List[Dict],
        field_records: List[Dict],
        global_method_name_map: Dict[str, List[str]],
        global_type_to_methods: Dict[str, Dict[str, str]],
        global_types: Set[str],
        extends_map: Dict[str, str],
    ) -> List[Dict]:
        source_text = safe_read_text(file_path)
        lines = source_text.splitlines()

        # Determine current class (best-effort: first class record)
        class_name = str(class_records[0]["name"]) if class_records else ""
        if not class_name:
            return []

        # Build same-class method lookup
        class_to_methods: Dict[str, Dict[str, str]] = {}
        for m in method_records:
            class_to_methods.setdefault(str(m["class"]), {})[str(m["name"])] = str(m["id"])

        relationships: List[Dict] = []

        # For each method with a span, scan its body lines for call patterns.
        for m in method_records:
            method_id = str(m["id"])
            start_line = int(m.get("start_line") or 0)
            end_line = int(m.get("end_line") or 0)
            if start_line <= 0 or end_line <= 0 or end_line < start_line:
                continue

            # Slice lines; include signature line to keep it simple.
            body_lines = lines[start_line - 1 : end_line]
            for offset, ln in enumerate(body_lines):
                lineno = start_line + offset

                # Qualified calls first
                for qm in self._RE_CALL_QUAL.finditer(ln):
                    recv = qm.group("recv")
                    name = qm.group("name")
                    if not name or name in self._JAVA_KEYWORDS_CALLLIKE:
                        continue

                    target_raw = ""
                    if recv == "this":
                        target_raw = f"{class_name}.{name}"
                    elif recv == "super":
                        parent = extends_map.get(class_name, "")
                        target_raw = f"{parent}.{name}" if parent else name
                    elif recv in global_types:
                        target_raw = f"{recv}.{name}"
                    else:
                        # unknown receiver type (variable instance) -> unresolved
                        target_raw = f"{recv}.{name}"

                    relationships.append({
                        "source": method_id,
                        "target": target_raw,
                        "type": "CALLS",
                        "file_path": file_path,
                        "line": lineno,
                    })

                # Unqualified calls
                for um in self._RE_CALL_UNQUAL.finditer(ln):
                    name = um.group("name")
                    if not name or name in self._JAVA_KEYWORDS_CALLLIKE:
                        continue
                    # Skip if this was part of a qualified call (already handled)
                    # e.g. 'LegacyScoringUtil.calculateBaseScore(' -> would match calculateBaseScore too
                    if "." in ln[max(0, um.start() - 2) : um.start()]:
                        continue

                    target_raw = ""
                    # Same class
                    same = class_to_methods.get(class_name, {}).get(name)
                    if same:
                        target_raw = same
                    else:
                        # Global unique name
                        cands = global_method_name_map.get(name, [])
                        if len(cands) == 1:
                            target_raw = cands[0]
                        else:
                            target_raw = name

                    relationships.append({
                        "source": method_id,
                        "target": target_raw,
                        "type": "CALLS",
                        "file_path": file_path,
                        "line": lineno,
                    })

        return relationships

