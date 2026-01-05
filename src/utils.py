"""
Utility helpers for Graph-RAG on code repositories.

This module is deliberately dependency-light so the demo can run offline.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


_CAMEL_1 = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_2 = re.compile(r"([A-Z]+)([A-Z][a-z])")


def normalize_code_text(text: str) -> str:
    """
    Normalize code-ish text for lexical vector search (TF-IDF/BM25 style).

    - split CamelCase: myMethodName -> my Method Name
    - split snake_case / dotted.names / :: names
    - lowercase
    """
    if not text:
        return ""
    t = text
    t = t.replace("::", " ")
    t = t.replace(".", " ").replace("_", " ").replace("/", " ")
    t = _CAMEL_2.sub(r"\1 \2", t)
    t = _CAMEL_1.sub(r"\1 \2", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


@dataclass(frozen=True)
class CommentSpan:
    start_line: int
    end_line: int
    text: str
    kind: str  # "javadoc" | "block" | "line"


_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//.*?$", re.MULTILINE)


def extract_comment_spans(source_text: str) -> List[CommentSpan]:
    """
    Extract comment spans by line numbers using regex.

    This is not a perfect Java lexer (comments inside strings are tricky),
    but it's good enough for a demo and for capturing Javadoc blocks.
    """
    spans: List[CommentSpan] = []

    # Block comments (including Javadoc)
    for m in _BLOCK_COMMENT_RE.finditer(source_text):
        block = m.group(0)
        kind = "javadoc" if block.startswith("/**") else "block"
        start = source_text.count("\n", 0, m.start()) + 1
        end = source_text.count("\n", 0, m.end()) + 1
        spans.append(CommentSpan(start_line=start, end_line=end, text=block, kind=kind))

    # Line comments
    for m in _LINE_COMMENT_RE.finditer(source_text):
        line_text = m.group(0)
        start = source_text.count("\n", 0, m.start()) + 1
        spans.append(CommentSpan(start_line=start, end_line=start, text=line_text, kind="line"))

    spans.sort(key=lambda s: (s.start_line, s.end_line))
    return spans


def find_attached_doc(comments: List[CommentSpan], decl_start_line: int, max_gap_lines: int = 1) -> Optional[str]:
    """
    Attach the nearest preceding comment block to a declaration start line.

    Common Java style:
      /** ... */
      public void foo() { ... }

    We accept a small gap (blank line) between comment and declaration.
    """
    best = None
    best_end = -1
    for c in comments:
        if c.end_line <= decl_start_line - 1:
            gap = (decl_start_line - 1) - c.end_line
            if gap <= max_gap_lines and c.end_line > best_end:
                best = c
                best_end = c.end_line
    if not best:
        return None

    # Prefer Javadoc when possible
    if best.kind == "javadoc":
        return best.text
    # If nearest isn't Javadoc, still return (could be // comment)
    return best.text


def safe_read_text(path: str) -> str:
    # Try UTF-8, fall back to latin-1 to avoid crashing on weird encodings.
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def load_snippet(path: str, start_line: int, end_line: int, max_lines: int = 120) -> str:
    """
    Load a snippet from file by 1-indexed line numbers.
    """
    if not path or not os.path.exists(path):
        return ""
    if start_line <= 0:
        start_line = 1
    if end_line < start_line:
        end_line = start_line

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)

    snippet_lines = lines[start_idx:end_idx]
    if len(snippet_lines) > max_lines:
        # Keep head+tail for readability
        head = snippet_lines[: max_lines // 2]
        tail = snippet_lines[-max_lines // 2 :]
        snippet_lines = head + ["\n... (snippet truncated) ...\n"] + tail

    return "".join(snippet_lines)

