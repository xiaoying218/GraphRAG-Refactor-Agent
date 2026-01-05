from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class HunkLine:
    tag: str  # ' ', '+', '-'
    text: str


@dataclass
class Hunk:
    src_start: int
    src_len: int
    dst_start: int
    dst_len: int
    lines: List[HunkLine]


@dataclass
class FilePatch:
    old_path: str
    new_path: str
    hunks: List[Hunk]


_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$")
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_OLD_RE = re.compile(r"^---\s+(?:a/)?(.+)$")
_NEW_RE = re.compile(r"^\+\+\+\s+(?:b/)?(.+)$")


class PatchApplyError(Exception):
    pass


def parse_unified_diff(diff_text: str) -> List[FilePatch]:
    """
    Parses a subset of unified diff enough for LLM-generated patches.
    Supports:
      diff --git a/... b/...
      --- a/...
      +++ b/...
      @@ -l,s +l,s @@ hunks
    """
    lines = diff_text.splitlines()
    i = 0
    patches: List[FilePatch] = []

    current_old = None
    current_new = None
    current_hunks: List[Hunk] = []

    def flush():
        nonlocal current_old, current_new, current_hunks
        if current_old and current_new:
            patches.append(FilePatch(old_path=current_old, new_path=current_new, hunks=current_hunks))
        current_old = None
        current_new = None
        current_hunks = []

    while i < len(lines):
        line = lines[i]

        m = _DIFF_GIT_RE.match(line)
        if m:
            flush()
            current_old, current_new = m.group(1), m.group(2)
            i += 1
            continue

        m = _OLD_RE.match(line)
        if m:
            if current_old is None and current_new is None:
                current_old = m.group(1)
            i += 1
            continue

        m = _NEW_RE.match(line)
        if m:
            if current_new is None:
                current_new = m.group(1)
            i += 1
            continue

        m = _HUNK_RE.match(line)
        if m:
            src_start = int(m.group(1))
            src_len = int(m.group(2) or "1")
            dst_start = int(m.group(3))
            dst_len = int(m.group(4) or "1")
            i += 1

            hunk_lines: List[HunkLine] = []
            while i < len(lines):
                hl = lines[i]
                if hl.startswith("diff --git ") or hl.startswith("@@ "):
                    break
                if hl.startswith("--- ") or hl.startswith("+++ "):
                    break
                if hl.startswith("\\ No newline"):
                    i += 1
                    continue

                if hl == "":
                    # treat empty as context blank line
                    hunk_lines.append(HunkLine(tag=" ", text=""))
                    i += 1
                    continue

                tag = hl[0]
                if tag not in (" ", "+", "-"):
                    break

                # normalize CRLF artifacts
                txt = hl[1:].rstrip("\r")
                hunk_lines.append(HunkLine(tag=tag, text=txt))
                i += 1

            current_hunks.append(Hunk(src_start, src_len, dst_start, dst_len, hunk_lines))
            continue

        i += 1

    flush()
    return patches


def _safe_join(root: Path, rel: str) -> Path:
    rel = rel.replace("\\", "/").lstrip("/")
    p = (root / rel).resolve()
    if not str(p).startswith(str(root.resolve())):
        raise PatchApplyError(f"Unsafe path in patch: {rel}")
    return p


def normalize_patch_path(root_dir: Path, rel: str) -> str:
    """
    Normalize patch file paths to be relative to root_dir.

    Handles cases like:
      data/marketing-demo/src/main/java/...  (but root_dir is already marketing-demo)
    """
    root_dir = Path(root_dir).resolve()
    rel = (rel or "").replace("\\", "/").lstrip("/")
    if not rel:
        return rel

    # 1) As-is
    try:
        p = _safe_join(root_dir, rel)
        if p.exists():
            return rel
    except PatchApplyError:
        pass

    parts = [x for x in rel.split("/") if x]
    if len(parts) <= 1:
        return rel

    # 2) Try suffixes that exist as files
    for i in range(1, len(parts)):
        cand = "/".join(parts[i:])
        try:
            p2 = _safe_join(root_dir, cand)
        except PatchApplyError:
            continue
        if p2.exists():
            return cand

    # 3) For new files: try suffixes whose parent exists
    for i in range(1, len(parts)):
        cand = "/".join(parts[i:])
        try:
            p2 = _safe_join(root_dir, cand)
        except PatchApplyError:
            continue
        if p2.parent.exists():
            return cand

    return rel


def _seek_line(lines: List[str], cursor: int, text: str, window: int = 60) -> Optional[int]:
    """
    Find `text` near `cursor` within +/- window.
    Returns best matched index (closest), or None.
    """
    n = len(lines)
    if n == 0:
        return None
    cursor = max(0, min(cursor, n - 1))

    best = None
    best_dist = 10**9

    start = max(0, cursor - window)
    end = min(n - 1, cursor + window)
    for i in range(start, end + 1):
        if lines[i] == text:
            dist = abs(i - cursor)
            if dist < best_dist:
                best_dist = dist
                best = i
                if dist == 0:
                    break
    return best


def apply_unified_diff(diff_text: str, *, root_dir: Path, dry_run: bool = False) -> List[Path]:
    """
    Applies unified diff to files under root_dir.
    Returns list of modified file paths (absolute).
    """
    root_dir = Path(root_dir).resolve()
    patches = parse_unified_diff(diff_text)
    if not patches:
        raise PatchApplyError("No file patches found in diff.")

    modified: List[Path] = []

    for fp in patches:
        old_path = fp.old_path
        new_path = fp.new_path

        if old_path == "/dev/null":
            old_path = ""
        if new_path == "/dev/null":
            new_path = ""

        target_rel = new_path or old_path
        if not target_rel:
            raise PatchApplyError("Patch missing target path.")

        target_rel = normalize_patch_path(root_dir, target_rel)
        target = _safe_join(root_dir, target_rel)

        if target.exists():
            old_text = target.read_text(encoding="utf-8", errors="replace")
            old_lines = [ln.rstrip("\r") for ln in old_text.splitlines()]
        else:
            old_lines = []

        new_lines = old_lines[:]

        line_offset = 0
        for h in fp.hunks:
            idx = max(0, h.src_start - 1 + line_offset)
            cursor = min(idx, max(0, len(new_lines) - 1)) if new_lines else 0

            for hl in h.lines:
                if hl.tag == " ":
                    if cursor >= len(new_lines) or new_lines[cursor] != hl.text:
                        found = _seek_line(new_lines, cursor, hl.text, window=60)
                        if found is None:
                            raise PatchApplyError(
                                f"Context mismatch while applying hunk to {target_rel}: "
                                f"expected '{hl.text}' near line {cursor+1}"
                            )
                        cursor = found
                    cursor += 1

                elif hl.tag == "-":
                    if cursor >= len(new_lines) or new_lines[cursor] != hl.text:
                        found = _seek_line(new_lines, cursor, hl.text, window=30)
                        if found is None:
                            raise PatchApplyError(
                                f"Removal mismatch while applying hunk to {target_rel}: "
                                f"expected to remove '{hl.text}' near line {cursor+1}"
                            )
                        cursor = found
                    del new_lines[cursor]
                    line_offset -= 1

                elif hl.tag == "+":
                    new_lines.insert(cursor, hl.text)
                    cursor += 1
                    line_offset += 1

        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
        modified.append(target)

    return modified


def list_target_files(diff_text: str) -> List[str]:
    patches = parse_unified_diff(diff_text)
    out: List[str] = []
    for fp in patches:
        target_rel = fp.new_path or fp.old_path
        if target_rel and target_rel not in out:
            out.append(target_rel)
    return out

