from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _strip_quotes(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        return v[1:-1]
    return v


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> bool:
    """A tiny .env loader (no external dependency).

    - Parses KEY=VALUE lines
    - Ignores blank lines and comments (# ...)
    - Supports quoted values ("..."/'...') and inline comments after a space + #
    - Writes into os.environ (unless already set and override=False)

    Returns True if file existed and was loaded.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        return False

    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip()

        # remove inline comment: only if there's an unquoted ' #' sequence
        if val and val[0] not in "'":
            # split on first ' #' (space hash) to avoid chopping URLs like https://...
            idx = val.find(" #")
            if idx != -1:
                val = val[:idx].rstrip()

        val = _strip_quotes(val)
        if not key:
            continue
        if (not override) and (key in os.environ) and (os.environ[key] != ""):
            continue
        os.environ[key] = val
    return True


def auto_load_dotenv(explicit_path: Optional[str] = None) -> Optional[str]:
    """Try to load dotenv from common locations; returns loaded path if any."""
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    # current working directory
    candidates.append(Path.cwd() / ".env")
    # repo root heuristic: walk up a few levels looking for .env
    cur = Path.cwd()
    for _ in range(5):
        candidates.append(cur / ".env")
        cur = cur.parent

    for c in candidates:
        try:
            if load_dotenv(c):
                return str(c)
        except Exception:
            # Never hard-fail on dotenv parsing
            continue
    return None

