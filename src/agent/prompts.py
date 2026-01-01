from __future__ import annotations

import json
from typing import Any, Dict, List


def _trim(s: str, max_chars: int) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= max_chars else (s[: max_chars - 3] + "...")


def context_pack_to_prompt(pack: Dict[str, Any], *, max_nodes: int = 18, max_snippet_chars: int = 2200) -> str:
    """
    Convert context_pack.json into a compact, LLM-friendly text block.
    """
    query = pack.get("query", "")
    focus = (pack.get("focus") or {}).get("node_id") or pack.get("focus_node") or ""
    seed_nodes = pack.get("seed_nodes", [])

    nodes = pack.get("nodes", [])
    def node_key(n):
        role = n.get("role", "")
        order = {"focus": 0, "seed": 1, "caller": 2, "callee": 3}.get(role, 9)
        return (order, n.get("type",""), n.get("node_id",""))

    nodes_sorted = sorted(nodes, key=node_key)[:max_nodes]

    out: List[str] = []
    out.append(f"Retrieval query: {query}")
    out.append(f"Focus node: {focus}")
    if seed_nodes:
        out.append("Seed nodes (vector search):")
        for s in seed_nodes[:8]:
            out.append(f"  - {s.get('node_id')} (score={s.get('score')})")
    out.append("")
    out.append("=== Context Nodes (with code excerpts) ===")
    for n in nodes_sorted:
        nid = n.get("node_id")
        typ = n.get("type")
        fp = n.get("file_path")
        sl = n.get("start_line")
        el = n.get("end_line")
        sig = _trim(n.get("signature",""), 240)
        roles = n.get("roles", [])
        why = n.get("why", [])
        snippet = _trim(n.get("snippet",""), max_snippet_chars)
        out.append(f"\n[{typ}] {nid}")
        if fp:
            out.append(f"Location: {fp}:{sl}-{el}")
        if sig:
            out.append(f"Signature: {sig}")
        if roles:
            out.append(f"Roles: {', '.join(map(str, roles))}")
        if why:
            why_short = [str(x) for x in why][:5]
            out.append("Why selected:")
            for w in why_short:
                out.append(f"  - {w}")
        if snippet:
            out.append("Code:")
            out.append(snippet)
    out.append("\n=== End Context ===")
    return "\n".join(out)


PLAN_SYSTEM = """You are a senior software engineer designing a safe automated refactoring plan.
You must be concise, risk-aware, and verification-driven.
Return STRICT JSON only (no markdown, no commentary outside JSON)."""

PLAN_USER_TEMPLATE = """Given the user request and the provided context pack, propose a refactoring plan.

Requirements:
- The goal is behavior-preserving refactoring.
- Only modify files that are necessary.
- Include a verification plan (commands) that can run in a sandbox (no shell operators like &&, |, >).
- Include risk notes.
- Optional: request additional information using tool_requests.
- IMPORTANT: All paths must be repo-root-relative, like "src/main/java/...". Do NOT prefix with "data/marketing-demo/".

Return JSON with this schema:
{{
  "objective": "...",
  "assumptions": ["..."],
  "steps": ["..."],
  "files_to_change": [{{"file_path":"...", "why":"...", "risk":"low|medium|high"}}],
  "verification": [{{"name":"...", "cmd":"..."}}],
  "tool_requests": [{{"tool":"ripgrep", "pattern":"...", "path":"..."}}]
}}

User request:
{request}

Context pack:
{context}
"""


EDIT_SYSTEM = """You are a senior software engineer.
You MUST output ONLY edit instructions using either Search/Replace blocks (preferred) or Full Rewrite blocks.
No markdown. No explanation. No code fences.

FORMAT (Search/Replace blocks - preferred):
FILE: <repo-relative-path>
<<<<<<< SEARCH
<exact text copied from the current file>
=======
<replacement text>
>>>>>>> REPLACE

You may include multiple SEARCH/REPLACE blocks under the same FILE.

FORMAT (Full Rewrite - allowed for small files):
FILE: <repo-relative-path>
<<<<<<< REWRITE
<entire new file content>
>>>>>>> REWRITE

RULES:
- Do NOT use "..." anywhere. Do NOT omit code.
- Do NOT abbreviate file paths.
- For Search/Replace: the SEARCH text MUST match EXACTLY ONE occurrence in the current file.
- Keep changes minimal and behavior-preserving.
- If a file is shown as MISSING in the 'EXACT REPO FILE CONTENTS' section, you MUST create it using a Full Rewrite block.
- You ARE allowed to create new files when required by the task; always use Full Rewrite for new files.
"""


EDIT_USER_TEMPLATE = """Task: {objective}

Refactoring plan (JSON):
{plan_json}

Context pack with code excerpts:
{context}

IMPORTANT:
- Use file paths relative to the repo root (e.g., "src/main/java/..."). Do NOT prefix with "data/marketing-demo/".
- Output MUST be ONLY edit instructions (no commentary).
- Prefer Search/Replace blocks (stable). Use Full Rewrite only when necessary.
- If a file is marked MISSING in the prompt, you MUST create it via Full Rewrite.
- Do NOT use "..." anywhere; do NOT shorten paths or code.

If the prompt includes a section named "EXACT REPO FILE CONTENTS (authoritative)", you MUST use that as the source of truth.

Return an UPDATED FULL set of edit instructions that should pass verification.
Return ONLY the edit instructions.
"""


REPAIR_SYSTEM = """You are a senior software engineer.
You MUST output ONLY edit instructions using either Search/Replace blocks (preferred) or Full Rewrite blocks.
No markdown. No explanation. No code fences.

You are given previous edit instructions and failure logs. Produce an UPDATED FULL set of instructions
(relative to the ORIGINAL codebase) that fixes the failures while keeping changes minimal and behavior-preserving.

FORMAT (Search/Replace blocks - preferred):
FILE: <repo-relative-path>
<<<<<<< SEARCH
<exact text copied from the current file>
=======
<replacement text>
>>>>>>> REPLACE

FORMAT (Full Rewrite - allowed for small files):
FILE: <repo-relative-path>
<<<<<<< REWRITE
<entire new file content>
>>>>>>> REWRITE

RULES:
- Do NOT use "..." anywhere. Do NOT omit code.
- Do NOT abbreviate file paths.
- For Search/Replace: the SEARCH text MUST match EXACTLY ONE occurrence in the current file.
- Keep changes minimal and behavior-preserving.
- If a file is shown as MISSING in the 'EXACT REPO FILE CONTENTS' section, you MUST create it using a Full Rewrite block.
- You ARE allowed to create new files when required by the task; always use Full Rewrite for new files.
"""


REPAIR_USER_TEMPLATE = """Task: {objective}

Refactoring plan (JSON):
{plan_json}

Previous edit instructions (that failed to apply or failed verification):
{prev_patch}

Failure / verification outputs:
{verify_logs}

Context pack with code excerpts:
{context}

IMPORTANT:
- Use file paths relative to the repo root (e.g., "src/main/java/..."). Do NOT prefix with "data/marketing-demo/".
- Output MUST be ONLY edit instructions (no commentary).
- Prefer Search/Replace blocks (stable). Use Full Rewrite only when necessary.
- If a file is marked MISSING in the prompt, you MUST create it via Full Rewrite.
- Do NOT use "..." anywhere; do NOT shorten paths or code.
- If apply failed because SEARCH did not match, adjust the SEARCH blocks to match the exact current file text (include more surrounding context).

If the prompt includes a section named "EXACT REPO FILE CONTENTS (authoritative)", you MUST use that as the source of truth.

Return an UPDATED FULL set of edit instructions that should pass verification.
Return ONLY the edit instructions.
"""
