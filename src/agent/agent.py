from __future__ import annotations

import json
import os
import re
import shutil
import time
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .llm import LLMClient
from .patcher import PatchApplyError  # reuse exception type
from .prompts import (
    PLAN_SYSTEM, PLAN_USER_TEMPLATE,
    EDIT_SYSTEM, EDIT_USER_TEMPLATE,
    REPAIR_SYSTEM, REPAIR_USER_TEMPLATE,
    context_pack_to_prompt,
)
from .sandbox import Sandbox, SandboxConfig, CommandResult


@dataclass
class AgentConfig:
    project_dir: Path
    work_dir: Optional[Path] = None  # if None, create .refactor_agent_runs/<ts> under project_dir
    max_iters: int = 3

    # Prompt control
    max_nodes_in_prompt: int = 18
    max_snippet_chars: int = 2200

    # Verification
    # If None/empty, RefactoringAgent will auto-detect sensible verification commands
    # based on repo contents (e.g. Maven/Gradle/Java demo runner).
    default_verify_cmds: Optional[List[str]] = None

    # Whether to trust/execute verification commands proposed by the LLM in plan.json.
    # Default False to avoid brittle verification steps.
    use_plan_verification: bool = False


    # Execution
    # - monolithic: one large patch per attempt (original behavior)
    # - stepwise: split into small steps (recommended; improves patch applicability and reduces truncation)
    execution_mode: str = "stepwise"  # "monolithic" or "stepwise"
    step_max_files: int = 10           # when auto-splitting, max files per step
    verify_each_step: bool = True     # run verification after each step
    enforce_file_whitelist: bool = True  # prevent edits outside plan files_to_change
    allow_new_files: bool = True      # allow creating new files via Full Rewrite

    # LLM output budgets (increase to reduce truncation)
    edit_max_tokens: int = 4000
    repair_max_tokens: int = 6000
    # Sandbox
    sandbox_timeout_s: int = 240
    use_docker: bool = False
    docker_image: str = "python:3.10-slim"
    allowed_commands: Optional[List[str]] = None


@dataclass
class SearchReplaceBlock:
    search: str
    replace: str


@dataclass
class FileEditInstruction:
    path: str
    blocks: List[SearchReplaceBlock] = None
    rewrite: Optional[str] = None



class RefactoringAgent:
    """
    Minimal refactoring agent (Plan → Edit → Verify/Repair) using git apply as the patch application engine.

    Key robustness features vs earlier versions:
    - Normalize plan paths (strip prefixes like data/marketing-demo/ if work_dir is already marketing-demo root)
    - Feed EXACT file contents into prompts so LLM doesn't guess context lines
    - Use git reset --hard + git clean -fd between iterations to keep index/worktree consistent
    - Support both git-style patches (diff --git a/ b/) and traditional patches (---/+++ without a/ b/) via -p level
    """
    def __init__(self, *, llm: LLMClient, cfg: AgentConfig):
        self.llm = llm
        self.cfg = cfg
        self.project_dir = Path(cfg.project_dir).resolve()
        if not self.project_dir.exists():
            raise FileNotFoundError(f"project_dir not found: {self.project_dir}")

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.work_dir = Path(cfg.work_dir or (self.project_dir / ".refactor_agent_runs" / ts)).resolve()

        if self.work_dir.exists():
            raise FileExistsError(f"work_dir already exists: {self.work_dir}")
        self.work_dir.parent.mkdir(parents=True, exist_ok=True)

        # Copy project into sandbox workspace (ignore heavy folders)
        shutil.copytree(
            self.project_dir,
            self.work_dir,
            dirs_exist_ok=False,
            ignore=shutil.ignore_patterns(".refactor_agent_runs", ".git", "target", "build", ".gradle", "node_modules", "__pycache__")
        )

        sandbox_cfg = SandboxConfig(
            root_dir=self.work_dir,
            timeout_s=cfg.sandbox_timeout_s,
            allowed_commands=(cfg.allowed_commands or SandboxConfig(root_dir=self.work_dir).allowed_commands),
            use_docker=cfg.use_docker,
            docker_image=cfg.docker_image,
        )
        self.sandbox = Sandbox(sandbox_cfg)

        # Auto-detect verification commands if not provided.
        if not (self.cfg.default_verify_cmds or []):
            self.cfg.default_verify_cmds = self._auto_detect_verify_cmds()

        self._git_initialized = False

    # ---------------------------
    # Utilities
    # ---------------------------


    def _auto_detect_verify_cmds(self) -> List[str]:
        """Choose deterministic verification commands based on repo contents.

        We intentionally avoid trusting LLM-generated verification commands by default
        because they are often brittle (wrong classpath, missing files, shell globs, etc.).
        """
        repo = self.work_dir

        # Java: if there is a DemoRunner, compile all sources and run it.
        src_root = repo / "src" / "main" / "java"
        if src_root.exists():
            demo = next(src_root.rglob("DemoRunner.java"), None)
            if demo:
                main_class = self._java_main_class_from_source(demo)
                # Compile into target/classes to avoid polluting source tree
                compile_cmd = "javac -encoding UTF-8 -d target/classes -cp src/main/java src/main/java/**/*.java"

                # Include resources if present
                cp = f"target/classes{os.pathsep}src/main/resources" if (repo / "src" / "main" / "resources").exists() else "target/classes"
                run_cmd = f"java -cp {cp} {main_class}"
                return [compile_cmd, run_cmd]

            # Fallback: just compile everything
            return ["javac -encoding UTF-8 -d target/classes -cp src/main/java src/main/java/**/*.java"]

        # Maven / Gradle (fallbacks)
        if (repo / "pom.xml").exists():
            if (repo / "mvnw").exists():
                return ["./mvnw -q test"]
            return ["mvn -q test"]

        if (repo / "build.gradle").exists() or (repo / "build.gradle.kts").exists():
            if (repo / "gradlew").exists():
                return ["./gradlew test"]
            return ["gradle test"]

        # Python (very generic fallback)
        if (repo / "pyproject.toml").exists() or (repo / "requirements.txt").exists():
            return ["pytest -q"]

        return []

    @staticmethod
    def _java_main_class_from_source(source_path: Path) -> str:
        """Derive a Java main class (FQN) from a .java file by reading its package statement."""
        try:
            lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return source_path.stem

        pkg = ""
        for ln in lines[:80]:
            s = ln.strip()
            if s.startswith("package ") and s.endswith(";"):
                pkg = s[len("package "):].rstrip(";").strip()
                break
        return f"{pkg}.{source_path.stem}" if pkg else source_path.stem

    def _ensure_output_dirs_for_cmds(self, cmds: List[str]) -> None:
        """Create output dirs used by javac -d to avoid spurious failures."""
        for cmd_str in cmds or []:
            m = re.search(r"\s-d\s+([^\s]+)", cmd_str)
            if not m:
                continue
            out_dir = m.group(1)
            # Only handle relative paths inside the workdir
            if out_dir.startswith(('/', '~')) or ':' in out_dir:
                continue
            try:
                (self.work_dir / out_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start:end + 1])
            raise

    @staticmethod
    def _sanitize_patch_text(patch: str) -> str:
        """
        Make LLM output more 'git apply' friendly:
        - remove ``` fences
        - trim everything before first diff header
        - normalize newlines and ensure trailing newline
        - fix corrupt diffs caused by blank lines inside hunks
        - fix common LLM mistakes where hunk lines miss the leading ' ' / '+' / '-' marker
        - reconstruct truncated 'diff --git' lines from subsequent ---/+++ headers
        """
        if patch is None:
            return ""

        patch = patch.replace("\r\n", "\n").replace("\r", "\n")
        raw_lines = patch.splitlines()

        # Drop code fences if any
        raw_lines = [ln for ln in raw_lines if not ln.strip().startswith("```")]

        # Trim everything before the first diff header
        start_idx = 0
        for i, ln in enumerate(raw_lines):
            if ln.startswith("diff --git ") or ln.startswith("--- "):
                start_idx = i
                break
        raw_lines = raw_lines[start_idx:]

        out: list[str] = []
        in_hunk = False

        for ln in raw_lines:
            if ln.startswith("@@ "):
                in_hunk = True
                out.append(ln)
                continue

            # leaving a hunk when a new file header starts
            if in_hunk and (ln.startswith("diff --git ") or ln.startswith("--- ") or ln.startswith("+++ ")):
                in_hunk = False

            if in_hunk:
                # In unified diff, every hunk line must start with ' ', '+', '-', or '\'
                if ln == "":
                    out.append(" ")
                    continue
                if ln.startswith("\\ No newline at end of file"):
                    out.append(ln)
                    continue
                if ln[:1] not in (" ", "+", "-", "\\"):
                    # Model forgot the prefix; assume it's a context line.
                    out.append(" " + ln)
                    continue

            out.append(ln)

        # Fix 'diff --git' lines that may be truncated by the LLM by reconstructing
        # them from the subsequent ---/+++ headers.
        fixed: list[str] = []
        i = 0
        while i < len(out):
            ln = out[i]
            if ln.startswith("diff --git "):
                a_path = None
                b_path = None
                j = i + 1
                while j < len(out) and (not out[j].startswith("diff --git ")):
                    if out[j].startswith("--- "):
                        a_path = out[j].split(maxsplit=1)[1].strip()
                    elif out[j].startswith("+++ "):
                        b_path = out[j].split(maxsplit=1)[1].strip()
                        break
                    j += 1
                if a_path and b_path:
                    fixed.append(f"diff --git {a_path} {b_path}")
                    i += 1
                    continue
            fixed.append(ln)
            i += 1

        return "\n".join(fixed).strip() + "\n"



    def _patch_has_git_header(self, patch: str) -> bool:
        return "diff --git " in (patch or "")

    def _guess_p_level(self, patch: str) -> int:
        """
        git apply by default behaves like -p1 (strip one path component).
        For patches without a/ b/ prefixes, we usually want -p0.
        """
        if re.search(r"^---\s+a\/", patch, re.M) or re.search(r"^\+\+\+\s+b\/", patch, re.M) or "diff --git a/" in patch:
            return 1
        return 0

    def _resolve_in_workdir(self, path: str, *, want_dir: bool = False) -> Optional[str]:
        """
        Resolve a possibly-prefixed path (e.g. data/marketing-demo/src/...) to a repo-root-relative path
        that exists under work_dir. Returns normalized relative path, or None if not resolvable.
        """
        if not path:
            return None
        p = str(path).replace("\\", "/").lstrip("/")

        # strip common diff prefixes
        if p.startswith("a/") or p.startswith("b/"):
            p = p[2:]

        # try as-is
        cand = (self.work_dir / p)
        if want_dir and cand.is_dir():
            return p
        if (not want_dir) and cand.is_file():
            return p

        parts = [x for x in p.split("/") if x]
        if not parts:
            return None

        # try suffixes
        for i in range(1, len(parts)):
            suf = "/".join(parts[i:])
            cand2 = (self.work_dir / suf)
            if want_dir and cand2.is_dir():
                return suf
            if (not want_dir) and cand2.is_file():
                return suf

        # directory fallback: if want_dir, also allow existing parent
        if want_dir:
            for i in range(1, len(parts)):
                suf = "/".join(parts[i:])
                cand2 = (self.work_dir / suf)
                if cand2.exists() and cand2.is_dir():
                    return suf
        return None

    def _normalize_plan_paths(self, plan: Dict[str, Any], context_pack: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize plan["files_to_change"][].file_path and plan["tool_requests"][].path so they work under work_dir.
        """
        plan = dict(plan)

        # files_to_change
        ftc = plan.get("files_to_change") or []
        new_ftc = []
        for item in ftc:
            item = dict(item)
            fp = (item.get("file_path") or "").strip()
            resolved = self._resolve_in_workdir(fp, want_dir=False)
            if resolved:
                item["file_path"] = resolved
            new_ftc.append(item)
        plan["files_to_change"] = new_ftc

        # tool_requests paths (dirs)
        tr = plan.get("tool_requests") or []
        new_tr = []
        for req in tr:
            req = dict(req)
            p = (req.get("path") or ".").strip() or "."
            resolved_dir = self._resolve_in_workdir(p, want_dir=True)
            req["path"] = resolved_dir or "."
            new_tr.append(req)
        plan["tool_requests"] = new_tr

        # verification commands: keep as-is
        return plan

    def _focus_file_from_context_pack(self, context_pack: Dict[str, Any]) -> Optional[str]:
        focus = (context_pack.get("focus") or {}).get("node_id") or context_pack.get("focus_node")
        if not focus:
            return None
        for n in (context_pack.get("nodes") or []):
            if n.get("node_id") == focus and n.get("file_path"):
                fp = n.get("file_path")
                resolved = self._resolve_in_workdir(fp, want_dir=False)
                return resolved or None
        return None

    def _dump_files_for_prompt(self, plan: Dict[str, Any], context_pack: Dict[str, Any], *, max_chars_per_file: int = 22000, only_files: Optional[List[str]] = None) -> str:
        """
        Append exact file contents from the sandbox workdir so the LLM can produce an apply-able patch.
        Uses normalized plan file paths; if missing, falls back to focus file.
        """
        rel_files: List[str] = []
        if only_files:
            rel_files.extend([str(x).strip() for x in only_files if str(x).strip()])
        else:
            for f in (plan.get("files_to_change") or []):
                fp = (f.get("file_path") or "").strip()
                if fp:
                    rel_files.append(fp)

        if not rel_files:
            focus_fp = self._focus_file_from_context_pack(context_pack)
            if focus_fp:
                rel_files.append(focus_fp)

        # de-dup
        seen = set()
        rel_files = [x for x in rel_files if not (x in seen or seen.add(x))]

        blocks = []
        max_files = len(rel_files) if only_files else max(6, self.cfg.step_max_files)
        for rel in rel_files[:max_files]:
            p = (self.work_dir / rel).resolve()
            if not str(p).startswith(str(self.work_dir)):
                continue
            if not p.exists() or not p.is_file():
                # Expose missing files to the LLM so it can create them via Full Rewrite.
                blocks.append(
                    f"\n--- FILE: {rel} (MISSING) ---\n"
                    "/* This file does not exist yet. Create it using a Full Rewrite (REWRITE) block. */\n"
                    f"--- END FILE: {rel} ---\n"
                )
                continue
            txt = p.read_text(encoding="utf-8", errors="replace")
            if len(txt) > max_chars_per_file:
                txt = txt[:max_chars_per_file] + "\n/* <TRUNCATED_FILE_CONTENT> */\n"
            blocks.append(f"\n--- FILE: {rel} ---\n{txt}\n--- END FILE: {rel} ---\n")
        return "\n".join(blocks)


    def _is_safe_repo_path(self, p: str) -> bool:
        """Conservative filter: only allow expanding scope to real source files."""
        if not p:
            return False
        p = p.replace("\\", "/").lstrip("./")
        # block path traversal
        if ".." in Path(p).parts:
            return False
        # adjust prefixes as needed for your repos
        return p.startswith(("src/", "lib/", "app/", "tests/", "test/"))

    def _extract_apply_search_mismatch_files(self, logs: str) -> List[str]:
        """Extract file paths from edit-apply errors like 'SEARCH block not found in <file>'"""
        if not logs:
            return []
        paths = re.findall(r"\[EDIT APPLY ERROR\] SEARCH block not found in ([^\s]+)", logs)
        out: List[str] = []
        seen = set()
        for p in paths:
            p = (p or "").strip()
            if not p:
                continue
            # normalize
            p = p.replace("\\", "/").lstrip("./")
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out

    # ---------------------------
    # Plan / Edit / Repair
    # ---------------------------

    def _plan(self, *, request: str, context_pack: Dict[str, Any]) -> Dict[str, Any]:
        context_txt = context_pack_to_prompt(
            context_pack,
            max_nodes=self.cfg.max_nodes_in_prompt,
            max_snippet_chars=self.cfg.max_snippet_chars,
        )
        messages = [
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": PLAN_USER_TEMPLATE.format(request=request, context=context_txt, hard_requirements=self._hard_requirements_text(context_pack))},
        ]
        out = self.llm.chat(messages, temperature=0.1, max_tokens=1400)
        plan = self._extract_json(out)
        return plan

    def _run_tool_requests(self, tool_requests: List[Dict[str, Any]]) -> str:
        if not tool_requests:
            return ""
        chunks: List[str] = []
        for req in tool_requests[:6]:
            tool = (req.get("tool") or "").lower()
            if tool in ("ripgrep", "rg"):
                pattern = str(req.get("pattern") or "").strip()
                path = str(req.get("path") or ".").strip() or "."
                if not pattern:
                    continue
                cmd = ["rg", "-n", pattern, path]
                try:
                    res = self.sandbox.run(cmd, cwd=self.work_dir)
                    chunks.append(f"[TOOL rg] cmd={' '.join(cmd)}\n{res.stdout}\n{res.stderr}".strip())
                except Exception as e:
                    # Fallback to grep if rg is not available in the environment.
                    try:
                        gcmd = ["grep", "-R", "-n", "--", pattern, path]
                        gres = self.sandbox.run(gcmd, cwd=self.work_dir)
                        chunks.append(f"[TOOL grep] cmd={' '.join(gcmd)}\n{gres.stdout}\n{gres.stderr}".strip())
                    except Exception as e2:
                        chunks.append(f"[TOOL rg ERROR] {e}\n[TOOL grep ERROR] {e2}")
        return "\n\n".join(chunks)

    def _edit(self, *, objective: str, plan: Dict[str, Any], context_pack: Dict[str, Any], extra_tool_info: str = "", only_files: Optional[List[str]] = None) -> str:
        context_txt = context_pack_to_prompt(
            context_pack,
            max_nodes=self.cfg.max_nodes_in_prompt,
            max_snippet_chars=self.cfg.max_snippet_chars,
        )
        file_dump = self._dump_files_for_prompt(plan, context_pack, only_files=only_files)
        if file_dump:
            context_txt += "\n\n=== EXACT REPO FILE CONTENTS (authoritative) ===\n" + file_dump + "\n=== END EXACT FILE CONTENTS ===\n"
        if extra_tool_info:
            context_txt += "\n\n=== Tool outputs ===\n" + extra_tool_info + "\n=== End tool outputs ===\n"

        plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": EDIT_SYSTEM},
            {"role": "user", "content": EDIT_USER_TEMPLATE.format(objective=objective, plan_json=plan_json, context=context_txt, hard_requirements=self._hard_requirements_text(context_pack))},
        ]
        patch = self.llm.chat(messages, temperature=0.2, max_tokens=self.cfg.edit_max_tokens)
        return (patch or "").strip()

    def _repair(self, *, objective: str, plan: Dict[str, Any], context_pack: Dict[str, Any], prev_patch: str, verify_logs: str, extra_tool_info: str = "", only_files: Optional[List[str]] = None) -> str:
        context_txt = context_pack_to_prompt(
            context_pack,
            max_nodes=self.cfg.max_nodes_in_prompt,
            max_snippet_chars=self.cfg.max_snippet_chars,
        )
        file_dump = self._dump_files_for_prompt(plan, context_pack, only_files=only_files)
        if file_dump:
            context_txt += "\n\n=== EXACT REPO FILE CONTENTS (authoritative) ===\n" + file_dump + "\n=== END EXACT FILE CONTENTS ===\n"
        if extra_tool_info:
            context_txt += "\n\n=== Tool outputs ===\n" + extra_tool_info + "\n=== End tool outputs ===\n"

        plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": REPAIR_SYSTEM},
            {"role": "user", "content": REPAIR_USER_TEMPLATE.format(
                objective=objective,
                plan_json=plan_json,
                prev_patch=prev_patch,
                verify_logs=verify_logs,
                context=context_txt,
                hard_requirements=self._hard_requirements_text(context_pack)
            )},
        ]
        patch = self.llm.chat(messages, temperature=0.2, max_tokens=self.cfg.repair_max_tokens)
        return (patch or "").strip()

    # ---------------------------
    # Git patch application
    # ---------------------------

    def _ensure_git_baseline(self, artifacts_dir: Path) -> None:
        if self._git_initialized:
            return
        if not (self.work_dir / ".git").exists():
            self.sandbox.run(["git", "init"], cwd=self.work_dir)
        # set identity to avoid "Please tell me who you are"
        self.sandbox.run(["git", "config", "user.email", "refactor-agent@example.com"], cwd=self.work_dir)
        self.sandbox.run(["git", "config", "user.name", "RefactorAgent"], cwd=self.work_dir)

        self.sandbox.run(["git", "add", "-A"], cwd=self.work_dir)
        # baseline commit (ignore non-zero if nothing to commit)
        _ = self.sandbox.run(["git", "commit", "-m", "baseline"], cwd=self.work_dir)
        self._git_initialized = True

    def _git_reset_clean(self) -> None:
        # restore to baseline commit
        self.sandbox.run(["git", "reset", "--hard", "HEAD"], cwd=self.work_dir)
        self.sandbox.run(["git", "clean", "-fd"], cwd=self.work_dir)

    def _git_head_hash(self) -> str:
        res = self.sandbox.run(["git", "rev-parse", "HEAD"], cwd=self.work_dir)
        return (res.stdout or "").strip()

    def _git_reset_clean_to(self, rev: str) -> None:
        self.sandbox.run(["git", "reset", "--hard", rev], cwd=self.work_dir)
        self.sandbox.run(["git", "clean", "-fd"], cwd=self.work_dir)

    def _git_commit_checkpoint(self, msg: str) -> None:
        # Commit current working tree changes. Ignore "nothing to commit".
        self.sandbox.run(["git", "add", "-A"], cwd=self.work_dir)
        res = self.sandbox.run(["git", "commit", "-m", msg], cwd=self.work_dir)
        _ = res  # keep for debugging if needed

    @staticmethod
    def _is_meta_path(p: str) -> bool:
        return (
            p.startswith(".agent_artifacts/")
            or p == ".agent_artifacts"
            or p.startswith(".git/")
            or p == ".git"
        )

    def _list_changed_files(self) -> List[str]:
        diff_names = self.sandbox.run(["git", "diff", "--name-only"], cwd=self.work_dir)
        files = [ln.strip() for ln in (diff_names.stdout or "").splitlines() if ln.strip()]
        return [f for f in files if not self._is_meta_path(f)]

    def _allowed_files_from_plan(self, plan: Dict[str, Any]) -> List[str]:
        rel_files: List[str] = []
        for f in (plan.get("files_to_change") or []):
            fp = (f.get("file_path") or "").strip()
            if fp:
                rel_files.append(fp)
        # de-dup keep order
        seen = set()
        rel_files = [x for x in rel_files if not (x in seen or seen.add(x))]
        return rel_files

    def _compute_allowed_files(self, plan: Dict[str, Any], context_pack: Dict[str, Any]) -> List[str]:
        allowed = self._allowed_files_from_plan(plan)
        for ef in (context_pack.get("expected_files") or []):
            if isinstance(ef, str) and ef.strip():
                allowed.append(ef.strip())
        # de-dup keep order
        seen: set[str] = set()
        return [x for x in allowed if not (x in seen or seen.add(x))]


    def _enforce_changed_files_whitelist(self, allowed: List[str]) -> Tuple[bool, str]:
        if not self.cfg.enforce_file_whitelist:
            return True, ""
        allowed_set = set(allowed or [])
        changed = self._list_changed_files()
        extra = [f for f in changed if f not in allowed_set]
        if extra:
            return False, (
                "[SCOPE ERROR] The patch modified files outside the planned scope.\n"
                f"Allowed: {sorted(allowed_set)}\n"
                f"Changed: {changed}\n"
                f"Out-of-scope: {extra}\n"
                "Fix by ONLY editing allowed files.\n"
            )
        return True, ""


    def _extract_apply_failure_context(self, stderr: str) -> str:
        """
        If stderr contains 'patch failed: <file>:<line>', append an excerpt of that file around the line.
        """
        if not stderr:
            return ""
        m = re.search(r"patch failed:\s+([^\s:]+):(\d+)", stderr)
        if not m:
            return ""
        rel = m.group(1).strip()
        line_no = int(m.group(2))
        resolved = self._resolve_in_workdir(rel, want_dir=False) or rel
        p = (self.work_dir / resolved)
        if not p.exists():
            return ""
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        lo = max(0, line_no - 1 - 20)
        hi = min(len(lines), line_no - 1 + 20)
        excerpt = "\n".join(f"{i+1:04d}: {lines[i]}" for i in range(lo, hi))
        return f"\n[FILE EXCERPT] {resolved} around line {line_no}\n{excerpt}\n"


    
    def _hard_requirements_text(self, context_pack: Dict[str, Any]) -> str:
        """Build strict, machine-checkable constraints to reduce 'guess-the-naming' failure modes."""
        lines: List[str] = []
        expected = context_pack.get("expected_files") or []
        expected = [str(x).strip() for x in expected if str(x).strip()]
        if expected:
            lines.append(f"- These files MUST exist at EXACT repo-relative paths after the refactor (do not rename/move):")
            for p in expected:
                lines.append(f"  - {p}")
            lines.append("- Do NOT satisfy expected files by using inner/nested classes; create top-level files at the exact paths.")
        focus_after = str(context_pack.get("focus_node_after") or "").strip()
        if focus_after:
            lines.append(f"- The post-refactor code MUST contain this method id exactly: {focus_after}")
            lines.append("  - Do not rename the class or method in that id. If you move code, keep this API as an adapter/wrapper.")
        if not lines:
            return "- (none)"
        return "\n".join(lines)

    def _extract_compiler_error_files(self, logs: str) -> List[str]:
        """Heuristically extract repo-relative file paths from compiler/test logs (javac/maven/gradle)."""
        if not logs:
            return []
        out: List[str] = []
        # Common patterns:
        # 1) javac: /abs/path/Foo.java:123: error:
        # 2) javac: src/main/java/.../Foo.java:123: error:
        # 3) maven: [ERROR] /abs/path/Foo.java:[123,45] cannot find symbol
        patterns = [
            r"(?P<path>(?:[A-Za-z]:)?[^\s:]+?\.java):(?P<line>\d+):\s*error",
            r"\[ERROR\]\s+(?P<path>(?:[A-Za-z]:)?[^\s:]+?\.java):\[(?P<line>\d+),\d+\]",
        ]
        for pat in patterns:
            for m in re.finditer(pat, logs):
                p = (m.group("path") or "").strip()
                if not p:
                    continue
                # Normalize to repo-relative if possible
                p_norm = p.replace("\\", "/")
                wd = str(self.work_dir).replace("\\", "/").rstrip("/")
                if p_norm.startswith(wd + "/"):
                    p_norm = p_norm[len(wd) + 1 :]
                # Filter build outputs (target/, build/) unless it's in source tree
                if "/target/" in p_norm or p_norm.startswith("target/") or "/build/" in p_norm or p_norm.startswith("build/"):
                    # ignore compiled artifacts
                    continue
                if p_norm not in out:
                    out.append(p_norm)
        return out


# ---------------------------
    # Apply LLM edits (Search/Replace Blocks or Full Rewrite)
    # ---------------------------

    def _apply_llm_output(self, llm_text: str, diff_file: Path) -> Tuple[bool, str]:
        """
        Apply LLM output to the repo working tree.
        Preferred format:
          - FILE: <path>
            <<<<<<< SEARCH
            ...
            =======
            ...
            >>>>>>> REPLACE
          - or:
            FILE: <path>
            <<<<<<< REWRITE
            <full new file content>
            >>>>>>> REWRITE

        Fallback: if the output looks like a unified diff (diff --git), apply via git apply.
        On success, writes `git diff` (against baseline) to diff_file.
        Returns (ok, logs). Logs are suitable to feed to repair.
        """
        (diff_file.parent / (diff_file.stem + "_raw.txt")).write_text(llm_text, encoding="utf-8")

        looks_like_diff = bool(re.search(r"^diff --git ", llm_text.strip(), re.M)) or bool(re.search(r"^---\s+", llm_text.strip(), re.M))
        looks_like_blocks = ("FILE:" in llm_text) or ("<<<<<<< SEARCH" in llm_text) or ("<<<<<<< REWRITE" in llm_text)

        if looks_like_diff and not looks_like_blocks:
            # unified diff path
            tmp_patch = diff_file.parent / (diff_file.stem + "_unified.diff")
            ok, logs = self._git_apply_unified_diff(llm_text, tmp_patch)
            if not ok:
                return False, logs
        else:
            edits, perr = self._parse_edit_instructions(llm_text)
            if perr:
                return False, "[EDIT PARSE ERROR]\n" + perr
            ok, logs = self._apply_edit_instructions(edits)
            if not ok:
                return False, logs

        # On success, persist the actual diff vs baseline
        diff = self.sandbox.run(["git", "diff"], cwd=self.work_dir)
        diff_file.write_text(diff.stdout or "", encoding="utf-8")
        return True, ""

    def _parse_edit_instructions(self, llm_text: str) -> Tuple[List[FileEditInstruction], str]:
        """
        Parse Search/Replace blocks and Full Rewrite blocks from LLM output.
        Expected:
          FILE: path
          <<<<<<< SEARCH
          ...
          =======
          ...
          >>>>>>> REPLACE

        Or:
          FILE: path
          <<<<<<< REWRITE
          ...
          >>>>>>> REWRITE

        Returns (edits, error_message). error_message == "" when ok.
        """
        if llm_text is None:
            return [], "Empty output."

        t = llm_text.replace("\r\n", "\n").replace("\r", "\n")
        lines = t.split("\n")

        edits: List[FileEditInstruction] = []
        cur: Optional[FileEditInstruction] = None

        def finish_current():
            nonlocal cur
            if cur is not None:
                if cur.blocks is None:
                    cur.blocks = []
                edits.append(cur)
                cur = None

        i = 0
        while i < len(lines):
            ln = lines[i]
            m = re.match(r"^\s*FILE:\s*(.+?)\s*$", ln)
            if m:
                finish_current()
                cur = FileEditInstruction(path=m.group(1).strip(), blocks=[], rewrite=None)
                i += 1
                continue

            if cur is None:
                i += 1
                continue

            if ln.strip() == "<<<<<<< SEARCH":
                i += 1
                search_lines = []
                while i < len(lines) and lines[i].strip() != "=======":
                    search_lines.append(lines[i])
                    i += 1
                if i >= len(lines) or lines[i].strip() != "=======":
                    return [], f"Missing '=======' after SEARCH block in file {cur.path}."
                i += 1
                replace_lines = []
                while i < len(lines) and lines[i].strip() != ">>>>>>> REPLACE":
                    replace_lines.append(lines[i])
                    i += 1
                if i >= len(lines) or lines[i].strip() != ">>>>>>> REPLACE":
                    return [], f"Missing '>>>>>>> REPLACE' after REPLACE block in file {cur.path}."
                i += 1
                cur.blocks.append(SearchReplaceBlock(search="\n".join(search_lines), replace="\n".join(replace_lines)))
                continue

            if ln.strip() == "<<<<<<< REWRITE":
                i += 1
                rewrite_lines = []
                while i < len(lines) and lines[i].strip() != ">>>>>>> REWRITE":
                    rewrite_lines.append(lines[i])
                    i += 1
                if i >= len(lines) or lines[i].strip() != ">>>>>>> REWRITE":
                    return [], f"Missing '>>>>>>> REWRITE' for file {cur.path}."
                i += 1
                cur.rewrite = "\n".join(rewrite_lines)
                continue

            i += 1

        finish_current()

        if not edits:
            return [], "No FILE: sections found. Output must start with one or more 'FILE: <path>' headers."
        # Validate
        bad = [e.path for e in edits if not e.path]
        if bad:
            return [], f"Empty FILE path in sections: {bad}"
        return edits, ""

    def _apply_edit_instructions(self, edits: List[FileEditInstruction]) -> Tuple[bool, str]:
        """
        Apply parsed edits to the working tree. Strict by default:
          - SEARCH text must match EXACTLY once
          - REWRITE replaces entire file contents
        """
        logs: List[str] = []
        for e in edits:
            rel = e.path.strip()
            resolved = self._resolve_in_workdir(rel, want_dir=False) or rel.lstrip("/")
            p = self.work_dir / resolved
            if not p.exists() and e.rewrite is None:
                logs.append(f"[EDIT APPLY ERROR] File not found: {resolved}")
                continue

            if e.rewrite is not None:
                p.parent.mkdir(parents=True, exist_ok=True)
                new_text = e.rewrite.replace("\r\n", "\n").replace("\r", "\n")
                if not new_text.endswith("\n"):
                    new_text += "\n"
                p.write_text(new_text, encoding="utf-8")
                continue

            # Search/Replace blocks
            file_text = p.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
            for bi, blk in enumerate(e.blocks or [], start=1):
                search = blk.search.replace("\r\n", "\n").replace("\r", "\n")
                replace = blk.replace.replace("\r\n", "\n").replace("\r", "\n")
                if search == "":
                    logs.append(f"[EDIT APPLY ERROR] Empty SEARCH block in {resolved} (block #{bi}).")
                    continue

                occ = file_text.count(search)
                if occ == 0:
                    # Provide fuzzy hint
                    hint = self._best_fuzzy_match_hint(file_text, search)
                    logs.append(
                        f"[EDIT APPLY ERROR] SEARCH block not found in {resolved} (block #{bi}).\n"
                        f"[SEARCH]\n{search[:800]}\n"
                        f"{hint}"
                    )
                    continue
                if occ > 1:
                    logs.append(
                        f"[EDIT APPLY ERROR] SEARCH block matched {occ} times in {resolved} (block #{bi}). "
                        "Make SEARCH more specific.\n"
                        f"[SEARCH]\n{search[:800]}\n"
                    )
                    continue

                file_text = file_text.replace(search, replace, 1)

            # Only write back if no errors for this file
            if not any(msg.startswith("[EDIT APPLY ERROR]") and f" {resolved}" in msg for msg in logs):
                if not file_text.endswith("\n"):
                    file_text += "\n"
                p.write_text(file_text, encoding="utf-8")

        if logs:
            return False, "\n".join(logs)
        return True, ""

    def _best_fuzzy_match_hint(self, file_text: str, search: str) -> str:
        """
        Best-effort hint for repair: find the most similar window in the file for the given search block.
        """
        try:
            file_lines = file_text.split("\n")
            search_lines = search.split("\n")
            n = len(search_lines)
            if n == 0:
                return ""

            # Limit scanning for performance on huge files
            max_lines = min(len(file_lines), 4000)
            file_lines = file_lines[:max_lines]

            best = (0.0, 0)  # (ratio, start_idx)
            search_join = "\n".join(search_lines)
            for i in range(0, max_lines - n + 1):
                window = "\n".join(file_lines[i:i+n])
                r = difflib.SequenceMatcher(None, search_join, window).ratio()
                if r > best[0]:
                    best = (r, i)
            ratio, idx = best
            if ratio < 0.35:
                return ""
            start = max(idx - 2, 0)
            end = min(idx + n + 2, len(file_lines))
            snippet = "\n".join(file_lines[start:end])
            return f"[FUZZY HINT] Best approx match near line {idx+1} (similarity {ratio:.2f}):\n{snippet}\n"
        except Exception:
            return ""

    def _git_apply_unified_diff(self, patch_text: str, patch_file: Path) -> Tuple[bool, str]:
        """
        Apply patch in work_dir. Returns (ok, logs). Logs are suitable to feed to repair.
        """
        patch_text = self._sanitize_patch_text(patch_text)

        patch_file.write_text(patch_text, encoding="utf-8")

        p_level = self._guess_p_level(patch_text)
        has_git_header = self._patch_has_git_header(patch_text)

        check_cmd = ["git", "apply", "--check", f"-p{p_level}", "--whitespace=nowarn", str(patch_file)]
        check = self.sandbox.run(check_cmd, cwd=self.work_dir)

        # If it's corrupt patch, stop early
        if (not check.ok) and ("corrupt patch" in (check.stderr or "")):
            return False, f"[PATCH APPLY ERROR]\n{check.stdout}\n{check.stderr}"

        # Try apply (3way only if git header exists)
        apply_cmd = ["git", "apply", f"-p{p_level}", "--whitespace=nowarn"]
        if has_git_header:
            apply_cmd.insert(2, "--3way")

        res = self.sandbox.run(apply_cmd + [str(patch_file)], cwd=self.work_dir)
        if res.ok:
            return True, ""

        # Fallback: try toggling p-level once (useful if LLM omitted a/b prefixes)
        alt_p = 0 if p_level == 1 else 1
        apply_cmd2 = ["git", "apply", f"-p{alt_p}", "--whitespace=nowarn"]
        if has_git_header:
            apply_cmd2.insert(2, "--3way")
        res2 = self.sandbox.run(apply_cmd2 + [str(patch_file)], cwd=self.work_dir)
        if res2.ok:
            return True, ""

        extra = self._extract_apply_failure_context(res.stderr + "\n" + res2.stderr)
        logs = (
            "[PATCH APPLY ERROR]\n"
            f"[CHECK CMD] {' '.join(check_cmd)}\n{check.stdout}\n{check.stderr}\n"
            f"[APPLY CMD] {' '.join(apply_cmd)} {patch_file.name}\n{res.stdout}\n{res.stderr}\n"
            f"[APPLY CMD ALT] {' '.join(apply_cmd2)} {patch_file.name}\n{res2.stdout}\n{res2.stderr}\n"
            f"{extra}"
        )
        return False, logs

    # ---------------------------
    # Verify
    # ---------------------------

    def _verify(self, plan: Dict[str, Any]) -> Tuple[bool, str]:
        cmds: List[str] = []

        # Prefer deterministic verification commands from config / auto-detect.
        # LLM-generated verification commands are often incorrect (classpath, globs, etc.),
        # so we only use them when explicitly enabled.
        if self.cfg.use_plan_verification:
            for v in plan.get("verification", []) or []:
                cmd_str = (v.get("cmd") or "").strip()
                if cmd_str:
                    cmds.append(cmd_str)

        if not cmds:
            cmds = list(self.cfg.default_verify_cmds or [])

        # As a last resort, fall back to plan verification (if any)
        if not cmds:
            for v in plan.get("verification", []) or []:
                cmd_str = (v.get("cmd") or "").strip()
                if cmd_str:
                    cmds.append(cmd_str)

        # Pre-create output dirs used by javac -d ...
        self._ensure_output_dirs_for_cmds(cmds)

        logs: List[str] = []
        ok = True
        for cmd_str in cmds:
            cmd = self.sandbox.split_cmd(cmd_str)
            try:
                res = self.sandbox.run(cmd, cwd=self.work_dir)
                logs.append(_format_cmd_result(res))
                if not res.ok:
                    ok = False
            except Exception as e:
                ok = False
                logs.append(f"[VERIFY ERROR] cmd={cmd_str}\n{e}")
        return ok, "\n\n".join(logs)

    # ---------------------------
    # Main loop
    # ---------------------------

    def run(self, *, request: str, context_pack: Dict[str, Any]) -> Dict[str, Any]:
        run_dir = self.work_dir
        artifacts = run_dir / ".agent_artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)

        raw_plan = self._plan(request=request, context_pack=context_pack)
        plan = self._normalize_plan_paths(raw_plan, context_pack)
        objective = request

        tool_info = self._run_tool_requests(plan.get("tool_requests", []) or [])
        (artifacts / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        if tool_info:
            (artifacts / "tool_outputs.txt").write_text(tool_info, encoding="utf-8")

        # Git baseline
        self._ensure_git_baseline(artifacts)
        baseline_rev = self._git_head_hash()

        # Planned/allowed files (also include expected_files from context_pack if present)
        allowed_files = self._compute_allowed_files(plan, context_pack)

        modified_files: List[str] = []

        def collect_modified_files() -> List[str]:
            # report diff vs baseline commit; exclude agent artifacts
            res = self.sandbox.run(["git", "diff", "--name-only", baseline_rev, "HEAD"], cwd=self.work_dir)
            files = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]
            return [f for f in files if not self._is_meta_path(f)]

        # ---------------------------
        # Mode: monolithic (original)
        # ---------------------------
        if (self.cfg.execution_mode or "monolithic").lower() in ("monolithic", "one_shot", "oneshot"):
            prev_patch = ""
            last_logs = ""

            for it in range(1, self.cfg.max_iters + 1):
                # Always reset to baseline between attempts
                self._git_reset_clean()

                if it == 1:
                    patch = self._edit(objective=objective, plan=plan, context_pack=context_pack, extra_tool_info=tool_info)
                else:
                    patch = self._repair(
                        objective=objective,
                        plan=plan,
                        context_pack=context_pack,
                        prev_patch=prev_patch,
                        verify_logs=last_logs,
                        extra_tool_info=tool_info,
                    )

                llm_out_file = artifacts / f"llm_output_{it}.txt"
                llm_out_file.write_text(patch, encoding="utf-8")
                diff_file = artifacts / f"patch_attempt_{it}.diff"
                ok_apply, apply_logs = self._apply_llm_output(patch, diff_file)
                (artifacts / f"apply_attempt_{it}.txt").write_text(apply_logs, encoding="utf-8")
                if not ok_apply:
                    last_logs = apply_logs
                    prev_patch = patch
                    continue

                ok_scope, scope_logs = self._enforce_changed_files_whitelist(
    self._compute_allowed_files(step_plan, context_pack)
)

                if not ok_scope:
                    (artifacts / f"scope_step{si}_attempt{it}.txt").write_text(scope_logs, encoding="utf-8")

                    # 解析 out-of-scope
                    m = re.search(r"Out-of-scope:\s*\[(.*?)\]", scope_logs, flags=re.DOTALL)
                    oos_paths = re.findall(r"'([^']+)'", m.group(0)) if m else []

                    found_set = set((context_pack or {}).get("found_files") or [])
                    candidates = []
                    for p in oos_paths:
                        p = (p or "").strip()
                        if not p:
                            continue
                        # ✅ 建议用 work_dir 判断是否存在（而不是 project_dir）
                        if (p in found_set) or (self._is_safe_repo_path(p) and (self.work_dir / p).exists()):
                            candidates.append(p)

                    candidates = list(dict.fromkeys(candidates))
                    if candidates:
                        scope_expanded_files = list(dict.fromkeys(scope_expanded_files + candidates))
                        allowed_union = list(dict.fromkeys((files_in_step or []) + scope_expanded_files))
                        step_plan["files_to_change"] = [{"file_path": p} for p in allowed_union]

                        # ✅ 关键：扩容后立刻复查 scope（不要直接 continue）
                        ok2, scope_logs2 = self._enforce_changed_files_whitelist(
                            self._compute_allowed_files(step_plan, context_pack)
                        )
                        if ok2:
                            (artifacts / f"scope_step{si}_attempt{it}_expanded.txt").write_text(
                                scope_logs + "\n\n[ALLOWLIST EXPANDED]\n" + scope_logs2,
                                encoding="utf-8",
                            )
                            # ✅ 继续往下走 verify/commit
                        else:
                            step_last_logs = scope_logs2
                            step_prev_patch = patch
                            continue
                    else:
                        step_last_logs = scope_logs
                        step_prev_patch = patch
                        continue


                verify_ok, verify_logs = self._verify(plan)
                (artifacts / f"verify_attempt_{it}.txt").write_text(verify_logs, encoding="utf-8")

                if verify_ok:
                    modified_files = collect_modified_files()
                    summary = {
                        "status": "success",
                        "attempts": it,
                        "work_dir": str(run_dir),
                        "artifacts_dir": str(artifacts),
                        "modified_files": modified_files,
                        "objective": objective,
                    }
                    (artifacts / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                    return summary

                last_logs = verify_logs
                prev_patch = patch

            summary = {
                "status": "failed",
                "attempts": self.cfg.max_iters,
                "work_dir": str(run_dir),
                "artifacts_dir": str(artifacts),
                "modified_files": modified_files,
                "objective": objective,
            }
            (artifacts / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            return summary

        # ---------------------------
        # Mode: stepwise (recommended)
        # ---------------------------
        # Build steps: use plan steps if provided, else split files_to_change into small chunks.
        step_files: List[str] = []
        for f in (plan.get("files_to_change") or []):
            fp = (f.get("file_path") or "").strip()
            if fp:
                step_files.append(fp)
        if not step_files:
            focus_fp = self._focus_file_from_context_pack(context_pack)
            if focus_fp:
                step_files.append(focus_fp)

        # Keep a stable "global" allowlist from the original plan (avoid chunking-induced scope errors)
        global_plan_files = list(dict.fromkeys(step_files))


        # chunk by step_max_files
        chunk = max(1, int(self.cfg.step_max_files or 1))
        steps: List[List[str]] = [step_files[i:i+chunk] for i in range(0, len(step_files), chunk)]
        if not steps:
            steps = [[]]

        for si, files_in_step in enumerate(steps, start=1):
            step_start_rev = self._git_head_hash()
            step_prev_patch = ""
            step_last_logs = ""
            scope_expanded_files: List[str] = []

            step_objective = objective
            if files_in_step:
                step_objective += f"\n\n[STEP {si}/{len(steps)}] Primary edit targets: {', '.join(files_in_step)}"
            else:
                step_objective += f"\n\n[STEP {si}/{len(steps)}]"

            step_plan = dict(plan)

            # Allow edits to: global plan files ∪ current step primary targets ∪ (scope-expanded files)
            allowed_union = list(dict.fromkeys((files_in_step or []) + scope_expanded_files))
            if allowed_union:
                step_plan["files_to_change"] = [{"file_path": p} for p in allowed_union]


            for it in range(1, self.cfg.max_iters + 1):
                # Reset to start of this step (NOT to baseline) so we can accumulate progress.
                self._git_reset_clean_to(step_start_rev)

                if it == 1:
                    patch = self._edit(
                        objective=step_objective,
                        plan=step_plan,
                        context_pack=context_pack,
                        extra_tool_info=tool_info,
                        only_files=allowed_union,
                    )
                else:
                    patch = self._repair(
                        objective=step_objective,
                        plan=step_plan,
                        context_pack=context_pack,
                        prev_patch=step_prev_patch,
                        verify_logs=step_last_logs,
                        extra_tool_info=tool_info,
                        only_files=allowed_union,
                    )

                llm_out_file = artifacts / f"llm_output_step{si}_attempt{it}.txt"
                llm_out_file.write_text(patch, encoding="utf-8")
                diff_file = artifacts / f"patch_step{si}_attempt{it}.diff"

                ok_apply, apply_logs = self._apply_llm_output(patch, diff_file)
                (artifacts / f"apply_step{si}_attempt{it}.txt").write_text(apply_logs, encoding="utf-8")
                if not ok_apply:
                    # If apply failed because SEARCH block didn't match, it often means the model
                    # did not have the authoritative file contents (e.g., missed Javadoc/comments).
                    # Extract the file path(s) and include them in the next prompt + allowlist.
                    missing = self._extract_apply_search_mismatch_files(apply_logs)
                    expanded: List[str] = []
                    for p in missing:
                        if self._is_safe_repo_path(p) and (self.work_dir / p).exists():
                            expanded.append(p)

                    if expanded:
                        scope_expanded_files = list(dict.fromkeys(scope_expanded_files + expanded))
                        allowed_union = list(dict.fromkeys((files_in_step or []) + scope_expanded_files))
                        step_plan["files_to_change"] = [{"file_path": p} for p in allowed_union]

                    step_last_logs = apply_logs
                    step_prev_patch = patch
                    continue

                ok_scope, scope_logs = self._enforce_changed_files_whitelist(
    self._compute_allowed_files(step_plan, context_pack)
)

                if not ok_scope:
                    (artifacts / f"scope_step{si}_attempt{it}.txt").write_text(scope_logs, encoding="utf-8")

                    # 解析 out-of-scope
                    m = re.search(r"Out-of-scope:\s*\[(.*?)\]", scope_logs, flags=re.DOTALL)
                    oos_paths = re.findall(r"'([^']+)'", m.group(0)) if m else []

                    found_set = set((context_pack or {}).get("found_files") or [])
                    candidates = []
                    for p in oos_paths:
                        p = (p or "").strip()
                        if not p:
                            continue
                        # ✅ 建议用 work_dir 判断是否存在（而不是 project_dir）
                        if (p in found_set) or (self._is_safe_repo_path(p) and (self.work_dir / p).exists()):
                            candidates.append(p)

                    candidates = list(dict.fromkeys(candidates))
                    if candidates:
                        scope_expanded_files = list(dict.fromkeys(scope_expanded_files + candidates))
                        allowed_union = list(dict.fromkeys((files_in_step or []) + scope_expanded_files))
                        step_plan["files_to_change"] = [{"file_path": p} for p in allowed_union]

                        # ✅ 关键：扩容后立刻复查 scope（不要直接 continue）
                        ok2, scope_logs2 = self._enforce_changed_files_whitelist(
                            self._compute_allowed_files(step_plan, context_pack)
                        )
                        if ok2:
                            (artifacts / f"scope_step{si}_attempt{it}_expanded.txt").write_text(
                                scope_logs + "\n\n[ALLOWLIST EXPANDED]\n" + scope_logs2,
                                encoding="utf-8",
                            )
                            # ✅ 继续往下走 verify/commit
                        else:
                            step_last_logs = scope_logs2
                            step_prev_patch = patch
                            continue
                    else:
                        step_last_logs = scope_logs
                        step_prev_patch = patch
                        continue


                if self.cfg.verify_each_step:
                    verify_ok, verify_logs = self._verify(step_plan)
                    (artifacts / f"verify_step{si}_attempt{it}.txt").write_text(verify_logs, encoding="utf-8")
                    if not verify_ok:
                        # Error-driven scope expansion: when compilation/test fails due to missing symbols,
                        # automatically include the referenced files in the next iteration's edit scope and
                        # dump their full contents to the model (to avoid API skew across files).
                        extra_files = [p for p in self._extract_compiler_error_files(verify_logs) if self._is_safe_repo_path(p) and (self.work_dir / p).exists()]
                        if extra_files:
                            cur_files = [str(x.get("file_path") or "").strip() for x in (step_plan.get("files_to_change") or []) if str(x.get("file_path") or "").strip()]
                            merged = cur_files + [p for p in extra_files if p not in cur_files]
                            step_plan["files_to_change"] = [{"file_path": p} for p in merged]
                            step_objective += "\n\n[ERROR-DRIVEN SCOPE EXPANSION] Verification failed. You may (and likely must) also edit: " + ", ".join(extra_files)
                        step_last_logs = verify_logs
                        step_prev_patch = patch
                        continue

                # Step success: checkpoint commit
                try:
                    self._git_commit_checkpoint(f"step {si}")
                except Exception:
                    # ignore if nothing to commit
                    pass
                modified_files = collect_modified_files()
                break
            else:
                # step failed
                summary = {
                    "status": "failed",
                    "attempts": self.cfg.max_iters,
                    "failed_step": si,
                    "work_dir": str(run_dir),
                    "artifacts_dir": str(artifacts),
                    "modified_files": modified_files,
                    "objective": objective,
                }
                (artifacts / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                return summary

        # Final verification (even if verify_each_step=False, do it once at the end)
        final_ok, final_logs = self._verify(plan)
        (artifacts / "verify_final.txt").write_text(final_logs, encoding="utf-8")
        if final_ok:
            modified_files = collect_modified_files()
            summary = {
                "status": "success",
                "attempts": len(steps),
                "work_dir": str(run_dir),
                "artifacts_dir": str(artifacts),
                "modified_files": modified_files,
                "objective": objective,
            }
            (artifacts / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            return summary

        summary = {
            "status": "failed",
            "attempts": len(steps),
            "work_dir": str(run_dir),
            "artifacts_dir": str(artifacts),
            "modified_files": modified_files,
            "objective": objective,
        }
        (artifacts / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary


def _format_cmd_result(res: CommandResult) -> str:
    cmd_str = " ".join(res.cmd)
    return (
        f"[CMD] {cmd_str}\n"
        f"[RET] {res.returncode}\n"
        f"[STDOUT]\n{res.stdout}\n"
        f"[STDERR]\n{res.stderr}\n"
    )

