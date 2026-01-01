from __future__ import annotations

import os
import glob
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class CommandResult:
    cmd: List[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class SandboxConfig:
    root_dir: Path
    timeout_s: int = 240
    allowed_commands: List[str] = field(default_factory=lambda: [
        # Search / inspection
        "rg", "grep", "find", "ls", "cat", "sed", "python", "python3",
        # Build / test
        "mvn", "./mvnw", "gradle", "./gradlew", "npm", "pnpm", "yarn",
        "pytest",
        # Format / lint (optional)
        "black", "ruff", "prettier", "eslint", "google-java-format",
        # git (optional)
        "git",
        # java tools
        "javac", "java"
    ])
    use_docker: bool = False
    docker_image: str = "python:3.10-slim"  # user should override for Java/Maven projects
    docker_workdir: str = "/repo"


class Sandbox:
    """
    Runs commands in a controlled way:
    - shell=False (no pipes/&&)
    - only whitelisted base commands
    - optionally run inside a docker container with repo mounted read-write
    """
    def __init__(self, cfg: SandboxConfig):
        self.cfg = cfg
        self.cfg.root_dir = Path(self.cfg.root_dir).resolve()

    def _is_allowed(self, cmd: List[str]) -> bool:
        if not cmd:
            return False
        base = cmd[0]
        return base in set(self.cfg.allowed_commands)

    def _expand_globs(self, cmd: List[str], *, cwd: Path) -> List[str]:
        """Expand shell-style globs (* ? [..]) because we run with shell=False.

        This matches typical shell behavior for arguments like 'src/**/*.java'.
        If a pattern has no matches, it is kept as-is.
        """
        expanded: List[str] = []
        for arg in cmd:
            # Skip options (e.g. '-cp') and URLs etc.; only expand likely file patterns
            if any(ch in arg for ch in ("*", "?", "[")) and not arg.startswith("-"):
                matches = glob.glob(str(cwd / arg), recursive=True)
                if matches:
                    # Make matches relative to cwd for nicer logs and compatibility
                    for mpath in sorted(matches):
                        expanded.append(os.path.relpath(mpath, str(cwd)))
                    continue
            expanded.append(arg)
        return expanded


    def run(self, cmd: List[str], *, cwd: Optional[Path] = None) -> CommandResult:
        if not self._is_allowed(cmd):
            raise PermissionError(f"Command not allowed: {cmd}. Allowed={self.cfg.allowed_commands}")

        cwd = (cwd or self.cfg.root_dir).resolve()

        # Expand globs like *.java because we run with shell=False
        cmd = self._expand_globs(cmd, cwd=cwd)

        # Ensure cwd stays inside root_dir
        if not str(cwd).startswith(str(self.cfg.root_dir)):
            raise PermissionError(f"cwd must be inside sandbox root: {cwd}")

        if self.cfg.use_docker:
            return self._run_in_docker(cmd, cwd=cwd)

        # Local run
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            shell=False,
            capture_output=True,
            text=True,
            timeout=self.cfg.timeout_s,
        )
        return CommandResult(cmd=cmd, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)

    def _run_in_docker(self, cmd: List[str], *, cwd: Path) -> CommandResult:
        # Map cwd relative to repo root into container workdir
        rel = cwd.relative_to(self.cfg.root_dir)
        workdir = str(Path(self.cfg.docker_workdir) / rel)

        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{self.cfg.root_dir}:{self.cfg.docker_workdir}",
            "-w", workdir,
            self.cfg.docker_image
        ] + cmd

        # NOTE: we intentionally do NOT allow arbitrary docker args from the LLM; only from config.
        p = subprocess.run(
            docker_cmd,
            shell=False,
            capture_output=True,
            text=True,
            timeout=self.cfg.timeout_s,
        )
        return CommandResult(cmd=docker_cmd, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)

    @staticmethod
    def split_cmd(cmd_str: str) -> List[str]:
        # shlex split (still safe because we run shell=False)
        return shlex.split(cmd_str)
