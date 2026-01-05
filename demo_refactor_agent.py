#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from src.agent.agent import AgentConfig, RefactoringAgent
from src.agent.llm import OpenAICompatibleChatClient, OpenAICompatibleConfig, DummyEchoLLM

def load_dotenv(dotenv_path: Path, *, override: bool = False) -> None:
    """
    Minimal .env loader (no extra dependency).
    - Supports KEY=VALUE
    - Ignores blank lines / comments (# ...)
    - Strips surrounding quotes
    - Expands ${VAR} using current env
    - By default does NOT override existing env vars (override=False)
    """
    if not dotenv_path.exists():
        return

    text = dotenv_path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()

        if not key:
            continue

        # strip quotes
        if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            val = val[1:-1]

        # allow \n in env values
        val = val.replace("\\n", "\n")

        # expand ${VAR}
        val = re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1), ""), val)

        if override or key not in os.environ:
            os.environ[key] = val


def env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def env_list(key: str) -> list[str]:
    """
    Reads a list from env. Split by ';' or newline.
    """
    v = (os.environ.get(key) or "").strip()
    if not v:
        return []
    parts = re.split(r"[;\n]+", v)
    return [p.strip() for p in parts if p.strip()]


def require(value: str | None, name: str, ap: argparse.ArgumentParser, hint: str) -> str:
    if value is None or str(value).strip() == "":
        ap.error(f"Missing {name}. {hint}")
    return str(value)


def load_context_pack(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()

    # 先让你可以指定 .env 文件位置（默认当前目录 .env）
    ap.add_argument("--env", default=".env", help="Path to .env file. Default: .env in current directory.")

    # CLI 不再 required，全部可从 .env 读取；CLI 仍可覆盖
    ap.add_argument("--project", default=None, help="Project repo root. Fallback: REFAC_PROJECT in .env")
    ap.add_argument("--context-pack", default=None, help="Path to context_pack.json. Fallback: REFAC_CONTEXT_PACK")
    ap.add_argument("--request", default=None, help="Refactoring request. Fallback: REFAC_REQUEST")

    ap.add_argument("--max-iters", type=int, default=None, help="Fallback: REFAC_MAX_ITERS (default 3)")

    ap.add_argument("--model", default=None, help="Fallback: OPENAI_MODEL (default gpt-4.1-mini)")
    ap.add_argument("--base-url", default=None, help="Fallback: OPENAI_BASE_URL (default https://api.openai.com)")
    ap.add_argument("--api-key", default=None, help="Fallback: OPENAI_API_KEY")

    ap.add_argument("--use-docker", action="store_true",
                    help="Run verification in docker. Fallback: REFAC_USE_DOCKER=true/false")
    ap.add_argument("--docker-image", default=None, help="Fallback: REFAC_DOCKER_IMAGE")

    ap.add_argument("--verify-cmd", action="append", default=None,
                    help="Repeatable verification cmd. Fallback: REFAC_VERIFY_CMDS separated by ';' or newlines")
    ap.add_argument("--allow-cmd", action="append", default=None,
                    help="Extend whitelist. Fallback: REFAC_ALLOW_CMDS separated by ';' or newlines")

    ap.add_argument("--dry-llm", action="store_true",
                    help="Do not call real LLM. Fallback: REFAC_DRY_LLM=true/false")

    ap.add_argument("--print-effective-config", action="store_true",
                    help="Print resolved config (after .env) and exit.")

    args = ap.parse_args()

    # 加载 .env（不覆盖已存在的系统环境变量）
    dotenv_path = Path(args.env).expanduser().resolve()
    load_dotenv(dotenv_path, override=False)

    # 优先级：CLI > .env（os.environ）> 默认值
    project = args.project or os.environ.get("REFAC_PROJECT")
    context_pack = args.context_pack or os.environ.get("REFAC_CONTEXT_PACK")
    request = args.request or os.environ.get("REFAC_REQUEST")

    max_iters = args.max_iters if args.max_iters is not None else int(os.environ.get("REFAC_MAX_ITERS", "3"))

    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")

    # bool：CLI 只负责把 true 打开；否则走 env
    use_docker = args.use_docker or env_bool("REFAC_USE_DOCKER", default=False)
    docker_image = args.docker_image or os.environ.get("REFAC_DOCKER_IMAGE", "python:3.10-slim")

    verify_cmds = (args.verify_cmd or []) if args.verify_cmd else env_list("REFAC_VERIFY_CMDS")
    allow_cmds = (args.allow_cmd or []) if args.allow_cmd else env_list("REFAC_ALLOW_CMDS")

    dry_llm = args.dry_llm or env_bool("REFAC_DRY_LLM", default=False)

    # 必填项：CLI 或 env 至少要有一个
    project = require(project, "--project / REFAC_PROJECT", ap, "Set it in .env or pass --project.")
    context_pack = require(context_pack, "--context-pack / REFAC_CONTEXT_PACK", ap, "Set it in .env or pass --context-pack.")
    request = require(request, "--request / REFAC_REQUEST", ap, "Set it in .env or pass --request.")

    if args.print_effective_config:
        print(json.dumps({
            "env_file": str(dotenv_path),
            "project": project,
            "context_pack": context_pack,
            "request": request,
            "max_iters": max_iters,
            "model": model,
            "base_url": base_url,
            "api_key_set": bool(api_key),
            "use_docker": use_docker,
            "docker_image": docker_image,
            "verify_cmds": verify_cmds,
            "allow_cmds": allow_cmds,
            "dry_llm": dry_llm,
        }, ensure_ascii=False, indent=2))
        return

    project_dir = Path(project).resolve()
    context_pack_path = Path(context_pack).resolve()
    pack = load_context_pack(context_pack_path)

    if dry_llm:
        llm = DummyEchoLLM()
    else:
        llm = OpenAICompatibleChatClient(OpenAICompatibleConfig(
            base_url=base_url,
            api_key=api_key or None,
            model=model,
        ))

    cfg = AgentConfig(
        project_dir=project_dir,
        max_iters=max_iters,
        use_docker=use_docker,
        docker_image=docker_image,
        default_verify_cmds=verify_cmds if verify_cmds else None,
        allowed_commands=None,
    )

    if allow_cmds:
        # 注意：只有 verify/tool 用到的 base command 才需要加白名单
        cfg.allowed_commands = list(set(allow_cmds + [
            "rg", "grep", "find", "ls", "cat", "sed",
            "python", "python3",
            "mvn", "./mvnw",
            "gradle", "./gradlew",
            "npm", "pnpm", "yarn",
            "pytest",
            "git",
            "javac", "java",
            "black", "ruff", "prettier", "eslint", "google-java-format",
        ]))

    agent = RefactoringAgent(llm=llm, cfg=cfg)
    summary = agent.run(request=request, context_pack=pack)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

