from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from ..dotenv import auto_load_dotenv
from typing import Any, Dict, List, Optional, Protocol, Tuple


class LLMClient(Protocol):
    def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.2, max_tokens: int = 1800) -> str:
        ...


@dataclass
class OpenAICompatibleConfig:
    """
    A minimal OpenAI-compatible Chat Completions config.

    Works with:
      - OpenAI (if /v1/chat/completions is enabled)
      - Many self-hosted servers (vLLM / LM Studio / OpenRouter-like gateways) that expose OpenAI-style endpoints.
    """
    base_url: str = "https://api.openai.com"
    api_key: Optional[str] = None
    model: str = "gpt-4.1-mini"
    timeout_s: int = 120
    max_retries: int = 2
    retry_backoff_s: float = 1.5
    # If True, do not verify TLS certs (NOT recommended). Kept for corporate proxies edge cases.
    insecure_skip_verify: bool = False


class OpenAICompatibleChatClient:
    def __init__(self, cfg: OpenAICompatibleConfig):
        self.cfg = cfg
        if not self.cfg.api_key:
            # Try loading from a local .env file (common in research demos).
            auto_load_dotenv(os.environ.get("REFAC_DOTENV") or os.environ.get("DOTENV_PATH"))
            self.cfg.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")

    def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.2, max_tokens: int = 1800) -> str:
        if not self.cfg.api_key:
            raise RuntimeError(
                "Missing API key. Set OpenAICompatibleConfig.api_key or environment variable OPENAI_API_KEY."
            )
        url = self.cfg.base_url.rstrip("/") + "/v1/chat/completions"

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        data = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key}",
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                ctx = None
                if self.cfg.insecure_skip_verify:
                    import ssl
                    ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_s, context=ctx) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                obj = json.loads(raw)
                # OpenAI-style response: choices[0].message.content
                return obj["choices"][0]["message"]["content"]
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
                last_err = e
                if attempt >= self.cfg.max_retries:
                    break
                time.sleep(self.cfg.retry_backoff_s * (attempt + 1))
        raise RuntimeError(f"LLM request failed after retries: {last_err}") from last_err


class DummyEchoLLM:
    """
    For dry-runs without an API key.

    This dummy client tries to be *structurally compatible* with the agent:
      - For planning prompts that require STRICT JSON, it returns a minimal valid plan.
      - For edit/repair prompts, it returns a harmless "create a new file" rewrite
        so the patch parser/apply step succeeds and verification can run.

    IMPORTANT: This does NOT attempt to accomplish the user's refactoring request.
    """
    def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.2, max_tokens: int = 1800) -> str:
        system = ""
        for m in messages:
            if m.get("role") == "system":
                system = str(m.get("content") or "")
                break

        # 1) Plan stage: must be valid JSON
        if "Return STRICT JSON" in system and "refactoring plan" in system.lower():
            plan = {
                "objective": "DRY RUN (no-op)",
                "assumptions": ["This is a dry run using DummyEchoLLM."],
                "steps": ["Do not change code."],
                "files_to_change": [],
                "verification": [],
                "tool_requests": [],
            }
            return json.dumps(plan)

        # 2) Edit/repair stage: return a valid edit instruction that applies cleanly
        if "output ONLY edit instructions" in system:
            return (
                "FILE: DRY_RUN_NOOP.txt\n"
                "<<<<<<< REWRITE\n"
                "DRY RUN: DummyEchoLLM created this file so the patch-apply step is exercised.\n"
                ">>>>>>> REWRITE\n"
            )

        # Fallback
        return "{}"