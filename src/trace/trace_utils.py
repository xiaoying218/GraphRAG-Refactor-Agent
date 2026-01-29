# src/trace/trace_utils.py
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


_CURRENT_TRACER: contextvars.ContextVar[Optional["TraceLogger"]] = contextvars.ContextVar(
    "CURRENT_TRACER", default=None
)


def get_tracer() -> Optional["TraceLogger"]:
    return _CURRENT_TRACER.get()


@contextlib.contextmanager
def using_tracer(tracer: "TraceLogger"):
    token = _CURRENT_TRACER.set(tracer)
    try:
        yield tracer
    finally:
        _CURRENT_TRACER.reset(token)


def _stable_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(obj)


def digest_obj(obj: Any, *, limit: int = 20000) -> str:
    """
    Digest large input/output objects so trace stays light.
    """
    if obj is None:
        return ""
    s = _stable_json(obj)
    if len(s) > limit:
        s = s[:limit] + f"...(truncated,{len(s)} chars)"
    h = hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()
    return h


def approx_token_count(text: str) -> int:
    # 粗估：英文 ~4 chars/token；中文更密一点也差不多够用做趋势分析
    if not text:
        return 0
    return max(1, len(text) // 4)


def _guess_token_est(input_obj: Any, output_obj: Any) -> int:
    # 尽量给个 token_est，没法精确但能做对比/趋势
    if isinstance(output_obj, str):
        return approx_token_count(output_obj)
    if isinstance(input_obj, str):
        return approx_token_count(input_obj)
    try:
        return approx_token_count(_stable_json(output_obj))
    except Exception:
        return 0


@dataclass
class TraceSpan:
    tracer: "TraceLogger"
    stage: str
    tool: str
    input_obj: Any = None
    extra: Optional[Dict[str, Any]] = None
    token_est: Optional[int] = None

    _start: float = 0.0
    _output_obj: Any = None
    _ok: bool = True
    _error_type: Optional[str] = None

    def set_output(self, output_obj: Any) -> None:
        self._output_obj = output_obj

    def add_extra(self, **kwargs: Any) -> None:
        if self.extra is None:
            self.extra = {}
        self.extra.update(kwargs)

    def __enter__(self) -> "TraceSpan":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        latency_ms = int((time.perf_counter() - self._start) * 1000)
        ok = exc is None
        error_type = None if ok else (exc_type.__name__ if exc_type else "Exception")

        token_est = self.token_est
        if token_est is None:
            token_est = _guess_token_est(self.input_obj, self._output_obj)

        self.tracer.log(
            stage=self.stage,
            tool=self.tool,
            input_obj=self.input_obj,
            output_obj=self._output_obj,
            latency_ms=latency_ms,
            ok=ok,
            error_type=error_type,
            token_est=token_est,
            extra=self.extra,
        )
        # 不吞异常
        return False


class TraceLogger:
    """
    One JSON per line:
      run_id, stage, tool, input_digest, output_digest, latency_ms, ok, error_type, token_est, extra
    """

    def __init__(self, *, run_id: str, trace_path: Path, base_extra: Optional[Dict[str, Any]] = None) -> None:
        self.run_id = run_id
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.base_extra = base_extra or {}
        self._lock = threading.Lock()

    def span(
        self,
        *,
        stage: str,
        tool: str,
        input_obj: Any = None,
        extra: Optional[Dict[str, Any]] = None,
        token_est: Optional[int] = None,
    ) -> TraceSpan:
        return TraceSpan(
            tracer=self,
            stage=stage,
            tool=tool,
            input_obj=input_obj,
            extra=extra,
            token_est=token_est,
        )

    def log(
        self,
        *,
        stage: str,
        tool: str,
        input_obj: Any = None,
        output_obj: Any = None,
        latency_ms: Optional[int] = None,
        ok: bool = True,
        error_type: Optional[str] = None,
        token_est: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "run_id": self.run_id,
            "stage": stage,
            "tool": tool,
            "input_digest": digest_obj(input_obj),
            "output_digest": digest_obj(output_obj),
            "latency_ms": latency_ms,
            "ok": bool(ok),
            "error_type": error_type or "",
            "token_est": int(token_est) if token_est is not None else 0,
            "extra": {**self.base_extra, **(extra or {})},
        }
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
