from __future__ import annotations

import json
import sys
import time
import uuid
import threading
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# -----------------------------
# Repo root detection
# -----------------------------
def find_repo_root(start: Path) -> Path:
    """
    Walk up from start to find a directory containing demo_context_pack.py.
    Fallback to start if not found.
    """
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "demo_context_pack.py").exists():
            return p
    return cur


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(THIS_DIR)

RUNS_DIR = REPO_ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_BIN = sys.executable  # ensure same venv


def now_ms() -> int:
    return int(time.time() * 1000)


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_relpath(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(p)


# -----------------------------
# Schemas
# -----------------------------
class ContextPackRequest(BaseModel):
    project: str = Field(..., description="Java project root directory (relative to repo root or absolute).")
    query: str = Field(..., description="Natural language query OR exact node id.")
    # demo_context_pack.py doesn't accept mode yet; keep it for API compatibility & future wiring
    mode: str = Field("graph_rag", description="Reserved for future: graph_rag|vector_only|bm25_only|hybrid ...")
    hops: int = Field(1, ge=0, le=4)
    seed_top_k: int = Field(8, ge=1, le=50)
    max_nodes: int = Field(30, ge=5, le=200)
    out_path: str = Field("context_pack.json", description="Output filename under runs/<run_id>/.")


class ContextPackResponse(BaseModel):
    run_id: str
    ok: bool
    mode: str
    hops: int
    latency_ms: int
    output_path: str
    pack_preview: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkRequest(BaseModel):
    project: str = Field(..., description="Java project root directory.")
    tasks_path: str = Field("data/bench_tasks.sample.json", description="Path to tasks JSON.")
    modes: str = Field("graph_rag,vector_only", description="Comma-separated: graph_rag,vector_only")
    out_dir: str = Field("", description="Optional. If empty -> runs/<run_id>/bench")
    async_run: bool = Field(True, description="Run in background thread.")

    # Optional passthrough flags (aligned to your demo_benchmark.py)
    dotenv: str = Field("", description="Optional path to .env for runner (empty = auto).")
    vector_only_no_search_tools: bool = Field(False)
    seed_top_k: int = Field(8, ge=1, le=50)
    hops: int = Field(1, ge=0, le=4)
    max_nodes: int = Field(30, ge=5, le=200)
    max_iters: int = Field(3, ge=1, le=20)
    dry_llm: bool = Field(False)
    accept_mode: str = Field("strict", description="strict|semantic")
    force_regex_parser: bool = Field(False)


class BenchmarkResponse(BaseModel):
    run_id: str
    ok: bool
    status: str  # started|running|finished|failed
    out_dir: str
    message: str = ""


class RefactorRequest(BaseModel):
    project: str = Field(..., description="Java project root directory.")
    request: str = Field(..., description="Natural language refactor request.")
    max_iters: int = Field(3, ge=1, le=20)

    # reserved for future: allow building context pack before refactor
    query: str = Field("", description="Optional query used to build context pack if needed.")
    hops: int = Field(1, ge=0, le=4)
    seed_top_k: int = Field(8, ge=1, le=50)
    max_nodes: int = Field(30, ge=5, le=200)


class RefactorResponse(BaseModel):
    run_id: str
    ok: bool
    status: str
    output_dir: str
    message: str = ""


# -----------------------------
# Subprocess adapters (aligned to your CLI)
# -----------------------------
def run_demo_context_pack(req: ContextPackRequest, run_dir: Path) -> Path:
    """
    demo_context_pack.py --project --query [--seed_top_k] [--hops] [--max_nodes] [--out]
    """
    script = REPO_ROOT / "demo_context_pack.py"
    if not script.exists():
        raise RuntimeError(f"demo_context_pack.py not found under repo root: {REPO_ROOT}")

    out_path = run_dir / req.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_BIN,
        str(script),
        "--project",
        req.project,
        "--query",
        req.query,
        "--seed_top_k",
        str(req.seed_top_k),
        "--hops",
        str(req.hops),
        "--max_nodes",
        str(req.max_nodes),
        "--out",
        str(out_path),
    ]

    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    if not out_path.exists():
        raise RuntimeError(f"context pack not generated: {out_path}")
    return out_path


def run_demo_benchmark(req: BenchmarkRequest, out_dir: Path) -> None:
    """
    demo_benchmark.py [--dotenv] --project --tasks --out [--modes] [--vector-only-no-search-tools]
                     [--seed-top-k] [--hops] [--max-nodes] [--max-iters] [--dry-llm]
                     [--accept-mode] [--force-regex-parser]
    """
    script = REPO_ROOT / "demo_benchmark.py"
    if not script.exists():
        raise RuntimeError(f"demo_benchmark.py not found under repo root: {REPO_ROOT}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_BIN,
        str(script),
        "--project",
        req.project,
        "--tasks",
        req.tasks_path,
        "--out",
        str(out_dir),
        "--modes",
        req.modes,
        "--seed-top-k",
        str(req.seed_top_k),
        "--hops",
        str(req.hops),
        "--max-nodes",
        str(req.max_nodes),
        "--max-iters",
        str(req.max_iters),
        "--accept-mode",
        req.accept_mode,
    ]

    if req.dotenv:
        cmd += ["--dotenv", req.dotenv]
    if req.vector_only_no_search_tools:
        cmd.append("--vector-only-no-search-tools")
    if req.dry_llm:
        cmd.append("--dry-llm")
    if req.force_regex_parser:
        cmd.append("--force-regex-parser")

    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


# -----------------------------
# Status registries
# -----------------------------
BENCH_STATUS: Dict[str, Dict[str, Any]] = {}
REFACTOR_STATUS: Dict[str, Dict[str, Any]] = {}


def bench_worker(run_id: str, req: BenchmarkRequest, out_dir: Path) -> None:
    BENCH_STATUS[run_id] = {"status": "running", "out_dir": str(out_dir), "error": ""}
    try:
        run_demo_benchmark(req, out_dir=out_dir)
        BENCH_STATUS[run_id]["status"] = "finished"
    except Exception as e:
        BENCH_STATUS[run_id]["status"] = "failed"
        BENCH_STATUS[run_id]["error"] = str(e)


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="GraphRAG Context Engine API", version="0.1.0")


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True, "repo_root": str(REPO_ROOT), "time_ms": now_ms()}


@app.post("/context_pack", response_model=ContextPackResponse)
def context_pack(req: ContextPackRequest) -> ContextPackResponse:
    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = now_ms()
    try:
        pack_path = run_demo_context_pack(req, run_dir=run_dir)
        latency = now_ms() - t0

        pack = read_json_if_exists(pack_path)

        # keep payload small: preview key fields if present
        preview: Dict[str, Any] = {}
        if isinstance(pack, dict):
            for k in ["meta", "summary", "stats"]:
                if k in pack:
                    preview[k] = pack[k]
            # some implementations store "context" or "pack"
            for k in ["context_pack", "pack", "context"]:
                if k in pack and k not in preview:
                    preview[k] = pack[k]
            if not preview:
                preview = {k: pack[k] for k in list(pack.keys())[:5]}

        return ContextPackResponse(
            run_id=run_id,
            ok=True,
            mode=req.mode,
            hops=req.hops,
            latency_ms=latency,
            output_path=safe_relpath(pack_path),
            pack_preview=preview,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"demo_context_pack failed: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/benchmark", response_model=BenchmarkResponse)
def benchmark(req: BenchmarkRequest) -> BenchmarkResponse:
    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(req.out_dir) if req.out_dir else (run_dir / "bench")
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    if req.async_run:
        th = threading.Thread(target=bench_worker, args=(run_id, req, out_dir), daemon=True)
        th.start()
        BENCH_STATUS[run_id] = {"status": "started", "out_dir": str(out_dir), "error": ""}
        return BenchmarkResponse(run_id=run_id, ok=True, status="started", out_dir=safe_relpath(out_dir))
    else:
        try:
            BENCH_STATUS[run_id] = {"status": "running", "out_dir": str(out_dir), "error": ""}
            run_demo_benchmark(req, out_dir=out_dir)
            BENCH_STATUS[run_id]["status"] = "finished"
            return BenchmarkResponse(run_id=run_id, ok=True, status="finished", out_dir=safe_relpath(out_dir))
        except Exception as e:
            BENCH_STATUS[run_id]["status"] = "failed"
            BENCH_STATUS[run_id]["error"] = str(e)
            raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/benchmark/{run_id}", response_model=BenchmarkResponse)
def benchmark_status(run_id: str) -> BenchmarkResponse:
    st = BENCH_STATUS.get(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_id not found")
    status = st.get("status", "unknown")
    out_dir = st.get("out_dir", "")
    err = st.get("error", "")
    ok = status in ("started", "running", "finished")
    msg = err if status == "failed" else ""
    return BenchmarkResponse(run_id=run_id, ok=ok, status=status, out_dir=safe_relpath(Path(out_dir)), message=msg)


# Optional endpoint: stub for now (so curl always returns JSON)
@app.post("/run_refactor", response_model=RefactorResponse)
def run_refactor(req: RefactorRequest) -> RefactorResponse:
    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # You can wire your agent entrypoint later.
    # For Day2, we just return a structured response.
    REFACTOR_STATUS[run_id] = {"status": "failed", "out_dir": str(run_dir), "error": "Not wired yet"}
    return RefactorResponse(
        run_id=run_id,
        ok=False,
        status="failed",
        output_dir=safe_relpath(run_dir),
        message="run_refactor not wired yet. Implement agent entrypoint later.",
    )


@app.get("/run_refactor/{run_id}", response_model=RefactorResponse)
def refactor_status(run_id: str) -> RefactorResponse:
    st = REFACTOR_STATUS.get(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_id not found")
    status = st.get("status", "unknown")
    out_dir = st.get("out_dir", "")
    err = st.get("error", "")
    ok = status in ("started", "running", "finished")
    msg = err if status == "failed" else ""
    return RefactorResponse(run_id=run_id, ok=ok, status=status, output_dir=safe_relpath(Path(out_dir)), message=msg)
