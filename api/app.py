"""Chemical-Graph-RAG HTTP 服务：薄封装 DHMF.query / 多跳问答 / agentic_query / retrieve_items。

启动（项目根目录，worker 必须为 1，避免多份 FAISS）：

    export DHMF_CONFIG=example/a/config_open.yaml
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 1

或：

    python -m api
"""
from __future__ import annotations

import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PATH = os.environ.get("DHMF_CONFIG", "example/a/config_open.yaml")
MAX_WORKERS = max(1, int(os.environ.get("DHMF_API_WORKERS", "4")))
CONTENT_CHARS = max(50, int(os.environ.get("DHMF_API_CONTENT_CHARS", "500")))

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="dhmf-api")


def _resolve_config_path() -> Path:
    raw = os.environ.get("DHMF_CONFIG", CONFIG_PATH)
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _jsonable(value: Any) -> Any:
    """Make DHMF / numpy values JSON-serializable."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value)
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.DHMF import DHMF
    from src.utils.config import Config

    os.chdir(ROOT)
    config_path = _resolve_config_path()
    if not config_path.is_file():
        raise FileNotFoundError(f"DHMF config not found: {config_path}")

    config = Config.from_yaml(str(config_path))
    graph = DHMF(config)
    graph.pin_retrieve_indexes()
    app.state.graph = graph
    app.state.config_path = str(config_path)
    try:
        yield
    finally:
        try:
            graph.unpin_retrieve_indexes()
        except Exception:
            pass
        app.state.graph = None
        _executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="Chemical Graph RAG",
    description="化工超图 RAG 在线问答。构建请继续用 python main.py。",
    version="0.1.0",
    lifespan=lifespan,
)

_cors = os.environ.get("DHMF_API_CORS", "*")
_origins = [o.strip() for o in _cors.split(",") if o.strip()] or ["*"]
_allow_all = _origins == ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_all else _origins,
    allow_credentials=not _allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    mode: Literal["dual_path"] = "dual_path"


class MultihopQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")


class AgenticQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    chunk_candidate_k: Optional[int] = None
    node_candidate_k: Optional[int] = None


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: int
    answer: str
    retrieval_sources: List[str] = []
    retrieval_doc_ids: List[Any] = []
    latency_s: Optional[float] = None
    retrieve_latency_s: Optional[float] = None
    retrieve_timing: Optional[Dict[str, float]] = None
    usage_prompt_tokens: Optional[int] = None
    usage_completion_tokens: Optional[int] = None
    usage_total_tokens: Optional[int] = None
    plan: Optional[List[dict]] = None
    steps: Optional[List[dict]] = None
    turns: Optional[List[dict]] = None
    protocol: Optional[str] = None
    trace_path: Optional[str] = None
    reasoning_content: Optional[str] = None


def _get_graph(request: Request):
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="DHMF 尚未加载完成")
    return graph


async def _run(func, /, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, partial(func, *args, **kwargs))


def _normalize_respond(respond: Any) -> dict:
    if not isinstance(respond, dict):
        raise HTTPException(status_code=500, detail="unexpected return type")
    out = _jsonable(respond)
    out.setdefault("status", 0)
    out["answer"] = str(out.get("answer") or "")
    out.setdefault("retrieval_sources", [])
    out.setdefault("retrieval_doc_ids", [])
    return out


def _slim_item(item: dict, content_chars: int = CONTENT_CHARS) -> dict:
    chunk = item.get("chunk") or {}
    if not isinstance(chunk, dict):
        chunk = {}
    content = str(chunk.get("content") or "")
    if len(content) > content_chars:
        content = content[:content_chars] + "…"
    return _jsonable({
        "doc_id": item.get("doc_id"),
        "source": item.get("source"),
        "score": item.get("score"),
        "match_type": item.get("match_type"),
        "role": item.get("role"),
        "chunk_id": chunk.get("id"),
        "content": content,
        "node_id": item.get("node_id"),
        "node_name": item.get("node_name"),
    })


@app.get("/")
def root():
    return {
        "service": "Chemical Graph RAG",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query",
        "multihop-query": "POST /multihop-query",
        "agentic-query": "POST /agentic-query",
        "retrieve": "POST /retrieve",
    }


@app.get("/health")
def health(request: Request):
    graph = _get_graph(request)
    return {
        "ok": True,
        "config_path": getattr(request.app.state, "config_path", str(_resolve_config_path())),
        "working_path": graph.config.settings.working_path,
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request):
    """单跳：三路召回 + 可选 rerank + 一次生成。对应 graph.query。"""
    graph = _get_graph(request)
    try:
        respond = await _run(graph.query, req.query, mode=req.mode, pretty=False)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"query failed: {e}") from e
    return _normalize_respond(respond)


@app.post("/multihop-query", response_model=QueryResponse)
async def multihop_query(req: MultihopQueryRequest, request: Request):
    """多跳问答（multihop-query）。单跳一次检索作答，多跳按计划展开。"""
    graph = _get_graph(request)
    try:
        respond = await _run(graph.agent_query, req.query, pretty=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"multihop-query failed: {e}") from e
    return _normalize_respond(respond)


@app.post("/agentic-query", response_model=QueryResponse)
async def agentic_query(req: AgenticQueryRequest, request: Request):
    """Tool-calling 检索问答。模型边想边调 search / read_doc / graph_neighbors。"""
    graph = _get_graph(request)
    try:
        respond = await _run(graph.agentic_query, req.query, pretty=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"agentic-query failed: {e}") from e
    return _normalize_respond(respond)


@app.post("/retrieve")
async def retrieve(req: RetrieveRequest, request: Request):
    """只要召回证据，不调用生成。对应 retrieve_module.retrieve_items。"""
    graph = _get_graph(request)
    kwargs = {}
    if req.chunk_candidate_k is not None:
        kwargs["chunk_candidate_k"] = req.chunk_candidate_k
    if req.node_candidate_k is not None:
        kwargs["node_candidate_k"] = req.node_candidate_k

    def _do():
        items = graph.retrieve_module.retrieve_items(req.query, **kwargs)
        try:
            timing = dict(graph.retrieve_module.get_last_timing() or {})
        except Exception:
            timing = {}
        return items, timing

    try:
        items, timing = await _run(_do)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"retrieve failed: {e}") from e
    return {
        "query": req.query,
        "items": [_slim_item(it) for it in (items or [])],
        "retrieve_timing": _jsonable(timing),
    }


def run() -> None:
    import uvicorn

    host = os.environ.get("DHMF_API_HOST", "0.0.0.0")
    port = int(os.environ.get("DHMF_API_PORT", "8000"))
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        workers=1,
        reload=False,
    )


if __name__ == "__main__":
    run()
