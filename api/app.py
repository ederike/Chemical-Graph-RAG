"""Chemical-Graph-RAG HTTP 服务：薄封装 DHMF.query / 多跳问答 / agentic_query / retrieve_items。

启动（项目根目录，uvicorn 进程数必须为 1，避免多份 FAISS）：

    export DHMF_CONFIG=example/a/config_open.yaml
    python -m api

线程池、端口、CORS 写在配置文件的 app_config: 段，不设环境变量。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WEB_DIR = ROOT / "chemical-rag-web" / "web"
WEB_ONLY = os.environ.get("CGR_WEB_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}

from src.utils.config import AppConfig, Config
from api.store import (
    MAX_PASSWORD_LEN,
    MAX_USERNAME_LEN,
    add_turn,
    authenticate,
    create_conversation,
    delete_conversation,
    get_conversation,
    init_db,
    issue_token,
    list_conversations,
    rename_conversation,
    revoke_token,
    user_from_token,
)

CONFIG_PATH = os.environ.get("DHMF_CONFIG", "example/a/config_open.yaml")
QUERY_MAX_CHARS = 8000


def _resolve_config_path() -> Path:
    raw = os.environ.get("DHMF_CONFIG", CONFIG_PATH)
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _load_config() -> Optional[Config]:
    path = _resolve_config_path()
    if not path.is_file():
        return None
    return Config.from_yaml(str(path))


_dhmf_config = _load_config()
_app_cfg = (
    _dhmf_config.app_config if _dhmf_config is not None else AppConfig()
)
MAX_WORKERS = _app_cfg.workers
CONTENT_CHARS = _app_cfg.content_chars

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="dhmf-api")


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
    os.chdir(ROOT)
    init_db()
    app.state.web_only = WEB_ONLY
    app.state.llm_context = probe_llm_context(_dhmf_config)
    if WEB_ONLY:
        app.state.graph = None
        app.state.config_path = str(_resolve_config_path())
        try:
            yield
        finally:
            _executor.shutdown(wait=False, cancel_futures=True)
        return

    from src.DHMF import DHMF

    config_path = _resolve_config_path()
    if not config_path.is_file():
        raise FileNotFoundError(f"DHMF config not found: {config_path}")

    config = _dhmf_config if _dhmf_config is not None else Config.from_yaml(
        str(config_path)
    )
    graph = DHMF(config)
    graph.pin_retrieve_indexes()
    app.state.graph = graph
    app.state.config_path = str(config_path)
    app.state.llm_context = probe_llm_context(config)
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

_cors = _app_cfg.cors or "*"
_origins = [o.strip() for o in _cors.split(",") if o.strip()] or ["*"]
_allow_all = _origins == ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_all else _origins,
    allow_credentials=not _allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HistoryTurn(BaseModel):
    query: str = ""
    answer: str = ""
    last_prompt_tokens: Optional[int] = None
    usage_prompt_tokens: Optional[int] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=QUERY_MAX_CHARS, description="用户问题")
    mode: Literal["dual_path"] = "dual_path"
    history: Optional[List[HistoryTurn]] = None


class MultihopQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=QUERY_MAX_CHARS, description="用户问题")
    history: Optional[List[HistoryTurn]] = None


class AgenticQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=QUERY_MAX_CHARS, description="用户问题")
    history: Optional[List[HistoryTurn]] = None


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=QUERY_MAX_CHARS)
    chunk_candidate_k: Optional[int] = None
    node_candidate_k: Optional[int] = None
    history: Optional[List[HistoryTurn]] = None


class StreamRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=QUERY_MAX_CHARS)
    mode: Literal["dual", "agent", "agentic", "retrieve"] = "dual"
    chunk_candidate_k: Optional[int] = None
    node_candidate_k: Optional[int] = None
    history: Optional[List[HistoryTurn]] = None


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
    if getattr(request.app.state, "web_only", False):
        raise HTTPException(
            status_code=503,
            detail="当前是网页试点模式（CGR_WEB_ONLY=1），未加载知识库。要问答请去掉该环境变量后重启 python -m api。",
        )
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="DHMF 尚未加载完成")
    return graph


def _bearer_token(request: Request) -> str:
    auth = request.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return ""


def _require_user(request: Request) -> dict:
    user = user_from_token(_bearer_token(request))
    if user is None:
        raise HTTPException(status_code=401, detail="未登录或登录已失效")
    return user


def _history_dicts(history) -> Optional[List[dict]]:
    if not history:
        return None
    out = []
    for t in history:
        q = (getattr(t, "query", None) or "").strip()
        a = getattr(t, "answer", None) or ""
        if not q:
            continue
        row = {"query": q, "answer": a}
        for key in ("last_prompt_tokens", "usage_prompt_tokens"):
            try:
                n = int(getattr(t, key, None) or 0)
            except (TypeError, ValueError):
                n = 0
            if n > 0:
                row[key] = n
        out.append(row)
    return out or None


def probe_llm_context(config) -> dict:
    """Read answering LLM max_model_len from the OpenAI-compatible /models API."""
    agentic = getattr(config, "agentic", None) if config is not None else None
    settings = getattr(config, "settings", None) if config is not None else None
    base = ""
    model = ""
    reserve = 4096
    cfg_cap = 0
    if agentic is not None:
        base = str(getattr(agentic, "base_url", "") or "")
        ma = getattr(agentic, "model_args", None) or {}
        if isinstance(ma, dict):
            model = str(ma.get("model") or "")
        reserve = int(getattr(agentic, "prompt_token_reserve", 4096) or 4096)
        cfg_cap = int(getattr(agentic, "max_prompt_tokens", 0) or 0)
    if not base and settings is not None:
        base = str(getattr(settings, "base_url", "") or "")
    out = {
        "model": model,
        "base_url": base,
        "max_context_tokens": cfg_cap or 131072,
        "reserve_tokens": max(0, reserve),
        "source": "config max_prompt_tokens" if cfg_cap else "default",
    }
    if not base:
        return out
    url = base.rstrip("/") + "/models"
    try:
        import urllib.request

        req = urllib.request.Request(
            url,
            headers={"Authorization": "Bearer EMPTY", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return out
        picked = None
        for m in models:
            if isinstance(m, dict) and str(m.get("id") or "") == model:
                picked = m
                break
        if picked is None:
            for m in models:
                if isinstance(m, dict) and m.get("max_model_len"):
                    picked = m
                    break
        if not isinstance(picked, dict):
            return out
        raw = picked.get("max_model_len") or picked.get("context_length")
        if raw:
            out["max_context_tokens"] = int(raw)
            out["model"] = str(picked.get("id") or model)
            out["source"] = "vllm max_model_len"
    except Exception as e:
        out["probe_error"] = str(e)[:240]
    return out


def _llm_context(request: Optional[Request] = None) -> dict:
    if request is not None:
        cached = getattr(request.app.state, "llm_context", None)
        if isinstance(cached, dict) and cached.get("max_context_tokens"):
            return cached
    cfg = _dhmf_config
    return probe_llm_context(cfg)


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
    index = WEB_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {
        "service": "Chemical Graph RAG",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/status")
def status():
    return {
        "service": "Chemical Graph RAG",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query",
        "multihop-query": "POST /multihop-query",
        "agentic-query": "POST /agentic-query",
        "retrieve": "POST /retrieve",
        "web_only": WEB_ONLY,
    }


@app.get("/context")
def llm_context(request: Request):
    _require_user(request)
    info = dict(_llm_context(request))
    info.pop("base_url", None)
    return info


@app.get("/health")
def health(request: Request):
    if getattr(request.app.state, "web_only", False):
        return {
            "ok": True,
            "web_only": True,
            "config_path": getattr(
                request.app.state, "config_path", str(_resolve_config_path())
            ),
        }
    graph = _get_graph(request)
    return {
        "ok": True,
        "web_only": False,
        "config_path": getattr(request.app.state, "config_path", str(_resolve_config_path())),
        "working_path": graph.config.settings.working_path,
    }


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=MAX_USERNAME_LEN)
    password: str = Field(..., min_length=1, max_length=MAX_PASSWORD_LEN)


class ConversationCreate(BaseModel):
    title: str = ""


class ConversationRename(BaseModel):
    title: str = Field(..., min_length=1, max_length=80)


class TurnCreate(BaseModel):
    query: str = Field(..., min_length=1, max_length=QUERY_MAX_CHARS)
    mode: Literal["dual", "agent", "agentic", "retrieve"] = "agentic"
    answer: str = ""
    status: int = 0
    latency_s: Optional[float] = None
    sources: Optional[List[Any]] = None
    result: Optional[Dict[str, Any]] = None
    stream_steps: Optional[List[Any]] = None


@app.post("/auth/login")
def auth_login(req: LoginRequest):
    user = authenticate(req.username, req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密钥不正确")
    token = issue_token(user["id"])
    return {"token": token, "username": user["username"]}


@app.post("/auth/logout")
def auth_logout(request: Request):
    token = _bearer_token(request)
    if token:
        revoke_token(token)
    return {"ok": True}


@app.get("/auth/me")
def auth_me(request: Request):
    user = _require_user(request)
    return {"username": user["username"], "is_admin": user["is_admin"]}


@app.get("/conversations")
def conversations_list(request: Request, q: str = ""):
    user = _require_user(request)
    return {"items": list_conversations(user["id"], q=q)}


@app.post("/conversations")
def conversations_create(req: ConversationCreate, request: Request):
    user = _require_user(request)
    return create_conversation(user["id"], req.title)


@app.get("/conversations/{conv_id}")
def conversations_get(conv_id: str, request: Request):
    user = _require_user(request)
    row = get_conversation(user["id"], conv_id)
    if row is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return row


@app.patch("/conversations/{conv_id}")
def conversations_rename(conv_id: str, req: ConversationRename, request: Request):
    user = _require_user(request)
    try:
        row = rename_conversation(user["id"], conv_id, req.title)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if row is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return row


@app.delete("/conversations/{conv_id}")
def conversations_delete(conv_id: str, request: Request):
    user = _require_user(request)
    if not delete_conversation(user["id"], conv_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True}


@app.post("/conversations/{conv_id}/turns")
def conversations_add_turn(conv_id: str, req: TurnCreate, request: Request):
    user = _require_user(request)
    row = add_turn(
        user["id"],
        conv_id,
        query=req.query,
        mode=req.mode,
        answer=req.answer,
        status=req.status,
        latency_s=req.latency_s,
        sources=req.sources,
        result=req.result,
        stream_steps=req.stream_steps,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return row


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request):
    """单跳：三路召回 + 可选 rerank + 一次生成。对应 graph.query。"""
    _require_user(request)
    _reject_if_over_context(request, _history_dicts(req.history), req.query)
    graph = _get_graph(request)
    try:
        respond = await _run(
            graph.query, req.query, mode=req.mode, pretty=False,
            history=_history_dicts(req.history),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"query failed: {e}") from e
    return _normalize_respond(respond)


@app.post("/multihop-query", response_model=QueryResponse)
async def multihop_query(req: MultihopQueryRequest, request: Request):
    """多跳问答（multihop-query）。单跳一次检索作答，多跳按计划展开。"""
    _require_user(request)
    _reject_if_over_context(request, _history_dicts(req.history), req.query)
    graph = _get_graph(request)
    try:
        respond = await _run(
            graph.agent_query, req.query, pretty=False,
            history=_history_dicts(req.history),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"multihop-query failed: {e}") from e
    return _normalize_respond(respond)


@app.post("/agentic-query", response_model=QueryResponse)
async def agentic_query(req: AgenticQueryRequest, request: Request):
    """Tool-calling 检索问答。模型边想边调 search / read_doc / graph_neighbors。"""
    _require_user(request)
    _reject_if_over_context(request, _history_dicts(req.history), req.query)
    graph = _get_graph(request)
    try:
        respond = await _run(
            graph.agentic_query, req.query, pretty=False,
            history=_history_dicts(req.history),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"agentic-query failed: {e}") from e
    return _normalize_respond(respond)


@app.post("/retrieve")
async def retrieve(req: RetrieveRequest, request: Request):
    """只要召回证据，不调用生成。对应 retrieve_module.retrieve_items。"""
    _require_user(request)
    graph = _get_graph(request)
    kwargs = {}
    if req.chunk_candidate_k is not None:
        kwargs["chunk_candidate_k"] = req.chunk_candidate_k
    if req.node_candidate_k is not None:
        kwargs["node_candidate_k"] = req.node_candidate_k

    def _do():
        from src.utils.chat_history import expand_retrieve_query
        q = expand_retrieve_query(req.query, _history_dicts(req.history))
        items = graph.retrieve_module.retrieve_items(q, **kwargs)
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


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(_jsonable(payload), ensure_ascii=False) + "\n\n"


def _reject_if_over_context(request: Request, history, query: str) -> None:
    info = _llm_context(request)
    max_ctx = int(info.get("max_context_tokens") or 0)
    reserve = int(info.get("reserve_tokens") or 4096)
    if max_ctx <= 0:
        return
    from src.utils.chat_history import estimate_next_prompt_tokens

    used = estimate_next_prompt_tokens(history, query)
    limit = max(1, max_ctx - max(0, reserve))
    if used >= limit:
        raise HTTPException(
            status_code=400,
            detail=(
                f"当前对话上下文约 {used} tokens，已达到回答模型上限 "
                f"{max_ctx}（需预留 {reserve}）。请点击「新对话」再继续。"
            ),
        )


@app.post("/stream")
async def stream(req: StreamRequest, request: Request):
    """SSE 进度流。前端用于过程框实时打印。"""
    _require_user(request)
    if req.mode != "retrieve":
        _reject_if_over_context(request, _history_dicts(req.history), req.query)
    graph = _get_graph(request)
    q: Queue = Queue()

    def push(ev: dict) -> None:
        q.put(ev)

    def work() -> None:
        from src.utils.progress import bind
        from src.utils.chat_history import expand_retrieve_query

        hist = _history_dicts(req.history)
        try:
            with bind(push):
                if req.mode == "dual":
                    respond = graph.query(
                        req.query, mode="dual_path", pretty=False, history=hist,
                    )
                    q.put({"type": "done", "data": _normalize_respond(respond)})
                elif req.mode == "agent":
                    respond = graph.agent_query(req.query, pretty=False, history=hist)
                    q.put({"type": "done", "data": _normalize_respond(respond)})
                elif req.mode == "agentic":
                    respond = graph.agentic_query(req.query, pretty=False, history=hist)
                    q.put({"type": "done", "data": _normalize_respond(respond)})
                else:
                    kwargs = {}
                    if req.chunk_candidate_k is not None:
                        kwargs["chunk_candidate_k"] = req.chunk_candidate_k
                    if req.node_candidate_k is not None:
                        kwargs["node_candidate_k"] = req.node_candidate_k
                    rq = expand_retrieve_query(req.query, hist)
                    items = graph.retrieve_module.retrieve_items(rq, **kwargs)
                    try:
                        timing = dict(graph.retrieve_module.get_last_timing() or {})
                    except Exception:
                        timing = {}
                    q.put({
                        "type": "done",
                        "data": {
                            "query": req.query,
                            "items": [_slim_item(it) for it in (items or [])],
                            "retrieve_timing": _jsonable(timing),
                            "status": 1,
                            "answer": "",
                        },
                    })
        except Exception as e:
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(None)

    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(_executor, work)

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break

                def _take():
                    try:
                        return q.get(timeout=12)
                    except Empty:
                        return {"_ping": True}

                item = await loop.run_in_executor(None, _take)
                if item is None:
                    break
                if item.get("_ping"):
                    yield ": ping\n\n"
                    continue
                yield _sse(item)
        finally:
            await fut

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _alias_api_prefix() -> None:
    """Local (no nginx): frontend calls /api/*, same handlers as /health, /query, ..."""
    seen = {(r.path, tuple(r.methods or [])) for r in app.routes if hasattr(r, "path")}
    extras = []
    for route in list(app.routes):
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        endpoint = getattr(route, "endpoint", None)
        if not path or not methods or endpoint is None:
            continue
        if path.startswith("/api"):
            continue
        if path in {"/", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}:
            continue
        alias = "/api" + path
        key = (alias, tuple(methods))
        if key in seen:
            continue
        extras.append((alias, endpoint, methods, route))
        seen.add(key)
    for alias, endpoint, methods, route in extras:
        kw = {}
        name = getattr(route, "name", None)
        if name:
            kw["name"] = "api_" + name
        if getattr(route, "response_model", None) is not None:
            kw["response_model"] = route.response_model
        app.add_api_route(alias, endpoint, methods=list(methods), **kw)


_alias_api_prefix()
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def run() -> None:
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host=_app_cfg.host or "0.0.0.0",
        port=_app_cfg.port,
        workers=1,
        reload=False,
    )


if __name__ == "__main__":
    run()
