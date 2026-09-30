"""WeKnora HTTP 客户端：登录、解析知识库、SSE 问答。

只走本包配置里的地址；不读 example/a。默认用 agent-chat + 内置「快速问答」，
避免 knowledge-chat 把额度耗在 thinking 上、最终答不出正文。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urljoin


def _opt_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("null", "none", "~"):
        return None
    return s


def _normalize_base(url: str) -> str:
    s = str(url or "").strip().rstrip("/")
    if s.endswith("/login"):
        s = s[: -len("/login")].rstrip("/")
    if s.endswith("/api/v1"):
        s = s[: -len("/api/v1")].rstrip("/")
    return s


def _join_answer(parts: Sequence[str]) -> str:
    if not parts:
        return ""
    # 流式分片通常是增量；偶发整段重复时去重相邻相同块
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        if out and p == out[-1]:
            continue
        if out and p.startswith(out[-1]) and len(p) > len(out[-1]):
            out[-1] = p
            continue
        out.append(p)
    return "".join(out).strip()


def _ref_filename(ref: Dict[str, Any]) -> Optional[str]:
    if not isinstance(ref, dict):
        return None
    for k in (
        "knowledge_filename",
        "knowledge_title",
        "file_name",
        "filename",
        "title",
        "source",
        "name",
    ):
        v = ref.get(k)
        if v is None:
            continue
        s = str(v).strip().replace("\\", "/").split("/")[-1]
        if s:
            return s
    return None


class WeKnoraClient:
    """登录一次，每题可新建会话。ask() 返回与评测块对齐的字典。"""

    def __init__(
        self,
        *,
        base_url: str,
        email: str,
        password: str,
        knowledge_base_id: Optional[str] = None,
        knowledge_base_name: Optional[str] = None,
        chat_mode: str = "agent",
        agent_id: str = "builtin-quick-answer",
        agent_enabled: bool = True,
        timeout: float = 300.0,
        max_retries: int = 2,
        new_session_per_question: bool = True,
        delete_session: bool = True,
        session_title_prefix: str = "benchmark_agentic_5",
    ):
        self.base_url = _normalize_base(base_url)
        if not self.base_url:
            raise ValueError("weknora.base_url 为空")
        self.api_root = self.base_url + "/api/v1"
        self.email = str(email or "").strip()
        self.password = str(password or "")
        if not self.email or not self.password:
            raise ValueError("weknora.email / password 不能为空")
        self.knowledge_base_id = _opt_str(knowledge_base_id)
        self.knowledge_base_name = _opt_str(knowledge_base_name)
        mode = str(chat_mode or "agent").strip().lower()
        if mode in ("agent", "agent-chat", "agent_chat", "quick-answer", "quick"):
            self.chat_mode = "agent"
        elif mode in ("knowledge", "knowledge-chat", "knowledge_chat", "qa"):
            self.chat_mode = "knowledge"
        else:
            raise ValueError(
                f"未知 weknora.chat_mode={chat_mode!r}，"
                "支持 agent / knowledge"
            )
        self.agent_id = str(agent_id or "builtin-quick-answer").strip()
        self.agent_enabled = bool(agent_enabled)
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))
        self.new_session_per_question = bool(new_session_per_question)
        self.delete_session = bool(delete_session)
        self.session_title_prefix = str(
            session_title_prefix or "benchmark_agentic_5"
        ).strip()

        self._token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._kb: Optional[Dict[str, Any]] = None
        self._session_id: Optional[str] = None
        self._indexed_titles: Optional[List[str]] = None

    @classmethod
    def from_config(cls, cfg: Any) -> "WeKnoraClient":
        return cls(
            base_url=getattr(cfg, "weknora_base_url", "") or "",
            email=getattr(cfg, "weknora_email", None) or "",
            password=getattr(cfg, "weknora_password", None) or "",
            knowledge_base_id=getattr(cfg, "weknora_knowledge_base_id", None),
            knowledge_base_name=getattr(cfg, "weknora_knowledge_base_name", None),
            chat_mode=getattr(cfg, "weknora_chat_mode", "agent") or "agent",
            agent_id=getattr(cfg, "weknora_agent_id", None) or "builtin-quick-answer",
            agent_enabled=bool(getattr(cfg, "weknora_agent_enabled", True)),
            timeout=float(getattr(cfg, "weknora_timeout", 300) or 300),
            max_retries=int(getattr(cfg, "weknora_max_retries", 2) or 2),
            new_session_per_question=bool(
                getattr(cfg, "weknora_new_session_per_question", True)
            ),
            delete_session=bool(getattr(cfg, "weknora_delete_session", True)),
            session_title_prefix=getattr(cfg, "weknora_session_title_prefix", None)
            or "benchmark_agentic_5",
        )

    # ------------------------------------------------------------------ http
    def _headers(self, *, stream: bool = False) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        if stream:
            h["Accept"] = "text/event-stream"
        return h

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return urljoin(self.api_root + "/", path.lstrip("/"))

    def _request_json(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        *,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        req = urllib.request.Request(
            self._url(path),
            data=data,
            method=method,
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
                if not raw:
                    return {"success": True, "http_status": resp.status}
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    obj.setdefault("http_status", resp.status)
                    return obj
                return {"success": True, "data": obj, "http_status": resp.status}
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                obj = json.loads(raw) if raw else {}
            except Exception:
                obj = {"raw": raw[:800]}
            if not isinstance(obj, dict):
                obj = {"data": obj}
            obj["http_status"] = e.code
            obj["success"] = False
            if e.code == 401:
                obj.setdefault("error", {})
            return obj

    def login(self, *, force: bool = False) -> str:
        if self._token and not force:
            return self._token
        obj = self._request_json(
            "POST",
            "/auth/login",
            {"email": self.email, "password": self.password},
            timeout=min(60.0, self.timeout),
        )
        token = obj.get("token")
        if not token or obj.get("success") is False:
            raise RuntimeError(
                f"WeKnora 登录失败 http={obj.get('http_status')} "
                f"msg={obj.get('message') or obj.get('error') or obj}"
            )
        self._token = str(token)
        self._refresh_token = obj.get("refresh_token")
        return self._token

    def _ensure_auth(self) -> None:
        if not self._token:
            self.login()

    def _with_reauth(self, fn):
        self._ensure_auth()
        out = fn()
        if isinstance(out, dict) and out.get("http_status") == 401:
            self.login(force=True)
            out = fn()
        return out

    # --------------------------------------------------------------- knowledge
    def list_knowledge_bases(self) -> List[Dict[str, Any]]:
        def _call():
            return self._request_json("GET", "/knowledge-bases")

        obj = self._with_reauth(_call)
        data = obj.get("data")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for k in ("items", "list", "knowledge_bases"):
                if isinstance(data.get(k), list):
                    return data[k]
        return []

    def resolve_knowledge_base(self, *, force: bool = False) -> Dict[str, Any]:
        if self._kb is not None and not force:
            return self._kb
        kbs = self.list_knowledge_bases()
        if not kbs:
            raise RuntimeError("WeKnora 当前空间没有知识库")

        wanted_id = self.knowledge_base_id
        wanted_name = self.knowledge_base_name
        hit = None
        if wanted_id:
            for kb in kbs:
                if str(kb.get("id") or "") == wanted_id:
                    hit = kb
                    break
            if hit is None:
                raise RuntimeError(f"WeKnora 找不到 knowledge_base_id={wanted_id}")
        elif wanted_name:
            name_l = wanted_name.lower()
            for kb in kbs:
                if str(kb.get("name") or "").strip().lower() == name_l:
                    hit = kb
                    break
            if hit is None:
                raise RuntimeError(f"WeKnora 找不到知识库名 {wanted_name!r}")
        else:
            scored = sorted(
                kbs,
                key=lambda kb: int(kb.get("knowledge_count") or 0),
                reverse=True,
            )
            hit = scored[0]

        self._kb = hit
        self.knowledge_base_id = str(hit.get("id") or "")
        self.knowledge_base_name = str(hit.get("name") or "") or self.knowledge_base_name
        print(
            f"[weknora] kb id={self.knowledge_base_id} "
            f"name={self.knowledge_base_name!r} "
            f"docs={hit.get('knowledge_count')}",
            file=sys.stderr,
        )
        return hit

    def list_indexed_titles(self, *, force: bool = False) -> List[str]:
        if self._indexed_titles is not None and not force:
            return list(self._indexed_titles)
        kb = self.resolve_knowledge_base()
        kid = str(kb.get("id") or "")
        titles: List[str] = []
        page = 1
        total = None
        while page <= 50:
            def _call(p=page):
                return self._request_json(
                    "GET",
                    f"/knowledge-bases/{kid}/knowledge?page={p}&page_size=100",
                )

            obj = self._with_reauth(_call)
            rows = obj.get("data") or []
            if isinstance(rows, dict):
                rows = rows.get("items") or rows.get("list") or []
            if not isinstance(rows, list):
                break
            for it in rows:
                if not isinstance(it, dict):
                    continue
                name = (
                    it.get("title")
                    or it.get("file_name")
                    or it.get("filename")
                    or it.get("name")
                )
                if name:
                    titles.append(str(name).replace("\\", "/").split("/")[-1])
            total = obj.get("total")
            if not rows:
                break
            if total is not None and len(titles) >= int(total):
                break
            page += 1
        self._indexed_titles = titles
        return list(titles)

    # ---------------------------------------------------------------- session
    def create_session(self, title: str) -> str:
        kb = self.resolve_knowledge_base()
        body = {"title": title, "knowledge_base_id": kb.get("id")}

        def _call():
            return self._request_json("POST", "/sessions", body)

        obj = self._with_reauth(_call)
        data = obj.get("data") if isinstance(obj.get("data"), dict) else obj
        sid = (data or {}).get("id") if isinstance(data, dict) else None
        if not sid:
            raise RuntimeError(f"WeKnora 创建会话失败: {obj}")
        return str(sid)

    def delete_session_id(self, session_id: str) -> None:
        if not session_id:
            return

        def _call():
            return self._request_json("DELETE", f"/sessions/{session_id}")

        try:
            self._with_reauth(_call)
        except Exception:
            pass

    def _session_for(self, tag: str) -> str:
        if self.new_session_per_question or not self._session_id:
            title = f"{self.session_title_prefix}-{tag or int(time.time())}"
            sid = self.create_session(title[:80])
            if not self.new_session_per_question:
                self._session_id = sid
            return sid
        return self._session_id

    # ------------------------------------------------------------------- SSE
    def _parse_sse_events(self, raw: bytes) -> Iterable[Dict[str, Any]]:
        text = raw.decode("utf-8", "replace")
        chunks = text.split("\n\n")
        for chunk in chunks:
            if not chunk.strip():
                continue
            data_lines: List[str] = []
            event_name = None
            for line in chunk.splitlines():
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            payload = "\n".join(data_lines).strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if isinstance(obj, dict):
                if event_name and "response_type" not in obj:
                    obj.setdefault("event", event_name)
                yield obj

    def _chat_body(self, question: str) -> Dict[str, Any]:
        kb = self.resolve_knowledge_base()
        body: Dict[str, Any] = {
            "query": question,
            "knowledge_base_ids": [kb.get("id")],
        }
        if self.chat_mode == "agent":
            body["agent_enabled"] = bool(self.agent_enabled)
            body["agent_id"] = self.agent_id
        else:
            body["thinking"] = False
        return body

    def _chat_path(self, session_id: str) -> str:
        if self.chat_mode == "agent":
            return f"/agent-chat/{session_id}"
        return f"/knowledge-chat/{session_id}"

    def _read_sse(self, session_id: str, question: str) -> Dict[str, Any]:
        self._ensure_auth()
        req = urllib.request.Request(
            self._url(self._chat_path(session_id)),
            data=json.dumps(self._chat_body(question), ensure_ascii=False).encode(),
            method="POST",
            headers=self._headers(stream=True),
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                http_status = resp.status
        except urllib.error.HTTPError as e:
            if e.code == 401:
                self.login(force=True)
                req.add_header("Authorization", f"Bearer {self._token}")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    http_status = resp.status
            else:
                body = e.read().decode("utf-8", "replace")
                return {
                    "ok": False,
                    "http_status": e.code,
                    "error": body[:500],
                    "latency_s": time.perf_counter() - t0,
                    "answer": "",
                    "retrieval_sources": [],
                    "events": [],
                }
        latency = time.perf_counter() - t0
        events = list(self._parse_sse_events(raw))
        return self._reduce_events(events, latency_s=latency, http_status=http_status)

    def _reduce_events(
        self,
        events: Sequence[Dict[str, Any]],
        *,
        latency_s: float,
        http_status: int,
    ) -> Dict[str, Any]:
        answers: List[str] = []
        thinking: List[str] = []
        sources: List[str] = []
        knowledge_ids: List[str] = []
        seen_src = set()
        seen_kid = set()
        n_tool_call = 0
        n_search = 0
        retrieve_ms = 0.0
        last_err = None
        complete = False
        for ev in events:
            rt = str(ev.get("response_type") or ev.get("event") or "").strip()
            data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
            if rt == "answer":
                piece = ev.get("content")
                if piece is None:
                    piece = data.get("content") or data.get("answer")
                if piece:
                    answers.append(str(piece))
            elif rt == "thinking":
                piece = ev.get("content") or ""
                if piece:
                    thinking.append(str(piece))
            elif rt == "tool_call":
                n_tool_call += 1
                name = str((data or {}).get("tool_name") or "")
                if name == "knowledge_search":
                    n_search += 1
            elif rt == "tool_result":
                name = str((data or {}).get("tool_name") or "")
                if name == "knowledge_search":
                    n_search = max(n_search, 1)
                    try:
                        retrieve_ms += float(data.get("duration_ms") or 0)
                    except (TypeError, ValueError):
                        pass
                err = data.get("error")
                if err:
                    last_err = str(err)
                if data.get("success") is False and not last_err:
                    last_err = str(data.get("output") or "tool failed")
            elif rt == "references":
                refs = ev.get("knowledge_references")
                if not isinstance(refs, list):
                    refs = data.get("knowledge_references") or data.get("references") or []
                for ref in refs or []:
                    fn = _ref_filename(ref if isinstance(ref, dict) else {})
                    if fn and fn not in seen_src:
                        seen_src.add(fn)
                        sources.append(fn)
                    kid = None
                    if isinstance(ref, dict):
                        kid = ref.get("knowledge_id")
                    if kid is not None and kid not in seen_kid:
                        seen_kid.add(kid)
                        knowledge_ids.append(kid)
            elif rt == "complete":
                complete = True
            elif rt in ("error", "fail"):
                last_err = str(ev.get("content") or data.get("message") or last_err or rt)

        answer = _join_answer(answers)
        truncated = (
            "per-response output limit" in answer
            or "hit the model's" in answer.lower()
        )
        ok = bool(answer) and not truncated and last_err is None
        error = last_err
        if truncated:
            error = answer
            ok = False
        if not answer and not error:
            error = "empty weknora answer"
        return {
            "ok": ok,
            "http_status": http_status,
            "complete": complete,
            "answer": answer,
            "raw_answer": answer,
            "thinking": "".join(thinking),
            "thinking_chars": sum(len(x) for x in thinking),
            "retrieval_sources": sources,
            "retrieval_doc_ids": knowledge_ids,
            "n_tool_calls": n_tool_call,
            "n_search": n_search,
            "retrieve_latency_s": (retrieve_ms / 1000.0) if retrieve_ms else None,
            "latency_s": latency_s,
            "error": error,
            "n_events": len(events),
        }

    def ask(self, question: str, *, session_tag: str = "") -> Dict[str, Any]:
        self.resolve_knowledge_base()
        last: Dict[str, Any] = {}
        for attempt in range(1, self.max_retries + 1):
            sid = None
            try:
                sid = self._session_for(session_tag or f"q{attempt}")
                last = self._read_sse(sid, question)
            except Exception as e:
                last = {
                    "ok": False,
                    "answer": "",
                    "raw_answer": "",
                    "retrieval_sources": [],
                    "retrieval_doc_ids": [],
                    "error": str(e),
                    "latency_s": None,
                    "n_search": 0,
                    "n_tool_calls": 0,
                }
            finally:
                if sid and self.delete_session and (
                    self.new_session_per_question or attempt >= self.max_retries
                ):
                    self.delete_session_id(sid)
                    if sid == self._session_id:
                        self._session_id = None
            if last.get("ok"):
                return last
            # 空答 / 截断可再试；明确 4xx 不再试
            status = last.get("http_status")
            if isinstance(status, int) and 400 <= status < 500 and status != 401:
                break
        return last
