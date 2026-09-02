"""
Agentic 工具：search / read_doc / graph_neighbors。

search 走 retrieve_items，覆盖参数全部来自 agentic 配置，不读 retrieve/agent。
read_doc / graph_neighbors 读同一套 SQLite（doc / chunk / node / hyperedge）。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..module.retrieve import add_retrieve_timing, empty_retrieve_timing

if TYPE_CHECKING:
    from ..DHMF import DHMF
    from ..utils.config import AgenticConfig


def _preview(text: str, n: int) -> str:
    s = (text or "").strip()
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + "…"


def _as_int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return dict(raw)
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {"_raw": raw}
        except Exception:
            return {"_raw": raw}
    return {"_raw": str(raw)}


def tool_schemas(cfg: "AgenticConfig") -> list:
    """OpenAI function tools；按配置开关裁剪。"""
    tools = []
    if cfg.enable_search:
        tools.append({
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    "在化工产品知识库中检索。返回若干条短摘要以及 doc_id / chunk_id。"
                    "需要核对规格、配方或安全数据时，再用 read_doc 阅读命中文档。"
                    "查询尽量具体，可含牌号、CAS、指标与数值；同一主体上的多项约束写在一次查询里。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "检索语句",
                        },
                    },
                    "required": ["query"],
                },
            },
        })
    if cfg.enable_read_doc:
        tools.append({
            "type": "function",
            "function": {
                "name": "read_doc",
                "description": (
                    "按 doc_id 阅读一份资料的摘要头块与正文。"
                    "只对 search 或 graph_neighbors 返回过的 doc_id 使用。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {
                            "type": "integer",
                            "description": "文档 id",
                        },
                    },
                    "required": ["doc_id"],
                },
            },
        })
    if cfg.enable_graph_neighbors:
        tools.append({
            "type": "function",
            "function": {
                "name": "graph_neighbors",
                "description": (
                    "查看超图邻居：同一超边或同一文档上的其它实体"
                    "（产品、公司、原料等）。用于从产品跳到生产商，或从公司找其它产品。"
                    "name / node_id / doc_id 至少提供一个。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "实体名、牌号或公司名",
                        },
                        "node_id": {
                            "type": "integer",
                            "description": "节点 id",
                        },
                        "doc_id": {
                            "type": "integer",
                            "description": "文档 id，列出该文档上的实体",
                        },
                    },
                },
            },
        })
    return tools


def allowed_tool_names(cfg: "AgenticConfig") -> List[str]:
    names = []
    if cfg.enable_search:
        names.append("search")
    if cfg.enable_read_doc:
        names.append("read_doc")
    if cfg.enable_graph_neighbors:
        names.append("graph_neighbors")
    return names


@dataclass
class ToolContext:
    dhmf: "DHMF"
    cfg: "AgenticConfig"
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    retrieve_latency_s: float = 0.0
    retrieve_timing: Dict[str, float] = field(default_factory=empty_retrieve_timing)
    sources: List[str] = field(default_factory=list)
    doc_ids: List[Any] = field(default_factory=list)
    _seen_src: set = field(default_factory=set)
    _seen_did: set = field(default_factory=set)

    def _remember_ref(self, source=None, doc_id=None) -> None:
        if source and source not in self._seen_src:
            self._seen_src.add(source)
            self.sources.append(str(source))
        if doc_id is not None and doc_id not in self._seen_did:
            self._seen_did.add(doc_id)
            self.doc_ids.append(doc_id)

    def _source_of(self, doc_id=None, chunk=None) -> str:
        retrieve = self.dhmf.retrieve_module
        if chunk:
            try:
                return retrieve._source_label(chunk) or ""
            except Exception:
                pass
        if doc_id is not None:
            try:
                retrieve._ensure_precompute()
                return retrieve._source_label({"doc_id": doc_id}) or ""
            except Exception:
                pass
            try:
                rows = self.dhmf.db["doc"].search("id", doc_id) or []
                if rows and rows[0].get("name"):
                    return str(rows[0]["name"])
            except Exception:
                pass
        return ""

    def execute(self, name: str, arguments) -> str:
        args = _parse_args(arguments)
        fn = {
            "search": self.search,
            "read_doc": self.read_doc,
            "graph_neighbors": self.graph_neighbors,
        }.get(name)
        if fn is None:
            return json.dumps(
                {"error": f"未知工具 {name!r}", "allowed": allowed_tool_names(self.cfg)},
                ensure_ascii=False,
            )
        try:
            payload = fn(args)
        except Exception as e:
            self.logger.exception(f"[agentic.tool] {name} failed: {e}")
            payload = {"error": f"{name} 失败: {e}"}
        if not isinstance(payload, str):
            payload = json.dumps(payload, ensure_ascii=False, indent=2)
        return payload

    def search(self, args: dict) -> dict:
        query = str(args.get("query") or args.get("q") or args.get("_raw") or "").strip()
        if not query:
            return {"error": "search 需要 query"}

        cfg = self.cfg
        retrieve = self.dhmf.retrieve_module
        t0 = time.perf_counter()
        items = retrieve.retrieve_items(
            query,
            chunk_candidate_k=int(cfg.chunk_candidate_k),
            node_candidate_k=int(cfg.node_candidate_k),
            enable_query_rewrite=bool(cfg.enable_query_rewrite),
            enable_keyword_exact=bool(cfg.enable_keyword_exact),
            enable_keyword_minority=bool(cfg.enable_keyword_minority),
            enable_keyword_majority=bool(cfg.enable_keyword_majority),
            keyword_candidate_k=int(cfg.keyword_candidate_k),
            keyword_top_k=int(cfg.keyword_top_k),
            enable_parallel_paths=bool(cfg.enable_parallel_paths),
            rerank_top_k=int(cfg.rerank_top_k),
            enable_rerank=bool(cfg.enable_rerank),
            enable_full_body_context=cfg.enable_full_body_context,
            enable_slice_family_expand=bool(cfg.enable_slice_family_expand),
        )
        dt = time.perf_counter() - t0
        self.retrieve_latency_s += dt
        try:
            timing = dict(retrieve.get_last_timing() or {})
        except Exception:
            timing = empty_retrieve_timing()
        self.retrieve_timing = add_retrieve_timing(self.retrieve_timing, timing)

        hits = self._compact_hits(items)
        for h in hits:
            self._remember_ref(h.get("source"), h.get("doc_id"))

        last_kw = getattr(retrieve, "last_keyword", None) or {}
        last_rw = getattr(retrieve, "last_rewrite", None) or {}
        if self.cfg.log_trace:
            try:
                raw_ctx = retrieve._format_retrieved_chunks(items)
            except Exception:
                raw_ctx = ""
            self.logger.info("[agentic.trace] ----- search raw retrieval -----")
            self.logger.info(
                f"[agentic.trace] rewritten={last_rw.get('rewritten')!r} "
                f"timing={timing} keyword={last_kw.get('minority')!r}"
            )
            for line in str(raw_ctx).splitlines() or [""]:
                self.logger.info(f"[agentic.trace] {line}")
            self.logger.info("[agentic.trace] ----- end search raw -----")
        return {
            "query": query,
            "rewritten": last_rw.get("rewritten") or None,
            "n_raw": len(items or []),
            "n_hits": len(hits),
            "hits": hits,
            "keyword": {
                "minority": list(last_kw.get("minority") or []),
                "majority": list(last_kw.get("majority") or []),
            },
            "timing_s": round(dt, 3),
        }

    def _compact_hits(self, items: list) -> list:
        preview_n = int(self.cfg.search_preview_chars)
        max_hits = int(self.cfg.search_max_hits)
        by_doc: Dict[Any, dict] = {}
        order = []
        for it in items or []:
            chunk = it.get("chunk") or it.get("result") or {}
            if not isinstance(chunk, dict):
                chunk = {}
            doc_id = it.get("doc_id")
            if doc_id is None:
                doc_id = chunk.get("doc_id")
            key = doc_id if doc_id is not None else f"chunk:{chunk.get('id')}"
            score = float(it.get("score") or 0.0)
            content = (chunk.get("content") or "").strip()
            cur = by_doc.get(key)
            if cur is None:
                source = it.get("source") or self._source_of(doc_id, chunk)
                by_doc[key] = {
                    "doc_id": doc_id,
                    "chunk_id": chunk.get("id") or it.get("chunk_id"),
                    "source": source,
                    "score": score,
                    "match_type": it.get("match_type") or "",
                    "node_id": it.get("node_id"),
                    "node_name": it.get("node_name"),
                    "preview": _preview(content, preview_n),
                }
                order.append(key)
                continue
            if score > float(cur.get("score") or 0.0):
                cur["score"] = score
                if content:
                    cur["preview"] = _preview(content, preview_n)
                if chunk.get("id") is not None:
                    cur["chunk_id"] = chunk.get("id")
            if not cur.get("node_name") and it.get("node_name"):
                cur["node_name"] = it.get("node_name")
                cur["node_id"] = it.get("node_id")
            mt = it.get("match_type") or ""
            if mt and mt not in str(cur.get("match_type") or ""):
                cur["match_type"] = "+".join(
                    p for p in (str(cur.get("match_type") or ""), mt) if p
                )

        ranked = sorted(
            (by_doc[k] for k in order),
            key=lambda h: -float(h.get("score") or 0.0),
        )
        if max_hits > 0:
            ranked = ranked[:max_hits]
        for h in ranked:
            try:
                h["score"] = round(float(h.get("score") or 0.0), 4)
            except (TypeError, ValueError):
                pass
        return ranked

    def read_doc(self, args: dict) -> dict:
        doc_id = _as_int(args.get("doc_id") if "doc_id" in args else args.get("_raw"))
        if doc_id is None:
            return {"error": "read_doc 需要整数 doc_id"}

        retrieve = self.dhmf.retrieve_module
        try:
            retrieve._ensure_precompute()
        except Exception:
            pass

        chunks = list(retrieve.chunks_by_doc.get(doc_id) or [])
        if not chunks:
            try:
                chunks = list(self.dhmf.db["chunk"].search("doc_id", doc_id) or [])
            except Exception as e:
                return {"error": f"读文档失败: {e}", "doc_id": doc_id}
        if not chunks:
            return {"error": f"没有 doc_id={doc_id} 的块", "doc_id": doc_id}

        try:
            chunks.sort(key=retrieve._chunk_order_key)
        except Exception:
            chunks.sort(key=lambda c: c.get("id") or 0)

        source = self._source_of(doc_id, chunks[0] if chunks else None)
        self._remember_ref(source, doc_id)

        he = None
        try:
            he = retrieve._hyperedge_for_doc(doc_id)
        except Exception:
            try:
                rows = self.dhmf.db["hyperedge"].search("doc_id", doc_id) or []
                he = rows[0] if rows else None
            except Exception:
                he = None

        max_chars = int(self.cfg.read_doc_max_chars)
        parts = [f"doc_id={doc_id}", f"source={source or '未知'}"]
        if he:
            hid = he.get("id")
            hname = (he.get("name") or "").strip()
            hcontent = (he.get("content") or "").strip()
            if hid is not None:
                parts.append(f"hyperedge_id={hid}")
            if hname:
                parts.append(f"hyperedge_name={hname}")
            if hcontent:
                parts.append("### 超边摘要")
                parts.append(hcontent)

        for i, c in enumerate(chunks):
            name = (c.get("name") or "").strip() or f"chunk_{c.get('id')}"
            body = (c.get("content") or "").strip()
            parts.append(f"### {name} (chunk_id={c.get('id')})")
            parts.append(body)
            if max_chars > 0 and sum(len(p) for p in parts) >= max_chars:
                extra = len(chunks) - i - 1
                if extra > 0:
                    parts.append(f"…(其余 {extra} 块已截断)")
                break

        text = "\n".join(parts)
        truncated = False
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars] + "\n…(truncated)"
            truncated = True
        return {
            "doc_id": doc_id,
            "source": source,
            "n_chunks": len(chunks),
            "truncated": truncated,
            "content": text,
        }

    def graph_neighbors(self, args: dict) -> dict:
        name = str(args.get("name") or "").strip()
        node_id = _as_int(args.get("node_id"))
        doc_id = _as_int(args.get("doc_id"))
        if not name and node_id is None and doc_id is None:
            return {"error": "graph_neighbors 需要 name / node_id / doc_id 之一"}

        node_db = self.dhmf.db["node"]
        limit = max(1, int(self.cfg.neighbors_limit))
        seeds = []
        if node_id is not None:
            seeds = list(node_db.search("id", node_id) or [])
        elif doc_id is not None:
            seeds = list(node_db.search("doc_id", doc_id) or [])[:limit]
        elif name:
            seeds = list(node_db.search("name", name) or [])
            if not seeds:
                try:
                    seeds = list(node_db.db.execute(
                        "SELECT * FROM node WHERE instr(name, ?) > 0 LIMIT ?",
                        (name, limit),
                    ) or [])
                except Exception:
                    seeds = []

        if not seeds:
            return {
                "query": {"name": name or None, "node_id": node_id, "doc_id": doc_id},
                "seeds": [],
                "neighbors": [],
                "note": "未找到对应实体",
            }

        preview_n = int(self.cfg.search_preview_chars)
        seed_ids = set()
        seed_out = []
        he_ids = set()
        doc_ids = set()
        for n in seeds[:limit]:
            nid = n.get("id")
            seed_ids.add(nid)
            did = n.get("doc_id")
            hid = n.get("hyperedge_id")
            if hid is not None:
                he_ids.add(hid)
            if did is not None:
                doc_ids.add(did)
            src = self._source_of(did)
            self._remember_ref(src, did)
            seed_out.append({
                "node_id": nid,
                "name": n.get("name"),
                "doc_id": did,
                "hyperedge_id": hid,
                "source": src,
                "preview": _preview(n.get("content") or "", preview_n),
            })

        neighbors = []
        seen = set(seed_ids)

        def _absorb(rows):
            for n in rows or []:
                nid = n.get("id")
                if nid in seen:
                    continue
                seen.add(nid)
                did = n.get("doc_id")
                src = self._source_of(did)
                self._remember_ref(src, did)
                neighbors.append({
                    "node_id": nid,
                    "name": n.get("name"),
                    "doc_id": did,
                    "hyperedge_id": n.get("hyperedge_id"),
                    "source": src,
                    "preview": _preview(n.get("content") or "", preview_n),
                })
                if len(neighbors) >= limit:
                    return True
            return False

        try:
            for hid in list(he_ids)[:8]:
                if _absorb(node_db.search("hyperedge_id", hid) or []):
                    break
            if len(neighbors) < limit:
                for did in list(doc_ids)[:8]:
                    if _absorb(node_db.search("doc_id", did) or []):
                        break
        except Exception as e:
            return {
                "query": {"name": name or None, "node_id": node_id, "doc_id": doc_id},
                "seeds": seed_out,
                "neighbors": neighbors,
                "error": f"展开邻居失败: {e}",
            }

        return {
            "query": {"name": name or None, "node_id": node_id, "doc_id": doc_id},
            "seeds": seed_out,
            "neighbors": neighbors[:limit],
        }
