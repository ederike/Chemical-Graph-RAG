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


SEARCH_MODES = ("hybrid", "chunk", "node", "keyword")
_SEARCH_MODE_ALIASES = {
    "hybrid": "hybrid",
    "mix": "hybrid",
    "mixed": "hybrid",
    "all": "hybrid",
    "default": "hybrid",
    "union": "hybrid",
    "chunk": "chunk",
    "chunks": "chunk",
    "passage": "chunk",
    "text": "chunk",
    "block": "chunk",
    "node": "node",
    "nodes": "node",
    "entity": "node",
    "entities": "node",
    "keyword": "keyword",
    "keywords": "keyword",
    "exact": "keyword",
    "fts": "keyword",
    "kw": "keyword",
}


def normalize_search_mode(raw) -> str:
    """hybrid / chunk / node / keyword。空或未知 → hybrid。"""
    s = str(raw or "").strip().lower()
    if not s:
        return "hybrid"
    return _SEARCH_MODE_ALIASES.get(s, "")


def search_retrieve_kwargs(cfg: "AgenticConfig", mode: str) -> dict:
    """
    按 mode 打开/关闭三路。宽度仍用 agentic 配置，不读 retrieve:。
    单路模式下若对应 k 配成 0，回落到该路默认宽度，避免模型显式选路却空跑。
    """
    chunk_k = int(cfg.chunk_candidate_k)
    node_k = int(cfg.node_candidate_k)
    kw_cand = int(cfg.keyword_candidate_k)
    kw_top = int(cfg.keyword_top_k)
    kw_on = bool(cfg.enable_keyword_exact)
    kw_min = bool(cfg.enable_keyword_minority)
    kw_maj = bool(cfg.enable_keyword_majority)

    if mode == "chunk":
        return {
            "chunk_candidate_k": chunk_k if chunk_k > 0 else 30,
            "node_candidate_k": 0,
            "enable_keyword_exact": False,
            "enable_keyword_minority": False,
            "enable_keyword_majority": False,
            "keyword_candidate_k": 0,
            "keyword_top_k": 0,
        }
    if mode == "node":
        return {
            "chunk_candidate_k": 0,
            "node_candidate_k": node_k if node_k > 0 else 30,
            "enable_keyword_exact": False,
            "enable_keyword_minority": False,
            "enable_keyword_majority": False,
            "keyword_candidate_k": 0,
            "keyword_top_k": 0,
        }
    if mode == "keyword":
        if not kw_min and not kw_maj:
            kw_min = True
        return {
            "chunk_candidate_k": 0,
            "node_candidate_k": 0,
            "enable_keyword_exact": True,
            "enable_keyword_minority": kw_min,
            "enable_keyword_majority": kw_maj,
            "keyword_candidate_k": kw_cand if kw_cand > 0 else 20,
            "keyword_top_k": kw_top if kw_top > 0 else 10,
        }
    return {
        "chunk_candidate_k": chunk_k,
        "node_candidate_k": node_k,
        "enable_keyword_exact": kw_on,
        "enable_keyword_minority": kw_min,
        "enable_keyword_majority": kw_maj,
        "keyword_candidate_k": kw_cand,
        "keyword_top_k": kw_top,
    }


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
                    "在化工产品知识库中检索。返回短摘要和 doc_id / chunk_id。"
                    "query 必须是一句完整的自然语言问题或陈述，禁止只传几个关键词。"
                    "mode=keyword：后端从问句抽取牌号/CAS/货号再精确匹配，问句里要带这些标识。"
                    "mode=node：实体节点向量检索，问一个主体（产品/公司/物质）。"
                    "mode=chunk：正文块语义检索，问用途/工艺/配方等段落内容。"
                    "mode=hybrid 或省略：三路混合，拿不准或既有标识又有规格时用。"
                    "核对数值或原文时再调用 read_doc。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "一句完整问句。正确：「CAS 号 13463-67-7 对应什么产品」、"
                                "「外墙乳胶漆提高耐沾污性常用哪些助剂」。"
                                "错误：「R-902 13463-67-7」或「耐沾污 硅丙」。"
                            ),
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["hybrid", "keyword", "node", "chunk"],
                            "description": (
                                "hybrid（默认）=块+节点+抽词精确匹配；"
                                "keyword=从完整问句抽少数值再 FTS；"
                                "node=实体节点向量；"
                                "chunk=正文块向量。"
                            ),
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
    trace: Any = None
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

        raw_mode = args.get("mode")
        if raw_mode is None or str(raw_mode).strip() == "":
            mode = "hybrid"
        else:
            mode = normalize_search_mode(raw_mode)
            if not mode:
                return {
                    "error": f"未知 search mode={raw_mode!r}",
                    "allowed": list(SEARCH_MODES),
                }

        cfg = self.cfg
        retrieve = self.dhmf.retrieve_module
        path_kw = search_retrieve_kwargs(cfg, mode)
        t0 = time.perf_counter()
        items = retrieve.retrieve_items(
            query,
            enable_query_rewrite=bool(cfg.enable_query_rewrite),
            enable_parallel_paths=bool(cfg.enable_parallel_paths),
            rerank_top_k=int(cfg.rerank_top_k),
            enable_rerank=bool(cfg.enable_rerank),
            enable_full_body_context=cfg.enable_full_body_context,
            enable_slice_family_expand=bool(cfg.enable_slice_family_expand),
            **path_kw,
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
        if self.trace is not None and getattr(self.trace, "enabled", False):
            try:
                raw_ctx = retrieve._format_retrieved_chunks(items)
            except Exception:
                raw_ctx = ""
            self.trace.line(
                f"search raw  mode={mode} rewritten={last_rw.get('rewritten')!r} "
                f"timing={timing} keyword={last_kw.get('minority')!r}"
            )
            self.trace.block("search raw retrieval", raw_ctx)
        return {
            "query": query,
            "mode": mode,
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
