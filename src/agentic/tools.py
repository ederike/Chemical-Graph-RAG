"""
Agentic 工具：search / read_doc / read_chunk / graph_neighbors。

search 走 retrieve_items，覆盖参数全部来自 agentic 配置，不读 retrieve/agent。
read_doc / read_chunk / graph_neighbors 读同一套 SQLite（doc / chunk / node / hyperedge）。
块阅读顺序按 chunk_index / body_N，不是入库 chunk_id。
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
                    "在化工产品知识库中检索。每条命中含 chunk_id、doc_id、"
                    "同切片阅读序 prev_chunk_id/next_chunk_id（不是 id 数值序），"
                    "以及切开 PDF 的 siblings（其它片的 doc_id / first_chunk_id）。"
                    "query 必须是一句完整的自然语言问题或陈述，禁止只传几个关键词。"
                    "mode=keyword：后端从问句抽取牌号/CAS/货号再精确匹配，问句里要带这些标识。"
                    "mode=node：实体节点向量检索，问一个主体（产品/公司/物质）。"
                    "mode=chunk：正文块语义检索，问用途/工艺/配方等段落内容。"
                    "mode=hybrid 或省略：三路混合，拿不准或既有标识又有规格时用。"
                    "要读命中块或邻近块用 read_chunk(chunk_id)；整片用 read_doc。"
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
                    "可对 search 命中的 doc_id，以及命中里 siblings 列出的同族切开文档使用。"
                    "返回该片按阅读序排列的 chunk_ids。sliced=true 时按 siblings 换片。"
                    "邻近细节优先 read_chunk(prev/next_chunk_id)，不要整片重读。"
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
    if cfg.enable_read_chunk:
        tools.append({
            "type": "function",
            "function": {
                "name": "read_chunk",
                "description": (
                    "按 chunk_id 读一块正文。块 id 不是阅读顺序："
                    "用返回的 prev_chunk_id / next_chunk_id 沿同一切片（同一 doc_id）往前后读；"
                    "正文不在这片时，用 siblings[].first_chunk_id 跳到其它切片再沿 next 走。"
                    "可对 search / read_doc / graph_neighbors 给出的任意 chunk_id 调用。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {
                            "type": "integer",
                            "description": "块 id",
                        },
                    },
                    "required": ["chunk_id"],
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
    if cfg.enable_read_chunk:
        names.append("read_chunk")
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

    def _ordered_chunks(self, doc_id) -> list:
        """该 doc（一片）内按阅读序的块。入库 chunk_id 可能乱序。"""
        if doc_id is None:
            return []
        retrieve = self.dhmf.retrieve_module
        try:
            retrieve._ensure_precompute()
        except Exception:
            pass
        chunks = list((retrieve.chunks_by_doc or {}).get(doc_id) or [])
        if not chunks:
            try:
                chunks = list(self.dhmf.db["chunk"].search("doc_id", doc_id) or [])
            except Exception:
                chunks = []
        if not chunks:
            return []
        try:
            chunks.sort(key=retrieve._chunk_order_key)
        except Exception:
            chunks.sort(key=lambda c: (c.get("id") or 0))
        return chunks

    def _chunk_row(self, chunk_id) -> Optional[dict]:
        if chunk_id is None:
            return None
        retrieve = self.dhmf.retrieve_module
        try:
            retrieve._ensure_precompute()
        except Exception:
            pass
        row = (getattr(retrieve, "chunk_dict", None) or {}).get(chunk_id)
        if row:
            return row
        try:
            rows = self.dhmf.db["chunk"].search("id", chunk_id) or []
            return rows[0] if rows else None
        except Exception:
            return None

    def _slice_info(self, doc_id, *, with_chunk_ids: bool = False) -> Optional[dict]:
        """切开 PDF 的同族切片。未切开返回 None。"""
        if doc_id is None:
            return None
        retrieve = self.dhmf.retrieve_module
        try:
            retrieve._ensure_precompute()
        except Exception:
            return None
        doc = (retrieve.doc_dict or {}).get(doc_id)
        if not doc:
            try:
                rows = self.dhmf.db["doc"].search("id", doc_id) or []
                doc = rows[0] if rows else None
            except Exception:
                doc = None
        if not doc:
            return None
        try:
            meta = retrieve._doc_slice_meta(doc)
        except Exception:
            return None
        n_slices = int(meta.get("n_slices") or 1)
        slice_index = int(meta.get("slice_index") or 0)
        family_key = meta.get("family_key") or ""
        sibling_ids = []
        if family_key:
            try:
                sibling_ids = list(
                    retrieve._ordered_family_doc_ids(
                        family_key, fallback=[doc_id],
                    ) or []
                )
            except Exception:
                sibling_ids = [doc_id]
        if n_slices <= 1 and len(sibling_ids) <= 1:
            return None
        siblings = []
        for did in sibling_ids:
            src = self._source_of(did)
            sdoc = (retrieve.doc_dict or {}).get(did) or {}
            try:
                sm = retrieve._doc_slice_meta(sdoc) if sdoc else {}
            except Exception:
                sm = {}
            och = self._ordered_chunks(did)
            cids = [c.get("id") for c in och if c.get("id") is not None]
            sib = {
                "doc_id": did,
                "slice_index": int(sm.get("slice_index") or 0),
                "source": src,
                "n_chunks": len(cids),
                "first_chunk_id": cids[0] if cids else None,
                "last_chunk_id": cids[-1] if cids else None,
            }
            if with_chunk_ids:
                sib["chunk_ids"] = cids
            siblings.append(sib)
        return {
            "sliced": True,
            "slice_index": slice_index,
            "n_slices": max(n_slices, len(siblings)),
            "family": meta.get("source_name") or family_key,
            "siblings": siblings,
        }

    def _chunk_nav(self, chunk: dict, *, with_sibling_chunk_ids: bool = False) -> dict:
        """阅读序导航：同片 prev/next，切开则带 siblings 的 first/last_chunk_id。"""
        cid = chunk.get("id")
        doc_id = chunk.get("doc_id")
        ordered = self._ordered_chunks(doc_id)
        idx = None
        for i, c in enumerate(ordered):
            if c.get("id") == cid:
                idx = i
                break
        prev_id = ordered[idx - 1].get("id") if idx is not None and idx > 0 else None
        next_id = (
            ordered[idx + 1].get("id")
            if idx is not None and idx + 1 < len(ordered)
            else None
        )
        nav = {
            "chunk_id": cid,
            "doc_id": doc_id,
            "chunk_index": idx,
            "n_chunks": len(ordered),
            "prev_chunk_id": prev_id,
            "next_chunk_id": next_id,
        }
        sl = self._slice_info(doc_id, with_chunk_ids=with_sibling_chunk_ids)
        if sl:
            nav.update(sl)
        return nav

    def _doc_locator(self, doc_id) -> dict:
        ordered = self._ordered_chunks(doc_id)
        ids = [c.get("id") for c in ordered if c.get("id") is not None]
        loc = {
            "doc_id": doc_id,
            "n_chunks": len(ids),
            "first_chunk_id": ids[0] if ids else None,
            "last_chunk_id": ids[-1] if ids else None,
        }
        sl = self._slice_info(doc_id)
        if sl:
            loc.update({
                "sliced": True,
                "slice_index": sl.get("slice_index"),
                "n_slices": sl.get("n_slices"),
                "family": sl.get("family"),
                "siblings": sl.get("siblings"),
            })
        return loc

    def execute(self, name: str, arguments) -> str:
        args = _parse_args(arguments)
        fn = {
            "search": self.search,
            "read_doc": self.read_doc,
            "read_chunk": self.read_chunk,
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

        try:
            last_kw = dict(retrieve.get_last_keyword() or {})
        except Exception:
            last_kw = {}
        try:
            last_rw = dict(retrieve.get_last_rewrite() or {})
        except Exception:
            last_rw = {}
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
                hit = {
                    "doc_id": doc_id,
                    "chunk_id": chunk.get("id") or it.get("chunk_id"),
                    "source": source,
                    "score": score,
                    "match_type": it.get("match_type") or "",
                    "node_id": it.get("node_id"),
                    "node_name": it.get("node_name"),
                    "preview": _preview(content, preview_n),
                    "matched_chunk_ids": [],
                }
                cid0 = chunk.get("id") or it.get("chunk_id")
                if cid0 is not None:
                    hit["matched_chunk_ids"].append(cid0)
                if chunk.get("id") is not None:
                    saved = {
                        "score": hit["score"],
                        "preview": hit["preview"],
                        "source": hit["source"],
                        "match_type": hit["match_type"],
                        "node_id": hit["node_id"],
                        "node_name": hit["node_name"],
                        "matched_chunk_ids": hit["matched_chunk_ids"],
                    }
                    hit.update(self._chunk_nav(chunk))
                    hit.update(saved)
                    hit["chunk_id"] = chunk.get("id")
                    if chunk.get("doc_id") is not None:
                        hit["doc_id"] = chunk.get("doc_id")
                else:
                    sl = self._slice_info(doc_id)
                    if sl:
                        hit.update(sl)
                by_doc[key] = hit
                order.append(key)
                continue
            cid = chunk.get("id") or it.get("chunk_id")
            if cid is not None and cid not in (cur.get("matched_chunk_ids") or []):
                cur.setdefault("matched_chunk_ids", []).append(cid)
            if score > float(cur.get("score") or 0.0):
                cur["score"] = score
                if content:
                    cur["preview"] = _preview(content, preview_n)
                if chunk.get("id") is not None:
                    keep = {
                        "score": score,
                        "preview": cur.get("preview"),
                        "source": cur.get("source"),
                        "match_type": cur.get("match_type"),
                        "node_id": cur.get("node_id"),
                        "node_name": cur.get("node_name"),
                        "matched_chunk_ids": cur.get("matched_chunk_ids"),
                    }
                    cur.update(self._chunk_nav(chunk))
                    cur.update(keep)
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

        chunks = self._ordered_chunks(doc_id)
        if not chunks:
            return {"error": f"没有 doc_id={doc_id} 的块", "doc_id": doc_id}

        source = self._source_of(doc_id, chunks[0] if chunks else None)
        self._remember_ref(source, doc_id)
        slice_info = self._slice_info(doc_id, with_chunk_ids=True)
        chunk_ids = [c.get("id") for c in chunks if c.get("id") is not None]

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
        parts = [
            f"doc_id={doc_id}",
            f"source={source or '未知'}",
            f"chunk_ids(阅读序)={chunk_ids}",
        ]
        if slice_info:
            nsl = slice_info.get("n_slices")
            sidx = slice_info.get("slice_index")
            fam = slice_info.get("family") or ""
            parts.append(
                f"sliced=true  这是长 PDF「{fam}」的第 {int(sidx) + 1}/{nsl} 片"
                f"（slice_index={sidx}，从 0 计）。"
                "当前正文只是这一段，后文可能在 siblings 的下一片。"
            )
            sibs = slice_info.get("siblings") or []
            if sibs:
                bits = [
                    (
                        f"doc_id={s.get('doc_id')}[slice={s.get('slice_index')}] "
                        f"first_chunk={s.get('first_chunk_id')} "
                        f"last_chunk={s.get('last_chunk_id')} {s.get('source')}"
                    )
                    for s in sibs
                ]
                parts.append("siblings: " + " | ".join(bits))
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
            nav = self._chunk_nav(c)
            parts.append(
                f"### {name} (chunk_id={c.get('id')} doc_id={doc_id} "
                f"idx={nav.get('chunk_index')} "
                f"prev={nav.get('prev_chunk_id')} next={nav.get('next_chunk_id')})"
            )
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
        out = {
            "doc_id": doc_id,
            "source": source,
            "n_chunks": len(chunks),
            "chunk_ids": chunk_ids,
            "first_chunk_id": chunk_ids[0] if chunk_ids else None,
            "last_chunk_id": chunk_ids[-1] if chunk_ids else None,
            "truncated": truncated,
            "content": text,
        }
        if slice_info:
            out.update(slice_info)
        return out

    def read_chunk(self, args: dict) -> dict:
        chunk_id = _as_int(
            args.get("chunk_id") if "chunk_id" in args else args.get("_raw")
        )
        if chunk_id is None:
            return {"error": "read_chunk 需要整数 chunk_id"}

        chunk = self._chunk_row(chunk_id)
        if not chunk:
            return {"error": f"没有 chunk_id={chunk_id}", "chunk_id": chunk_id}

        doc_id = chunk.get("doc_id")
        source = self._source_of(doc_id, chunk)
        self._remember_ref(source, doc_id)
        nav = self._chunk_nav(chunk, with_sibling_chunk_ids=True)
        ordered = self._ordered_chunks(doc_id)
        this_ids = [c.get("id") for c in ordered if c.get("id") is not None]

        body = (chunk.get("content") or "").strip()
        max_chars = int(getattr(self.cfg, "read_chunk_max_chars", 0) or 0)
        truncated = False
        if max_chars > 0 and len(body) > max_chars:
            body = body[:max_chars] + "\n…(truncated)"
            truncated = True

        extra = {}
        try:
            extra = self.dhmf.retrieve_module._parse_extra(chunk.get("extra")) or {}
        except Exception:
            extra = {}

        out = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "source": source,
            "name": (chunk.get("name") or "").strip() or None,
            "chunk_index": nav.get("chunk_index"),
            "n_chunks": nav.get("n_chunks"),
            "prev_chunk_id": nav.get("prev_chunk_id"),
            "next_chunk_id": nav.get("next_chunk_id"),
            "chunk_ids": this_ids,
            "truncated": truncated,
            "content": body,
        }
        if extra.get("chunk_index") is not None:
            out["stored_chunk_index"] = extra.get("chunk_index")
        for k in ("sliced", "slice_index", "n_slices", "family", "siblings"):
            if nav.get(k) is not None:
                out[k] = nav[k]
        return out

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
            loc = self._doc_locator(did) if did is not None else {}
            seed_out.append({
                "node_id": nid,
                "name": n.get("name"),
                "doc_id": did,
                "hyperedge_id": hid,
                "source": src,
                "first_chunk_id": loc.get("first_chunk_id"),
                "last_chunk_id": loc.get("last_chunk_id"),
                "n_chunks": loc.get("n_chunks"),
                "slice_index": loc.get("slice_index"),
                "n_slices": loc.get("n_slices"),
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
                loc = self._doc_locator(did) if did is not None else {}
                neighbors.append({
                    "node_id": nid,
                    "name": n.get("name"),
                    "doc_id": did,
                    "hyperedge_id": n.get("hyperedge_id"),
                    "source": src,
                    "first_chunk_id": loc.get("first_chunk_id"),
                    "last_chunk_id": loc.get("last_chunk_id"),
                    "n_chunks": loc.get("n_chunks"),
                    "slice_index": loc.get("slice_index"),
                    "n_slices": loc.get("n_slices"),
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
