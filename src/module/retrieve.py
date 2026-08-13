from typing import Dict
import logging
from ..utils.database import BaseDB, BaseVDB
from ..utils.config import Config

from ..utils.OpenAIAPI import Embedding, LLM, Reranker
from ..utils.prompt import PROMPT
from ..utils.config import resolve_credentials
import json
import re
import threading
import time
from collections import defaultdict

import numpy as np

# retrieve_items 分阶段耗时字段（秒）；thread-local 保证并发评测不串号
RETRIEVE_TIMING_KEYS = (
    'precompute_s',
    'rewrite_s',
    'embed_s',
    'chunk_s',
    'node_s',
    'keyword_s',
    'expand_s',
    'rerank_s',
    'total_s',
)


def empty_retrieve_timing() -> dict:
    return {k: 0.0 for k in RETRIEVE_TIMING_KEYS}


def add_retrieve_timing(a, b) -> dict:
    """逐项累加两份 retrieve_timing（缺省按 0）。"""
    out = empty_retrieve_timing()
    for k in RETRIEVE_TIMING_KEYS:
        out[k] = float((a or {}).get(k) or 0.0) + float((b or {}).get(k) or 0.0)
    return out

class Retrieve:
    """
    三路检索（查询改写 / query instruct 可选）：
      0) LLM 将用户查询改写得更具体（可选）
      0.5) 仅查询向量加 Qwen3 Instruct 前缀（文档侧不加）
      1) 改写 query 向量 ↔ chunk 内容向量 → 取 chunk_candidate_k（0=跳过）
      2) 改写 query 向量 ↔ node 内容向量 → 取 node_candidate_k（0=跳过）→ 映射所属 chunk
      3) 关键词精确匹配（可选 enable_keyword_exact；candidate/top 任一为 0 则跳过）：
         LLM 抽取 minority/majority 关键词 → chunk.content 精确包含匹配
         → 少数值优先、多数值命中数排序取 keyword_candidate_k
         → 文档级首轮 rerank → keyword_top_k（只截关键词路，不伤向量路）
      4) 关键词路 ∪ 双路（chunk∪node）按 chunk 并集合并（只增不减；score 取 max）
      5) 文档扩展（enable_full_body_context）+ 文档级终轮 Reranker → rerank_top_k（0=跳过截断）
         默认每文档：头块 + 命中索引块（原文序、块去重）
         enable_full_body_context=True 时：每命中文档写入该文全部 body 索引块，不含头块
    """
    def vector_match(self, db_name, vector, topk=10):
        vdb = self.vdb[db_name]
        db = self.db[db_name]
        vdb_res = vdb.search(vector, topk)
        res = [
            {'distance': item['distance'], 'result': db.search('id', item['id'])[0]}
            for item in vdb_res
        ]
        return res

    def __init__(
        self,
        db: Dict[str, BaseDB],
        vdb: Dict[str, BaseVDB],
        logger: logging.Logger,
        config: Config,
    ):
        self.config = config
        self.logger = logger
        self.db = db
        self.vdb = vdb
        llm_key, llm_url = resolve_credentials(self.config, self.config.retrieve)
        self.llmmodel = LLM(llm_key, llm_url)
        emb_key = (getattr(self.config.retrieve, 'embedding_api_key', None) or '').strip()
        emb_url = (getattr(self.config.retrieve, 'embedding_base_url', None) or '').strip()
        if not emb_key or not emb_url:
            v_key, v_url = resolve_credentials(self.config, self.config.vectorization)
            emb_key = emb_key or v_key
            emb_url = emb_url or v_url
        if not emb_key:
            emb_key = self.config.settings.api_key or 'EMPTY'
        if not emb_url:
            emb_url = self.config.settings.base_url or ''
        emb_timeout = 120.0
        emb_retries = 3
        emb_wait = 0.5
        try:
            v_retry = getattr(self.config.vectorization, 'retry', None)
            if v_retry is not None:
                emb_timeout = float(getattr(v_retry, 'timeout', emb_timeout) or emb_timeout)
                emb_retries = int(getattr(v_retry, 'max_attempt', emb_retries) or emb_retries)
                emb_wait = float(getattr(v_retry, 'wait', emb_wait) or emb_wait)
        except Exception:
            pass
        self.embedding = Embedding(
            api_key=emb_key,
            base_url=emb_url,
            timeout=emb_timeout,
            max_retries=emb_retries,
            retry_wait=max(0.5, emb_wait),
        )
        rr_key = (getattr(self.config.retrieve, 'rerank_api_key', None) or '').strip()
        rr_url = (getattr(self.config.retrieve, 'rerank_base_url', None) or '').strip()
        rr_key = rr_key or emb_key
        rr_url = rr_url or emb_url
        self.reranker = Reranker(
            api_key=rr_key or 'EMPTY',
            base_url=rr_url or '',
            timeout=emb_timeout,
            max_retries=emb_retries,
            retry_wait=max(0.5, emb_wait),
        )
        self._precomputed = False
        self.all_chunks = []
        self.chunk_dict = {}
        self.all_hyperedges = []
        self.hyperedge_dict = {}
        self.hyperedge_by_doc = {}
        self.hyperedge_by_chunk = {}
        self.chunks_by_doc = defaultdict(list)
        self.doc_head_chunk = {}
        self.doc_dict = {}
        self.last_rewrite = {'original': '', 'rewritten': ''}
        self.last_rerank = {'enabled': False, 'n_in': 0, 'n_out': 0, 'scores': []}
        self.last_keyword = {
            'enabled': False,
            'minority': [],
            'majority': [],
            'pool_chunks': 0,
            'top_docs': 0,
            'doc_ids': [],
        }
        # 最近一次 retrieve_items 的分阶段耗时（thread-local）
        self._tls = threading.local()
        self.last_timing = empty_retrieve_timing()

    def get_last_timing(self) -> dict:
        """当前线程最近一次 retrieve_items 分阶段耗时（秒）。"""
        t = getattr(self._tls, 'timing', None)
        if isinstance(t, dict) and t:
            return dict(t)
        return dict(self.last_timing or empty_retrieve_timing())

    def _set_last_timing(self, timing: dict) -> None:
        cleaned = empty_retrieve_timing()
        for k in RETRIEVE_TIMING_KEYS:
            try:
                cleaned[k] = round(float((timing or {}).get(k) or 0.0), 6)
            except (TypeError, ValueError):
                cleaned[k] = 0.0
        self._tls.timing = cleaned
        self.last_timing = cleaned

    def _ensure_precompute(self):
        if self._precomputed:
            return

        self.all_chunks = list(self.db['chunk'].search_all() or [])
        self.chunk_dict = {c['id']: c for c in self.all_chunks}

        self.all_hyperedges = list(self.db['hyperedge'].search_all() or [])
        self.hyperedge_dict = {h['id']: h for h in self.all_hyperedges}
        self.hyperedge_by_doc = {}
        self.hyperedge_by_chunk = {}
        for h in self.all_hyperedges:
            did = h.get('doc_id')
            if did is not None and did not in self.hyperedge_by_doc:
                self.hyperedge_by_doc[did] = h
            cid = h.get('chunk_id')
            if cid is not None:
                self.hyperedge_by_chunk[cid] = h

        self.chunks_by_doc = defaultdict(list)
        for c in self.all_chunks:
            self.chunks_by_doc[c.get('doc_id')].append(c)
        for did in self.chunks_by_doc:
            self.chunks_by_doc[did].sort(key=self._chunk_order_key)

        self.doc_head_chunk = {}
        for did, clist in self.chunks_by_doc.items():
            head = None
            for c in clist:
                if self._is_head_chunk(c):
                    head = c
                    break
            if head is None and clist:
                head = clist[0]
            if head is not None:
                self.doc_head_chunk[did] = head

        docs = list(self.db['doc'].search_all() or [])
        self.doc_dict = {d['id']: d for d in docs}

        self._precomputed = True

    @staticmethod
    def _parse_extra(raw):
        if not raw:
            return {}
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _is_head_chunk(self, chunk: dict) -> bool:
        if not chunk:
            return False
        name = (chunk.get('name') or '').strip().lower()
        if name == 'head':
            return True
        extra = self._parse_extra(chunk.get('extra'))
        if extra.get('is_head') is True or extra.get('role') == 'head':
            return True
        cid = chunk.get('id')
        if cid is not None and cid in self.hyperedge_by_chunk:
            return True
        return False

    def _chunk_order_key(self, chunk: dict):
        extra = self._parse_extra(chunk.get('extra'))
        idx = extra.get('chunk_index')
        if idx is None:
            name = (chunk.get('name') or '').strip().lower()
            if name == 'head':
                idx = 0
            elif name.startswith('body_'):
                try:
                    idx = int(name.split('_', 1)[1])
                except Exception:
                    idx = 10**9
            else:
                idx = 0 if self._is_head_chunk(chunk) else 10**9
        try:
            idx = int(idx)
        except Exception:
            idx = 10**9
        return (idx, chunk.get('id') or 0)

    def distance_to_dpr_similarity(self, distances):
        distances = np.asarray(distances, dtype=float)
        similarities = 1 - (distances ** 2) / 2
        return float(np.clip(similarities, 0, 1))

    def _source_label(self, chunk: dict) -> str:
        if not chunk:
            return '未知来源'
        doc_id = chunk.get('doc_id')
        if doc_id is not None:
            doc = self.doc_dict.get(doc_id)
            if doc is None:
                try:
                    docs = self.db['doc'].search('id', doc_id) or []
                    if docs:
                        doc = docs[0]
                        self.doc_dict[doc_id] = doc
                except Exception:
                    doc = None
            if doc and doc.get('name'):
                return str(doc['name'])
        if chunk.get('name'):
            return str(chunk['name'])
        if chunk.get('id') is not None:
            return f"chunk_{chunk['id']}"
        return '未知来源'

    def _hyperedge_for_doc(self, doc_id):
        if doc_id is None:
            return None
        he = self.hyperedge_by_doc.get(doc_id)
        if he is not None:
            return he
        rows = self.db['hyperedge'].search('doc_id', doc_id) or []
        if rows:
            self.hyperedge_by_doc[doc_id] = rows[0]
            return rows[0]
        return None

    def _rewrite_model_args(self) -> dict:
        """
        Merge retrieve.model_args + rewrite_model_args.
        Always force enable_thinking=False unless user explicitly sets true
        in rewrite_model_args (Qwen3.5 Flash thinking ≈ 20–50s for one rewrite).
        """
        model_args = dict(self.config.retrieve.model_args or {})
        rewrite_over = dict(
            getattr(self.config.retrieve, 'rewrite_model_args', None) or {}
        )
        model_args.update(rewrite_over)
        model_args.setdefault('temperature', 0.0)
        model_args.setdefault('max_tokens', 128)
        if 'enable_thinking' not in model_args:
            model_args['enable_thinking'] = False
        return model_args

    def rewrite_query(self, query: str, *, enabled: bool | None = None) -> str:
        """
        将用户查询改写得更具体，利于向量检索。
        失败时回退原文。

        enabled:
          None  → 读 config.retrieve.enable_query_rewrite
          True/False → 本次调用覆盖配置（不改共享 config，多线程安全）
        """
        original = (query or '').strip()
        self.last_rewrite = {'original': original, 'rewritten': original}
        if not original:
            return original

        if enabled is None:
            enable = bool(getattr(self.config.retrieve, 'enable_query_rewrite', True))
        else:
            enable = bool(enabled)
        if not enable:
            return original

        t0 = time.perf_counter()
        prompt = PROMPT.get('query_rewrite', '').format(query=original)
        model_args = self._rewrite_model_args()

        try:
            response = self.llmmodel.generate(
                prompt={'system': '', 'user': prompt},
                model_args=model_args,
                use_cache=getattr(self.config.retrieve, 'use_cache', True),
            )
            if response.get('status') != 1:
                self.logger.warning(
                    f"[retrieve] query rewrite failed status={response.get('status')} "
                    f"answer={str(response.get('answer'))[:200]!r}"
                )
                return original

            text = (response.get('answer') or '').strip()
            if text.startswith('```'):
                lines = text.splitlines()
                if lines and lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                text = '\n'.join(lines).strip()
            text = text.splitlines()[0].strip() if text else ''
            for prefix in ('改写查询：', '改写查询:', '改写：', '改写:', '查询：', '查询:'):
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
            if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'", '“', '”'):
                text = text[1:-1].strip()

            if not text:
                self.logger.warning('[retrieve] query rewrite empty, use original')
                return original

            self.last_rewrite = {'original': original, 'rewritten': text}
            dt = time.perf_counter() - t0
            cache_hit = bool(response.get('_cache_hit'))
            reasoning_tok = response.get('usage_reasoning_tokens')
            self.logger.info(
                f"[retrieve] query rewrite "
                f"dt={dt:.3f}s cache_hit={cache_hit} "
                f"tokens prompt={response.get('usage_prompt_tokens')} "
                f"completion={response.get('usage_completion_tokens')} "
                f"reasoning={reasoning_tok} "
                f"orig={original!r} → rewritten={text!r}"
            )
            if dt > 5.0 and not cache_hit:
                self.logger.warning(
                    f"[retrieve] query rewrite slow ({dt:.1f}s). "
                    f"If using Qwen3.5, ensure enable_thinking=false "
                    f"(reasoning_tokens={reasoning_tok})."
                )
            return text
        except Exception as e:
            self.logger.warning(f"[retrieve] query rewrite error: {e}")
            return original

    def _query_instruct_text(self) -> str:
        raw = getattr(self.config.retrieve, 'query_instruct', None)
        return (raw or '').strip()

    def _query_instruct_enabled(self) -> bool:
        if not bool(getattr(self.config.retrieve, 'enable_query_instruct', True)):
            return False
        return bool(self._query_instruct_text())

    def _apply_query_instruct(self, query: str) -> str:
        """
        Wrap the search query for Qwen3-Embedding only.
        Documents are stored without this prefix; do not apply to rewrite /
        keyword extract / rerank / answer prompts.
        """
        q = (query or '').strip()
        if not q or not self._query_instruct_enabled():
            return q
        # Already wrapped (manual call or cached rewritten text).
        if q.startswith('Instruct:'):
            return q
        instruct = self._query_instruct_text()
        return f'Instruct: {instruct}\nQuery:{q}'

    def _keyword_extract_model_args(self) -> dict:
        """Merge retrieve.model_args + keyword_extract_model_args; default thinking off."""
        model_args = dict(self.config.retrieve.model_args or {})
        over = dict(
            getattr(self.config.retrieve, 'keyword_extract_model_args', None) or {}
        )
        model_args.update(over)
        model_args.setdefault('temperature', 0.0)
        model_args.setdefault('max_tokens', 256)
        if 'enable_thinking' not in model_args:
            model_args['enable_thinking'] = False
        if 'response_format' not in model_args:
            model_args['response_format'] = {'type': 'json_object'}
        return model_args

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        text = (text or '').strip()
        if not text.startswith('```'):
            return text
        lines = text.splitlines()
        if lines and lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        return '\n'.join(lines).strip()

    @staticmethod
    def _normalize_keyword_list(raw) -> list:
        """Normalize LLM keyword list: strip, drop empty, dedupe (casefold), preserve order."""
        if raw is None:
            return []
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            return []
        out = []
        seen = set()
        for item in raw:
            if item is None:
                continue
            s = str(item).strip()
            if not s:
                continue
            # 过短的多数值键易误伤（如 "NO" 单独出现），但仍允许少数值短编号；
            # 统一下限 2 字符；单字符几乎无检索价值。
            if len(s) < 2:
                continue
            key = s.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    def extract_keywords(self, query: str) -> dict:
        """
        LLM 抽取 minority / majority 关键词。
        返回 {'minority': [...], 'majority': [...]}；失败时空列表。
        """
        original = (query or '').strip()
        empty = {'minority': [], 'majority': []}
        if not original:
            return empty

        prompt = PROMPT.get('keyword_extract', '').format(query=original)
        if not prompt:
            self.logger.warning('[retrieve] keyword_extract prompt missing')
            return empty

        model_args = self._keyword_extract_model_args()
        t0 = time.perf_counter()
        try:
            response = self.llmmodel.generate(
                prompt={'system': '', 'user': prompt},
                model_args=model_args,
                use_cache=getattr(self.config.retrieve, 'use_cache', True),
            )
            if response.get('status') != 1:
                self.logger.warning(
                    f"[retrieve] keyword extract failed status={response.get('status')} "
                    f"answer={str(response.get('answer'))[:200]!r}"
                )
                return empty

            text = self._strip_code_fence(response.get('answer') or '')
            data = None
            try:
                data = json.loads(text)
            except Exception:
                # 尝试截取首个 JSON 对象
                m = re.search(r'\{[\s\S]*\}', text)
                if m:
                    try:
                        data = json.loads(m.group(0))
                    except Exception:
                        data = None
            if not isinstance(data, dict):
                self.logger.warning(
                    f"[retrieve] keyword extract non-json: {text[:200]!r}"
                )
                return empty

            minority = self._normalize_keyword_list(
                data.get('minority') or data.get('rare') or data.get('values')
            )
            majority = self._normalize_keyword_list(
                data.get('majority') or data.get('common') or data.get('keys')
            )
            # 同一词同时出现在两类时保留 minority（高区分度优先）
            min_cf = {k.casefold() for k in minority}
            majority = [k for k in majority if k.casefold() not in min_cf]

            dt = time.perf_counter() - t0
            self.logger.info(
                f"[retrieve] keyword extract dt={dt:.3f}s "
                f"cache_hit={bool(response.get('_cache_hit'))} "
                f"minority={minority!r} majority={majority!r}"
            )
            return {'minority': minority, 'majority': majority}
        except Exception as e:
            self.logger.warning(f"[retrieve] keyword extract error: {e}")
            return empty

    @staticmethod
    def _content_contains(content: str, keyword: str) -> bool:
        if not content or not keyword:
            return False
        return keyword.casefold() in content.casefold()

    def _search_chunks_by_keywords(
        self,
        minority: list,
        majority: list,
        candidate_k: int,
    ) -> list:
        """
        在预加载的 chunk.content 上做精确包含匹配，构造候选池：
          1) 优选取 minority 命中块（有少数值时）
          2) 再按 majority 命中个数降序补齐
          3) 截到 candidate_k；candidate_k<=0 时返回空（跳过）

        返回 hit 列表，含 score / match_type / keyword 统计字段。
        """
        try:
            candidate_k = int(candidate_k)
        except (TypeError, ValueError):
            candidate_k = 0
        if candidate_k <= 0:
            return []

        minority = [k for k in (minority or []) if k]
        majority = [k for k in (majority or []) if k]
        if not minority and not majority:
            return []

        scored = []
        for chunk in self.all_chunks:
            content = chunk.get('content') or ''
            if not content:
                continue
            min_hits = [k for k in minority if self._content_contains(content, k)]
            maj_hits = [k for k in majority if self._content_contains(content, k)]
            if not min_hits and not maj_hits:
                continue
            n_min = len(min_hits)
            n_maj = len(maj_hits)
            # 排序键：少数值优先，再多数值命中数；score 供后续展示
            # minority 权重远高于 majority，便于 max 合并时保留
            score = float(n_min) * 10.0 + float(n_maj)
            scored.append({
                'chunk_id': chunk.get('id'),
                'chunk': chunk,
                'doc_id': chunk.get('doc_id'),
                'score': score,
                'match_type': 'keyword',
                'keyword_minority_hits': min_hits,
                'keyword_majority_hits': maj_hits,
                'n_minority': n_min,
                'n_majority': n_maj,
                '_sort': (
                    1 if n_min > 0 else 0,
                    n_min,
                    n_maj,
                    score,
                ),
            })

        # 少数值块优先，再按少数值命中数、多数值命中数
        scored.sort(
            key=lambda h: (
                -h['_sort'][0],
                -h['_sort'][1],
                -h['_sort'][2],
                -h['_sort'][3],
                h.get('chunk_id') if h.get('chunk_id') is not None else 0,
            )
        )
        out = []
        for h in scored[:candidate_k]:
            h = dict(h)
            h.pop('_sort', None)
            out.append(h)
        return out

    def _hits_to_doc_passages(self, hits: list) -> list:
        """
        将 chunk hit 列表转成可供文档级 rerank 的临时 passages
        （按文档聚合，块序拼接由 _rerank_materials 完成）。
        """
        by_doc = {}
        for hit in hits or []:
            chunk = hit.get('chunk') or self.chunk_dict.get(hit.get('chunk_id')) or {}
            doc_id = hit.get('doc_id')
            if doc_id is None:
                doc_id = chunk.get('doc_id')
            if doc_id is None:
                continue
            g = by_doc.get(doc_id)
            if g is None:
                by_doc[doc_id] = {
                    'doc_id': doc_id,
                    'score': float(hit.get('score') or 0.0),
                    'chunks': [],
                    'seen': set(),
                }
                g = by_doc[doc_id]
            sc = float(hit.get('score') or 0.0)
            if sc > g['score']:
                g['score'] = sc
            cid = chunk.get('id') if chunk else hit.get('chunk_id')
            if cid is not None and cid in g['seen']:
                continue
            if cid is not None:
                g['seen'].add(cid)
            if chunk:
                g['chunks'].append(chunk)

        passages = []
        # 稳定顺序：高分文档在前
        ordered = sorted(
            by_doc.values(),
            key=lambda g: (-float(g['score']), g['doc_id'] if g['doc_id'] is not None else 0),
        )
        for mat_i, g in enumerate(ordered, start=1):
            doc_id = g['doc_id']
            source = self._source_label(g['chunks'][0] if g['chunks'] else {'doc_id': doc_id})
            # 用全文块（若有预计算）做更稳的文档级 rerank 文本
            full_chunks = list(self.chunks_by_doc.get(doc_id) or [])
            use_chunks = full_chunks if full_chunks else sorted(
                g['chunks'], key=self._chunk_order_key
            )
            for c in use_chunks:
                if self._is_head_chunk(c) and full_chunks:
                    # 关键词首轮 rerank 优先 body；无 body 时仍用 head
                    body_only = [x for x in use_chunks if not self._is_head_chunk(x)]
                    if body_only:
                        continue
                passages.append({
                    'chunk': c,
                    'score': float(g['score']),
                    'match_type': 'keyword',
                    'doc_id': doc_id,
                    'role': 'index' if not self._is_head_chunk(c) else 'head',
                    'material_id': mat_i,
                    'material_score': float(g['score']),
                    'source': source,
                })
        return passages

    def _keyword_path_retrieve(
        self,
        query: str,
        *,
        candidate_k: int = 50,
        top_k: int = 10,
    ) -> dict:
        """
        关键词精确匹配整条支路：
          extract → content 精确匹配 → 候选池排序截断
          → 文档级首轮 rerank → keyword_top_k 文档
        返回 {
          hits_by_chunk: {cid: hit},
          doc_ids: set,
          minority, majority,
          has_minority: bool,
          pool_n, top_n,
        }
        """
        empty = {
            'hits_by_chunk': {},
            'doc_ids': set(),
            'minority': [],
            'majority': [],
            'has_minority': False,
            'pool_n': 0,
            'top_n': 0,
        }
        try:
            candidate_k = int(candidate_k)
        except (TypeError, ValueError):
            candidate_k = 0
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 0
        # 任一 k<=0：跳过整条关键词路
        if candidate_k <= 0 or top_k <= 0:
            self.logger.info(
                f"[retrieve] keyword path skipped "
                f"(candidate_k={candidate_k}, top_k={top_k})"
            )
            return empty

        q = (query or '').strip()
        if not q:
            return empty

        kw = self.extract_keywords(q)
        minority = kw.get('minority') or []
        majority = kw.get('majority') or []
        if not minority and not majority:
            self.logger.info('[retrieve] keyword path: no keywords extracted')
            return {**empty, 'minority': minority, 'majority': majority}

        pool = self._search_chunks_by_keywords(
            minority, majority, candidate_k=candidate_k
        )
        if not pool:
            self.logger.info(
                f"[retrieve] keyword path: no content hits "
                f"minority={minority!r} majority={majority!r}"
            )
            return {
                **empty,
                'minority': minority,
                'majority': majority,
                'has_minority': bool(minority),
            }

        # 文档级首轮 rerank（关键词路强制启用，不受 enable_rerank 开关影响）
        passages = self._hits_to_doc_passages(pool)
        reranked = self._rerank_materials(
            q,
            passages,
            top_k=top_k,
            enable=True,
            update_last=False,
            stage='keyword',
        )
        # 取保留文档
        keep_docs = set()
        for p in reranked or []:
            did = p.get('doc_id')
            if did is not None:
                keep_docs.add(did)

        # 若 rerank 失败/关闭回退仍可能有 passages
        if not keep_docs:
            for p in passages:
                did = p.get('doc_id')
                if did is not None:
                    keep_docs.add(did)
            # 按 material 分截断（top_k 已保证 >0）
            keep_docs = set(list(keep_docs)[: top_k])

        hits_by_chunk = {}
        for hit in pool:
            did = hit.get('doc_id')
            chunk = hit.get('chunk') or {}
            if did is None:
                did = chunk.get('doc_id')
            if did not in keep_docs:
                continue
            cid = hit.get('chunk_id') or chunk.get('id')
            if cid is None:
                continue
            item = dict(hit)
            item['chunk_id'] = cid
            item['doc_id'] = did
            item['match_type'] = 'keyword'
            hits_by_chunk[cid] = item

        # 文档被 top_k 选中但 pool 里该文块被截掉时：补一条代表 hit（头块或首块）
        docs_with_hits = {
            (h.get('doc_id') or (h.get('chunk') or {}).get('doc_id'))
            for h in hits_by_chunk.values()
        }
        for did in keep_docs:
            if did in docs_with_hits:
                continue
            head = self.doc_head_chunk.get(did)
            c = head or ((self.chunks_by_doc.get(did) or [None])[0])
            if not c:
                continue
            cid = c.get('id')
            if cid is None:
                continue
            hits_by_chunk[cid] = {
                'chunk_id': cid,
                'chunk': c,
                'doc_id': did,
                'score': 1.0,
                'match_type': 'keyword',
                'keyword_minority_hits': minority,
                'keyword_majority_hits': [],
                'n_minority': len(minority),
                'n_majority': 0,
            }

        self.logger.info(
            f"[retrieve] keyword path pool={len(pool)} docs_in={len({h.get('doc_id') for h in pool})} "
            f"rerank_top_docs={len(keep_docs)} hits={len(hits_by_chunk)} "
            f"minority={minority!r} majority={majority!r}"
        )
        return {
            'hits_by_chunk': hits_by_chunk,
            'doc_ids': keep_docs,
            'minority': minority,
            'majority': majority,
            'has_minority': any(
                (h.get('n_minority') or 0) > 0 for h in pool
            ) or bool(minority and hits_by_chunk),
            'pool_n': len(pool),
            'top_n': len(keep_docs),
        }

    def _union_hits_by_docs(
        self,
        dual_hits: dict,
        keyword_hits: dict,
        keyword_doc_ids: set = None,
        *,
        has_minority: bool = False,
    ) -> dict:
        """
        关键词路与双路按 chunk 并集合并（关键词只增不减）：
          merged = dual ∪ keyword
        - 同 chunk：score 取 max，match_type 合并（chunk+node+keyword）
        - 关键词路内部仍可 candidate → 文档 rerank → keyword_top_k
        - 终轮文档 rerank 在并集结果上做（见 retrieve_items）
        不会用关键词文档集过滤掉纯向量命中。
        """
        dual_hits = dual_hits or {}
        keyword_hits = keyword_hits or {}
        merged = {}

        def _doc_ids_from_hits(hits: dict) -> set:
            ids = set()
            for h in hits.values():
                did = h.get('doc_id')
                if did is None:
                    chunk = h.get('chunk') or {}
                    did = chunk.get('doc_id')
                if did is not None:
                    ids.add(did)
            return ids

        def _absorb(hit):
            cid = hit.get('chunk_id')
            if cid is None:
                chunk = hit.get('chunk') or {}
                cid = chunk.get('id')
            if cid is None:
                return
            did = hit.get('doc_id')
            if did is None:
                did = (hit.get('chunk') or {}).get('doc_id')
            cur = merged.get(cid)
            if cur is None:
                item = dict(hit)
                item['chunk_id'] = cid
                if did is not None:
                    item['doc_id'] = did
                item.setdefault(
                    'chunk', hit.get('chunk') or self.chunk_dict.get(cid)
                )
                merged[cid] = item
                return
            score = float(hit.get('score') or 0.0)
            if score > float(cur.get('score') or 0.0):
                cur['score'] = score
            mt_a = cur.get('match_type') or ''
            mt_b = hit.get('match_type') or ''
            parts = set()
            for mt in (mt_a, mt_b):
                if not mt:
                    continue
                for p in str(mt).split('+'):
                    p = p.strip()
                    if p:
                        parts.add(p)
            if parts:
                order = {'chunk': 0, 'node': 1, 'keyword': 2}
                cur['match_type'] = '+'.join(
                    sorted(parts, key=lambda x: (order.get(x, 9), x))
                )
            for key in (
                'node_id', 'node_name', 'hyperedge_id', 'doc_id',
                'keyword_minority_hits', 'keyword_majority_hits',
                'n_minority', 'n_majority',
            ):
                if cur.get(key) is None and hit.get(key) is not None:
                    cur[key] = hit[key]

        for h in dual_hits.values():
            _absorb(h)
        for h in keyword_hits.values():
            _absorb(h)

        dual_docs = _doc_ids_from_hits(dual_hits)
        kw_docs = set(keyword_doc_ids or ()) or _doc_ids_from_hits(keyword_hits)
        merged_docs = _doc_ids_from_hits(merged)
        self.logger.info(
            f"[retrieve] keyword∪dual chunks={len(merged)} "
            f"docs={len(merged_docs)} "
            f"(dual_docs={len(dual_docs)} kw_docs={len(kw_docs)} "
            f"dual_chunks={len(dual_hits)} kw_chunks={len(keyword_hits)}"
            f"{' minority' if has_minority else ''})"
        )
        return merged

    # 兼容旧调用名
    def _intersect_hits_by_docs(self, *args, **kwargs):
        return self._union_hits_by_docs(*args, **kwargs)

    def _search_chunks_by_query(self, query_embedding, topk: int) -> list:
        """改写 query ↔ chunk 向量。topk<=0 时跳过本路。"""
        try:
            topk = int(topk)
        except (TypeError, ValueError):
            topk = 0
        if topk <= 0:
            return []
        raw = self.vector_match('chunk', query_embedding, topk=topk)
        hits = []
        for r in raw:
            dist = r.get('distance')
            sim = self.distance_to_dpr_similarity(dist)
            chunk = r.get('result') or {}
            hits.append({
                'chunk_id': chunk.get('id'),
                'chunk': chunk,
                'score': sim,
                'match_type': 'chunk',
            })
        return hits

    def _search_nodes_by_query(self, query_embedding, topk: int) -> list:
        """改写 query ↔ node 内容向量 → 映射到所属 chunk。topk<=0 时跳过本路。"""
        try:
            topk = int(topk)
        except (TypeError, ValueError):
            topk = 0
        if topk <= 0:
            return []
        raw = self.vector_match('node', query_embedding, topk=topk)
        best_by_chunk = {}

        for r in raw:
            node = r.get('result') or {}
            dist = r.get('distance')
            sim = self.distance_to_dpr_similarity(dist)
            cid = node.get('chunk_id')
            if cid is None:
                continue
            chunk = self.chunk_dict.get(cid)
            if chunk is None:
                rows = self.db['chunk'].search('id', cid) or []
                chunk = rows[0] if rows else {
                    'id': cid,
                    'doc_id': node.get('doc_id'),
                    'content': '',
                }
            prev = best_by_chunk.get(cid)
            if prev is None or sim > prev['score']:
                best_by_chunk[cid] = {
                    'chunk_id': cid,
                    'chunk': chunk,
                    'score': sim,
                    'match_type': 'node',
                    'node_id': node.get('id'),
                    'node_name': node.get('name'),
                    'hyperedge_id': node.get('hyperedge_id'),
                    'doc_id': node.get('doc_id') or chunk.get('doc_id'),
                }

        return list(best_by_chunk.values())

    def _merge_chunk_hits(self, chunk_hits: list, node_hits: list) -> dict:
        """按 chunk_id 合并两路命中；score 取 max；match_type 合并。"""
        by_cid = {}

        def _absorb(hit):
            cid = hit.get('chunk_id')
            if cid is None:
                chunk = hit.get('chunk') or {}
                cid = chunk.get('id')
            if cid is None:
                return
            cur = by_cid.get(cid)
            if cur is None:
                item = dict(hit)
                item['chunk_id'] = cid
                item.setdefault('chunk', hit.get('chunk') or self.chunk_dict.get(cid))
                by_cid[cid] = item
                return
            score = hit.get('score') or 0.0
            if score > (cur.get('score') or 0.0):
                cur['score'] = score
            mt_a = cur.get('match_type') or ''
            mt_b = hit.get('match_type') or ''
            if mt_a and mt_b and mt_a != mt_b:
                cur['match_type'] = 'chunk+node'
            elif mt_b and not mt_a:
                cur['match_type'] = mt_b
            for key in ('node_id', 'node_name', 'hyperedge_id', 'doc_id'):
                if cur.get(key) is None and hit.get(key) is not None:
                    cur[key] = hit[key]

        for h in chunk_hits:
            _absorb(h)
        for h in node_hits:
            _absorb(h)
        return by_cid

    def _body_chunks_for_doc(self, doc_id) -> list:
        """该文档全部非头块索引块，按原文序。"""
        if doc_id is None:
            return []
        out = []
        for c in self.chunks_by_doc.get(doc_id) or []:
            if self._is_head_chunk(c):
                continue
            out.append(c)
        return out

    def _build_materials(self, hits_by_chunk: dict) -> list:
        """
        按文档聚合（不再做全局 top_k 截断）：
          默认：资料 = 头块 + 命中索引块（按原文顺序，块去重）
          enable_full_body_context：资料 = 该文档全部 body 索引块（不含头块）
            —— 命中 head 或任一 body 均同样扩全篇 body
          文档组之间按组内最高分降序（仅排序，全部进入上下文）

        召回宽度由双路各自的 chunk_candidate_k / node_candidate_k 决定。
        """
        full_body = bool(
            getattr(self.config.retrieve, 'enable_full_body_context', False)
        )

        groups = {}
        for cid, hit in hits_by_chunk.items():
            chunk = hit.get('chunk') or self.chunk_dict.get(cid) or {}
            doc_id = hit.get('doc_id') or chunk.get('doc_id')
            if doc_id is None:
                continue
            g = groups.get(doc_id)
            if g is None:
                head = self.doc_head_chunk.get(doc_id)
                he = self._hyperedge_for_doc(doc_id)
                g = {
                    'doc_id': doc_id,
                    'score': 0.0,
                    'index_ids': set(),
                    'head': head,
                    'hyperedge': he,
                    'hit_meta': {},
                }
                groups[doc_id] = g
            score = float(hit.get('score') or 0.0)
            if score > g['score']:
                g['score'] = score
            g['hit_meta'][cid] = hit
            head = g['head']
            if head is not None and cid == head.get('id'):
                continue
            if self._is_head_chunk(chunk):
                continue
            g['index_ids'].add(cid)

        # 全篇 body 模式：无论命中 head 还是 body，索引集合 = 该文档全部 body
        if full_body:
            for g in groups.values():
                body_ids = set()
                for c in self._body_chunks_for_doc(g['doc_id']):
                    cid = c.get('id')
                    if cid is not None:
                        body_ids.add(cid)
                g['index_ids'] = body_ids

        ordered_docs = sorted(
            groups.values(),
            key=lambda g: (-float(g['score']), g['doc_id'] if g['doc_id'] is not None else 0),
        )
        # 双路合并后的全部资料组均进入上下文（不按全局 top_k 再截）

        passages = []
        for mat_i, g in enumerate(ordered_docs, start=1):
            doc_id = g['doc_id']
            he = g['hyperedge']
            hid = he.get('id') if he else None
            head = g['head']
            source = self._source_label(head or {'doc_id': doc_id})

            # 全篇 body 模式：不把头块写入上下文
            if head is not None and not full_body:
                head_hit = g['hit_meta'].get(head['id'], {})
                passages.append({
                    'chunk': head,
                    'score': float(g['score']),
                    'match_type': head_hit.get('match_type') or 'head',
                    'hyperedge_id': hid,
                    'doc_id': doc_id,
                    'role': 'head',
                    'material_id': mat_i,
                    'material_score': float(g['score']),
                    'source': source,
                })

            index_chunks = []
            for cid in g['index_ids']:
                c = self.chunk_dict.get(cid) or (g['hit_meta'].get(cid) or {}).get('chunk')
                if c is None:
                    continue
                if full_body and self._is_head_chunk(c):
                    continue
                index_chunks.append(c)
            index_chunks.sort(key=self._chunk_order_key)

            seen = set()
            if head is not None and not full_body:
                seen.add(head['id'])
            for c in index_chunks:
                cid = c.get('id')
                if cid in seen:
                    continue
                seen.add(cid)
                hit = g['hit_meta'].get(cid, {})
                if hit:
                    match_type = hit.get('match_type') or 'index'
                    score = float(hit.get('score') or g['score'])
                    node_id = hit.get('node_id')
                    node_name = hit.get('node_name')
                    he_id = hid or hit.get('hyperedge_id')
                else:
                    # 非检索命中、由全篇 body 扩展补入
                    match_type = 'full_body_expand' if full_body else 'index'
                    score = float(g['score'])
                    node_id = None
                    node_name = None
                    he_id = hid
                passages.append({
                    'chunk': c,
                    'score': score,
                    'match_type': match_type,
                    'hyperedge_id': he_id,
                    'doc_id': doc_id,
                    'role': 'index',
                    'material_id': mat_i,
                    'material_score': float(g['score']),
                    'source': source,
                    'node_id': node_id,
                    'node_name': node_name,
                })

        return passages

    def _format_retrieved_chunks(self, chunks_with_meta) -> str:
        """
        按资料组输出（Markdown 标题，不用全角括号）：
          ----- Material 1 | source: xxx -----
          ### Head
          ...
          ### Index block 1
          ...
        """
        if not chunks_with_meta:
            return '(no relevant product materials retrieved)\n'

        by_mat = defaultdict(list)
        has_mat = False
        for item in chunks_with_meta:
            mid = item.get('material_id')
            if mid is not None:
                has_mat = True
            by_mat[mid if mid is not None else 0].append(item)

        if not has_mat:
            parts = []
            for i, item in enumerate(chunks_with_meta, start=1):
                chunk = item.get('chunk') or item.get('result') or {}
                source = item.get('source') or self._source_label(chunk)
                content = (chunk.get('content') or '').strip()
                meta_bits = []
                if item.get('match_type'):
                    meta_bits.append(f"match={item['match_type']}")
                # 块得分仅用于内部排序，不写入上下文
                meta_str = (' | ' + ' | '.join(meta_bits)) if meta_bits else ''
                parts.append(
                    f'----- Passage {i} | source: {source}{meta_str} -----\n'
                    f'{content}\n'
                )
            return '\n'.join(parts)

        # 主资料组与推荐扩展（role=recommendation）分开排版
        main_groups = {}
        rec_items = []
        for mid, items in by_mat.items():
            recs = [it for it in items if it.get('role') == 'recommendation']
            mains = [it for it in items if it.get('role') != 'recommendation']
            rec_items.extend(recs)
            if mains:
                main_groups[mid if mid is not None else 0] = mains

        parts = []
        for mid in sorted(main_groups.keys(), key=lambda x: (x is None, x or 0)):
            items = main_groups[mid]
            if not items:
                continue
            first = items[0]
            source = first.get('source') or self._source_label(first.get('chunk') or {})
            hid = first.get('hyperedge_id')
            meta_bits = []
            if hid is not None:
                meta_bits.append(f'hyperedge_id={hid}')
            # 块/资料得分仅用于内部排序，不写入上下文
            meta_str = (' | ' + ' | '.join(meta_bits)) if meta_bits else ''
            block_lines = [
                f'----- Material {mid} | source: {source}{meta_str} -----'
            ]

            index_i = 0
            for item in items:
                chunk = item.get('chunk') or {}
                content = (chunk.get('content') or '').strip()
                role = item.get('role') or (
                    'head' if self._is_head_chunk(chunk) else 'index'
                )
                mt = item.get('match_type') or ''
                side = []
                if mt:
                    side.append(f'match={mt}')
                if item.get('node_name'):
                    side.append(f"node={item['node_name']}")
                side_str = (' (' + '; '.join(side) + ')') if side else ''

                if role == 'head':
                    block_lines.append(f'### Head{side_str}')
                else:
                    index_i += 1
                    block_lines.append(f'### Index block {index_i}{side_str}')
                block_lines.append(content)
                block_lines.append('')

            parts.append('\n'.join(block_lines).rstrip() + '\n')

        # 推荐扩展：仅超边 content，无分数
        for i, item in enumerate(rec_items, start=1):
            chunk = item.get('chunk') or {}
            content = (chunk.get('content') or item.get('content') or '').strip()
            source = item.get('source') or self._source_label(chunk)
            hid = item.get('hyperedge_id')
            meta_bits = ['recommendation']
            if hid is not None:
                meta_bits.append(f'hyperedge_id={hid}')
            meta_str = ' | '.join(meta_bits)
            parts.append(
                f'----- Recommendation {i} | source: {source} | {meta_str} -----\n'
                f'{content}\n'
            )

        return '\n'.join(parts)

    def _enrich_passage_ids(self, items: list) -> list:
        out = []
        for item in items or []:
            it = dict(item)
            chunk = it.get('chunk') or it.get('result') or {}
            doc_id = it.get('doc_id') or (chunk.get('doc_id') if isinstance(chunk, dict) else None)
            it['doc_id'] = doc_id
            if it.get('hyperedge_id') is None and doc_id is not None:
                he = self._hyperedge_for_doc(doc_id)
                if he:
                    it['hyperedge_id'] = he.get('id')
            if it.get('source') is None:
                it['source'] = self._source_label(chunk if isinstance(chunk, dict) else {})
            out.append(it)
        return out

    @staticmethod
    def _parse_recommendation_ids(raw) -> list:
        """Parse '1,2,3' / list → int ids（去空、去重、保序）。"""
        if raw is None:
            return []
        if isinstance(raw, (list, tuple, set)):
            parts = list(raw)
        else:
            text = str(raw).strip()
            if not text:
                return []
            parts = re.split(r'[,，;\s]+', text)
        out = []
        seen = set()
        for p in parts:
            if p is None:
                continue
            s = str(p).strip()
            if not s:
                continue
            try:
                i = int(s)
            except Exception:
                continue
            if i in seen:
                continue
            seen.add(i)
            out.append(i)
        return out

    def _hyperedge_by_id(self, hid):
        if hid is None:
            return None
        he = self.hyperedge_dict.get(hid)
        if he is not None:
            return he
        rows = self.db['hyperedge'].search('id', hid) or []
        if rows:
            self.hyperedge_dict[hid] = rows[0]
            return rows[0]
        return None

    def _expand_recommendations(self, passages: list) -> list:
        """
        主检索资料之后追加一层推荐扩展：
          - 仅附加 hyperedge.content
          - 与主检索按 hyperedge_id / doc_id 去重：重合保留主检索（head-body 顺序）
          - 不对推荐再扩推荐
          - 无分数；role=recommendation
        """
        enable = bool(
            getattr(self.config.retrieve, 'enable_recommendation_expand', False)
        )
        if not enable or not passages:
            return passages

        main_he_ids = set()
        main_doc_ids = set()
        for p in passages:
            hid = p.get('hyperedge_id')
            if hid is not None:
                main_he_ids.add(int(hid))
            did = p.get('doc_id')
            if did is not None:
                main_doc_ids.add(int(did))

        # 触发源：主检索中出现过的超边（直接命中或连带头块）
        seed_he_ids = sorted(main_he_ids)
        candidate_ids = []
        seen_cand = set()
        for hid in seed_he_ids:
            he = self._hyperedge_by_id(hid)
            if not he:
                continue
            for rid in self._parse_recommendation_ids(he.get('recommendation')):
                if rid in main_he_ids or rid in seen_cand:
                    continue
                seen_cand.add(rid)
                candidate_ids.append(rid)

        if not candidate_ids:
            return passages

        extra = []
        added_he = set()
        for rid in candidate_ids:
            if rid in main_he_ids or rid in added_he:
                continue
            he = self._hyperedge_by_id(rid)
            if not he:
                continue
            doc_id = he.get('doc_id')
            # 同一文档已在主检索中 → 只保留主检索（保证 head-body 顺序）
            if doc_id is not None and int(doc_id) in main_doc_ids:
                continue
            content = (he.get('content') or '').strip()
            if not content:
                continue
            source = self._source_label({'doc_id': doc_id, 'name': he.get('name')})
            extra.append({
                'chunk': {
                    'id': None,
                    'doc_id': doc_id,
                    'name': he.get('name') or 'recommendation',
                    'content': content,
                },
                'score': None,
                'match_type': 'recommendation',
                'hyperedge_id': rid,
                'doc_id': doc_id,
                'role': 'recommendation',
                'material_id': None,
                'material_score': None,
                'source': source,
                'recommendation_label': '推荐扩展',
            })
            added_he.add(rid)
            if doc_id is not None:
                main_doc_ids.add(int(doc_id))

        if extra:
            self.logger.info(
                f"[retrieve] recommendation expand "
                f"seeds={len(seed_he_ids)} candidates={len(candidate_ids)} "
                f"added={len(extra)}"
            )
        return list(passages) + extra

    @staticmethod
    def _concat_doc_text(items: list, *, max_chars: int = -1) -> str:
        """
        将同一文档的块按当前列表顺序拼接为纯文本。
        不加「资料N / 头块 / 文档1」等标注，块与块之间仅用换行连接。
        """
        parts = []
        for it in items or []:
            chunk = it.get('chunk') or {}
            text = (chunk.get('content') or it.get('content') or '').strip()
            if text:
                parts.append(text)
        body = '\n'.join(parts)
        if max_chars is not None and int(max_chars) > 0 and len(body) > int(max_chars):
            body = body[: int(max_chars)]
        return body

    def _group_main_materials(self, passages: list) -> list:
        """
        主检索资料按 material_id（缺省 doc_id）分组，保留块原有顺序。
        推荐扩展（role=recommendation）不参与文档级重排。
        返回 [{key, doc_id, source, items, score}, ...]，顺序 = 当前资料序。
        """
        groups = {}
        order = []
        for p in passages or []:
            if p.get('role') == 'recommendation':
                continue
            mid = p.get('material_id')
            did = p.get('doc_id')
            key = mid if mid is not None else (f'doc:{did}' if did is not None else id(p))
            g = groups.get(key)
            if g is None:
                g = {
                    'key': key,
                    'doc_id': did,
                    'source': p.get('source') or self._source_label(p.get('chunk') or {}),
                    'items': [],
                    'score': float(p.get('material_score') or p.get('score') or 0.0),
                }
                groups[key] = g
                order.append(key)
            g['items'].append(p)
            sc = float(p.get('material_score') or p.get('score') or 0.0)
            if sc > g['score']:
                g['score'] = sc
        return [groups[k] for k in order]

    def _rerank_materials(
        self,
        query: str,
        passages: list,
        *,
        top_k: int | None = None,
        enable: bool | None = None,
        update_last: bool = True,
        stage: str = 'final',
    ) -> list:
        """
        文档级重排：
          1) 主资料按文档聚合成完整文本（块序拼接、无标注）
          2) Qwen3-Reranker 打分排序
          3) 只保留 top_k 文档；重排 material_id = 1..k
          4) 推荐扩展不进入最终上下文（上下文仅 top_k 完整文档）
        失败时回退：按原 material 分截 top_k。

        top_k / enable: 覆盖配置（关键词路首轮 rerank 用）。
        update_last: 是否写入 self.last_rerank（中间轮可关，避免覆盖终轮统计）。
        stage: 日志标签 final / keyword。
        """
        if enable is None:
            enable = bool(getattr(self.config.retrieve, 'enable_rerank', False))
        else:
            enable = bool(enable)
        if top_k is None:
            raw_k = getattr(self.config.retrieve, 'rerank_top_k', 4)
            try:
                top_k = int(raw_k) if raw_k is not None else 4
            except (TypeError, ValueError):
                top_k = 4
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 4
        # top_k<=0：跳过截断/重排，保留全部（与 enable=False 同效）
        if top_k <= 0:
            enable = False
        max_chars = int(getattr(self.config.retrieve, 'rerank_max_chars', -1) or -1)
        model_args = dict(
            getattr(self.config.retrieve, 'rerank_model_args', None) or {}
        )
        model_args.setdefault('model', 'Qwen3-Reranker-0.6B')

        groups = self._group_main_materials(passages)
        meta = {
            'enabled': enable,
            'n_in': len(groups),
            'n_out': 0,
            'scores': [],
            'top_k': top_k,
            'stage': stage,
        }
        if update_last:
            self.last_rerank = dict(meta)
        if not groups:
            return []

        if not enable:
            # 关闭重排 / top_k=0：保留全部主资料 + 推荐（调用方已拼好）
            if update_last:
                self.last_rerank.update({
                    'n_out': len(groups),
                    'skipped': top_k <= 0,
                })
            return passages

        docs_text = [
            self._concat_doc_text(g['items'], max_chars=max_chars) for g in groups
        ]
        # 空文档仍占位，避免 index 错位
        docs_text = [t if t else ' ' for t in docs_text]

        t0 = time.perf_counter()
        try:
            ranked = self.reranker.rerank(
                query,
                docs_text,
                model=model_args.get('model') or 'Qwen3-Reranker-0.6B',
                top_n=top_k,
                model_args=model_args,
            )
        except Exception as e:
            self.logger.warning(
                f"[retrieve] rerank({stage}) failed, fallback to retrieval score "
                f"top_k={top_k}: {e}"
            )
            # 回退：按已有 material_score 截断
            ranked_groups = sorted(groups, key=lambda g: -float(g['score']))[:top_k]
            out = []
            scores = []
            for new_i, g in enumerate(ranked_groups, start=1):
                scores.append({
                    'rank': new_i,
                    'doc_id': g['doc_id'],
                    'source': g['source'],
                    'relevance_score': float(g['score']),
                    'fallback': True,
                })
                for it in g['items']:
                    row = dict(it)
                    row['material_id'] = new_i
                    row['material_score'] = float(g['score'])
                    row['rerank_score'] = float(g['score'])
                    out.append(row)
            meta.update({
                'n_out': len(ranked_groups),
                'scores': scores,
                'fallback': True,
                'latency_s': round(time.perf_counter() - t0, 3),
            })
            if update_last:
                self.last_rerank.update(meta)
            return out

        # 按 rerank 结果重装 passages
        out = []
        scores = []
        seen_idx = set()
        for new_i, item in enumerate(ranked, start=1):
            idx = int(item.get('index', -1))
            if idx < 0 or idx >= len(groups) or idx in seen_idx:
                continue
            seen_idx.add(idx)
            g = groups[idx]
            rel = float(item.get('relevance_score') or 0.0)
            scores.append({
                'rank': new_i,
                'doc_id': g['doc_id'],
                'source': g['source'],
                'relevance_score': rel,
                'index': idx,
            })
            for it in g['items']:
                row = dict(it)
                row['material_id'] = new_i
                row['material_score'] = rel
                row['rerank_score'] = rel
                out.append(row)
            if len(seen_idx) >= top_k:
                break

        meta.update({
            'n_out': len(seen_idx),
            'scores': scores,
            'fallback': False,
            'latency_s': round(time.perf_counter() - t0, 3),
            'model': model_args.get('model'),
        })
        if update_last:
            self.last_rerank.update(meta)
        self.logger.info(
            f"[retrieve] rerank({stage}) model={model_args.get('model')!r} "
            f"in={len(groups)} out={len(seen_idx)} top_k={top_k} "
            f"dt={meta.get('latency_s', 0):.3f}s "
            # f"top={[(s.get('source'), round(s.get('relevance_score', 0), 4)) for s in scores[:top_k]]}"
        )
        return out

    @staticmethod
    def _resolve_topk(override, config_val, default: int) -> int:
        """
        解析 topk：显式参数优先，否则配置，否则 default。
        允许 0（跳过该路）；负数钳到 0；非法回 default。
        注意：不能用 ``x or default``，否则 0 会被误判为缺省。
        """
        raw = override if override is not None else config_val
        if raw is None:
            return max(0, int(default))
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return max(0, int(default))

    def retrieve_items(
        self,
        query,
        chunk_candidate_k=None,
        node_candidate_k=None,
        enable_query_rewrite=None,
        enable_keyword_exact=None,
        keyword_candidate_k=None,
        keyword_top_k=None,
    ) -> list:
        """
        查询改写 → 双路向量 topk ∪ 关键词精确匹配（关键词路内部可先文档 rerank）
        → 并集混合 → 扩展 → 终轮 rerank。

        关键词路只增加候选，不删减向量命中；同 chunk 合并 score/match_type。

        各路 topk：
          - >0：正常截断
          - 0：跳过该路（chunk / node / keyword 候选与 top / rerank_top_k）

        enable_query_rewrite / enable_keyword_exact:
          None 用配置；True/False 仅本次覆盖（不改共享 config）。

        分阶段耗时写入 self.last_timing / get_last_timing()：
          precompute / rewrite / embed / chunk / node / keyword / expand / rerank / total
        """
        t_all = time.perf_counter()
        t0 = time.perf_counter()
        self._ensure_precompute()
        t_precompute = time.perf_counter() - t0

        chunk_cand = self._resolve_topk(
            chunk_candidate_k,
            getattr(self.config.retrieve, 'chunk_candidate_k', None),
            10,
        )
        node_cand = self._resolve_topk(
            node_candidate_k,
            getattr(self.config.retrieve, 'node_candidate_k', None),
            10,
        )
        kw_cand = self._resolve_topk(
            keyword_candidate_k,
            getattr(self.config.retrieve, 'keyword_candidate_k', None),
            50,
        )
        kw_top = self._resolve_topk(
            keyword_top_k,
            getattr(self.config.retrieve, 'keyword_top_k', None),
            10,
        )
        rerank_top = self._resolve_topk(
            None,
            getattr(self.config.retrieve, 'rerank_top_k', None),
            4,
        )

        if enable_keyword_exact is None:
            enable_kw = bool(
                getattr(self.config.retrieve, 'enable_keyword_exact', True)
            )
        else:
            enable_kw = bool(enable_keyword_exact)
        if kw_cand <= 0 or kw_top <= 0:
            enable_kw = False

        use_cache = getattr(self.config.retrieve, 'use_cache', True)
        full_body = bool(
            getattr(self.config.retrieve, 'enable_full_body_context', False)
        )
        enable_rerank = bool(
            getattr(self.config.retrieve, 'enable_rerank', False)
        )
        if rerank_top <= 0:
            enable_rerank = False

        need_vector = chunk_cand > 0 or node_cand > 0

        t0 = time.perf_counter()
        rewritten = self.rewrite_query(query, enabled=enable_query_rewrite)
        t_rewrite = time.perf_counter() - t0
        search_query = rewritten or query

        t0 = time.perf_counter()
        query_embedding = None
        embed_query = self._apply_query_instruct(search_query)
        if need_vector:
            emb_resp = self.embedding.generate(
                embed_query,
                model_args=self.config.retrieve.embedding_model_args,
                use_cache=use_cache,
            )
            query_embedding = emb_resp['answer']
            if not isinstance(query_embedding, list):
                self.logger.error(
                    f"[retrieve] embedding failed: {query_embedding!r}"
                )
                query_embedding = None
        t_emb = time.perf_counter() - t0

        t0 = time.perf_counter()
        if chunk_cand > 0 and query_embedding is not None:
            chunk_hits = self._search_chunks_by_query(
                query_embedding, topk=chunk_cand
            )
        else:
            chunk_hits = []
            if chunk_cand <= 0:
                self.logger.info('[retrieve] chunk path skipped (chunk_candidate_k=0)')
        t_chunk = time.perf_counter() - t0

        t0 = time.perf_counter()
        if node_cand > 0 and query_embedding is not None:
            node_hits = self._search_nodes_by_query(
                query_embedding, topk=node_cand
            )
        else:
            node_hits = []
            if node_cand <= 0:
                self.logger.info('[retrieve] node path skipped (node_candidate_k=0)')
        t_node = time.perf_counter() - t0

        dual_merged = self._merge_chunk_hits(chunk_hits, node_hits)

        t0 = time.perf_counter()
        self.last_keyword = {
            'enabled': enable_kw,
            'minority': [],
            'majority': [],
            'pool_chunks': 0,
            'top_docs': 0,
            'doc_ids': [],
        }
        kw_result = {
            'hits_by_chunk': {},
            'doc_ids': set(),
            'minority': [],
            'majority': [],
            'has_minority': False,
            'pool_n': 0,
            'top_n': 0,
        }
        if enable_kw:
            kw_result = self._keyword_path_retrieve(
                query,
                candidate_k=kw_cand,
                top_k=kw_top,
            )
            self.last_keyword = {
                'enabled': True,
                'minority': list(kw_result.get('minority') or []),
                'majority': list(kw_result.get('majority') or []),
                'pool_chunks': int(kw_result.get('pool_n') or 0),
                'top_docs': int(kw_result.get('top_n') or 0),
                'doc_ids': sorted(
                    d for d in (kw_result.get('doc_ids') or set())
                    if d is not None
                ),
                'has_minority': bool(kw_result.get('has_minority')),
            }
        elif kw_cand <= 0 or kw_top <= 0:
            self.logger.info(
                f"[retrieve] keyword path skipped "
                f"(keyword_candidate_k={kw_cand}, keyword_top_k={kw_top})"
            )
        t_keyword = time.perf_counter() - t0

        if enable_kw and (
            kw_result.get('hits_by_chunk') or kw_result.get('doc_ids')
        ):
            # 并集：向量命中全部保留，关键词命中只追加/增强
            merged = self._union_hits_by_docs(
                dual_merged,
                kw_result.get('hits_by_chunk') or {},
                kw_result.get('doc_ids') or set(),
                has_minority=bool(kw_result.get('has_minority')),
            )
        else:
            merged = dual_merged

        t0 = time.perf_counter()
        passages = self._build_materials(merged)
        passages = self._enrich_passage_ids(passages)

        if not enable_rerank:
            passages = self._expand_recommendations(passages)
        t_expand = time.perf_counter() - t0

        t0 = time.perf_counter()
        passages = self._rerank_materials(
            search_query or query,
            passages,
            top_k=rerank_top,
            enable=enable_rerank,
            stage='final',
        )
        t_rerank = time.perf_counter() - t0

        t_total = time.perf_counter() - t_all
        timing = {
            'precompute_s': t_precompute,
            'rewrite_s': t_rewrite,
            'embed_s': t_emb,
            'chunk_s': t_chunk,
            'node_s': t_node,
            'keyword_s': t_keyword,
            'expand_s': t_expand,
            'rerank_s': t_rerank,
            'total_s': t_total,
        }
        self._set_last_timing(timing)

        n_mat = len({p.get('material_id') for p in passages if p.get('material_id') is not None})
        n_rec = sum(1 for p in passages if p.get('role') == 'recommendation')
        n_head = sum(1 for p in passages if p.get('role') == 'head')
        n_index = sum(1 for p in passages if p.get('role') == 'index')
        n_merged_chunks = len(merged)
        self.logger.info(
            f"[retrieve] multi-path "
            f"rewrite={rewritten!r} "
            f"query_instruct={self._query_instruct_enabled()} "
            f"chunk_k={chunk_cand} node_k={node_cand} "
            f"keyword={enable_kw} kw_cand={kw_cand} kw_top={kw_top} "
            f"kw_minority={self.last_keyword.get('minority')!r} "
            f"kw_majority={self.last_keyword.get('majority')!r} "
            f"kw_pool={self.last_keyword.get('pool_chunks')} "
            f"kw_docs={self.last_keyword.get('top_docs')} "
            f"full_body={full_body} rerank={enable_rerank} rerank_k={rerank_top} "
            f"chunk_hits={len(chunk_hits)} node_hits={len(node_hits)} "
            f"merged_chunks={n_merged_chunks} "
            f"materials={n_mat} heads={n_head} index={n_index} "
            f"rec_expand={n_rec} passages={len(passages)} "
            f"timing precompute={t_precompute:.3f}s rewrite={t_rewrite:.3f}s "
            f"embed={t_emb:.3f}s chunk={t_chunk:.3f}s node={t_node:.3f}s "
            f"keyword={t_keyword:.3f}s expand={t_expand:.3f}s "
            f"rerank={t_rerank:.3f}s total={t_total:.3f}s"
        )
        return passages

    def retrieve(self, query, **kwargs):
        items = self.retrieve_items(query, **kwargs)
        return self._format_retrieved_chunks(items)
