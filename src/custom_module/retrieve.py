from ..module.retrieve import BaseRetrieve
from ..utils.OpenAIAPI import Embedding, LLM, Reranker
from ..utils.prompt import PROMPT
from ..utils.config import resolve_credentials
import json
import re
import time
from collections import defaultdict

import numpy as np


class Retrieve(BaseRetrieve):
    """
    双路检索（先查询改写，再两路各自 topk 截断后合并）：
      0) LLM 将用户查询改写得更具体
      1) 改写 query 向量 ↔ chunk 内容向量 → 取 chunk_candidate_k
      2) 改写 query 向量 ↔ node 内容向量 → 取 node_candidate_k → 映射所属 chunk
      3) 两路按 chunk 合并去重 + 文档扩展
      4) 可选：文档级 Reranker 重排，只保留 rerank_top_k 份完整文档进上下文
         默认每文档：头块 + 命中索引块（原文序、块去重）
         enable_full_body_context=True 时：每命中文档写入该文全部 body 索引块，不含头块
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Answer / rewrite LLM: retrieve.* → settings
        llm_key, llm_url = resolve_credentials(self.config, self.config.retrieve)
        self.llmmodel = LLM(llm_key, llm_url)
        # Embedding: retrieve.embedding_* → vectorization.* → settings
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
        # 超时/重试：优先 vectorization.retry，避免默认 connect=5s 导致检索嵌入频繁超时
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
        # Reranker：默认与 embedding 同机（xinference /v1/rerank）
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
        # 最近一次检索的改写 / 重排结果（便于调试）
        self.last_rewrite = {'original': '', 'rewritten': ''}
        self.last_rerank = {'enabled': False, 'n_in': 0, 'n_out': 0, 'scores': []}

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

    # ------------------------------------------------------------------
    # chunk / head helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # query rewrite（双路检索之前）
    # ------------------------------------------------------------------
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
        # Safety: rewrite is a trivial task; default thinking OFF
        if 'enable_thinking' not in model_args:
            model_args['enable_thinking'] = False
        return model_args

    def rewrite_query(self, query: str) -> str:
        """
        将用户查询改写得更具体，利于向量检索。
        失败时回退原文。
        """
        original = (query or '').strip()
        self.last_rewrite = {'original': original, 'rewritten': original}
        if not original:
            return original

        enable = bool(getattr(self.config.retrieve, 'enable_query_rewrite', True))
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
            # 只取第一行有效文本，去掉可能的「改写：」前缀
            text = text.splitlines()[0].strip() if text else ''
            for prefix in ('改写查询：', '改写查询:', '改写：', '改写:', '查询：', '查询:'):
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
            # 去掉首尾引号
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

    # ------------------------------------------------------------------
    # dual-path vector search
    # ------------------------------------------------------------------
    def _search_chunks_by_query(self, query_embedding, topk: int) -> list:
        """改写 query ↔ chunk 向量。"""
        raw = self.vector_match('chunk', query_embedding, topk=max(1, topk))
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
        """改写 query ↔ node 内容向量 → 映射到所属 chunk。"""
        topk = max(1, int(topk))
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
        按资料组输出：
          ----- 资料 1 【来源：xxx】 -----
          【头块】
          ...
          【索引块 1】
          ...
        """
        if not chunks_with_meta:
            return '（未检索到相关产品资料）\n'

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
                    meta_bits.append(f"检索={item['match_type']}")
                # 块得分仅用于内部排序，不写入上下文
                meta_str = (' | ' + ' | '.join(meta_bits)) if meta_bits else ''
                parts.append(
                    f'----- 资料段落 {i} 【来源：{source}】{meta_str} -----\n'
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
                f'----- 资料 {mid} 【来源：{source}】{meta_str} -----'
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
                    side.append(f'检索={mt}')
                if item.get('node_name'):
                    side.append(f"节点={item['node_name']}")
                side_str = (' (' + '; '.join(side) + ')') if side else ''

                if role == 'head':
                    block_lines.append(f'【头块】{side_str}')
                else:
                    index_i += 1
                    block_lines.append(f'【索引块 {index_i}】{side_str}')
                block_lines.append(content)
                block_lines.append('')

            parts.append('\n'.join(block_lines).rstrip() + '\n')

        # 推荐扩展：仅超边 content，无分数
        for i, item in enumerate(rec_items, start=1):
            chunk = item.get('chunk') or {}
            content = (chunk.get('content') or item.get('content') or '').strip()
            source = item.get('source') or self._source_label(chunk)
            hid = item.get('hyperedge_id')
            meta_bits = ['推荐扩展']
            if hid is not None:
                meta_bits.append(f'hyperedge_id={hid}')
            meta_str = ' | '.join(meta_bits)
            parts.append(
                f'----- 推荐扩展 {i} 【来源：{source}】 | {meta_str} -----\n'
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

    # ------------------------------------------------------------------
    # recommendation expand（一层；仅 hyperedge.content）
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # document-level rerank
    # ------------------------------------------------------------------
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

    def _rerank_materials(self, query: str, passages: list) -> list:
        """
        文档级重排：
          1) 主资料按文档聚合成完整文本（块序拼接、无标注）
          2) Qwen3-Reranker 打分排序
          3) 只保留 top_k 文档；重排 material_id = 1..k
          4) 推荐扩展不进入最终上下文（上下文仅 top_k 完整文档）
        失败时回退：按原 material 分截 top_k。
        """
        enable = bool(getattr(self.config.retrieve, 'enable_rerank', False))
        top_k = int(getattr(self.config.retrieve, 'rerank_top_k', 4) or 4)
        top_k = max(1, top_k)
        max_chars = int(getattr(self.config.retrieve, 'rerank_max_chars', -1) or -1)
        model_args = dict(
            getattr(self.config.retrieve, 'rerank_model_args', None) or {}
        )
        model_args.setdefault('model', 'Qwen3-Reranker-0.6B')

        groups = self._group_main_materials(passages)
        self.last_rerank = {
            'enabled': enable,
            'n_in': len(groups),
            'n_out': 0,
            'scores': [],
            'top_k': top_k,
        }
        if not groups:
            return []

        if not enable:
            # 关闭重排：保留全部主资料 + 推荐（调用方已拼好）
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
                f"[retrieve] rerank failed, fallback to retrieval score top_k={top_k}: {e}"
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
            self.last_rerank.update({
                'n_out': len(ranked_groups),
                'scores': scores,
                'fallback': True,
                'latency_s': round(time.perf_counter() - t0, 3),
            })
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

        self.last_rerank.update({
            'n_out': len(seen_idx),
            'scores': scores,
            'fallback': False,
            'latency_s': round(time.perf_counter() - t0, 3),
            'model': model_args.get('model'),
        })
        self.logger.info(
            f"[retrieve] rerank model={model_args.get('model')!r} "
            f"in={len(groups)} out={len(seen_idx)} top_k={top_k} "
            f"dt={self.last_rerank['latency_s']:.3f}s "
            f"top={[(s.get('source'), round(s.get('relevance_score', 0), 4)) for s in scores[:top_k]]}"
        )
        return out

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def retrive_items(
        self,
        query,
        chunk_candidate_k=None,
        node_candidate_k=None,
    ) -> list:
        """
        查询改写 → 双路各自 topk 截取 → 合并扩展去重 → 文档级重排(top_k) → 上下文。
        """
        t_all = time.perf_counter()
        self._ensure_precompute()

        chunk_cand = int(
            chunk_candidate_k
            if chunk_candidate_k is not None
            else (getattr(self.config.retrieve, 'chunk_candidate_k', None) or 10)
        )
        node_cand = int(
            node_candidate_k
            if node_candidate_k is not None
            else (getattr(self.config.retrieve, 'node_candidate_k', None) or 10)
        )
        chunk_cand = max(1, chunk_cand)
        node_cand = max(1, node_cand)
        use_cache = getattr(self.config.retrieve, 'use_cache', True)
        full_body = bool(
            getattr(self.config.retrieve, 'enable_full_body_context', False)
        )
        enable_rerank = bool(
            getattr(self.config.retrieve, 'enable_rerank', False)
        )

        # 0) 查询改写
        t0 = time.perf_counter()
        rewritten = self.rewrite_query(query)
        t_rewrite = time.perf_counter() - t0

        # 1) 对改写查询做一次 embedding，两路共用
        t0 = time.perf_counter()
        emb_resp = self.embedding.generate(
            rewritten,
            model_args=self.config.retrieve.embedding_model_args,
            use_cache=use_cache,
        )
        query_embedding = emb_resp['answer']
        t_emb = time.perf_counter() - t0
        if not isinstance(query_embedding, list):
            self.logger.error(
                f"[retrieve] embedding failed: {query_embedding!r}"
            )
            return []

        # 2) query ↔ chunk（本路 topk 截断）
        t0 = time.perf_counter()
        chunk_hits = self._search_chunks_by_query(query_embedding, topk=chunk_cand)
        t_chunk = time.perf_counter() - t0

        # 3) query ↔ node（本路 topk 截断）
        t0 = time.perf_counter()
        node_hits = self._search_nodes_by_query(query_embedding, topk=node_cand)
        t_node = time.perf_counter() - t0

        # 4) 合并去重 → 按文档成组扩展
        merged = self._merge_chunk_hits(chunk_hits, node_hits)
        passages = self._build_materials(merged)
        passages = self._enrich_passage_ids(passages)

        # 5) 可选：一层推荐扩展（仅 content；与主检索去重）
        #    若开启文档重排，推荐扩展不进最终上下文，故跳过以省开销
        if not enable_rerank:
            passages = self._expand_recommendations(passages)

        # 6) 文档级重排：完整文档纯文本拼接 → top_k
        t0 = time.perf_counter()
        passages = self._rerank_materials(rewritten or query, passages)
        t_rerank = time.perf_counter() - t0

        n_mat = len({p.get('material_id') for p in passages if p.get('material_id') is not None})
        n_rec = sum(1 for p in passages if p.get('role') == 'recommendation')
        n_head = sum(1 for p in passages if p.get('role') == 'head')
        n_index = sum(1 for p in passages if p.get('role') == 'index')
        n_merged_chunks = len(merged)
        self.logger.info(
            f"[retrieve] dual-path "
            f"rewrite={rewritten!r} "
            f"chunk_k={chunk_cand} node_k={node_cand} "
            f"full_body={full_body} rerank={enable_rerank} "
            f"chunk_hits={len(chunk_hits)} node_hits={len(node_hits)} "
            f"merged_chunks={n_merged_chunks} "
            f"materials={n_mat} heads={n_head} index={n_index} "
            f"rec_expand={n_rec} passages={len(passages)} "
            f"timing rewrite={t_rewrite:.3f}s embed={t_emb:.3f}s "
            f"chunk={t_chunk:.3f}s node={t_node:.3f}s "
            f"rerank={t_rerank:.3f}s "
            f"total={time.perf_counter()-t_all:.3f}s"
        )
        return passages

    def retrive(self, query, **kwargs):
        items = self.retrive_items(query, **kwargs)
        return self._format_retrieved_chunks(items)
