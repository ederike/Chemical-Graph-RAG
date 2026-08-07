from typing import Dict
import logging
from ..utils.database import BaseDB
from ..utils.config import Config
from tqdm import tqdm
from ..utils.utils import TQDM_BAR_FORMAT

import json
import re
import time
import tiktoken

class Chunk:
    """
    分块（summary 之后）：
      - hyperedge.content → 1 个头块 name=head（不分块）
      - doc.content（全文识别）→ 按 token 切成 body_1, body_2, …
      - 切块逻辑：按句子打包到 max_tokens，相邻块保留 overlap_ratio 重叠
    """
    def _flush_every(self) -> int:
        try:
            n = int(getattr(self.config.chunk, 'flush_every', 1000) or 1000)
        except (TypeError, ValueError):
            n = 1000
        return max(1, n)

    def _maybe_flush(self, force: bool = False):
        n = len(self.chunk_db.buffer)
        if n <= 0 and not self.doc_db.buffer:
            return
        if not force and n < self._flush_every():
            return
        self.save()

    def processing(self):
        n = len(self.tasks)
        bar = tqdm(
            self.tasks,
            desc='chunk',
            unit='doc',
            bar_format=TQDM_BAR_FORMAT,
        )
        for task in bar:
            self.processing_single_task(task)
            bar.set_postfix(total=n)

    def __init__(self, db: Dict[str, BaseDB], logger: logging.Logger, config: Config):
        self.config = config
        self.doc_db = db['doc']
        self.chunk_db = db['chunk']
        self.logger = logger
        self._flushed_count = 0
        self.tokener = tiktoken.get_encoding('cl100k_base')
        self.hyperedge_db = db['hyperedge']
        self.metrics = None
        self._pending_he_bind = []

    @staticmethod
    def _normalize_overlap_ratio(overlap_ratio) -> float:
        try:
            x = float(overlap_ratio or 0.0)
        except (TypeError, ValueError):
            x = 0.0
        if x > 1.0:
            x = x / 100.0
        return max(0.0, min(x, 0.9))

    @staticmethod
    def _join_units(units: list) -> str:
        """Join sentence units without gluing English words together."""
        if not units:
            return ''
        out = units[0]
        for u in units[1:]:
            if not u:
                continue
            if not out:
                out = u
                continue
            if re.search(r'[A-Za-z0-9]$', out) and re.search(r'^[A-Za-z0-9]', u):
                out = f'{out} {u}'
            else:
                out = f'{out}{u}'
        return out.strip()

    @staticmethod
    def _split_sentences(document: str) -> list:
        """
        Split on sentence terminators and newlines only.
        Do not split on commas/semicolons — that produced mid-phrase cuts
        and made overlaps look broken.
        """
        text = (document or '').replace('\r\n', '\n').replace('\r', '\n')
        if not text.strip():
            return []
        parts = re.split(r'(?<=[。！？.!?])\s*|\n+', text)
        return [p.strip() for p in parts if p and p.strip()]

    def split_document_into_chunks(self, document, max_tokens=512, overlap_ratio=0.1):
        """
        Pack sentences into chunks of at most max_tokens.
        Adjacent chunks share ~overlap_ratio * max_tokens of trailing content
        (prefer whole sentences; fall back to token-level tail when needed).
        """
        encoding = tiktoken.get_encoding('cl100k_base')
        max_tokens = max(1, int(max_tokens or 512))
        overlap_ratio = self._normalize_overlap_ratio(overlap_ratio)
        overlap_tokens = int(max_tokens * overlap_ratio) if overlap_ratio > 0 else 0
        step = max(1, max_tokens - overlap_tokens) if overlap_tokens > 0 else max_tokens

        def n_tokens(text: str) -> int:
            return len(encoding.encode(text or ''))

        def tail_by_tokens(text: str, budget: int) -> str:
            """
            Longest character suffix of text whose token count <= budget.
            Guarantees a true string suffix (safe for next-chunk prefix match).
            Avoids tiktoken mid-token decode artifacts (U+FFFD).
            """
            text = text or ''
            if not text or budget <= 0:
                return ''
            if n_tokens(text) <= budget:
                return text
            lo, hi = 1, len(text)
            best = ''
            while lo <= hi:
                mid = (lo + hi) // 2
                suf = text[-mid:]
                if n_tokens(suf) <= budget:
                    best = suf
                    lo = mid + 1
                else:
                    hi = mid - 1
            return best

        def window_by_tokens(text: str, start_hint: int, budget: int) -> tuple:
            """
            Take a substring of text starting near start_hint with <= budget tokens.
            Returns (piece, char_end_index).
            """
            if not text or budget <= 0:
                return '', start_hint
            # Expand from start_hint until token budget is exhausted
            lo = max(0, min(start_hint, len(text) - 1))
            # Binary search end index
            end_lo, end_hi = lo + 1, len(text)
            best_end = lo
            while end_lo <= end_hi:
                mid = (end_lo + end_hi) // 2
                piece = text[lo:mid]
                if piece and n_tokens(piece) <= budget:
                    best_end = mid
                    end_lo = mid + 1
                else:
                    end_hi = mid - 1
            piece = text[lo:best_end].strip()
            return piece, best_end

        # 1) sentence units; expand oversize units with sliding char windows
        units = []
        for sent in self._split_sentences(document):
            if n_tokens(sent) <= max_tokens:
                units.append(sent)
                continue
            # Sliding windows by approximate token step
            pos = 0
            while pos < len(sent):
                piece, end = window_by_tokens(sent, pos, max_tokens)
                if not piece:
                    # force at least one char forward to avoid infinite loop
                    pos += 1
                    continue
                units.append(piece)
                if end >= len(sent):
                    break
                # step forward by ~ (max_tokens - overlap) tokens ≈ char ratio
                if overlap_tokens > 0 and overlap_tokens < max_tokens:
                    # next window starts so that tail ~overlap_tokens is retained
                    ov = tail_by_tokens(piece, overlap_tokens)
                    # restart near end of piece minus overlap chars
                    pos = max(pos + 1, end - len(ov) if ov else end)
                else:
                    pos = end

        if not units:
            return []

        chunks = []
        cur = []
        cur_tok = 0

        def _keep_overlap_tail(source_units: list) -> tuple:
            """
            Keep trailing units totaling ~overlap_tokens for the next chunk.
            Prefer whole sentences so boundaries stay readable.
            """
            if overlap_tokens <= 0 or not source_units:
                return [], 0

            last_ut = n_tokens(source_units[-1])
            soft_budget = overlap_tokens
            # Prefer whole last sentence when it leaves room for new content
            prefer_unit_budget = max(overlap_tokens, (max_tokens * 2) // 3)
            if last_ut <= prefer_unit_budget:
                soft_budget = max(soft_budget, last_ut)

            tail = []
            ttok = 0
            for u in reversed(source_units):
                ut = n_tokens(u)
                if tail and ttok + ut > soft_budget:
                    break
                if not tail and ut > soft_budget:
                    piece = tail_by_tokens(u, soft_budget)
                    if not piece:
                        return [], 0
                    return [piece], n_tokens(piece)
                tail.insert(0, u)
                ttok += ut
            return tail, ttok

        def _emit():
            nonlocal cur, cur_tok
            if not cur:
                return
            text = self._join_units(cur)
            if text:
                chunks.append(text)
            cur, cur_tok = _keep_overlap_tail(cur)

        for u in units:
            ut = n_tokens(u)
            if cur and cur_tok + ut > max_tokens:
                _emit()
                # After keeping overlap, still no room for u: shrink overlap to fit
                if cur and cur_tok + ut > max_tokens:
                    if ut >= max_tokens:
                        cur, cur_tok = [], 0
                    else:
                        room = max_tokens - ut
                        joined = self._join_units(cur)
                        piece = tail_by_tokens(joined, room)
                        if piece:
                            cur, cur_tok = [piece], n_tokens(piece)
                        else:
                            cur, cur_tok = [], 0

            cur.append(u)
            cur_tok += ut
            if cur_tok >= max_tokens:
                _emit()

        if cur:
            text = self._join_units(cur)
            if text and (not chunks or text != chunks[-1]):
                chunks.append(text)

        return chunks

    def resolve_head_body(self, task: dict) -> tuple:
        """
        head = 该文档超边 content（summary 写入）
        body = doc.content（全文识别）
        返回 (head, body, hyperedge_row_or_None)
        """
        doc_id = task.get('id')
        body = (task.get('content') or '').strip()
        head = ''
        he_rows = self.hyperedge_db.search('doc_id', doc_id) or []
        he_row = he_rows[0] if he_rows else None
        if he_row:
            head = (he_row.get('content') or '').strip()
        return head, body, he_row

    def processing_single_task(self, task):
        t0 = time.perf_counter()
        doc_id = task['id']
        max_tokens = int(getattr(self.config.chunk, 'chunk_size_max', 512) or 512)
        overlap_ratio = self._normalize_overlap_ratio(
            getattr(self.config.chunk, 'chunk_overlap', 0.1)
        )

        head, body, he_row = self.resolve_head_body(task)
        if not head and not body:
            self.logger.warning(
                f"Skip chunk doc_id={doc_id} name={task.get('name')!r}: "
                f"empty head and body"
            )
            return
        if he_row is None:
            self.logger.warning(
                f"Doc {task.get('name') or doc_id}: no hyperedge; "
                f"run summary first. Skip."
            )
            return

        pieces = []  # (content, is_head, chunk_index)

        pieces.append((head or '', True, 0))

        if body:
            body_parts = self.split_document_into_chunks(
                body, max_tokens=max_tokens, overlap_ratio=overlap_ratio
            )
            for i, part in enumerate(body_parts, start=1):
                if part and part.strip():
                    pieces.append((part.strip(), False, i))

        for content, is_head, chunk_index in pieces:
            if not content and not is_head:
                continue
            extra = {
                'is_head': bool(is_head),
                'chunk_index': int(chunk_index),
                'role': 'head' if is_head else 'body',
                'chunk_overlap': overlap_ratio,
                'source': 'hyperedge.summary' if is_head else 'doc.recognition',
            }
            add_chunk = {
                'doc_id': doc_id,
                'content': content or '',
                'name': 'head' if is_head else f'body_{chunk_index}',
                'extra': json.dumps(extra, ensure_ascii=False),
            }
            if self.config.chunk.count_token:
                add_chunk['tokens'] = len(
                    self.tokener.encode(add_chunk['content'] or '')
                )
            self.chunk_db.buffer.append(add_chunk)

        self.doc_db.buffer.append({
            'id': doc_id,
            'status': 'chunk',
        })

        self._pending_he_bind.append({
            'hyperedge_id': he_row['id'],
            'doc_id': doc_id,
        })

        if self.metrics is not None:
            self.metrics.record(
                'chunk',
                time.perf_counter() - t0,
                cache_hit=False,
                name=task.get('name') or f'doc_{doc_id}',
                extra=(
                    f'n_chunks={len(pieces)} head=1 '
                    f'body={max(0, len(pieces) - 1)}'
                ),
                log=False,
            )
        self._maybe_flush()

    def prepare(self):
        """Only docs already summarized (status=summary), unless debug."""
        if self.config.settings.debug:
            summarized = self.doc_db.search('status', 'summary') or []
            chunked = self.doc_db.search('status', 'chunk') or []
            all_docs = self.doc_db.search_all() or []
            seen = set()
            tasks = []
            for r in list(summarized) + list(chunked) + list(all_docs):
                rid = r.get('id')
                if rid in seen:
                    continue
                he = self.hyperedge_db.search('doc_id', rid) or []
                if not he:
                    continue
                seen.add(rid)
                tasks.append(r)
            self.tasks = tasks
        else:
            self.tasks = self.doc_db.search('status', 'summary') or []
        self.doc_db.buffer_clear()
        self.chunk_db.buffer_clear()
        self._flushed_count = 0
        self._pending_he_bind = []
        self.logger.debug(f"The number of documents to be chunk: {len(self.tasks)}")

    def save(self):
        n_chunk = len(self.chunk_db.buffer)
        n_doc = len(self.doc_db.buffer)
        if n_chunk <= 0 and n_doc <= 0:
            return

        pending = list(self._pending_he_bind or [])
        he_by_doc = {p['doc_id']: p['hyperedge_id'] for p in pending}
        head_bind = []

        if self.doc_db.buffer:
            self.doc_db.update(self.doc_db.buffer)
            self.doc_db.buffer_clear()

        if self.chunk_db.buffer:
            # INSERT then lastrowid — never predict ids with MAX(id)+offset
            chunk_rows = list(self.chunk_db.buffer)
            inserted_ids = self.chunk_db.add(chunk_rows, return_ids=True) or []
            if len(inserted_ids) != len(chunk_rows):
                self.logger.warning(
                    f"[chunk] return_ids length mismatch: "
                    f"rows={len(chunk_rows)} ids={len(inserted_ids)}"
                )
            for ch, cid in zip(chunk_rows, inserted_ids):
                if not cid:
                    continue
                if (ch.get('name') or '').strip().lower() != 'head':
                    continue
                did = ch.get('doc_id')
                hid = he_by_doc.get(did)
                if hid is not None:
                    head_bind.append({'id': hid, 'chunk_id': int(cid)})
            self._flushed_count += n_chunk
            self.chunk_db.buffer_clear()

        if head_bind:
            self.hyperedge_db.update(head_bind)

        self._pending_he_bind = []
        self.logger.info(
            f"[chunk] flush chunks={n_chunk} docs={n_doc} "
            f"he_bind={len(head_bind)} total_chunks_flushed={self._flushed_count}"
        )

    def clear(self):
        """
        清空 chunk 表；有超边的 doc → status=summary，否则 → new。
        """
        self.chunk_db.clear()
        docs = self.doc_db.search_all() or []
        updates = []
        for d in docs:
            he = self.hyperedge_db.search('doc_id', d['id']) or []
            updates.append({
                'id': d['id'],
                'status': 'summary' if he else 'new',
            })
        if updates:
            self.doc_db.update(updates)
        self.logger.info(
            f"[chunk] clear: chunks wiped, docs status reset n={len(updates)}"
        )
