from ..module.chunk import BaseChunk
import json
import re
import time
import tiktoken


class Chunk(BaseChunk):
    """
    分块（summary 之后）：
      - hyperedge.content → 1 个头块 name=head（不分块）
      - doc.content（全文识别）→ 按 token 切成 body_1, body_2, …
      - 切块逻辑：按句子打包到 max_tokens，相邻块保留 overlap_ratio 重叠
    """

    def __init__(self, db, logger, config):
        super().__init__(db, logger, config)
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

    def split_document_into_chunks(self, document, max_tokens=512, overlap_ratio=0.1):
        """
        按句子打包到 max_tokens；相邻块保留 overlap_ratio * max_tokens 的尾部重叠。
        """
        encoding = tiktoken.get_encoding('cl100k_base')
        max_tokens = max(1, int(max_tokens or 512))
        overlap_ratio = self._normalize_overlap_ratio(overlap_ratio)
        overlap_tokens = int(max_tokens * overlap_ratio) if overlap_ratio > 0 else 0
        step = max(1, max_tokens - overlap_tokens) if overlap_tokens > 0 else max_tokens

        sentences = re.split(r'(?<=[。！？\.\!\?，,；;])', document)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_token_count = 0

        def _emit_and_keep_overlap():
            nonlocal current_chunk, current_token_count
            if not current_chunk:
                return
            chunks.append(''.join(current_chunk))
            if overlap_tokens <= 0:
                current_chunk = []
                current_token_count = 0
                return
            tail = []
            tail_tokens = 0
            for s in reversed(current_chunk):
                st = len(encoding.encode(s))
                if tail and tail_tokens + st > overlap_tokens:
                    break
                if not tail and st > overlap_tokens:
                    s_tokens = encoding.encode(s)
                    piece = encoding.decode(s_tokens[-overlap_tokens:]).strip()
                    current_chunk = [piece] if piece else []
                    current_token_count = len(encoding.encode(piece)) if piece else 0
                    return
                tail.insert(0, s)
                tail_tokens += st
            current_chunk = tail
            current_token_count = tail_tokens

        for sentence in sentences:
            sentence_tokens = len(encoding.encode(sentence))
            if current_token_count + sentence_tokens <= max_tokens:
                current_chunk.append(sentence)
                current_token_count += sentence_tokens
                continue

            if current_chunk:
                _emit_and_keep_overlap()

            if sentence_tokens > max_tokens:
                tokens = encoding.encode(sentence)
                prefix_tokens = []
                if current_chunk:
                    prefix_tokens = encoding.encode(''.join(current_chunk))
                    current_chunk = []
                    current_token_count = 0
                room = max(1, max_tokens - len(prefix_tokens))
                first = encoding.decode(prefix_tokens + tokens[:room]).strip()
                if first:
                    chunks.append(first)
                start = max(0, room - overlap_tokens) if overlap_tokens > 0 else room
                while start < len(tokens):
                    end = min(len(tokens), start + max_tokens)
                    piece = encoding.decode(tokens[start:end]).strip()
                    if piece and (not chunks or piece != chunks[-1]):
                        chunks.append(piece)
                    if end >= len(tokens):
                        break
                    start += step
                if overlap_tokens > 0 and chunks:
                    tail = encoding.decode(
                        encoding.encode(chunks[-1])[-overlap_tokens:]
                    ).strip()
                    current_chunk = [tail] if tail else []
                    current_token_count = len(encoding.encode(tail)) if tail else 0
                else:
                    current_chunk = []
                    current_token_count = 0
            else:
                if current_chunk and current_token_count + sentence_tokens > max_tokens:
                    current_chunk = [sentence]
                    current_token_count = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_token_count += sentence_tokens

        if current_chunk:
            text = ''.join(current_chunk).strip()
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

        rows = self.chunk_db.db.execute("SELECT MAX(id) FROM chunk")
        max_id_before = (
            rows[0]['MAX(id)']
            if rows and rows[0]['MAX(id)'] is not None
            else 0
        )

        pending = list(self._pending_he_bind or [])
        he_by_doc = {p['doc_id']: p['hyperedge_id'] for p in pending}
        head_bind = []
        next_id = max_id_before
        for ch in self.chunk_db.buffer:
            next_id += 1
            if (ch.get('name') or '').strip().lower() == 'head':
                did = ch.get('doc_id')
                hid = he_by_doc.get(did)
                if hid is not None:
                    head_bind.append({'id': hid, 'chunk_id': next_id})

        if self.doc_db.buffer:
            self.doc_db.update(self.doc_db.buffer)
            self.doc_db.buffer_clear()
        if self.chunk_db.buffer:
            self.chunk_db.add(self.chunk_db.buffer)
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
