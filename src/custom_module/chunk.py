from ..module.chunk import BaseChunk
import json
import re
import time
import tiktoken


class Chunk(BaseChunk):
    """
    基于 PDF 识别 JSON 的分块：
      - doc.extra / content 提供 {"head","body"}
      - head → 1 个头块（name=head，写入 chunk 表；后续 build 写入 hyperedge）
      - body → 按 token 切成 body_1, body_2, … 索引块
      - head 与 body 块均进入后续 extract
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = None  # set by DHMF

    def split_document_into_chunks(self, document, max_tokens=512):
        encoding = tiktoken.get_encoding('cl100k_base')

        sentences = re.split(r'(?<=[。！？\.\!\?，,；;])', document)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_token_count = 0

        for sentence in sentences:
            sentence_tokens = len(encoding.encode(sentence))
            if current_token_count + sentence_tokens <= max_tokens:
                current_chunk.append(sentence)
                current_token_count += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                # 单句超长：硬切
                if sentence_tokens > max_tokens:
                    tokens = encoding.encode(sentence)
                    for i in range(0, len(tokens), max_tokens):
                        piece = encoding.decode(tokens[i:i + max_tokens]).strip()
                        if piece:
                            chunks.append(piece)
                    current_chunk = []
                    current_token_count = 0
                else:
                    current_chunk = [sentence]
                    current_token_count = sentence_tokens

        if current_chunk:
            chunks.append(''.join(current_chunk))

        return chunks

    @staticmethod
    def _parse_json_obj(raw) -> dict:
        if not raw:
            return {}
        if isinstance(raw, dict):
            return raw
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def resolve_head_body(self, task: dict) -> tuple:
        """
        从 doc 行解析 head / body。
        优先级：
          1) task['extra'] 中的 head/body
          2) task['content'] 作为 {"head","body"} JSON
          3) 顶层 task['head'] / task['body']
          4) 整段 content 当作 body（head 为空）——兜底，保证可分块
        """
        extra = self._parse_json_obj(task.get('extra'))
        head = (extra.get('head') if isinstance(extra.get('head'), str) else None)
        body = (extra.get('body') if isinstance(extra.get('body'), str) else None)

        if head is None and body is None:
            content = task.get('content') or ''
            try:
                from .doc import Doc as _Doc
                parsed = _Doc.parse_recognition_json(content)
            except Exception:
                parsed = None
            if parsed is not None:
                head = parsed.get('head') or ''
                body = parsed.get('body') or ''
            else:
                head = task.get('head') if isinstance(task.get('head'), str) else ''
                body = task.get('body') if isinstance(task.get('body'), str) else ''
                if not head and not body:
                    body = (content or '').strip()
                    head = ''

        head = (head or '').strip()
        body = (body or '').strip()
        return head, body

    def processing_single_task(self, task):
        t0 = time.perf_counter()
        doc_id = task['id']
        max_tokens = int(getattr(self.config.chunk, 'chunk_size_max', 512) or 512)

        head, body = self.resolve_head_body(task)

        pieces = []  # list of (content, is_head, chunk_index)

        # 头块：始终写入一块（即使 head 为空也占位，保证每文档有超边源）
        if head:
            pieces.append((head, True, 0))
        elif body:
            # head 缺失：用 body 首段充当头块，其余为索引块
            body_parts = self.split_document_into_chunks(body, max_tokens=max_tokens)
            if body_parts:
                pieces.append((body_parts[0], True, 0))
                body = '\n'.join(body_parts[1:])
            else:
                pieces.append((body, True, 0))
                body = ''
        else:
            pieces.append(('', True, 0))

        if body:
            body_parts = self.split_document_into_chunks(body, max_tokens=max_tokens)
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
            }
            add_chunk = {
                'doc_id': doc_id,
                'content': content or '',
                'name': 'head' if is_head else f'body_{chunk_index}',
                'extra': json.dumps(extra, ensure_ascii=False),
            }
            if self.config.chunk.count_token:
                add_chunk['tokens'] = len(self.tokener.encode(add_chunk['content'] or ''))
            self.chunk_db.buffer.append(add_chunk)

        update_doc = {
            'id': doc_id,
            'status': 'chunk',
        }
        self.doc_db.buffer.append(update_doc)

        if self.metrics is not None:
            self.metrics.record(
                'chunk',
                time.perf_counter() - t0,
                cache_hit=False,
                name=task.get('name') or f'doc_{doc_id}',
                extra=f'n_chunks={len(pieces)} head=1 body={max(0, len(pieces)-1)}',
                log=False,
            )
