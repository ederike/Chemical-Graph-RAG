from ..module.build import BaseBuild
from collections import defaultdict
import json
import time


class Build(BaseBuild):
    """
    构图逻辑（识别 JSON 管线）：
      - 识别结果 {"head","body"} → chunk 表已有 head 块 + body_* 块
      - 每文档一头块 → 写入 hyperedge 表，hyperedge.content = head 文本
      - 每个 chunk（含 head）上的抽取实体 → node 表
      - node.chunk_id = 所属块；node.hyperedge_id = 该文档唯一超边
      - 不再构建 edge 表
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = None  # set by DHMF
        self._doc_he_temp = {}

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
        name = (chunk.get('name') or '').strip().lower()
        if name == 'head':
            return True
        extra = self._parse_extra(chunk.get('extra'))
        if extra.get('is_head') is True:
            return True
        if extra.get('role') == 'head':
            return True
        if extra.get('chunk_index') == 0:
            return True
        return False

    def _chunk_sort_key(self, chunk: dict):
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

    def _merge_entities(self, extracts) -> dict:
        all_entities = {}
        for extract in extracts or []:
            entities = extract.get('entities') or {}
            if isinstance(entities, dict):
                for name, content in entities.items():
                    if content == '':
                        continue
                    if name not in all_entities or len(str(content)) > len(str(all_entities[name])):
                        all_entities[name] = content
            elif isinstance(entities, list):
                for name in entities:
                    if name and name not in all_entities:
                        all_entities[name] = ''
        return all_entities

    def _extracts_from_chunk(self, chunk: dict) -> list:
        extra = self._parse_extra(chunk.get('extra'))
        if not isinstance(extra, dict):
            return []
        if 'extract' in extra and isinstance(extra['extract'], dict):
            return extra['extract'].get('extract') or []
        if 'entities' in extra:
            return [extra]
        return []

    def processing_single_task(self, task):
        """兼容单任务调用：实际批量逻辑在 processing。"""
        self.processing_doc_chunks([task])

    def processing_doc_chunks(self, chunks: list):
        """处理同一 doc 下的全部 extract 完成块。"""
        if not chunks:
            return

        t0 = time.perf_counter()
        chunks = sorted(chunks, key=self._chunk_sort_key)
        doc_id = chunks[0]['doc_id']

        head = None
        for c in chunks:
            if self._is_head_chunk(c):
                head = c
                break
        if head is None:
            head = chunks[0]

        he_temp = self._doc_he_temp.get(doc_id)
        if he_temp is None:
            he_temp = self.hyperedge_id_temp
            self._doc_he_temp[doc_id] = he_temp
            self.hyperedge_id_temp += 1

            if 'hyperedge' in self.config.build.target:
                # 超边内容 = 识别 JSON 的 head（头块 content）
                head_text = (head.get('content') or '').strip()
                head_entities = self._merge_entities(self._extracts_from_chunk(head))
                hyperedge = {
                    'doc_id': doc_id,
                    'chunk_id': head['id'],
                    'name': head.get('name') or 'head',
                    'content': head_text,
                    'extra': json.dumps(
                        {
                            'is_head': True,
                            'role': 'head',
                            'source': 'recognition.head',
                            'entities': head_entities,
                            'source_chunk_id': head['id'],
                        },
                        ensure_ascii=False,
                    ),
                }
                self.hyperedge_db.buffer.append(hyperedge)

        entity_count = 0
        for task in chunks:
            extracts = self._extracts_from_chunk(task)
            all_entities = self._merge_entities(extracts)
            entity_count += len(all_entities)

            if 'node' in self.config.build.target:
                for node_name, content in all_entities.items():
                    emb_text = f'{node_name}\n{content}'.strip() if content else str(node_name)
                    node = {
                        'doc_id': doc_id,
                        'chunk_id': task['id'],
                        'hyperedge_id': he_temp,
                        'name': node_name,
                        'content': content or '',
                        'embedding_content': emb_text,
                    }
                    self.node_db.buffer.append(node)

            self.chunk_db.buffer.append({
                'id': task['id'],
                'status': 'build',
            })

        if self.metrics is not None:
            self.metrics.record(
                'build',
                time.perf_counter() - t0,
                cache_hit=False,
                name=f'doc_{doc_id}',
                extra=f'chunks={len(chunks)} entities≈{entity_count}',
                log=False,
            )
        # 整文档处理完后再判断分批落库，避免同一 doc 的 node/超边被拆批
        self._maybe_flush()

    def processing(self):
        """按文档聚合：一头块一超边，全块抽节点；按 flush_every 分批写库。"""
        self.hyperedge_id_temp = 0
        self._doc_he_temp = {}
        self._flushed_count = 0
        self.hyperedge_db.buffer_clear()
        self.node_db.buffer_clear()
        self.edge_db.buffer_clear()
        self.chunk_db.buffer_clear()

        by_doc = defaultdict(list)
        for task in self.tasks:
            by_doc[task['doc_id']].append(task)

        from tqdm import tqdm
        for doc_id in tqdm(sorted(by_doc.keys()), desc='build', unit='doc'):
            self.processing_doc_chunks(by_doc[doc_id])
