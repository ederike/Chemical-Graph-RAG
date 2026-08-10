from typing import Dict
import logging
from ..utils.database import BaseDB
from ..utils.config import Config

from collections import defaultdict
import json
import time

class Build:
    """
    构图（summary 已写入 hyperedge）：
      - 每文档已有唯一超边（content = LLM 总结）
      - 每个 chunk（含 head）上的抽取实体 → node 表
      - node.chunk_id = 所属块；node.hyperedge_id = 该文档已有超边真实 id
      - 不再新建 hyperedge 行；可选更新 hyperedge.extra 中的实体聚合
      - 不再构建 edge 表
    """
    def _flush_every(self) -> int:
        try:
            n = int(getattr(self.config.build, 'flush_every', 1000) or 1000)
        except (TypeError, ValueError):
            n = 1000
        return max(1, n)

    def _buffer_item_count(self) -> int:
        return (
            len(self.hyperedge_db.buffer)
            + len(self.node_db.buffer)
            + len(self.edge_db.buffer)
        )

    def _maybe_flush(self, force: bool = False):
        n = self._buffer_item_count()
        if n <= 0 and not self.chunk_db.buffer:
            return
        if not force and n < self._flush_every():
            return
        self.save()

    def prepare(self):
        if self.config.settings.debug:
            self.tasks = self.chunk_db.search_all()
        else:
            self.tasks = self.chunk_db.search('status', "extract")
        self.hyperedge_db.buffer_clear()
        self.node_db.buffer_clear()
        self.edge_db.buffer_clear()
        self.chunk_db.buffer_clear()
        self._flushed_count = 0
        self.hyperedge_id_temp = 0
        self.logger.debug(f"The number of chunks to be built :{len(self.tasks)}")

    def __init__(self, db: Dict[str, BaseDB], logger: logging.Logger, config: Config):
        self.config = config
        self.logger = logger
        self.chunk_db = db['chunk']
        self.hyperedge_db = db['hyperedge']
        self.node_db = db['node']
        self.edge_db = db['edge']
        self._flushed_count = 0
        self.hyperedge_id_temp = 0
        self.metrics = None
        self._doc_he_cache = {}

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
                    if name not in all_entities or len(str(content)) > len(
                        str(all_entities[name])
                    ):
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

    def _hyperedge_for_doc(self, doc_id):
        if doc_id in self._doc_he_cache:
            return self._doc_he_cache[doc_id]
        rows = self.hyperedge_db.search('doc_id', doc_id) or []
        he = rows[0] if rows else None
        self._doc_he_cache[doc_id] = he
        return he

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

        he = self._hyperedge_for_doc(doc_id)
        if he is None:
            self.logger.error(
                f"build: no hyperedge for doc_id={doc_id}; "
                f"run summary before build. Skip."
            )
            return
        he_id = he['id']

        head = None
        for c in chunks:
            if self._is_head_chunk(c):
                head = c
                break
        if head is None:
            head = chunks[0]

        if 'hyperedge' in self.config.build.target:
            head_entities = self._merge_entities(self._extracts_from_chunk(head))
            prev_extra = self._parse_extra(he.get('extra'))
            prev_extra.update({
                'is_head': True,
                'role': 'head',
                'source': prev_extra.get('source') or 'summary',
                'entities': head_entities,
                'source_chunk_id': head['id'],
            })
            self.hyperedge_db.buffer.append({
                'id': he_id,
                'chunk_id': head['id'],
                'extra': json.dumps(prev_extra, ensure_ascii=False),
            })

        entity_count = 0
        for task in chunks:
            extracts = self._extracts_from_chunk(task)
            all_entities = self._merge_entities(extracts)
            entity_count += len(all_entities)

            if 'node' in self.config.build.target:
                for node_name, content in all_entities.items():
                    emb_text = (
                        f'{node_name}\n{content}'.strip()
                        if content
                        else str(node_name)
                    )
                    node = {
                        'doc_id': doc_id,
                        'chunk_id': task['id'],
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
        self._maybe_flush()

    def save(self):
        n_he = len(self.hyperedge_db.buffer)
        n_node = len(self.node_db.buffer)
        n_edge = len(self.edge_db.buffer)
        n_chunk = len(self.chunk_db.buffer)
        if n_he <= 0 and n_node <= 0 and n_edge <= 0 and n_chunk <= 0:
            return

        if self.chunk_db.buffer:
            self.chunk_db.update(self.chunk_db.buffer)
            self.chunk_db.buffer_clear()

        if self.hyperedge_db.buffer:
            self.hyperedge_db.update(self.hyperedge_db.buffer)
            self.hyperedge_db.buffer_clear()

        if self.node_db.buffer:
            self.node_db.add(self.node_db.buffer)
            self.node_db.buffer_clear()

        if self.edge_db.buffer:
            self.edge_db.add(self.edge_db.buffer)
            self.edge_db.buffer_clear()

        flushed = n_he + n_node + n_edge
        self._flushed_count += flushed
        # self.logger.info(
        #     f"[build] flush hyperedge_upd={n_he} node={n_node} edge={n_edge} "
        #     f"chunk_status={n_chunk} total_items_flushed={self._flushed_count}"
        # )

    def clear(self):
        """
        清除构图产物（node/edge），重置 chunk status→extract。
        不删除 hyperedge（由 summary 维护）；不碰 doc/chunk 正文。
        """
        self.chunk_db.update_key('status', 'extract')
        self.node_db.clear()
        self.edge_db.clear()

    def processing(self):
        """按文档聚合：绑定已有超边，全块抽节点；按 flush_every 分批写库。"""
        self._doc_he_cache = {}
        self._flushed_count = 0
        self.hyperedge_db.buffer_clear()
        self.node_db.buffer_clear()
        self.edge_db.buffer_clear()
        self.chunk_db.buffer_clear()

        by_doc = defaultdict(list)
        for task in self.tasks:
            by_doc[task['doc_id']].append(task)

        from tqdm import tqdm
        from ..utils.utils import TQDM_BAR_FORMAT

        doc_ids = sorted(by_doc.keys())
        n = len(doc_ids)
        bar = tqdm(
            doc_ids,
            desc='build',
            unit='doc',
            bar_format=TQDM_BAR_FORMAT,
        )
        for doc_id in bar:
            self.processing_doc_chunks(by_doc[doc_id])
            bar.set_postfix(total=n)
