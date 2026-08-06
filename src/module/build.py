from tqdm import tqdm
import json
import logging
from typing import Dict
from ..utils.database import BaseDB
from ..utils.config import Config


class BaseBuild:
    def __init__(self, db: Dict[str, BaseDB], logger: logging.Logger, config: Config):
        self.config = config
        self.logger = logger

        self.chunk_db = db['chunk']
        self.hyperedge_db = db['hyperedge']
        self.node_db = db['node']
        self.edge_db = db['edge']
        self._flushed_count = 0
        self.hyperedge_id_temp = 0

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

    def prepare_save(self):
        """
        将本批 buffer 内的临时 hyperedge_id 映射为即将写入的真实自增 id。
        约定：每批 flush 后 hyperedge_id_temp 归零，本批 hyperedge 按 temp 0..n-1 顺序 append。
        """
        hyperedge_ids_map = {}
        if 'hyperedge' in self.config.build.target:
            rows = self.hyperedge_db.db.execute("SELECT MAX(id) FROM hyperedge")
            max_hyperedge_id = rows[0]['MAX(id)'] if rows else None
            if max_hyperedge_id is None:
                max_hyperedge_id = 0
            n_he = len(self.hyperedge_db.buffer)
            # 优先按本批 hyperedge 条数顺序映射 0..n-1
            for index in range(n_he):
                hyperedge_ids_map[index] = max_hyperedge_id + 1 + index
            # 若节点上仍有未落在 0..n-1 的 temp id（异常），按出现序补映射
            extra_temps = set()
            for node in self.node_db.buffer:
                tid = node.get('hyperedge_id')
                if tid is not None and tid not in hyperedge_ids_map:
                    extra_temps.add(tid)
            for edge in self.edge_db.buffer:
                tid = edge.get('hyperedge_id')
                if tid is not None and tid not in hyperedge_ids_map:
                    extra_temps.add(tid)
            next_id = max_hyperedge_id + 1 + n_he
            for tid in sorted(extra_temps):
                hyperedge_ids_map[tid] = next_id
                next_id += 1

        if 'node' in self.config.build.target:
            for node in self.node_db.buffer:
                node['hyperedge_id'] = hyperedge_ids_map.get(node['hyperedge_id'], None)

        if 'edge' in self.config.build.target:
            for edge in self.edge_db.buffer:
                edge['hyperedge_id'] = hyperedge_ids_map.get(edge['hyperedge_id'], None)

    def processing_single_task(self, task):
        extra = task['extra']

        extra = json.loads(extra)
        doc_id = task['doc_id']
        chunk_id = task['id']
        extracts = extra['extract']['extract']

        for extract in extracts:
            entities = extract.get('entities') or {}
            if 'hyperedge' in self.config.build.target:
                # 不再依赖 knowledge；内容优先用 chunk 原文
                hyperedge = {
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'content': task.get('content') or extract.get('knowledge') or '',
                    'extra': json.dumps({'entities': entities}, ensure_ascii=False),
                }
                self.hyperedge_db.buffer.append(hyperedge)

            if 'node' in self.config.build.target:
                if isinstance(entities, dict):
                    node_iter = entities.items()
                    for name, content in node_iter:
                        node = {
                            'doc_id': doc_id,
                            'chunk_id': chunk_id,
                            'hyperedge_id': self.hyperedge_id_temp,
                            'name': name,
                            'content': content or '',
                        }
                        self.node_db.buffer.append(node)
                else:
                    # 兼容旧 list 实体名
                    for name in entities:
                        node = {
                            'doc_id': doc_id,
                            'chunk_id': chunk_id,
                            'hyperedge_id': self.hyperedge_id_temp,
                            'name': name,
                            'content': '',
                        }
                        self.node_db.buffer.append(node)

            if 'edge' in self.config.build.target:
                names = list(entities.keys()) if isinstance(entities, dict) else list(entities or [])
                for test in range(max(0, len(names) - 1)):
                    edge = {
                        'doc_id': doc_id,
                        'chunk_id': chunk_id,
                        'hyperedge_id': self.hyperedge_id_temp,
                        'name': "aaa",
                    }
                    self.edge_db.buffer.append(edge)

            update_chunk = {
                'id': chunk_id,
                'status': 'build',
            }
            self.chunk_db.buffer.append(update_chunk)
            self.hyperedge_id_temp += 1

        self._maybe_flush()

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

    def processing(self):
        self.hyperedge_id_temp = 0
        for task in tqdm(self.tasks):
            self.processing_single_task(task)

    def save(self):
        n_he = len(self.hyperedge_db.buffer)
        n_node = len(self.node_db.buffer)
        n_edge = len(self.edge_db.buffer)
        n_chunk = len(self.chunk_db.buffer)
        if n_he <= 0 and n_node <= 0 and n_edge <= 0 and n_chunk <= 0:
            return

        self.prepare_save()

        if self.chunk_db.buffer:
            self.chunk_db.update(self.chunk_db.buffer)
            self.chunk_db.buffer_clear()

        if self.hyperedge_db.buffer:
            self.hyperedge_db.add(self.hyperedge_db.buffer)
            self.hyperedge_db.buffer_clear()

        if self.node_db.buffer:
            self.node_db.add(self.node_db.buffer)
            self.node_db.buffer_clear()

        if self.edge_db.buffer:
            self.edge_db.add(self.edge_db.buffer)
            self.edge_db.buffer_clear()

        flushed = n_he + n_node + n_edge
        self._flushed_count += flushed
        # 下一批临时 id 从 0 重新编号
        self.hyperedge_id_temp = 0
        if hasattr(self, '_doc_he_temp'):
            self._doc_he_temp = {}
        self.logger.info(
            f"[build] flush hyperedge={n_he} node={n_node} edge={n_edge} "
            f"chunk_status={n_chunk} total_items_flushed={self._flushed_count}"
        )

    def clear(self):
        self.chunk_db.update_key('status', 'extract')
        self.hyperedge_db.clear()
        self.node_db.clear()
        self.edge_db.clear()
