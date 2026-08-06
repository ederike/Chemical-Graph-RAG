import tiktoken
from tqdm import tqdm
from typing import Dict
import logging
from ..utils.utils import hash_str
from ..utils.database import BaseDB
from ..utils.config import Config


class BaseDoc:
    def __init__(self, db: Dict[str, BaseDB], logger: logging.Logger, config: Config):
        self.config = config
        self.doc_db = db['doc']
        self.logger = logger
        self._flushed_count = 0

        if self.config.doc.count_token:
            self.tokener = tiktoken.get_encoding("cl100k_base")

    def _flush_every(self) -> int:
        try:
            n = int(getattr(self.config.doc, 'flush_every', 1000) or 1000)
        except (TypeError, ValueError):
            n = 1000
        return max(1, n)

    def processing_single_task(self, task):
        text = task['content']
        file_name = task['name']
        hash = hash_str(text)
        add_doc = {
            'name': file_name,
            'content': text,
            'hash': hash,
        }
        if self.config.doc.count_token:
            add_doc['tokens'] = len(self.tokener.encode(text))
        self.doc_db.buffer.append(add_doc)
        self._maybe_flush()

    def _maybe_flush(self, force: bool = False):
        n = len(self.doc_db.buffer)
        if n <= 0:
            return
        if not force and n < self._flush_every():
            return
        self.save()

    def prepare(self, doc_list):
        self.tasks = doc_list
        self.doc_db.buffer_clear()
        self._flushed_count = 0
        self.logger.debug(f"Number of inserted documents :{len(self.tasks)}")

    def processing(self):
        for task in tqdm(self.tasks):
            self.processing_single_task(task)

    def save(self):
        buf = self.doc_db.buffer
        if not buf:
            return
        self.doc_db.add(buf)
        self._flushed_count += len(buf)
        self.logger.info(
            f"[doc] flush n={len(buf)} total_flushed={self._flushed_count}"
        )
        self.doc_db.buffer_clear()

    def clear(self):
        self.doc_db.clear()
