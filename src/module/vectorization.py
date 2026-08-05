from tqdm import tqdm
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..utils.OpenAIAPI import Embedding
from ..utils.database import BaseDB,BaseVDB
from ..utils.utils import Retry
from ..utils.config import Config, resolve_credentials

class BaseVectorization:
    def __init__(self,logger:logging.Logger,config:Config):
        self.config = config
        self.logger = logger
        # 边嵌边写批量：config.vectorization.flush_every（默认 500）
        try:
            fe = int(getattr(config.vectorization, 'flush_every', 500) or 500)
        except (TypeError, ValueError):
            fe = 500
        self.flush_every = max(1, fe)
        self._buffer_lock = threading.Lock()
        self._flushed_count = 0
        api_key, base_url = resolve_credentials(config, config.vectorization)
        emb_timeout, emb_retries, emb_wait = 120.0, 3, 0.5
        try:
            v_retry = getattr(config.vectorization, 'retry', None)
            if v_retry is not None:
                emb_timeout = float(getattr(v_retry, 'timeout', emb_timeout) or emb_timeout)
                emb_retries = int(getattr(v_retry, 'max_attempt', emb_retries) or emb_retries)
                emb_wait = float(getattr(v_retry, 'wait', emb_wait) or emb_wait)
        except Exception:
            pass
        self.embedding = Embedding(
            api_key=api_key,
            base_url=base_url,
            timeout=emb_timeout,
            max_retries=emb_retries,
            retry_wait=max(0.5, emb_wait),
        )

    def _append_embedding(self, task_id, emb):
        """Append one result; flush FAISS+SQLite when buffer reaches flush_every."""
        to_flush_db = None
        to_flush_vdb = None
        with self._buffer_lock:
            self.task_db.buffer.append({
                'id': task_id,
                'embedding_status': 'done',
            })
            # 直接存向量 list/ndarray，避免 json 字符串放大内存
            self.task_vdb.buffer.append({
                'id': task_id,
                'embedding': emb,
            })
            if len(self.task_vdb.buffer) >= self.flush_every:
                to_flush_db = self.task_db.buffer
                to_flush_vdb = self.task_vdb.buffer
                self.task_db.buffer = []
                self.task_vdb.buffer = []
        if to_flush_vdb is not None:
            self._flush_buffers(to_flush_db, to_flush_vdb)

    def _flush_buffers(self, db_buf, vdb_buf):
        """Write a detached buffer batch to VDB + DB and persist the index."""
        n = len(vdb_buf) if vdb_buf else 0
        if vdb_buf:
            self.task_vdb.add(vdb_buf)
            self.task_vdb.save()
        if db_buf:
            self.task_db.update(db_buf)
        if n:
            self._flushed_count += n
            table = getattr(self.task_db, 'table', '?')
            # self.logger.info(
            #     f"vectorization flush: table={table} batch={n} total_flushed={self._flushed_count}"
            # )

    @Retry(max_attempt=5, wait=0.1, timeout=600, config_attr='vectorization.retry')
    def processing_single_task(self,task,**kwargs):
        emb_content=task['embedding_content']
        if emb_content is None:
            emb_content=task['content']
        if emb_content is None:
            return
        
        model_args=self.config.vectorization.model_args

        response = self.embedding.generate(emb_content,model_args=model_args,use_cache=self.config.vectorization.use_cache)
        if response['status']!=1:
            raise Exception(f"Embedding failed, status: {response['status']}")
        
        emb = response['answer']
        self._append_embedding(task['id'], emb)

    def prepare(self,db: BaseDB, vdb: BaseVDB):
        """
        只拉 embedding_status='undone'，支持中断后续跑：
          - 已 flush 的行是 done，不会再进任务列表 → 不重复写
          - 未 flush 的尾巴仍是 undone，下次会重做（有 cache 则几乎只读缓存）
        debug=True 时也只做 undone（全量重算请 clear vdb + 重置 status）。
        """
        self.task_db=db
        self.task_vdb=vdb
        self.tasks=self.task_db.search('embedding_status','undone')
        self.task_db.buffer_clear()
        self.task_vdb.buffer_clear()
        self._flushed_count = 0
        self.logger.info(
            f"vectorization prepare: table={getattr(db, 'table', '?')} "
            f"undone={len(self.tasks)} flush_every={self.flush_every}"
        )

    def processing(self):
        if self.config.vectorization.num_thread<=1:
            for task in tqdm(self.tasks):
                self.processing_single_task(task)
        else:
            with ThreadPoolExecutor(max_workers=self.config.vectorization.num_thread) as executor:
                futures = [executor.submit(self.processing_single_task, task) for task in self.tasks]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()

    def save(self):
        """Flush residual buffer (last incomplete batch). Safe under concurrency."""
        with self._buffer_lock:
            db_buf = self.task_db.buffer
            vdb_buf = self.task_vdb.buffer
            self.task_db.buffer = []
            self.task_vdb.buffer = []
        self._flush_buffers(db_buf, vdb_buf)
        
    def clear(self,db: BaseDB,vdb: BaseVDB):
        db.update_key('embedding_status',None)
        db.clear()

        vdb.clear()
