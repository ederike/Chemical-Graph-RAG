import gc
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ..utils.OpenAIAPI import Embedding
from ..utils.database import BaseDB, BaseVDB
from ..utils.config import Config, resolve_credentials
from ..utils.utils import Retry, TQDM_BAR_FORMAT

# Columns required for embedding; never SELECT * on multi-million tables.
_TASK_COLUMNS = ('id', 'content', 'embedding_content')


class Vectorization:
    def _append_embedding(self, task_id, emb):
        """Append one result; flush in-memory when buffer reaches flush_every."""
        to_flush_db = None
        to_flush_vdb = None
        with self._buffer_lock:
            self.task_db.buffer.append({
                'id': task_id,
                'embedding_status': 'done',
            })
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

    def _flush_buffers(self, db_buf, vdb_buf, *, force_checkpoint: bool = False):
        """
        Add vectors to in-memory FAISS; defer expensive full-index disk write
        until index_save_every vectors (or force_checkpoint).

        Resume safety: SQLite embedding_status is only marked done after the
        FAISS index has been successfully written to disk for that batch group.
        Crash mid-buffer → those rows stay undone and are retried (cache hits).
        """
        if not vdb_buf and not db_buf and not force_checkpoint:
            return

        with self._flush_io_lock:
            n = len(vdb_buf) if vdb_buf else 0
            if vdb_buf:
                self.task_vdb.add(vdb_buf)
                if db_buf:
                    self._pending_status.extend(db_buf)
                self._vectors_since_save += n
                self._flushed_count += n

            if force_checkpoint or self._vectors_since_save >= self.index_save_every:
                self._checkpoint_disk()

    def _checkpoint_disk(self):
        """Write full FAISS index once, then mark corresponding SQLite rows done."""
        # Always persist index if we have anything unsaved in memory path,
        # or if there are pending status rows (vectors already in FAISS RAM).
        if self._vectors_since_save <= 0 and not self._pending_status:
            return

        table = getattr(self.task_db, 'table', '?')
        n_status = len(self._pending_status)
        n_vecs = self._vectors_since_save
        t0 = time.perf_counter()

        # Full-index write (costly on multi-GB files) — once per checkpoint.
        self.task_vdb.save()

        if self._pending_status:
            # SQLite update can itself be huge; chunk it to avoid one giant write.
            pending = self._pending_status
            self._pending_status = []
            chunk = max(self.flush_every, 1000)
            for i in range(0, len(pending), chunk):
                self.task_db.update(pending[i: i + chunk])
            del pending

        self._vectors_since_save = 0
        dt = time.perf_counter() - t0
        self.logger.info(
            f"[vectorization] checkpoint table={table} "
            f"vecs_since_save={n_vecs} status_rows={n_status} disk_time={dt:.2f}s "
            f"total_flushed={self._flushed_count}"
        )
        # Release peak memory after multi-GB index serialization on WSL.
        gc.collect()

    def prepare(self, db: BaseDB, vdb: BaseVDB):
        """
        Bind DB/VDB only. Do NOT load all undone rows into memory (node table
        can be millions of rows — SELECT * would OOM / crash WSL).

        Actual rows are keyset-paginated inside processing().
        """
        self.task_db = db
        self.task_vdb = vdb
        self.task_db.buffer_clear()
        self.task_vdb.buffer_clear()
        self._flushed_count = 0
        self._pending_status = []
        self._vectors_since_save = 0
        self.total_undone = self.task_db.count_by('embedding_status', 'undone')
        self.logger.info(
            f"vectorization prepare: table={getattr(db, 'table', '?')} "
            f"undone={self.total_undone} flush_every={self.flush_every} "
            f"index_save_every={self.index_save_every} "
            f"task_page_size={self.task_page_size}"
        )

    def save(self):
        """Flush residual buffer and force a final disk checkpoint."""
        with self._buffer_lock:
            db_buf = self.task_db.buffer
            vdb_buf = self.task_vdb.buffer
            self.task_db.buffer = []
            self.task_vdb.buffer = []
        self._flush_buffers(db_buf, vdb_buf, force_checkpoint=True)

    def clear(self, db: BaseDB, vdb: BaseVDB):
        """Reset embedding flags and wipe the vector index; do not delete SQL rows."""
        db.update_key('embedding_status', None)
        vdb.clear()

    def __init__(self, logger: logging.Logger, config: Config):
        self.config = config
        self.logger = logger
        try:
            fe = int(getattr(config.vectorization, 'flush_every', 1000) or 1000)
        except (TypeError, ValueError):
            fe = 1000
        self.flush_every = max(1, fe)

        # How often to rewrite the full FAISS file + mark SQLite done.
        # Default: 5× flush_every so we do far fewer multi-GB writes on WSL.
        try:
            ise = getattr(config.vectorization, 'index_save_every', None)
            if ise is None or ise == 0:
                ise = self.flush_every * 5
            else:
                ise = int(ise)
        except (TypeError, ValueError):
            ise = self.flush_every * 5
        self.index_save_every = max(self.flush_every, ise)

        # Keyset page size for loading undone rows (id/content only).
        try:
            tps = getattr(config.vectorization, 'task_page_size', None)
            if tps is None or tps == 0:
                tps = max(self.flush_every * 5, 5000)
            else:
                tps = int(tps)
        except (TypeError, ValueError):
            tps = max(self.flush_every * 5, 5000)
        self.task_page_size = max(self.flush_every, tps)

        self._buffer_lock = threading.Lock()
        self._flush_io_lock = threading.Lock()
        self._flushed_count = 0
        self._pending_status = []
        self._vectors_since_save = 0
        self.total_undone = 0
        self.tasks = []  # kept for compatibility; pages are ephemeral

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
        self.metrics = None
        self.usage_prompt_tokens = 0
        self.usage_total_tokens = 0

    def _fetch_task_page(self, after_id: int, limit: int):
        """Load one page of undone rows with only columns needed for embedding."""
        return self.task_db.search_by(
            'embedding_status',
            'undone',
            columns=list(_TASK_COLUMNS),
            after_id=after_id,
            limit=limit,
        )

    @Retry(max_attempt=3, wait=0.1, timeout=60, config_attr='vectorization.retry')
    def processing_single_task(self, task, **kwargs):
        emb_content = task.get('embedding_content')
        if emb_content is None:
            emb_content = task.get('content')
        if emb_content is None:
            return

        t0 = time.perf_counter()
        model_args = self.config.vectorization.model_args
        table = getattr(self.task_db, 'table', 'unknown')

        response = self.embedding.generate(
            emb_content,
            model_args=model_args,
            use_cache=self.config.vectorization.use_cache,
        )
        if response['status'] != 1:
            self.logger.error(f"Embedding failed, status: {response['status']}")
            raise Exception(f"Embedding failed, status: {response['status']}")

        cache_hit = bool(response.get('_cache_hit'))
        prompt_tok = response.get('usage_prompt_tokens') or 0
        completion_tok = response.get('usage_completion_tokens') or 0
        total_tok = response.get('usage_total_tokens')
        if total_tok is None:
            total_tok = (prompt_tok or 0) + (completion_tok or 0)

        if not cache_hit:
            self.usage_prompt_tokens += prompt_tok or 0
            self.usage_total_tokens += total_tok or 0

        dt = time.perf_counter() - t0
        if self.metrics is not None:
            self.metrics.record(
                f'vectorization:{table}',
                dt,
                cache_hit=cache_hit,
                prompt_tokens=prompt_tok if not cache_hit else 0,
                completion_tokens=completion_tok if not cache_hit else 0,
                total_tokens=total_tok if not cache_hit else 0,
                name=f"{table}_{task.get('id')}",
                log=False,
                accumulate_time=False,
            )

        self._append_embedding(task['id'], response['answer'])

    def _process_page(self, tasks, bar, stage):
        """Embed one page of tasks (thread pool or serial)."""
        if not tasks:
            return

        def _postfix():
            pf = {
                'total': self.total_undone,
                'flush': getattr(self, '_flushed_count', 0),
            }
            if self.metrics is not None:
                s = self.metrics.stage_snapshot(stage)
                pf['real'] = s['real']
            return pf

        if self.config.vectorization.num_thread <= 1:
            for task in tasks:
                self.processing_single_task(task)
                if bar is not None:
                    bar.update(1)
                    bar.set_postfix(**_postfix())
        else:
            with ThreadPoolExecutor(
                max_workers=self.config.vectorization.num_thread
            ) as executor:
                futures = [
                    executor.submit(self.processing_single_task, task)
                    for task in tasks
                ]
                for future in as_completed(futures):
                    future.result()
                    if bar is not None:
                        bar.update(1)
                        bar.set_postfix(**_postfix())

    def processing(self):
        """
        Paginated vectorization:
          - never holds all undone rows in RAM
          - flushes embeddings into FAISS RAM every flush_every
          - writes full FAISS file + SQLite status only every index_save_every
          - forces a checkpoint at the end of each page and at the end of the run
        """
        table = getattr(self.task_db, 'table', 'unknown')
        stage = f'vectorization:{table}'
        n = int(self.total_undone or 0)
        if n == 0:
            return

        t_wall = time.perf_counter()
        after_id = 0
        page_size = self.task_page_size
        processed = 0

        bar = tqdm(
            total=n,
            desc=f'vectorize:{table}',
            unit='item',
            bar_format=TQDM_BAR_FORMAT,
        )
        try:
            while True:
                tasks = self._fetch_task_page(after_id, page_size)
                if not tasks:
                    break

                after_id = int(tasks[-1]['id'])
                page_n = len(tasks)
                self._process_page(tasks, bar, stage)
                processed += page_n

                # Residual buffer → FAISS RAM; disk write only when
                # index_save_every is reached (avoids rewriting multi-GB
                # indexes on every small page).
                with self._buffer_lock:
                    db_buf = self.task_db.buffer
                    vdb_buf = self.task_vdb.buffer
                    self.task_db.buffer = []
                    self.task_vdb.buffer = []
                self._flush_buffers(db_buf, vdb_buf, force_checkpoint=False)

                # Drop page payload promptly (embeddings already in FAISS RAM).
                del tasks, db_buf, vdb_buf
                gc.collect()

                # Keep going until a page returns empty (count may be approximate).
                if processed >= n and page_n < page_size:
                    break
        finally:
            bar.close()

        # Final residual + forced disk checkpoint (SQLite status + FAISS file).
        self.save()

        if self.metrics is not None:
            self.metrics.finalize_stage_wall_time(
                stage, time.perf_counter() - t_wall
            )
            self.metrics.log_stage(stage)
