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
                add_info = self.task_vdb.add(vdb_buf) or {}
                add_s = float(add_info.get('add_seconds') or 0.0)
                self._index_add_seconds += add_s
                self._index_add_count += int(add_info.get('n') or n)
                if db_buf:
                    self._pending_status.extend(db_buf)
                self._vectors_since_save += n
                self._flushed_count += n
                # Per-batch HNSW/FAISS graph-build cost (add_with_ids wall time).
                if add_s > 0 or n > 0:
                    idx_type = getattr(self.task_vdb, 'index_type', '?')
                    self.logger.info(
                        f"[vectorization] index_add table={getattr(self.task_db, 'table', '?')} "
                        f"index_type={idx_type} n={n} add_time={add_s:.3f}s "
                        f"cum_add_time={self._index_add_seconds:.3f}s "
                        f"cum_n={self._index_add_count}"
                    )

            if force_checkpoint or self._vectors_since_save >= self.index_save_every:
                self._checkpoint_disk()

    def _checkpoint_disk(self):
        """
        Persist active FAISS shard, mark SQLite rows done, then (if sharded)
        seal+unload the active shard so RAM only holds the next empty batch.

        Resume safety: SQLite embedding_status is marked done only after the
        FAISS file for this batch has been successfully written.
        """
        # Always persist index if we have anything unsaved in memory path,
        # or if there are pending status rows (vectors already in FAISS RAM).
        if self._vectors_since_save <= 0 and not self._pending_status:
            return

        table = getattr(self.task_db, 'table', '?')
        n_status = len(self._pending_status)
        n_vecs = self._vectors_since_save
        t0 = time.perf_counter()

        # Write active (or mono) index — sharded mode: only the small write shard.
        self.task_vdb.save()

        if self._pending_status:
            # SQLite update can itself be huge; chunk it to avoid one giant write.
            pending = self._pending_status
            self._pending_status = []
            chunk = max(self.flush_every, 1000)
            for i in range(0, len(pending), chunk):
                self.task_db.update(pending[i: i + chunk])
            del pending

        # Per-batch memory release: seal active shard → disk only, open empty active.
        sealed = False
        if hasattr(self.task_vdb, 'seal_and_rotate'):
            try:
                sealed = bool(self.task_vdb.seal_and_rotate(force=True))
            except Exception as e:
                self.logger.warning(
                    f"[vectorization] seal_and_rotate failed (non-fatal): {e}"
                )

        self._vectors_since_save = 0
        dt = time.perf_counter() - t0
        stats = ''
        idx_type = getattr(self.task_vdb, 'index_type', '?')
        if hasattr(self.task_vdb, 'shard_stats'):
            try:
                s = self.task_vdb.shard_stats()
                if s.get('sharding'):
                    stats = (
                        f" shards_sealed={s.get('sealed_count')} "
                        f"active={s.get('active')} "
                        f"active_ntotal={s.get('active_ntotal')}"
                    )
            except Exception:
                pass
        self.logger.info(
            f"[vectorization] checkpoint table={table} index_type={idx_type} "
            f"vecs_since_save={n_vecs} status_rows={n_status} "
            f"disk_time={dt:.2f}s sealed={sealed} "
            f"total_flushed={self._flushed_count} "
            f"cum_index_add_time={self._index_add_seconds:.3f}s{stats}"
        )
        # Drop peak memory after seal (and any temp page-cache pressure).
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
        self._index_add_seconds = 0.0
        self._index_add_count = 0
        if hasattr(vdb, 'reset_index_timing'):
            vdb.reset_index_timing()

        # Enable FAISS sharding: each progress batch writes one shard, seals,
        # unloads — peak RAM ≈ one batch, not the full multi-GB index.
        smv = self.shard_max_vectors
        if smv is not None and int(smv) > 0 and hasattr(vdb, 'enable_sharding'):
            vdb.enable_sharding(int(smv))

        self.total_undone = self.task_db.count_by('embedding_status', 'undone')
        idx_type = getattr(vdb, 'index_type', getattr(self.config.vectorization, 'index_type', '?'))
        hnsw_info = ''
        if str(idx_type).lower() == 'hnsw':
            hnsw_info = (
                f" hnsw_M={getattr(vdb, 'hnsw_M', '?')}"
                f" efConstruction={getattr(vdb, 'hnsw_efConstruction', '?')}"
                f" efSearch={getattr(vdb, 'hnsw_efSearch', '?')}"
            )
        shard_info = ''
        if hasattr(vdb, 'shard_stats'):
            try:
                s = vdb.shard_stats()
                if s.get('sharding'):
                    shard_info = (
                        f" progress_batch={s.get('shard_max_vectors')} "
                        f"sealed={s.get('sealed_count')} "
                        f"active={s.get('active')}"
                    )
            except Exception:
                pass
        if self.shard_max_vectors:
            # One knob already aligned flush/index_save/page → log compactly.
            self.logger.info(
                f"vectorization prepare: table={getattr(db, 'table', '?')} "
                f"undone={self.total_undone} "
                f"index_type={idx_type}{hnsw_info} "
                f"progress_batch={self.shard_max_vectors} "
                f"(page=flush=save=seal) "
                f"api_batch={self.batch_size}×{self.config.vectorization.num_thread}"
                f"{shard_info}"
            )
        else:
            self.logger.info(
                f"vectorization prepare: table={getattr(db, 'table', '?')} "
                f"undone={self.total_undone} "
                f"index_type={idx_type}{hnsw_info} "
                f"flush_every={self.flush_every} "
                f"index_save_every={self.index_save_every} "
                f"task_page_size={self.task_page_size} "
                f"batch_size={self.batch_size} "
                f"num_thread={self.config.vectorization.num_thread} "
                f"(mono, no shard_max_vectors)"
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
        """Reset embedding flags and wipe the vector index; do not delete SQL rows.

        Must set status to 'undone' (not NULL) so the next vectorization pass
        re-selects all rows. Embeddings themselves are re-fetched from cache.
        """
        table = getattr(db, 'table', '?')
        idx_type = getattr(vdb, 'index_type', '?')
        db.update_key('embedding_status', 'undone')
        vdb.clear()
        if hasattr(vdb, 'reset_index_timing'):
            vdb.reset_index_timing()
        self.logger.info(
            f"[vectorization] clear table={table} index_type={idx_type} "
            f"embedding_status→undone, vdb wiped"
        )

    def __init__(self, logger: logging.Logger, config: Config):
        self.config = config
        self.logger = logger

        # --- Progress batch ---
        # Preferred single knob: shard_max_vectors (Config already aligns
        # flush_every / index_save_every / task_page_size to it when set).
        # Mono mode (no shard): flush_every + index_save_every + task_page_size.
        try:
            smv = getattr(config.vectorization, 'shard_max_vectors', None)
            if smv is None or smv == 0:
                smv = None
            else:
                smv = max(1, int(smv))
        except (TypeError, ValueError):
            smv = None
        self.shard_max_vectors = smv

        if smv is not None:
            # One durable batch = page load = FAISS write = disk seal size.
            self.flush_every = smv
            self.index_save_every = smv
            self.task_page_size = smv
        else:
            try:
                fe = int(getattr(config.vectorization, 'flush_every', 1000) or 1000)
            except (TypeError, ValueError):
                fe = 1000
            self.flush_every = max(1, fe)
            try:
                ise = getattr(config.vectorization, 'index_save_every', None)
                if ise is None or ise == 0:
                    ise = self.flush_every * 5
                else:
                    ise = int(ise)
            except (TypeError, ValueError):
                ise = self.flush_every * 5
            self.index_save_every = max(self.flush_every, ise)
            try:
                tps = getattr(config.vectorization, 'task_page_size', None)
                if tps is None or tps == 0:
                    tps = max(self.flush_every * 5, 5000)
                else:
                    tps = int(tps)
            except (TypeError, ValueError):
                tps = max(self.flush_every * 5, 5000)
            self.task_page_size = max(self.flush_every, tps)

        # Texts per embeddings API call (list input). 1 = one request per text.
        try:
            bs = int(getattr(config.vectorization, 'batch_size', 1) or 1)
        except (TypeError, ValueError):
            bs = 1
        self.batch_size = max(1, bs)

        self._buffer_lock = threading.Lock()
        self._flush_io_lock = threading.Lock()
        self._flushed_count = 0
        self._pending_status = []
        self._vectors_since_save = 0
        self._index_add_seconds = 0.0
        self._index_add_count = 0
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

    def _task_text(self, task):
        emb_content = task.get('embedding_content')
        if emb_content is None:
            emb_content = task.get('content')
        return emb_content

    def _record_one_result(self, task, response, dt, table, stage):
        """Metrics + append embedding for one task result dict."""
        if response is None or response.get('status') != 1:
            ans = (response or {}).get('answer')
            self.logger.error(f"Embedding failed, status: {(response or {}).get('status')}")
            raise Exception(f"Embedding failed, status: {(response or {}).get('status')}: {ans}")

        cache_hit = bool(response.get('_cache_hit'))
        prompt_tok = response.get('usage_prompt_tokens') or 0
        completion_tok = response.get('usage_completion_tokens') or 0
        total_tok = response.get('usage_total_tokens')
        if total_tok is None:
            total_tok = (prompt_tok or 0) + (completion_tok or 0)

        if not cache_hit:
            self.usage_prompt_tokens += prompt_tok or 0
            self.usage_total_tokens += total_tok or 0

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

    @Retry(max_attempt=3, wait=0.1, timeout=60, config_attr='vectorization.retry')
    def processing_single_task(self, task, **kwargs):
        emb_content = self._task_text(task)
        if emb_content is None:
            return 0

        t0 = time.perf_counter()
        model_args = self.config.vectorization.model_args
        table = getattr(self.task_db, 'table', 'unknown')
        stage = f'vectorization:{table}'

        response = self.embedding.generate(
            emb_content,
            model_args=model_args,
            use_cache=self.config.vectorization.use_cache,
        )
        self._record_one_result(
            task, response, time.perf_counter() - t0, table, stage
        )
        return 1

    @Retry(max_attempt=3, wait=0.1, timeout=120, config_attr='vectorization.retry')
    def processing_batch_task(self, tasks, **kwargs):
        """
        Embed a list of tasks in one API call (OpenAI input=list).
        Per-text cache still applies; only misses hit the server.
        Returns number of successfully embedded items.
        """
        if not tasks:
            return 0
        if len(tasks) == 1:
            return self.processing_single_task(tasks[0])

        pairs = []
        for task in tasks:
            text = self._task_text(task)
            if text is None:
                continue
            pairs.append((task, text))
        if not pairs:
            return 0

        t0 = time.perf_counter()
        model_args = self.config.vectorization.model_args
        table = getattr(self.task_db, 'table', 'unknown')
        stage = f'vectorization:{table}'
        texts = [t for _, t in pairs]

        responses = self.embedding.generate_batch(
            texts,
            model_args=model_args,
            use_cache=self.config.vectorization.use_cache,
        )
        dt = time.perf_counter() - t0
        # Wall time for the whole HTTP batch is shared across items for metrics.
        dt_each = dt / max(len(pairs), 1)

        if not isinstance(responses, list) or len(responses) != len(pairs):
            raise Exception(
                f"Embedding batch size mismatch: got "
                f"{len(responses) if isinstance(responses, list) else type(responses)} "
                f"want {len(pairs)}"
            )

        n_ok = 0
        for (task, _), response in zip(pairs, responses):
            self._record_one_result(task, response, dt_each, table, stage)
            n_ok += 1
        return n_ok

    def _process_page(self, tasks, bar, stage):
        """Embed one page of tasks (thread pool × batch_size)."""
        if not tasks:
            return

        batch_size = max(1, int(self.batch_size or 1))
        # Slice page into micro-batches for one API call each.
        batches = [
            tasks[i: i + batch_size]
            for i in range(0, len(tasks), batch_size)
        ]

        def _postfix():
            pf = {
                'total': self.total_undone,
                'flush': getattr(self, '_flushed_count', 0),
                'bs': batch_size,
            }
            if self.metrics is not None:
                s = self.metrics.stage_snapshot(stage)
                pf['real'] = s['real']
            return pf

        def _run_batch(batch):
            if batch_size <= 1 or len(batch) == 1:
                return self.processing_single_task(batch[0]) if batch else 0
            return self.processing_batch_task(batch)

        if self.config.vectorization.num_thread <= 1:
            for batch in batches:
                _run_batch(batch)
                if bar is not None:
                    bar.update(len(batch))
                    bar.set_postfix(**_postfix())
        else:
            with ThreadPoolExecutor(
                max_workers=self.config.vectorization.num_thread
            ) as executor:
                futures = {
                    executor.submit(_run_batch, batch): len(batch)
                    for batch in batches
                }
                for future in as_completed(futures):
                    future.result()
                    if bar is not None:
                        bar.update(futures[future])
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

        wall_s = time.perf_counter() - t_wall
        idx_type = getattr(self.task_vdb, 'index_type', '?')
        # Prefer VDB cumulative timing if available (same source of truth).
        add_s = self._index_add_seconds
        add_n = self._index_add_count
        if hasattr(self.task_vdb, 'index_add_seconds'):
            add_s = float(getattr(self.task_vdb, 'index_add_seconds', 0.0) or add_s)
            add_n = int(getattr(self.task_vdb, 'index_add_count', 0) or add_n)
        per_1k = (add_s / add_n * 1000.0) if add_n else 0.0
        hnsw_extra = ''
        if str(idx_type).lower() == 'hnsw':
            hnsw_extra = (
                f" hnsw_M={getattr(self.task_vdb, 'hnsw_M', '?')}"
                f" efConstruction={getattr(self.task_vdb, 'hnsw_efConstruction', '?')}"
                f" efSearch={getattr(self.task_vdb, 'hnsw_efSearch', '?')}"
            )
        self.logger.info(
            f"[vectorization] index_build_summary table={table} "
            f"index_type={idx_type}{hnsw_extra} "
            f"vectors={add_n} index_add_time={add_s:.3f}s "
            f"per_1k_vectors={per_1k:.3f}s wall_time={wall_s:.3f}s "
            f"(index_add_time = FAISS add_with_ids / HNSW graph build)"
        )

        if self.metrics is not None:
            self.metrics.finalize_stage_wall_time(stage, wall_s)
            self.metrics.log_stage(stage)
