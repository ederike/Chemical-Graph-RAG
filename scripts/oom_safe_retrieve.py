"""OOM guards for MSDS-scale retrieve/eval. Patches at runtime; does not edit src/.

Why this exists
---------------
example/a MSDS node id window is ~4.8M–24.3M (~195 × 100k HNSW shards, ~34GB).
The stock pin_retrieve_indexes() would faiss.read_index every overlapping shard
into RSS. _load_precompute() would also SELECT * every chunk/doc in range into
Python dicts (~1.4M chunks + ~200k docs). Either path OOMs a 9.7GB box.

Guards
------
1. pin_shards / pin_retrieve_for_eval become no-ops. Search already load/unload
   one sealed shard at a time when nothing is pinned.
2. Skip sealed shards whose filename-index × shard_max_vectors cannot overlap
   the query id window, so TDS/patent head shards are not opened.
3. Replace _load_precompute with SQLite-backed LRU maps (per-id / per-doc).
4. Cap sqlite mmap/cache; gc after each ANN search.
"""
from __future__ import annotations

import gc
import logging
import sys
from collections import OrderedDict
from typing import Any, Callable, Optional

_LOG = logging.getLogger("oom_safe_retrieve")
_APPLIED = False
_CACHE_CHUNK = 512
_CACHE_DOC = 256
_CACHE_HE = 256
_CACHE_DOC_CHUNKS = 64


class _LRUMap:
    """Mapping that fetches on miss. Always truthy so `obj or {}` stays this."""

    def __init__(self, fetch: Callable[[Any], Any], cache_size: int = 256):
        self._fetch = fetch
        self._cache_size = max(8, int(cache_size))
        self._cache: "OrderedDict[Any, Any]" = OrderedDict()

    def __bool__(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self._cache)

    def _put(self, key, val):
        self._cache[key] = val
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def get(self, key, default=None):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        try:
            val = self._fetch(key)
        except Exception as e:
            _LOG.warning("lazy fetch failed key=%r: %s", key, e)
            return default
        if val is None:
            return default
        self._put(key, val)
        return val

    def __getitem__(self, key):
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __contains__(self, key) -> bool:
        return self.get(key) is not None

    def items(self):
        return self._cache.items()

    def values(self):
        return self._cache.values()

    def keys(self):
        return self._cache.keys()


def _one(db, key, value) -> Optional[dict]:
    if db is None or value is None:
        return None
    try:
        rows = db.search(key, value) or []
    except Exception:
        return None
    return rows[0] if rows else None


def _many(db, key, value) -> list:
    if db is None or value is None:
        return []
    try:
        return list(db.search(key, value) or [])
    except Exception as e:
        _LOG.warning("lazy list fetch failed %s=%r: %s", key, value, e)
        return []


def apply_oom_guards() -> None:
    """Idempotent. Call before constructing DHMF / importing the bench workflow."""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    from src.utils import database as dbmod
    from src.module import retrieve as retrmod

    _patch_sqlite(dbmod)
    _patch_pin_and_search(dbmod)
    _patch_precompute(retrmod)
    _patch_bench_pin()
    print(
        "[oom-guard] applied: skip pin, per-shard ANN, lazy precompute, "
        "sqlite mmap/cache cap",
        file=sys.stderr,
        flush=True,
    )


def _patch_sqlite(dbmod) -> None:
    orig_load = dbmod.SQLiteDB.load

    def load(self):
        orig_load(self)
        try:
            self.conn.execute("PRAGMA mmap_size=268435456")  # 256 MiB
            self.conn.execute("PRAGMA cache_size=-131072")  # ~128 MiB
            self.conn.execute("PRAGMA temp_store=FILE")
            self.conn.execute("PRAGMA busy_timeout=600000")
        except Exception as e:
            print(f"[oom-guard] sqlite pragma failed: {e}", file=sys.stderr, flush=True)

    dbmod.SQLiteDB.load = load


def _patch_pin_and_search(dbmod) -> None:
    BaseVDB = dbmod.BaseVDB
    orig_search = BaseVDB.search
    orig_pin = BaseVDB.pin_shards

    def pin_shards(self, *, max_vectors: int = 0, min_vectors: int = 0):
        lo, hi = dbmod.normalize_id_range(min_vectors, max_vectors)
        print(
            f"[oom-guard] skip pin_shards {getattr(self, 'vdb_name', '?')} "
            f"id={lo}..{hi} (per-shard load/unload)",
            file=sys.stderr,
            flush=True,
        )
        return {
            "sharding": bool(getattr(self, "_sharding", False)),
            "pinned": False,
            "already": False,
            "skipped": True,
            "min_vectors": lo,
            "max_vectors": hi,
            "shards": [],
            "ntotal": 0,
            "bytes": 0,
            "seconds": 0.0,
        }

    def search(self, vector, topk: int = 10, *, max_vectors: int = 0, min_vectors: int = 0):
        lo, hi = dbmod.normalize_id_range(min_vectors, max_vectors)
        saved = None
        if getattr(self, "_sharding", False) and getattr(self, "_sealed", None):
            smv = int(getattr(self, "shard_max_vectors", 0) or 100000)
            smv = max(1, smv)
            kept = []
            for name in self._sealed:
                idx = self._shard_index(name)
                # Conservative: skip only when even (idx+2)*smv is still < lo.
                if lo > 0 and (idx + 2) * smv < lo:
                    continue
                if hi > 0 and idx * smv > hi + smv:
                    continue
                kept.append(name)
            saved = self._sealed
            self._sealed = kept
        try:
            return orig_search(
                self, vector, topk, max_vectors=max_vectors, min_vectors=min_vectors
            )
        finally:
            if saved is not None:
                self._sealed = saved
            gc.collect()

    BaseVDB.pin_shards = pin_shards
    BaseVDB.search = search
    # keep orig_pin referenced so linters don't think it is unused
    BaseVDB._oom_orig_pin_shards = orig_pin


def _patch_precompute(retrmod) -> None:
    Retrieve = retrmod.Retrieve
    orig_load = Retrieve._load_precompute

    def _load_precompute(self):
        chunk_db = self.db.get("chunk")
        if chunk_db is not None and hasattr(chunk_db, "ensure_fts"):
            try:
                self.last_fts_info = chunk_db.ensure_fts() or {}
                self.logger.info(f"[retrieve] chunk_fts {self.last_fts_info}")
            except Exception as e:
                self.logger.warning(f"[retrieve] chunk_fts ensure failed: {e}")
                self.last_fts_info = {"error": str(e)}
        else:
            self.last_fts_info = {}

        chunk_lo, chunk_hi = self._scope_vector_range("chunk")

        def _in_chunk_range(row):
            if not row:
                return None
            try:
                cid = int(row.get("id") or 0)
            except (TypeError, ValueError):
                return row
            if chunk_lo > 0 and cid < chunk_lo:
                return None
            if chunk_hi > 0 and cid > chunk_hi:
                return None
            return row

        def fetch_chunk(cid):
            return _in_chunk_range(_one(chunk_db, "id", cid))

        def fetch_chunks_for_doc(doc_id):
            rows = _many(chunk_db, "doc_id", doc_id)
            rows = [r for r in (_in_chunk_range(r) for r in rows) if r]
            try:
                rows.sort(key=self._chunk_order_key)
            except Exception:
                rows.sort(key=lambda c: (c.get("id") or 0))
            return rows

        doc_db = self.db.get("doc")
        he_db = self.db.get("hyperedge")

        self.chunk_dict = _LRUMap(fetch_chunk, _CACHE_CHUNK)
        self.chunks_by_doc = _LRUMap(fetch_chunks_for_doc, _CACHE_DOC_CHUNKS)
        self.doc_dict = _LRUMap(lambda i: _one(doc_db, "id", i), _CACHE_DOC)
        self.hyperedge_dict = _LRUMap(lambda i: _one(he_db, "id", i), _CACHE_HE)
        self.hyperedge_by_doc = _LRUMap(lambda i: _one(he_db, "doc_id", i), _CACHE_HE)
        self.hyperedge_by_chunk = _LRUMap(lambda i: _one(he_db, "chunk_id", i), _CACHE_HE)

        class _HeadMap:
            def __bool__(self_inner):
                return True

            def get(self_inner, doc_id, default=None):
                clist = self.chunks_by_doc.get(doc_id) or []
                head = None
                for c in clist:
                    if self._is_head_chunk(c):
                        head = c
                        break
                if head is None and clist:
                    head = clist[0]
                return head if head is not None else default

        self.doc_head_chunk = _HeadMap()
        # FTS path is used for keywords; a full-table Python scan would OOM.
        self.all_chunks = []
        self.all_hyperedges = []
        self.docs_by_family = {}
        self.family_key_by_doc = {}
        self._precomputed = True
        self.logger.info(
            f"[retrieve] precompute LAZY (oom-guard) "
            f"chunk_id_range={chunk_lo}..{chunk_hi} "
            f"no full-table load; per-id/per-doc LRU"
        )

    Retrieve._load_precompute = _load_precompute
    Retrieve._oom_orig_load_precompute = orig_load


def _patch_bench_pin() -> None:
    def skip_pin(dhmf) -> bool:
        print(
            "[oom-guard] skip pin_retrieve_for_eval "
            "(search_range too large to pin)",
            file=sys.stderr,
            flush=True,
        )
        return False

    def skip_unpin(dhmf, pinned: bool) -> None:
        return

    import benchmark.benchmark_agentic_1.utils as _u

    _u.pin_retrieve_for_eval = skip_pin
    _u.unpin_retrieve_for_eval = skip_unpin
    import benchmark.benchmark_agentic_1.workflow as _w

    _w.pin_retrieve_for_eval = skip_pin
    _w.unpin_retrieve_for_eval = skip_unpin

    try:
        from src.DHMF import DHMF

        def skip_dhmf_pin(
            self, db_name=None, chunk_max_vectors=None, node_max_vectors=None
        ):
            print(
                "[oom-guard] skip DHMF.pin_retrieve_indexes "
                "(per-shard load/unload)",
                file=sys.stderr,
                flush=True,
            )
            # Still build lazy precompute so the first query does not
            # fall through to the full-table original path.
            try:
                self.retrieve_module._ensure_precompute()
            except Exception as e:
                print(f"[oom-guard] lazy precompute: {e}", file=sys.stderr, flush=True)
            return {"skipped": True}

        def skip_dhmf_unpin(self, db_name=None):
            return {"skipped": True}

        DHMF.pin_retrieve_indexes = skip_dhmf_pin
        DHMF.unpin_retrieve_indexes = skip_dhmf_unpin
    except Exception as e:
        print(f"[oom-guard] DHMF pin patch skipped: {e}", file=sys.stderr, flush=True)
