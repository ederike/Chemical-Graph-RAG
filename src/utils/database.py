import faiss
import heapq
import os
import shutil
import sqlite3
import time
from pathlib import Path
import threading
import numpy as np
import json

from tqdm import tqdm

# Per-resource locks: same SQLite file / same FAISS file share one lock;
# SQL and FAISS never block each other.
_sqlite_locks = {}
_sqlite_locks_guard = threading.Lock()
_faiss_locks = {}
_faiss_locks_guard = threading.Lock()


def _lock_for(registry: dict, guard: threading.Lock, path) -> threading.Lock:
    key = str(Path(path).resolve())
    with guard:
        lk = registry.get(key)
        if lk is None:
            lk = threading.Lock()
            registry[key] = lk
        return lk


def sqlite_lock_for(db_path) -> threading.Lock:
    return _lock_for(_sqlite_locks, _sqlite_locks_guard, db_path)


def faiss_lock_for(vdb_path) -> threading.Lock:
    return _lock_for(_faiss_locks, _faiss_locks_guard, vdb_path)


def normalize_id_range(min_vectors=0, max_vectors=0) -> tuple:
    """Inclusive FAISS/SQL id range (lo, hi). 0 on a side = unbounded."""
    try:
        lo = max(0, int(min_vectors or 0))
    except (TypeError, ValueError):
        lo = 0
    try:
        hi = max(0, int(max_vectors or 0))
    except (TypeError, ValueError):
        hi = 0
    if lo > 0 and hi > 0 and lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _id_in_range(i, lo: int, hi: int) -> bool:
    try:
        n = int(i)
    except (TypeError, ValueError):
        return False
    if lo > 0 and n < lo:
        return False
    if hi > 0 and n > hi:
        return False
    return True


def _shard_overlaps_range(id_start: int, ntotal: int, lo: int, hi: int) -> str:
    """
    Shard covering ids [id_start, id_start+ntotal-1] vs [lo, hi].
    Return 'before' / 'after' / 'overlap' / 'empty'.
    """
    try:
        n = int(ntotal or 0)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return 'empty'
    shard_lo = int(id_start)
    shard_hi = shard_lo + n - 1
    if hi > 0 and shard_lo > hi:
        return 'after'
    if lo > 0 and shard_hi < lo:
        return 'before'
    return 'overlap'


def normalize_index_quant(v) -> str:
    """none (float32) | fp16 (FAISS ScalarQuantizer QT_fp16)."""
    if isinstance(v, bool):
        return 'fp16' if v else 'none'
    s = str(v or 'none').strip().lower().replace('-', '_')
    if s in (
        'fp16', 'float16', 'half', 'sq_fp16', 'sqfp16',
        'qt_fp16', 'qtfp16', 'sq16', 'true', 'on', '1', 'yes',
    ):
        return 'fp16'
    return 'none'


class BaseDB:
    def __init__(self,db_path: str,db_name: str,create_table_sql: str):
        self.db_path = Path(db_path)
        self.table = db_name
        self.create_table_sql = create_table_sql
        self.buffer = []
        self.load()

        table_columns = self.db.execute(f"PRAGMA table_info({self.table})")
        self.table_columns = [col['name'] for col in table_columns]

    def load(self):
        self.db= SQLiteDB(self.db_path,self.table,self.create_table_sql)

    def save(self):
        self.db.save()

    def clear(self):
        self.db.clear()

    def buffer_clear(self):
        self.buffer=[]

    def add(self, data_list: list, *, return_ids: bool = False):
        """
        Insert rows. When return_ids=True, return lastrowid list (one per input
        row; 0 means INSERT OR IGNORE skipped that row).
        """
        if data_list == []:
            return [] if return_ids else None

        keys = list(data_list[0].keys())
        keys = [k for k in keys if k in self.table_columns]
        columns = ','.join(keys)
        placeholders = ','.join('?' * len(keys))
        sql = f'INSERT OR IGNORE INTO {self.table} ({columns}) VALUES ({placeholders})'
        values_list = [tuple(data[k] for k in keys) for data in data_list]
        if return_ids:
            return self.db.execute_insert_returning_ids(sql, values_list)
        self.db.execute_batch(sql, values_list)
        return None

    def update(self,data_list: list):
        """
        批量 UPDATE。按行各自的字段集合更新（避免 buffer 里首行字段少导致后面 extra/status 写丢）。
        同一 id 多次出现时，后写覆盖先写（executemany 顺序执行）。
        """
        if data_list == []:
            return
        # 按「字段集合」分组，保证 SQL 列一致；组内 executemany
        groups = {}
        for data in data_list:
            if data is None or data.get('id') is None:
                continue
            keys = tuple(
                k for k in data.keys()
                if k in self.table_columns and k != 'id'
            )
            if not keys:
                continue
            groups.setdefault(keys, []).append(data)
        for keys, rows in groups.items():
            set_clause = ','.join([f"{k} = ?" for k in keys])
            sql = f"UPDATE {self.table} SET {set_clause} WHERE id = ?"
            values_list = [
                tuple(row[k] for k in keys) + (row['id'],)
                for row in rows
            ]
            self.db.execute_batch(sql, values_list)
    
    def update_key(self,key: str,value):
        sql = f"UPDATE {self.table} SET {key} = ?"
        self.db.execute(sql, (value,))

    def search(self,key: str,value):
        sql = f"SELECT * FROM {self.table} WHERE {key} = ?"
        return self.db.execute(sql, (value,))

    def count_by(self, key: str, value) -> int:
        """Count rows where key == value (no row payload loaded)."""
        sql = f"SELECT COUNT(*) AS cnt FROM {self.table} WHERE {key} = ?"
        rows = self.db.execute(sql, (value,))
        if not rows:
            return 0
        try:
            return int(rows[0].get('cnt') or 0)
        except (TypeError, ValueError, AttributeError):
            return 0

    def search_by(
        self,
        key: str,
        value,
        *,
        columns=None,
        after_id: int = 0,
        limit: int = 0,
    ):
        """
        Keyset-paginated SELECT for large tables.

        columns: optional list of column names (defaults to *).
        after_id: only rows with id > after_id (stable resume within a run).
        limit: max rows; 0/None means no limit (avoid on multi-million tables).
        """
        if columns:
            cols = [c for c in columns if c in self.table_columns]
            if 'id' not in cols:
                cols = ['id'] + cols
            if not cols:
                cols = list(self.table_columns)
            col_sql = ','.join(cols)
        else:
            col_sql = '*'
        sql = f"SELECT {col_sql} FROM {self.table} WHERE {key} = ? AND id > ?"
        params: list = [value, int(after_id or 0)]
        sql += " ORDER BY id ASC"
        if limit and int(limit) > 0:
            sql += " LIMIT ?"
            params.append(int(limit))
        return self.db.execute(sql, tuple(params))

    def search_all(self, *, max_id: int = 0, min_id: int = 0):
        """Load the table. min_id/max_id>0 限制 id >= min / id <= max。"""
        try:
            lo = max(0, int(min_id or 0))
        except (TypeError, ValueError):
            lo = 0
        try:
            hi = max(0, int(max_id or 0))
        except (TypeError, ValueError):
            hi = 0
        clauses = []
        params = []
        if lo > 0:
            clauses.append("id >= ?")
            params.append(lo)
        if hi > 0:
            clauses.append("id <= ?")
            params.append(hi)
        sql = f"SELECT * FROM {self.table}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
            return self.db.execute(sql, tuple(params))
        return self.db.execute(sql)

    def search_lte(self, key: str, value):
        """SELECT * WHERE key <= value. Unknown key → empty list."""
        return self.search_range(key, max_value=value)

    def search_range(self, key: str, *, min_value=0, max_value=0):
        """SELECT * WHERE min_value <= key <= max_value。0=该端不限制。"""
        if key not in self.table_columns:
            return []
        try:
            lo = max(0, int(min_value or 0))
        except (TypeError, ValueError):
            lo = 0
        try:
            hi = max(0, int(max_value or 0))
        except (TypeError, ValueError):
            hi = 0
        clauses = []
        params = []
        if lo > 0:
            clauses.append(f"{key} >= ?")
            params.append(lo)
        if hi > 0:
            clauses.append(f"{key} <= ?")
            params.append(hi)
        sql = f"SELECT * FROM {self.table}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
            return self.db.execute(sql, tuple(params))
        return self.db.execute(sql)

    def list_ids(self):
        """Primary-key ids only (compact / reconcile)."""
        rows = self.db.execute(f"SELECT id FROM {self.table}") or []
        out = []
        for row in rows:
            try:
                out.append(int(row['id']))
            except (TypeError, ValueError, KeyError):
                continue
        return out

    def delete(self, key: str, value):
        """Delete rows where key == value. Returns deleted row count (best-effort)."""
        before = self.search(key, value)
        sql = f"DELETE FROM {self.table} WHERE {key} = ?"
        self.db.execute(sql, (value,))
        return len(before)

    def delete_by_ids(self, ids):
        """Delete rows by primary key id list. Returns number of ids requested."""
        if not ids:
            return 0
        placeholders = ','.join('?' * len(ids))
        sql = f"DELETE FROM {self.table} WHERE id IN ({placeholders})"
        self.db.execute(sql, tuple(ids))
        return len(ids)

class BaseVDB:
    """
    FAISS vector store.

    Monolithic mode (default):
      {vdb_path}/{name}.vdb  — entire index always in RAM.

    Sharded mode (enable_sharding / shard_max_vectors):
      {vdb_path}/{name}.shards/
        meta.json
        0.vdb, 1.vdb, ...   — sealed shards on disk only
        N.vdb               — active write shard (only this in RAM)

    After each vectorization checkpoint the active shard is saved, sealed,
    unloaded, and a fresh empty active is opened. Peak RAM ≈ one shard
    (shard_max_vectors × dim × 4B, or × 2B when index_quant=fp16).
    Search merges sealed + active. Default: sealed shards load one-by-one
    and unload (build-friendly). pin_shards(min_vectors=A, max_vectors=B)
    keeps shards overlapping that id range in RAM; search uses resident
    handles and falls back to load/unload for anything not pinned.
    unpin_shards() releases them.

    index_type:
      flat_l2 — IndexFlatL2 (exact, full scan)
      hnsw    — IndexHNSWFlat / IndexHNSWSQ (approx ANN; build cost is add_with_ids)
    index_quant:
      none — float32 vectors
      fp16 — FAISS ScalarQuantizer QT_fp16 (half-precision storage + search)

    Deletion: FAISS HNSW has no remove_ids. remove() records tombstone ids
    (sidecar deleted.json) and search() drops them. The HNSW graph is unchanged;
    no rebuild. Re-adding an id clears its tombstone. clear() wipes tombstones.
    compact() is manual only: rewrite each shard keeping ids still in SQL,
    then drop tombstones. Callers pass live_ids from SQLite. Skip when every
    indexed id is still in SQL. Never runs from delete() / search().

    repack() rewrites the store to a new shard size (or one monolithic file)
    by reconstructing already-stored vectors. It does not call the embedding
    API. HNSW graphs are rebuilt from those vectors (CPU).
    """

    def __init__(
        self,
        vdb_path: str,
        vdb_name: str,
        vdb_dim: int,
        shard_max_vectors: int = None,
        *,
        index_type: str = 'hnsw',
        index_quant: str = 'none',
        hnsw_M: int = 32,
        hnsw_efConstruction: int = 200,
        hnsw_efSearch: int = 64,
    ):
        self.vdb_path = Path(vdb_path)
        self.vdb_name = vdb_name
        self.vdb_dim = vdb_dim
        self.buffer = []
        self.shard_max_vectors = None
        self._sharding = False
        self.shards_dir = None
        self._sealed = []          # list[str] filenames under shards_dir
        self._active_name = None   # str filename of write shard
        self._next_id = 0
        self.vdb_file_path = self.vdb_path / f'{self.vdb_name}.vdb'
        # self.vdb: FassiVDB | None — active (or sole mono) index handle
        self.vdb = None

        # Index backend (used when creating empty indexes)
        self.index_type = (index_type or 'hnsw').strip().lower()
        if self.index_type not in ('flat_l2', 'hnsw'):
            self.index_type = 'hnsw'
        self._index_quant_requested = normalize_index_quant(index_quant)
        self.index_quant = self._index_quant_requested
        self.hnsw_M = max(2, int(hnsw_M or 32))
        self.hnsw_efConstruction = max(1, int(hnsw_efConstruction or 200))
        self.hnsw_efSearch = max(1, int(hnsw_efSearch or 64))
        # Cumulative FAISS add_with_ids wall time (HNSW graph build cost).
        self.index_add_seconds = 0.0
        self.index_add_count = 0
        self.last_search_stats = {}
        self._resident = {}            # name -> FassiVDB (search pin)
        self._resident_lock = threading.RLock()
        self._resident_min_vectors = 0
        self._resident_max_vectors = 0
        self._deleted = set()
        self._deleted_lock = threading.Lock()

        if shard_max_vectors is not None and int(shard_max_vectors) > 0:
            self.enable_sharding(int(shard_max_vectors))
        else:
            self.load()
        self._load_deleted()

    def _make_faiss(self, path, *, create_empty: bool = True) -> "FassiVDB":
        """Construct a FassiVDB handle with this store's index settings."""
        return FassiVDB(
            path,
            self.vdb_dim,
            create_empty=create_empty,
            index_type=self.index_type,
            index_quant=self.index_quant,
            hnsw_M=self.hnsw_M,
            hnsw_efConstruction=self.hnsw_efConstruction,
            hnsw_efSearch=self.hnsw_efSearch,
        )

    def _deleted_path(self) -> Path:
        if self._sharding and self.shards_dir is not None:
            return self.shards_dir / 'deleted.json'
        return self.vdb_path / f'{self.vdb_name}.deleted.json'

    def _mono_deleted_path(self) -> Path:
        return self.vdb_path / f'{self.vdb_name}.deleted.json'

    @staticmethod
    def _read_deleted_file(path: Path) -> set:
        if not path.exists() or path.stat().st_size <= 0:
            return set()
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return set()
        raw = data.get('ids', data) if isinstance(data, dict) else data
        if not isinstance(raw, (list, tuple, set)):
            return set()
        out = set()
        for x in raw:
            try:
                i = int(x)
            except (TypeError, ValueError):
                continue
            if i >= 0:
                out.add(i)
        return out

    def _load_deleted(self):
        ids = self._read_deleted_file(self._deleted_path())
        if self._sharding:
            ids |= self._read_deleted_file(self._mono_deleted_path())
        self._deleted = ids

    def _save_deleted(self):
        path = self._deleted_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._deleted:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass
            return
        payload = {'ids': sorted(int(i) for i in self._deleted)}
        tmp = path.with_suffix('.json.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(str(tmp), str(path))

    def _clear_deleted(self):
        self._deleted = set()
        for path in {self._deleted_path(), self._mono_deleted_path()}:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass

    def _live_id(self, i) -> bool:
        try:
            i = int(i)
        except (TypeError, ValueError):
            return False
        if i < 0:
            return False
        return i not in self._deleted

    def _search_fetch_k(self, topk: int) -> int:
        """Over-fetch so tombstones in the ANN window do not starve topk."""
        n_del = len(self._deleted)
        if n_del <= 0:
            return topk
        extra = min(n_del, max(32, topk * 3))
        return int(topk) + extra

    @staticmethod
    def _dedupe_hits(pairs, topk: int):
        """Keep best (smallest L2) hit per id, then cut to topk."""
        best = {}
        order = []
        for dist, vid in pairs:
            if vid not in best:
                best[vid] = dist
                order.append(vid)
            elif dist < best[vid]:
                best[vid] = dist
        ranked = sorted(order, key=lambda i: best[i])[:topk]
        return [{'distance': best[i], 'id': i} for i in ranked]

    def index_stats(self) -> dict:
        return {
            'index_type': self.index_type,
            'index_quant': self.index_quant,
            'hnsw_M': self.hnsw_M,
            'hnsw_efConstruction': self.hnsw_efConstruction,
            'hnsw_efSearch': self.hnsw_efSearch,
            'index_add_seconds': float(self.index_add_seconds),
            'index_add_count': int(self.index_add_count),
            'deleted_count': len(self._deleted),
        }

    def reset_index_timing(self):
        self.index_add_seconds = 0.0
        self.index_add_count = 0

    # ------------------------------------------------------------------ load / meta
    def load(self):
        """Load monolithic file, or re-open sharded layout if already present."""
        shards_dir = self.vdb_path / f'{self.vdb_name}.shards'
        if shards_dir.is_dir() and (shards_dir / 'meta.json').exists():
            # Prefer existing shards even if caller did not pass shard_max.
            meta = self._read_meta(shards_dir)
            max_v = int(meta.get('shard_max_vectors') or 0) or 10000
            self.enable_sharding(max_v)
            return
        self._sharding = False
        self.shards_dir = None
        self._sealed = []
        self._active_name = None
        self.vdb_file_path = self.vdb_path / f'{self.vdb_name}.vdb'
        self.vdb = self._make_faiss(self.vdb_file_path)

    def enable_sharding(self, shard_max_vectors: int):
        """
        Switch to (or reconfigure) sharded layout.
        Existing monolithic {name}.vdb is moved to sealed 0.vdb and unloaded
        so RAM only holds the new empty active write shard.
        """
        shard_max_vectors = max(1, int(shard_max_vectors))
        if self._sharding:
            self.shard_max_vectors = shard_max_vectors
            self._write_meta()
            return

        self.shards_dir = self.vdb_path / f'{self.vdb_name}.shards'
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.shards_dir / 'meta.json'
        mono = self.vdb_path / f'{self.vdb_name}.vdb'

        if meta_path.exists():
            meta = self._read_meta(self.shards_dir)
            self._sealed = list(meta.get('sealed') or [])
            self._active_name = meta.get('active')
            self._next_id = int(meta.get('next_id') or 0)
            self.shard_max_vectors = shard_max_vectors
            # Keep new shards consistent with on-disk encoding.
            # Legacy meta without index_quant is float32.
            if 'index_quant' in meta:
                disk_q = normalize_index_quant(meta.get('index_quant'))
            elif self._sealed:
                disk_q = 'none'
            else:
                disk_q = self.index_quant
            self.index_quant = disk_q
            self._sharding = True
            # Drop any mono handle that was loaded at construction.
            if self.vdb is not None:
                self.vdb.unload()
                self.vdb = None
            self._open_active(create_if_missing=True)
            self._write_meta()
            self._migrate_mono_deleted()
            return

        # Migrate monolithic file → sealed shard 0 (keep data, free RAM).
        sealed = []
        next_id = 0
        if mono.exists() and mono.stat().st_size > 0:
            name = f'{next_id}.vdb'
            dest = self.shards_dir / name
            # Ensure mono is flushed then release RAM before rename.
            if self.vdb is not None:
                try:
                    self.vdb.save()
                except Exception:
                    pass
                self.vdb.unload()
                self.vdb = None
            os.replace(str(mono), str(dest))
            sealed.append(name)
            next_id = 1

        if self.vdb is not None:
            self.vdb.unload()
            self.vdb = None

        self._sealed = sealed
        self._next_id = next_id
        self._active_name = None
        self.shard_max_vectors = shard_max_vectors
        self._sharding = True
        self._open_active(create_if_missing=True)
        self._write_meta()
        self._migrate_mono_deleted()

    def _migrate_mono_deleted(self):
        """Move leftover {name}.deleted.json into shards/deleted.json."""
        if not self._sharding or self.shards_dir is None:
            return
        src = self._mono_deleted_path()
        dst = self._deleted_path()
        extra = self._read_deleted_file(src)
        if extra:
            self._deleted |= extra
            self._save_deleted()
        if src.exists() and src != dst:
            try:
                src.unlink()
            except OSError:
                pass

    def _read_meta(self, shards_dir: Path) -> dict:
        p = shards_dir / 'meta.json'
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f) or {}
        except Exception:
            return {}

    def _write_meta(self):
        if not self._sharding or self.shards_dir is None:
            return
        meta = {
            'version': 1,
            'dim': self.vdb_dim,
            'shard_max_vectors': self.shard_max_vectors,
            'next_id': self._next_id,
            'sealed': list(self._sealed),
            'active': self._active_name,
            'index_type': self.index_type,
            'index_quant': self.index_quant,
            'hnsw_M': self.hnsw_M,
            'hnsw_efConstruction': self.hnsw_efConstruction,
            'hnsw_efSearch': self.hnsw_efSearch,
        }
        p = self.shards_dir / 'meta.json'
        tmp = p.with_suffix('.json.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(str(tmp), str(p))

    def _open_active(self, create_if_missing: bool = True):
        if not self._sharding:
            return
        if self._active_name:
            path = self.shards_dir / self._active_name
            if path.exists() or create_if_missing:
                self.vdb = self._make_faiss(path)
                self.vdb_file_path = path
                return
        # New empty active shard.
        name = f'{self._next_id}.vdb'
        self._next_id += 1
        self._active_name = name
        path = self.shards_dir / name
        self.vdb = self._make_faiss(path)
        self.vdb_file_path = path
        self._write_meta()

    def _ensure_active(self):
        if self.vdb is not None and self.vdb.vdb is not None:
            return
        if self._sharding:
            self._open_active(create_if_missing=True)
        else:
            self.vdb = self._make_faiss(self.vdb_file_path)

    # ------------------------------------------------------------------ persist / rotate
    def save(self):
        """Persist active (or mono) index to disk. Does not unload."""
        if self.vdb is not None:
            self.vdb.save()
        if self._sharding:
            self._write_meta()

    def seal_and_rotate(self, *, force: bool = True) -> bool:
        """
        After a successful save(): seal the active write shard to disk-only
        storage and open a fresh empty active. Frees RAM for the previous batch.

        force=True  — always seal if active has any vectors (per-checkpoint batching).
        force=False — only when ntotal >= shard_max_vectors.
        Returns True if a seal/rotate happened.
        """
        if not self._sharding or self.vdb is None:
            return False
        n = self.vdb.ntotal
        if n <= 0:
            return False
        if not force and self.shard_max_vectors and n < self.shard_max_vectors:
            return False

        # File already written by save(); drop RAM handle and mark sealed.
        name = self._active_name
        self.vdb.unload()
        self.vdb = None
        if name and name not in self._sealed:
            self._sealed.append(name)
        self._active_name = None
        self._open_active(create_if_missing=True)
        self._write_meta()
        return True

    def clear(self):
        """Wipe all vectors (mono file and/or every shard)."""
        if self._sharding and self.shards_dir is not None:
            if self.vdb is not None:
                self.vdb.unload()
                self.vdb = None
            # Remove shard files + meta.
            try:
                for p in self.shards_dir.glob('*'):
                    try:
                        p.unlink()
                    except OSError:
                        pass
                try:
                    self.shards_dir.rmdir()
                except OSError:
                    pass
            except OSError:
                pass
            mono = self.vdb_path / f'{self.vdb_name}.vdb'
            if mono.exists():
                try:
                    mono.unlink()
                except OSError:
                    pass
            # Recreate empty sharded layout with the configured encoding
            # (not whatever the wiped shards used).
            max_v = self.shard_max_vectors or 10000
            self._sharding = False
            self._sealed = []
            self._active_name = None
            self._next_id = 0
            self.index_quant = getattr(
                self, '_index_quant_requested', self.index_quant
            )
            self._clear_deleted()
            self.enable_sharding(max_v)
            return

        self.index_quant = getattr(
            self, '_index_quant_requested', self.index_quant
        )
        if self.vdb is not None:
            self.vdb.unload()
            self.vdb = None
        if self.vdb_file_path.exists():
            try:
                self.vdb_file_path.unlink()
            except OSError:
                pass
        self._clear_deleted()
        self.load()

    def buffer_clear(self):
        self.buffer = []

    @property
    def ntotal(self) -> int:
        """Approximate total vectors (sealed on disk by file size not counted precisely)."""
        n = 0
        if self.vdb is not None:
            n += self.vdb.ntotal
        if self._sharding and self.shards_dir is not None:
            for name in self._sealed:
                if self._active_name and name == self._active_name:
                    continue
                path = self.shards_dir / name
                if not path.exists():
                    continue
                # Cheap: open header only via full load is heavy; use faiss read.
                # For logging we accept loading sealed once — callers rarely use ntotal.
                try:
                    tmp = self._make_faiss(path, create_empty=False)
                    n += tmp.ntotal
                    tmp.unload()
                except Exception:
                    pass
        return n

    def active_ntotal(self) -> int:
        if self.vdb is None:
            return 0
        return self.vdb.ntotal

    def shard_stats(self) -> dict:
        out = {
            'sharding': self._sharding,
            'shard_max_vectors': self.shard_max_vectors,
            'sealed_count': len(self._sealed) if self._sharding else 0,
            'active': self._active_name,
            'active_ntotal': self.active_ntotal(),
        }
        out.update(self.index_stats())
        return out

    def add(self, data_list: list) -> dict:
        """
        Add vectors to the active (or mono) index.
        Returns {'n': int, 'add_seconds': float} for build-log timing.
        """
        if data_list == []:
            return {'n': 0, 'add_seconds': 0.0}
        ids = [data['id'] for data in data_list]
        # 兼容：embedding 可为 list/ndarray，或历史 JSON 字符串
        vecs = []
        for data in data_list:
            emb = data['embedding']
            if isinstance(emb, str):
                emb = json.loads(emb)
            vecs.append(emb)
        vectors = np.asarray(vecs, dtype=np.float32)
        # Re-adding an id makes it searchable again (clear tombstone).
        # flat_l2 can also physically drop the old row before insert.
        int_ids = []
        for i in ids:
            try:
                int_ids.append(int(i))
            except (TypeError, ValueError):
                continue
        if int_ids:
            with self._deleted_lock:
                if self._deleted.intersection(int_ids):
                    self._deleted.difference_update(int_ids)
                    self._save_deleted()
        self._ensure_active()
        if self.vdb is not None and self.index_type != 'hnsw':
            self.vdb.remove(ids)
        add_s = float(self.vdb.add(ids, vectors) or 0.0)
        self.index_add_seconds += add_s
        self.index_add_count += len(ids)
        return {'n': len(ids), 'add_seconds': add_s}

    @staticmethod
    def _shard_index(name) -> int:
        stem = str(name or '').rsplit('.', 1)[0]
        try:
            return int(stem)
        except (TypeError, ValueError):
            return 10 ** 9

    def _ordered_sealed_names(self) -> list:
        names = [n for n in (self._sealed or []) if n]
        names.sort(key=self._shard_index)
        return names

    def pin_shards(self, *, max_vectors: int = 0, min_vectors: int = 0) -> dict:
        """
        Load overlapping shards into RAM and keep the handles.

        min_vectors / max_vectors: same meaning as search() (0 = unbounded).
        Idempotent when already pinned with the same range.
        """
        t0 = time.perf_counter()
        lo, hi = normalize_id_range(min_vectors, max_vectors)

        if not self._sharding:
            n = int(self.vdb.ntotal or 0) if self.vdb is not None else 0
            return {
                'sharding': False,
                'pinned': True,
                'already': True,
                'min_vectors': lo,
                'max_vectors': hi,
                'shards': ['mono'] if n else [],
                'ntotal': n,
                'bytes': 0,
                'seconds': round(time.perf_counter() - t0, 3),
            }

        with self._resident_lock:
            if (
                self._resident
                and int(self._resident_min_vectors or 0) == lo
                and int(self._resident_max_vectors or 0) == hi
            ):
                ntotal = sum(int(h.ntotal or 0) for h in self._resident.values())
                return {
                    'sharding': True,
                    'pinned': True,
                    'already': True,
                    'min_vectors': lo,
                    'max_vectors': hi,
                    'shards': list(self._resident.keys()),
                    'ntotal': ntotal,
                    'bytes': 0,
                    'seconds': round(time.perf_counter() - t0, 3),
                }

            self._unpin_unlocked()
            loaded = []
            scanned = 0
            nbytes = 0
            id_start = 0
            for name in self._ordered_sealed_names():
                path = self.shards_dir / name
                if not path.exists() or path.stat().st_size < 1024:
                    continue
                handle = self._make_faiss(path, create_empty=False)
                n = int(handle.ntotal or 0)
                if n <= 0:
                    handle.unload()
                    continue
                rel = _shard_overlaps_range(id_start, n, lo, hi)
                if rel == 'after':
                    handle.unload()
                    break
                if rel == 'before':
                    handle.unload()
                    id_start += n
                    continue
                self._resident[name] = handle
                scanned += n
                nbytes += int(path.stat().st_size)
                loaded.append(name)
                id_start += n

            self._resident_min_vectors = lo
            self._resident_max_vectors = hi
            return {
                'sharding': True,
                'pinned': bool(loaded),
                'already': False,
                'min_vectors': lo,
                'max_vectors': hi,
                'shards': loaded,
                'ntotal': scanned,
                'bytes': nbytes,
                'seconds': round(time.perf_counter() - t0, 3),
            }

    def unpin_shards(self) -> dict:
        """Unload all pin_shards handles. No-op if nothing is pinned."""
        with self._resident_lock:
            n = len(self._resident)
            self._unpin_unlocked()
            return {'unpinned': n}

    def _unpin_unlocked(self):
        for handle in self._resident.values():
            try:
                handle.unload()
            except Exception:
                pass
        self._resident = {}
        self._resident_min_vectors = 0
        self._resident_max_vectors = 0

    def resident_stats(self) -> dict:
        with self._resident_lock:
            names = list(self._resident.keys())
            ntotal = sum(int(h.ntotal or 0) for h in self._resident.values())
            return {
                'pinned': bool(names),
                'min_vectors': int(self._resident_min_vectors or 0),
                'max_vectors': int(self._resident_max_vectors or 0),
                'shards': names,
                'ntotal': ntotal,
            }

    def search(
        self,
        vector: list,
        topk: int = 10,
        *,
        max_vectors: int = 0,
        min_vectors: int = 0,
    ):
        """
        ANN search. min_vectors/max_vectors 限制 FAISS id 闭区间；
        分片模式下只打开与该区间重叠的分片（跳过更前/更后的片）。
        某一端 0 / None = 该端不限制。
        """
        vector = np.array(vector, dtype=np.float32).reshape(1, -1)
        lo, hi = normalize_id_range(min_vectors, max_vectors)
        if topk <= 0:
            self.last_search_stats = {
                'min_vectors': lo,
                'max_vectors': hi,
                'shards_opened': 0,
                'shards_total': 0,
                'ntotal_scanned': 0,
                'skipped_head': False,
                'skipped_tail': False,
            }
            return []

        fetch_k = self._search_fetch_k(topk)

        def _keep_id(i) -> bool:
            if not self._live_id(i):
                return False
            return _id_in_range(i, lo, hi)

        if not self._sharding:
            if self.vdb is None:
                self.last_search_stats = {
                    'min_vectors': lo,
                    'max_vectors': hi,
                    'shards_opened': 0,
                    'shards_total': 0,
                    'ntotal_scanned': 0,
                    'skipped_head': False,
                    'skipped_tail': False,
                }
                return []
            distances, ids = self.vdb.search(vector, fetch_k)
            if ids is None:
                self.last_search_stats = {
                    'min_vectors': lo,
                    'max_vectors': hi,
                    'shards_opened': 1,
                    'shards_total': 1,
                    'ntotal_scanned': int(self.vdb.ntotal or 0),
                    'skipped_head': False,
                    'skipped_tail': False,
                }
                return []
            pairs = [
                (float(d), int(i))
                for d, i in zip(distances[0], ids[0])
                if _keep_id(i)
            ]
            self.last_search_stats = {
                'min_vectors': lo,
                'max_vectors': hi,
                'shards_opened': 1,
                'shards_total': 1,
                'ntotal_scanned': int(self.vdb.ntotal or 0),
                'skipped_head': bool(lo > 0),
                'skipped_tail': bool(hi > 0),
            }
            return self._dedupe_hits(pairs, topk)

        heap = []  # (distance, id) keep best topk (smallest L2)
        sealed_names = self._ordered_sealed_names()
        n_total = len(sealed_names) + (
            1 if (self.vdb is not None and self.vdb.vdb is not None) else 0
        )
        opened = 0
        scanned = 0
        skipped_head = False
        skipped_tail = False
        resident_hits = 0
        resident_misses = 0
        id_start = 0

        def _consume(distances, ids):
            if ids is None:
                return
            for d, i in zip(distances[0], ids[0]):
                if not _keep_id(i):
                    continue
                i = int(i)
                d = float(d)
                if len(heap) < topk:
                    heapq.heappush(heap, (-d, i))
                elif d < -heap[0][0]:
                    heapq.heapreplace(heap, (-d, i))

        def _search_handle(handle, *, owned: bool):
            nonlocal opened, scanned, resident_hits, resident_misses
            distances, ids = handle.search(vector, fetch_k)
            _consume(distances, ids)
            scanned += int(handle.ntotal or 0)
            opened += 1
            if owned:
                resident_misses += 1
            else:
                resident_hits += 1

        for name in sealed_names:
            path = self.shards_dir / name
            if not path.exists():
                continue
            handle = None
            owned = False
            with self._resident_lock:
                pinned = self._resident.get(name)
                if pinned is not None and pinned.vdb is not None:
                    handle = pinned
                else:
                    handle = self._make_faiss(path, create_empty=False)
                    owned = True
            try:
                n = int(handle.ntotal or 0)
                rel = _shard_overlaps_range(id_start, n, lo, hi)
                if rel == 'after':
                    skipped_tail = True
                    break
                if rel == 'before' or rel == 'empty':
                    skipped_head = skipped_head or rel == 'before'
                    id_start += max(0, n)
                    continue
                _search_handle(handle, owned=owned)
                id_start += max(0, n)
            finally:
                if owned:
                    handle.unload()

        if not skipped_tail:
            if self.vdb is not None and self.vdb.vdb is not None:
                n = int(self.vdb.ntotal or 0)
                rel = _shard_overlaps_range(id_start, n, lo, hi)
                if rel == 'overlap':
                    _search_handle(self.vdb, owned=False)
                elif rel == 'after':
                    skipped_tail = True
                elif rel == 'before':
                    skipped_head = True
        else:
            skipped_tail = True

        self.last_search_stats = {
            'min_vectors': lo,
            'max_vectors': hi,
            'shards_opened': opened,
            'shards_total': n_total,
            'ntotal_scanned': scanned,
            'skipped_head': skipped_head,
            'skipped_tail': skipped_tail,
            'resident_hits': resident_hits,
            'resident_misses': resident_misses,
        }
        pairs = [(-neg_d, i) for neg_d, i in heap]
        return self._dedupe_hits(pairs, topk)

    def remove(self, ids, persist: bool = True):
        """Tombstone vectors by id so search no longer returns them.

        HNSW cannot drop graph nodes in-place; ids are recorded in deleted.json
        and filtered at search. flat_l2 also tries FAISS remove_ids.
        Returns the number of newly tombstoned ids.
        """
        if not ids:
            return 0
        id_list = []
        seen = set()
        for i in ids:
            try:
                iv = int(i)
            except (TypeError, ValueError):
                continue
            if iv < 0 or iv in seen:
                continue
            seen.add(iv)
            id_list.append(iv)
        if not id_list:
            return 0

        if self.index_type != 'hnsw':
            if not self._sharding:
                if self.vdb is not None:
                    n_phys = self.vdb.remove(id_list)
                    if persist and n_phys:
                        self.save()
            else:
                if self.vdb is not None and self.vdb.vdb is not None:
                    n_phys = self.vdb.remove(id_list)
                    if persist and n_phys:
                        self.vdb.save()
                for name in list(self._sealed):
                    path = self.shards_dir / name
                    if not path.exists():
                        continue
                    tmp = self._make_faiss(path, create_empty=False)
                    try:
                        r = tmp.remove(id_list)
                        if r:
                            tmp.save()
                    finally:
                        tmp.unload()

        with self._deleted_lock:
            before = len(self._deleted)
            self._deleted.update(id_list)
            added = len(self._deleted) - before
            if persist and added:
                self._save_deleted()
        return added

    def _index_tombstone_hits(self, index) -> dict:
        ntot = 0
        hits = 0
        if index is None:
            return {'ntotal': 0, 'tombstone_hits': 0}
        try:
            ntot = int(getattr(index, 'ntotal', 0) or 0)
        except Exception:
            ntot = 0
        try:
            id_map = faiss.vector_to_array(index.id_map)
        except Exception:
            return {'ntotal': ntot, 'tombstone_hits': 0}
        deleted = self._deleted
        for nid in id_map:
            try:
                i = int(nid)
            except (TypeError, ValueError):
                continue
            if i >= 0 and i in deleted:
                hits += 1
        return {'ntotal': ntot, 'tombstone_hits': int(hits)}

    def deleted_stats(self, *, per_shard: bool = True) -> dict:
        """Tombstone count / ratio. per_shard opens each shard to count hits."""
        ntot = int(self.ntotal or 0)
        ndel = len(self._deleted)
        if ntot > 0:
            ratio = ndel / ntot
        else:
            ratio = 1.0 if ndel else 0.0
        out = {
            'name': self.vdb_name,
            'sharding': bool(self._sharding),
            'ntotal': ntot,
            'deleted_count': ndel,
            'deleted_ratio': float(ratio),
            'has_tombstones': ndel > 0,
            'shards': [],
        }
        if not per_shard:
            return out

        if not self._sharding:
            idx = self.vdb.vdb if self.vdb is not None else None
            one = self._index_tombstone_hits(idx)
            out['shards'].append({'shard': 'mono', **one})
            return out

        if self.vdb is not None and self.vdb.vdb is not None:
            one = self._index_tombstone_hits(self.vdb.vdb)
            out['shards'].append({
                'shard': self._active_name or 'active',
                **one,
            })
        for name in list(self._sealed):
            if self._active_name and name == self._active_name:
                continue
            path = self.shards_dir / name
            if not path.exists():
                continue
            tmp = self._make_faiss(path, create_empty=False)
            try:
                idx = tmp.vdb if tmp is not None else None
                one = self._index_tombstone_hits(idx)
            finally:
                tmp.unload()
            out['shards'].append({'shard': name, **one})
        return out

    def _index_dead_hits(self, index, is_live) -> int:
        if index is None:
            return 0
        try:
            id_map = faiss.vector_to_array(index.id_map)
        except Exception:
            return 0
        dead = 0
        for nid in id_map:
            try:
                ok = bool(is_live(nid))
            except Exception:
                ok = False
            if not ok:
                dead += 1
        return dead

    def _count_not_live(self, is_live) -> int:
        if not self._sharding:
            idx = self.vdb.vdb if self.vdb is not None else None
            return self._index_dead_hits(idx, is_live)
        dead = 0
        if self.vdb is not None and self.vdb.vdb is not None:
            dead += self._index_dead_hits(self.vdb.vdb, is_live)
        for name in list(self._sealed):
            if self._active_name and name == self._active_name:
                continue
            path = self.shards_dir / name
            if not path.exists():
                continue
            tmp = self._make_faiss(path, create_empty=False)
            try:
                idx = tmp.vdb if tmp is not None else None
                dead += self._index_dead_hits(idx, is_live)
            finally:
                tmp.unload()
        return dead

    def compact(self, *, live_ids=None, batch_size: int = 8192) -> dict:
        """
        Manual rewrite keeping only ids still present in SQL (live_ids).

        live_ids is the source of truth. Tombstones alone are not enough:
        a SQL-deleted id that missed tombstone is still dropped here.
        Shards with no dead ids are left untouched. Does not re-embed.
        """
        try:
            batch_size = max(256, int(batch_size or 8192))
        except (TypeError, ValueError):
            batch_size = 8192

        live_set = None
        if live_ids is not None:
            live_set = set()
            for i in live_ids:
                try:
                    live_set.add(int(i))
                except (TypeError, ValueError):
                    continue

        def is_live(i):
            try:
                i = int(i)
            except (TypeError, ValueError):
                return False
            if i < 0:
                return False
            if live_set is not None:
                return i in live_set
            return i not in self._deleted

        stats = self.deleted_stats(per_shard=False)
        summary = {
            'name': self.vdb_name,
            'skipped': False,
            'reason': '',
            'ntotal_before': stats['ntotal'],
            'deleted_before': stats['deleted_count'],
            'deleted_ratio': stats['deleted_ratio'],
            'sql_live_ids': len(live_set) if live_set is not None else None,
            'shards': [],
            'kept': 0,
            'dropped': 0,
            'rewritten': 0,
        }

        dead_in_index = self._count_not_live(is_live)
        if dead_in_index <= 0:
            summary['skipped'] = True
            summary['reason'] = (
                'all_ids_in_sql' if live_set is not None else 'no_tombstones'
            )
            if live_set is not None:
                with self._deleted_lock:
                    keep = self._deleted.intersection(live_set)
                    if keep != self._deleted:
                        self._deleted = keep
                        self._save_deleted()
            return summary

        if not self._sharding:
            self._ensure_active()
            one = self.vdb.compact_live(is_live, batch_size=batch_size)
            summary['shards'].append({'shard': 'mono', **one})
            summary['kept'] += int(one.get('kept') or 0)
            summary['dropped'] += int(one.get('dropped') or 0)
            if one.get('rewritten'):
                summary['rewritten'] += 1
            self._clear_deleted()
            summary['ntotal_after'] = int(self.vdb.ntotal if self.vdb else 0)
            summary['deleted_after'] = 0
            return summary

        # Active
        if self.vdb is not None and self.vdb.vdb is not None:
            one = self.vdb.compact_live(is_live, batch_size=batch_size)
            summary['shards'].append({
                'shard': self._active_name or 'active',
                **one,
            })
            summary['kept'] += int(one.get('kept') or 0)
            summary['dropped'] += int(one.get('dropped') or 0)
            if one.get('rewritten'):
                summary['rewritten'] += 1

        still_sealed = []
        for name in list(self._sealed):
            if self._active_name and name == self._active_name:
                still_sealed.append(name)
                continue
            path = self.shards_dir / name
            if not path.exists():
                continue
            tmp = self._make_faiss(path, create_empty=False)
            try:
                one = tmp.compact_live(is_live, batch_size=batch_size)
            finally:
                tmp.unload()
            summary['shards'].append({'shard': name, **one})
            summary['kept'] += int(one.get('kept') or 0)
            summary['dropped'] += int(one.get('dropped') or 0)
            if one.get('rewritten'):
                summary['rewritten'] += 1
            if int(one.get('kept') or 0) == 0 and one.get('rewritten'):
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            still_sealed.append(name)

        self._sealed = still_sealed
        self._write_meta()
        self._clear_deleted()
        summary['ntotal_after'] = int(self.ntotal or 0)
        summary['deleted_after'] = 0
        return summary

    def _live_pred(self, live_ids):
        live_set = None
        if live_ids is not None:
            live_set = set()
            for i in live_ids:
                try:
                    live_set.add(int(i))
                except (TypeError, ValueError):
                    continue

        def is_live(i):
            try:
                i = int(i)
            except (TypeError, ValueError):
                return False
            if i < 0:
                return False
            if live_set is not None:
                return i in live_set
            return i not in self._deleted

        return is_live, live_set

    def _iter_source_vectors(self, is_live, batch_size: int):
        """Yield (ids, vecs) from sealed shards then active / mono."""
        if self._sharding:
            for name in list(self._sealed):
                if self._active_name and name == self._active_name:
                    continue
                path = self.shards_dir / name
                if not path.exists() or path.stat().st_size <= 0:
                    continue
                tmp = self._make_faiss(path, create_empty=False)
                try:
                    yield from tmp.iter_live_vectors(is_live, batch_size)
                finally:
                    tmp.unload()
            if self.vdb is not None:
                yield from self.vdb.iter_live_vectors(is_live, batch_size)
            return
        if self.vdb is not None:
            yield from self.vdb.iter_live_vectors(is_live, batch_size)

    def _add_trained(self, handle: "FassiVDB", ids, vecs):
        if handle is None or handle.vdb is None:
            return
        if vecs is None or getattr(vecs, 'shape', (0,))[0] <= 0:
            return
        inner = getattr(handle.vdb, 'index', None)
        if inner is not None and hasattr(inner, 'is_trained') and not inner.is_trained:
            inner.train(vecs)
        handle.add(ids, vecs)

    def _minimal_shard_count(self, ntotal: int, shard_max: int) -> int:
        if ntotal <= 0:
            return 0
        return (int(ntotal) + int(shard_max) - 1) // int(shard_max)

    def repack(
        self,
        *,
        shard_max_vectors=None,
        live_ids=None,
        batch_size: int = 8192,
        force: bool = False,
    ) -> dict:
        """
        Rewrite this store to a new shard size without re-embedding.

        Reconstructs vectors already in FAISS (fp16 stores decode to float32
        then re-quantize). HNSW is rebuilt; embedding API is not called.

        shard_max_vectors:
          None / 0 — one file {name}.vdb (monolithic)
          N > 0    — shards of at most N vectors; leftover goes in the last
                     sealed shard, plus an empty active write shard
        live_ids: if given, drop ids not in this set (same as compact).
        force: rewrite even when the current layout already matches.
        """
        try:
            batch_size = max(256, int(batch_size or 8192))
        except (TypeError, ValueError):
            batch_size = 8192

        dest_max = None
        if shard_max_vectors is not None:
            try:
                dest_max = int(shard_max_vectors)
            except (TypeError, ValueError):
                dest_max = None
            if dest_max is not None and dest_max <= 0:
                dest_max = None

        is_live, live_set = self._live_pred(live_ids)
        n_before = int(self.ntotal or 0)
        n_sealed = len(self._sealed) if self._sharding else 0
        want_shards = dest_max is not None

        summary = {
            'name': self.vdb_name,
            'skipped': False,
            'reason': '',
            'ntotal_before': n_before,
            'sharding_before': bool(self._sharding),
            'shard_max_before': self.shard_max_vectors,
            'sealed_before': n_sealed,
            'sharding_after': want_shards,
            'shard_max_after': dest_max,
            'kept': 0,
            'dropped': 0,
            'shards_written': 0,
        }

        if n_before <= 0:
            summary['skipped'] = True
            summary['reason'] = 'empty'
            return summary

        if not force and not self._deleted:
            if not want_shards and not self._sharding:
                summary['skipped'] = True
                summary['reason'] = 'already_mono'
                return summary
            if (
                want_shards
                and self._sharding
                and self.shard_max_vectors == dest_max
                and n_sealed == self._minimal_shard_count(n_before, dest_max)
            ):
                summary['skipped'] = True
                summary['reason'] = 'already_packed'
                return summary

        # Flush active so disk is complete, then stream reconstruct → new files.
        if self.vdb is not None:
            try:
                self.vdb.save()
            except Exception:
                pass

        work = self.vdb_path / f'{self.vdb_name}.repack.tmp'
        bak = self.vdb_path / f'{self.vdb_name}.repack.bak'
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)

        dest_dir = work / 'shards'
        dest_mono = work / f'{self.vdb_name}.vdb'
        if want_shards:
            dest_dir.mkdir(parents=True)

        seen = set()
        kept = 0
        next_id = 0
        sealed_names = []
        dest = None
        active_name = None

        def _open_dest():
            nonlocal dest, active_name, next_id
            if dest is not None:
                dest.unload()
                dest = None
            if want_shards:
                active_name = f'{next_id}.vdb'
                next_id += 1
                dest = self._make_faiss(dest_dir / active_name, create_empty=True)
            else:
                active_name = None
                dest = self._make_faiss(dest_mono, create_empty=True)
            return dest

        def _seal_dest():
            nonlocal dest, active_name
            if dest is None or dest.ntotal <= 0:
                return
            dest.save()
            dest.unload()
            dest = None
            if want_shards and active_name:
                sealed_names.append(active_name)
            active_name = None

        _open_dest()
        bar = tqdm(
            total=n_before,
            desc=f'repack {self.vdb_name}',
            unit='vec',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]',
        )
        try:
            for ids, vecs in self._iter_source_vectors(is_live, batch_size):
                if ids is None or len(ids) == 0:
                    continue
                ids = np.asarray(ids, dtype=np.int64)
                vecs = np.asarray(vecs, dtype=np.float32)
                mask = []
                for i in ids:
                    ii = int(i)
                    if ii in seen:
                        mask.append(False)
                    else:
                        seen.add(ii)
                        mask.append(True)
                if not any(mask):
                    continue
                if not all(mask):
                    sel = np.asarray(mask, dtype=bool)
                    ids = ids[sel]
                    vecs = vecs[sel]
                offset = 0
                n = int(ids.shape[0])
                while offset < n:
                    if want_shards and dest is not None and dest_max:
                        room = dest_max - int(dest.ntotal or 0)
                        if room <= 0:
                            _seal_dest()
                            _open_dest()
                            room = dest_max
                        take = min(n - offset, room)
                    else:
                        take = n - offset
                    self._add_trained(
                        dest, ids[offset:offset + take], vecs[offset:offset + take]
                    )
                    offset += take
                kept += n
                bar.update(n)
            bar.close()
        except Exception:
            bar.close()
            if dest is not None:
                dest.unload()
            shutil.rmtree(work, ignore_errors=True)
            raise

        if dest is not None and dest.ntotal > 0:
            _seal_dest()
        elif dest is not None:
            dest.unload()
            dest = None

        dropped = max(0, n_before - kept)
        summary['kept'] = kept
        summary['dropped'] = dropped
        summary['shards_written'] = len(sealed_names) if want_shards else (1 if kept else 0)

        if kept <= 0:
            shutil.rmtree(work, ignore_errors=True)
            summary['skipped'] = True
            summary['reason'] = 'no_live_vectors'
            return summary

        if want_shards:
            # Empty active write shard so next DHMF init does not load all vectors.
            empty_name = f'{next_id}.vdb'
            next_id += 1
            empty = self._make_faiss(dest_dir / empty_name, create_empty=True)
            empty.save()
            empty.unload()
            meta = {
                'version': 1,
                'dim': self.vdb_dim,
                'shard_max_vectors': dest_max,
                'next_id': next_id,
                'sealed': list(sealed_names),
                'active': empty_name,
                'index_type': self.index_type,
                'index_quant': self.index_quant,
                'hnsw_M': self.hnsw_M,
                'hnsw_efConstruction': self.hnsw_efConstruction,
                'hnsw_efSearch': self.hnsw_efSearch,
            }
            meta_path = dest_dir / 'meta.json'
            tmp_meta = meta_path.with_suffix('.json.tmp')
            with open(tmp_meta, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            os.replace(str(tmp_meta), str(meta_path))

        # Unload RAM so files can be renamed.
        if self.vdb is not None:
            self.vdb.unload()
            self.vdb = None

        if bak.exists():
            shutil.rmtree(bak, ignore_errors=True)
        bak.mkdir(parents=True)

        old_shards = self.vdb_path / f'{self.vdb_name}.shards'
        old_mono = self.vdb_path / f'{self.vdb_name}.vdb'
        try:
            if old_shards.exists():
                os.rename(str(old_shards), str(bak / 'shards'))
            if old_mono.exists():
                os.rename(str(old_mono), str(bak / 'mono.vdb'))
            if want_shards:
                os.rename(str(dest_dir), str(old_shards))
            else:
                os.rename(str(dest_mono), str(old_mono))
        except Exception:
            # Best-effort restore
            try:
                bak_shards = bak / 'shards'
                bak_mono = bak / 'mono.vdb'
                if want_shards and old_shards.exists():
                    shutil.rmtree(old_shards, ignore_errors=True)
                if (not want_shards) and old_mono.exists():
                    try:
                        old_mono.unlink()
                    except OSError:
                        pass
                if bak_shards.exists() and not old_shards.exists():
                    os.rename(str(bak_shards), str(old_shards))
                if bak_mono.exists() and not old_mono.exists():
                    os.rename(str(bak_mono), str(old_mono))
            except OSError:
                pass
            shutil.rmtree(work, ignore_errors=True)
            raise

        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(bak, ignore_errors=True)

        self._clear_deleted()
        if want_shards:
            self._sharding = False
            self._sealed = []
            self._active_name = None
            self._next_id = 0
            self.enable_sharding(dest_max)
        else:
            self._sharding = False
            self.shards_dir = None
            self._sealed = []
            self._active_name = None
            self._next_id = 0
            self.shard_max_vectors = None
            self.vdb_file_path = old_mono
            self.vdb = self._make_faiss(self.vdb_file_path, create_empty=False)

        summary['ntotal_after'] = int(self.ntotal or 0)
        summary['sealed_after'] = len(self._sealed) if self._sharding else 0
        return summary

    def iter_faiss_indexes(self):
        """
        Yield (label, faiss.Index) for export. Caller must not
        retain indexes after the generator advances when sharded (sealed
        are unloaded after each yield via context — actually we load sealed
        fully; use reconstruct_id_map for safer export).
        """
        if not self._sharding:
            if self.vdb is not None and self.vdb.vdb is not None:
                yield 'mono', self.vdb.vdb
            return
        for name in self._sealed:
            path = self.shards_dir / name
            if not path.exists():
                continue
            tmp = self._make_faiss(path, create_empty=False)
            try:
                if tmp.vdb is not None:
                    yield name, tmp.vdb
            finally:
                tmp.unload()
        if self.vdb is not None and self.vdb.vdb is not None:
            yield self._active_name or 'active', self.vdb.vdb

    def id_to_vector_map(self) -> dict:
        """Build id → vector across all shards (may be large; for export)."""
        out = {}
        for _label, index in self.iter_faiss_indexes():
            try:
                id_map = faiss.vector_to_array(index.id_map)
            except Exception:
                continue
            inner = getattr(index, 'index', None)
            if inner is None:
                continue
            for pos, nid in enumerate(id_map):
                if not self._live_id(nid):
                    continue
                try:
                    vec = inner.reconstruct(int(pos))
                except Exception:
                    continue
                out[int(nid)] = np.asarray(vec, dtype=np.float64)
        return out

class SQLiteDB:
    def __init__(self,db_path,table,create_table_sql):
        self.db_path = Path(db_path)
        self.table = table
        self.create_table_sql = create_table_sql
        self._lock = sqlite_lock_for(self.db_path)
        self.load()
    
    def load(self):
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True)
        if not self.db_path.exists():
            self.db_path.touch()

        with self._lock:
            self.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=60.0,
            )
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            # WAL: readers don't block writers across connections on same file
            try:
                self.cursor.execute('PRAGMA journal_mode=WAL')
                self.cursor.execute('PRAGMA synchronous=NORMAL')
            except sqlite3.Error:
                pass
            self.cursor.execute(self.create_table_sql)
            self.conn.commit()

    def save(self):
        with self._lock:
            self.conn.commit()
    
    def clear(self):
        with self._lock:
            self.cursor.execute(f"DROP TABLE IF EXISTS {self.table}")
            self.conn.commit()
        self.load()

    def execute(self,SQL,values=()):
        with self._lock:
            self.cursor.execute(SQL,values)
            self.conn.commit()
            res=self.cursor.fetchall()
        res=[dict(row) for row in res]
        return res

    def execute_ids(self, SQL, values=()):
        """SELECT first-column ints as a set. No dict wrap, no commit."""
        with self._lock:
            self.cursor.execute(SQL, values)
            rows = self.cursor.fetchall()
        out = set()
        for row in rows:
            try:
                out.add(int(row[0]))
            except (TypeError, ValueError, IndexError):
                continue
        return out

    def execute_batch(self,SQL,values_list):
        with self._lock:
            self.cursor.executemany(SQL,values_list)
            self.conn.commit()

    def execute_insert_returning_ids(self, SQL, values_list):
        """
        Sequential INSERT; collect lastrowid per row.
        0 means the row was ignored (INSERT OR IGNORE) or insert failed.
        """
        ids = []
        if not values_list:
            return ids
        with self._lock:
            for values in values_list:
                self.cursor.execute(SQL, values)
                # INSERT OR IGNORE: lastrowid unchanged when row is skipped
                rowid = int(self.cursor.lastrowid or 0)
                ids.append(rowid)
            self.conn.commit()
        return ids


class FassiVDB:
    """
    Single FAISS index file handle.

    index_type:
      flat_l2 — IndexIDMap(IndexFlatL2 or IndexScalarQuantizer)
      hnsw    — IndexIDMap2(IndexHNSWFlat or IndexHNSWSQ)
    index_quant:
      none — float32
      fp16 — ScalarQuantizer QT_fp16
    """

    def __init__(
        self,
        vdb_path,
        vdb_dim,
        *,
        create_empty: bool = True,
        index_type: str = 'hnsw',
        index_quant: str = 'none',
        hnsw_M: int = 32,
        hnsw_efConstruction: int = 200,
        hnsw_efSearch: int = 64,
    ):
        self.vdb_path = Path(vdb_path)
        self.vdb_dim = vdb_dim
        self.index_type = (index_type or 'hnsw').strip().lower()
        if self.index_type not in ('flat_l2', 'hnsw'):
            self.index_type = 'hnsw'
        self.index_quant = normalize_index_quant(index_quant)
        self.hnsw_M = max(2, int(hnsw_M or 32))
        self.hnsw_efConstruction = max(1, int(hnsw_efConstruction or 200))
        self.hnsw_efSearch = max(1, int(hnsw_efSearch or 64))
        self._lock = faiss_lock_for(self.vdb_path)
        self.vdb = None
        self.last_add_seconds = 0.0
        self.load(create_empty=create_empty)

    @property
    def ntotal(self) -> int:
        if self.vdb is None:
            return 0
        try:
            return int(self.vdb.ntotal)
        except Exception:
            return 0

    def _apply_hnsw_search_params(self):
        """Set efSearch on HNSW (wrapped or bare) after load / create."""
        if self.vdb is None:
            return
        inner = getattr(self.vdb, 'index', None)
        hnsw = None
        if inner is not None and hasattr(inner, 'hnsw'):
            hnsw = inner.hnsw
        elif hasattr(self.vdb, 'hnsw'):
            hnsw = self.vdb.hnsw
        if hnsw is not None:
            try:
                hnsw.efSearch = int(self.hnsw_efSearch)
            except Exception:
                pass

    def _create_empty_index(self):
        """Build an empty FAISS index for the configured backend."""
        d = int(self.vdb_dim)
        use_fp16 = self.index_quant == 'fp16'
        if self.index_type == 'hnsw':
            if use_fp16:
                base = faiss.IndexHNSWSQ(
                    d, faiss.ScalarQuantizer.QT_fp16, int(self.hnsw_M)
                )
            else:
                base = faiss.IndexHNSWFlat(d, int(self.hnsw_M))
            base.hnsw.efConstruction = int(self.hnsw_efConstruction)
            base.hnsw.efSearch = int(self.hnsw_efSearch)
            # IDMap2: supports reconstruct by id.
            return faiss.IndexIDMap2(base)
        if use_fp16:
            return faiss.IndexIDMap(
                faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_fp16)
            )
        return faiss.IndexIDMap(faiss.IndexFlatL2(d))

    def load(self, *, create_empty: bool = True):
        if self.vdb_path.exists() and self.vdb_path.stat().st_size > 0:
            with self._lock:
                self.vdb = faiss.read_index(str(self.vdb_path))
            self._apply_hnsw_search_params()
        elif self.vdb_dim is not None and create_empty:
            self.vdb = self._create_empty_index()
            self.save()
        else:
            self.vdb = None

    def unload(self):
        """Release the in-memory index; file on disk is kept."""
        with self._lock:
            self.vdb = None

    def save(self):
        """
        Persist index atomically: write to *.vdb.tmp then os.replace.
        Avoids truncated/corrupt index if process dies mid-write; also
        reduces peak open-file risk vs in-place overwrite of multi-GB files.
        """
        if self.vdb is None:
            return
        with self._lock:
            if not self.vdb_path.parent.exists():
                self.vdb_path.parent.mkdir(parents=True)
            tmp_path = self.vdb_path.with_suffix(self.vdb_path.suffix + '.tmp')
            try:
                faiss.write_index(self.vdb, str(tmp_path))
                os.replace(str(tmp_path), str(self.vdb_path))
            except Exception:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass
                raise

    def clear(self):
        with self._lock:
            self.vdb = None
            file_path = Path(self.vdb_path)
            if file_path.exists():
                try:
                    file_path.unlink()
                except OSError:
                    pass
        self.load(create_empty=True)

    def add(self, ids, items) -> float:
        """
        Add vectors with ids. For HNSW this is the graph-construction cost.
        Returns wall seconds spent in add_with_ids.
        """
        if self.vdb is None:
            self.last_add_seconds = 0.0
            return 0.0
        id_array = np.array(ids, dtype=np.int64)
        t0 = time.perf_counter()
        with self._lock:
            self.vdb.add_with_ids(items, id_array)
        self.last_add_seconds = time.perf_counter() - t0
        return self.last_add_seconds

    def remove(self, ids):
        """Physical remove_ids. HNSW raises; caller should tombstone instead."""
        if self.vdb is None or not ids:
            return 0
        id_array = np.array(list(ids), dtype=np.int64)
        with self._lock:
            try:
                selector = faiss.IDSelectorBatch(id_array)
                removed = self.vdb.remove_ids(selector)
                return int(removed)
            except RuntimeError:
                return 0

    def search(self, item, topk):
        if self.vdb is None:
            return None, None
        with self._lock:
            self._apply_hnsw_search_params()
            distances, ids = self.vdb.search(item, topk)
        return distances, ids

    def _reconstruct_positions(self, inner, positions) -> np.ndarray:
        """Reconstruct float32 vectors at internal positions."""
        pos = np.asarray(positions, dtype=np.int64)
        if pos.size == 0:
            return np.zeros((0, int(self.vdb_dim)), dtype=np.float32)
        if hasattr(inner, 'reconstruct_batch'):
            try:
                out = inner.reconstruct_batch(pos)
                return np.asarray(out, dtype=np.float32)
            except Exception:
                pass
        out = np.zeros((pos.size, int(self.vdb_dim)), dtype=np.float32)
        for i, p in enumerate(pos):
            out[i] = inner.reconstruct(int(p))
        return out

    def compact_live(self, is_live, batch_size: int = 8192) -> dict:
        """
        Rebuild this index file from ids for which is_live(id) is true.
        Returns {kept, dropped, rewritten}. No-op if nothing is dead.
        """
        if self.vdb is None:
            return {'kept': 0, 'dropped': 0, 'rewritten': False}
        try:
            id_map = faiss.vector_to_array(self.vdb.id_map)
        except Exception:
            return {'kept': 0, 'dropped': 0, 'rewritten': False}

        live_pos = []
        dropped = 0
        for pos, nid in enumerate(id_map):
            try:
                ok = bool(is_live(nid))
            except Exception:
                ok = False
            if ok:
                live_pos.append(int(pos))
            else:
                dropped += 1

        if dropped <= 0:
            return {'kept': len(live_pos), 'dropped': 0, 'rewritten': False}

        inner = getattr(self.vdb, 'index', None)
        if inner is None:
            return {'kept': len(live_pos), 'dropped': dropped, 'rewritten': False}

        new_index = self._create_empty_index()
        new_inner = getattr(new_index, 'index', new_index)
        if live_pos and hasattr(new_inner, 'is_trained') and not new_inner.is_trained:
            sample_n = min(len(live_pos), 65536)
            sample = self._reconstruct_positions(inner, live_pos[:sample_n])
            if sample.shape[0] > 0:
                new_inner.train(sample)

        bs = max(256, int(batch_size or 8192))
        for start in range(0, len(live_pos), bs):
            chunk = live_pos[start:start + bs]
            vecs = self._reconstruct_positions(inner, chunk)
            ids = np.asarray([int(id_map[p]) for p in chunk], dtype=np.int64)
            if vecs.shape[0] == 0:
                continue
            new_index.add_with_ids(vecs, ids)

        with self._lock:
            self.vdb = new_index
            self._apply_hnsw_search_params()
        self.save()
        return {
            'kept': len(live_pos),
            'dropped': dropped,
            'rewritten': True,
        }

    def iter_live_vectors(self, is_live, batch_size: int = 8192):
        """Yield (ids: int64[N], vecs: float32[N, dim]) for rows passing is_live."""
        if self.vdb is None:
            return
        try:
            id_map = faiss.vector_to_array(self.vdb.id_map)
        except Exception:
            return
        inner = getattr(self.vdb, 'index', None)
        if inner is None:
            return
        try:
            bs = max(256, int(batch_size or 8192))
        except (TypeError, ValueError):
            bs = 8192
        live_pos = []
        live_ids = []
        for pos, nid in enumerate(id_map):
            try:
                ok = bool(is_live(nid))
            except Exception:
                ok = False
            if not ok:
                continue
            live_pos.append(int(pos))
            live_ids.append(int(nid))
            if len(live_pos) >= bs:
                vecs = self._reconstruct_positions(inner, live_pos)
                yield (
                    np.asarray(live_ids, dtype=np.int64),
                    np.asarray(vecs, dtype=np.float32),
                )
                live_pos = []
                live_ids = []
        if live_pos:
            vecs = self._reconstruct_positions(inner, live_pos)
            yield (
                np.asarray(live_ids, dtype=np.int64),
                np.asarray(vecs, dtype=np.float32),
            )
