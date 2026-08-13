import faiss
import os
import sqlite3
import time
from pathlib import Path
import threading
import numpy as np
import json

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

    def search_all(self):
        sql = f"SELECT * FROM {self.table}"
        return self.db.execute(sql)

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
    Search merges all sealed + active; sealed shards are loaded one-by-one.

    index_type:
      flat_l2 — IndexFlatL2 (exact, full scan)
      hnsw    — IndexHNSWFlat / IndexHNSWSQ (approx ANN; build cost is add_with_ids)
    index_quant:
      none — float32 vectors
      fp16 — FAISS ScalarQuantizer QT_fp16 (half-precision storage + search)
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

        if shard_max_vectors is not None and int(shard_max_vectors) > 0:
            self.enable_sharding(int(shard_max_vectors))
        else:
            self.load()

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

    def index_stats(self) -> dict:
        return {
            'index_type': self.index_type,
            'index_quant': self.index_quant,
            'hnsw_M': self.hnsw_M,
            'hnsw_efConstruction': self.hnsw_efConstruction,
            'hnsw_efSearch': self.hnsw_efSearch,
            'index_add_seconds': float(self.index_add_seconds),
            'index_add_count': int(self.index_add_count),
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
        # 幂等：flat_l2 可在 active 上先删再加。
        # HNSW 不支持 remove_ids，必须 vectorization_clear 后整库重建。
        self._ensure_active()
        if self.vdb is not None and self.index_type != 'hnsw':
            self.vdb.remove(ids)
        add_s = float(self.vdb.add(ids, vectors) or 0.0)
        self.index_add_seconds += add_s
        self.index_add_count += len(ids)
        return {'n': len(ids), 'add_seconds': add_s}

    def search(self, vector: list, topk: int = 10):
        vector = np.array(vector, dtype=np.float32).reshape(1, -1)
        if topk <= 0:
            return []

        if not self._sharding:
            if self.vdb is None:
                return []
            distances, ids = self.vdb.search(vector, topk)
            if ids is None:
                return []
            pairs = [(float(d), int(i)) for d, i in zip(distances[0], ids[0]) if i != -1]
            return [{'distance': p[0], 'id': p[1]} for p in pairs]

        # Merge top-k across sealed (load one-by-one) + active.
        heap = []  # (distance, id) keep best topk (smallest L2)
        import heapq

        def _consume(distances, ids):
            if ids is None:
                return
            for d, i in zip(distances[0], ids[0]):
                i = int(i)
                if i == -1:
                    continue
                d = float(d)
                if len(heap) < topk:
                    heapq.heappush(heap, (-d, i))  # max-heap via negation
                elif d < -heap[0][0]:
                    heapq.heapreplace(heap, (-d, i))

        for name in self._sealed:
            path = self.shards_dir / name
            if not path.exists():
                continue
            tmp = self._make_faiss(path, create_empty=False)
            try:
                distances, ids = tmp.search(vector, topk)
                _consume(distances, ids)
            finally:
                tmp.unload()

        if self.vdb is not None and self.vdb.vdb is not None:
            distances, ids = self.vdb.search(vector, topk)
            _consume(distances, ids)

        pairs = sorted(((-neg_d, i) for neg_d, i in heap), key=lambda x: x[0])
        return [{'distance': d, 'id': i} for d, i in pairs]

    def remove(self, ids, persist: bool = True):
        """Remove vectors by id list from active + all sealed shards.

        HNSW does not implement remove_ids; returns 0 and leaves vectors in place.
        For HNSW deletion / rebuild use vectorization_clear + vectorization.
        """
        if not ids:
            return 0
        if self.index_type == 'hnsw':
            return 0
        n = 0
        if not self._sharding:
            if self.vdb is None:
                return 0
            n = self.vdb.remove(ids)
            if persist and n:
                self.save()
            return n

        # Active
        if self.vdb is not None and self.vdb.vdb is not None:
            n += self.vdb.remove(ids)
            if persist and n:
                self.vdb.save()

        # Sealed: load → remove → save if hit → unload
        for name in list(self._sealed):
            path = self.shards_dir / name
            if not path.exists():
                continue
            tmp = self._make_faiss(path, create_empty=False)
            try:
                r = tmp.remove(ids)
                if r:
                    tmp.save()
                    n += r
            finally:
                tmp.unload()
        return n

    def iter_faiss_indexes(self):
        """
        Yield (label, faiss.Index) for recommend/export. Caller must not
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
        """Build id → vector across all shards (may be large; for recommend)."""
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
            # IDMap2: supports reconstruct by id (recommend path).
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
        """Remove vectors by ids. Returns number removed; 0 if unsupported (HNSW)."""
        if self.vdb is None or not ids:
            return 0
        id_array = np.array(list(ids), dtype=np.int64)
        with self._lock:
            try:
                selector = faiss.IDSelectorBatch(id_array)
                removed = self.vdb.remove_ids(selector)
                return int(removed)
            except RuntimeError:
                # IndexHNSWFlat does not implement remove_ids.
                return 0

    def search(self, item, topk):
        if self.vdb is None:
            return None, None
        with self._lock:
            self._apply_hnsw_search_params()
            distances, ids = self.vdb.search(item, topk)
        return distances, ids
