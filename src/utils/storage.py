import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from .database import BaseDB, BaseVDB

_log = logging.getLogger(__name__)

_FTS_TABLE = 'chunk_fts'
_FTS_META = 'chunk_fts_meta'
_FTS_TOKENIZER = 'trigram'
# trigram 对不足 3 个字符的 token 不建索引；短词改走 instr。
_FTS_MIN_CHARS = 3

class DocDB(BaseDB):
    def __init__(self,db_path):
        name='doc'
        create_table_sql = \
            f"""
            CREATE TABLE IF NOT EXISTS {name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT DEFAULT 'new',
                name TEXT,
                content TEXT,
                extra TEXT,
                tokens INT,
                embedding_content TEXT,
                embedding_status TEXT DEFAULT 'undone', 
                hash TEXT UNIQUE
            );
            """
        super().__init__(db_path,name,create_table_sql)

class ChunkDB(BaseDB):
    def __init__(self, db_path):
        name = 'chunk'
        create_table_sql = \
            f"""
            CREATE TABLE IF NOT EXISTS {name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT DEFAULT 'new',
                doc_id INTEGER,
                name TEXT,
                content TEXT,
                extra TEXT,
                tokens INT,
                embedding_content TEXT,
                embedding_status TEXT DEFAULT 'undone'
            );
            """
        super().__init__(db_path, name, create_table_sql)

    def add(self, data_list: list, *, return_ids: bool = False):
        ids = super().add(data_list, return_ids=True) or []
        pairs = []
        for row, cid in zip(data_list or [], ids):
            if not cid:
                continue
            try:
                pairs.append((int(cid), (row.get('content') or '').casefold()))
            except (TypeError, ValueError):
                continue
        if pairs:
            try:
                self.fts_upsert(pairs)
            except Exception as e:
                _log.warning(f"[chunk_fts] upsert after add failed: {e}")
        return ids if return_ids else None

    def delete_by_ids(self, ids):
        n = super().delete_by_ids(ids)
        try:
            self.fts_delete_ids(ids)
        except Exception as e:
            _log.warning(f"[chunk_fts] delete_ids failed: {e}")
        return n

    def clear(self):
        try:
            self.fts_drop()
        except Exception as e:
            _log.warning(f"[chunk_fts] drop on clear failed: {e}")
        super().clear()

    def fts_drop(self):
        self.db.execute(f"DROP TABLE IF EXISTS {_FTS_TABLE}")
        self.db.execute(f"DROP TABLE IF EXISTS {_FTS_META}")

    def fts_schema_ok(self) -> bool:
        rows = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (_FTS_TABLE,),
        ) or []
        return bool(rows)

    def ensure_fts(self, *, force: bool = False) -> dict:
        """
        保证 chunk_fts 存在且与 chunk 行数一致。
        索引正文为 content.casefold()，供 MATCH / instr 做大小写无关子串检索。
        """
        t0 = time.perf_counter()
        self._fts_ensure_schema()
        n_chunk = self._count_table(self.table)
        n_fts = self._fts_meta_n()
        rebuilt = False
        if force or n_fts != n_chunk:
            self._fts_rebuild()
            n_fts = self._fts_meta_n()
            rebuilt = True
        dt = time.perf_counter() - t0
        info = {
            'n_chunk': n_chunk,
            'n_fts': n_fts,
            'rebuilt': rebuilt,
            'dt_s': round(dt, 3),
            'tokenizer': _FTS_TOKENIZER,
        }
        _log.info(
            f"[chunk_fts] ensure rebuilt={rebuilt} chunk={n_chunk} "
            f"fts={n_fts} dt={dt:.3f}s"
        )
        return info

    def fts_upsert(self, pairs):
        """pairs: iterable of (chunk_id, casefolded_content)."""
        pairs = [(int(cid), content or '') for cid, content in (pairs or [])]
        if not pairs:
            return
        self._fts_ensure_schema()
        ids = [p[0] for p in pairs]
        self.fts_delete_ids(ids)
        sql = f"INSERT INTO {_FTS_TABLE}(rowid, content) VALUES (?, ?)"
        self.db.execute_batch(sql, pairs)
        self._fts_touch_meta()

    def fts_delete_ids(self, ids):
        if not ids or not self.fts_schema_ok():
            return
        ids = [int(i) for i in ids if i is not None]
        if not ids:
            return
        for i in range(0, len(ids), 400):
            part = ids[i:i + 400]
            ph = ','.join('?' * len(part))
            self.db.execute(
                f"DELETE FROM {_FTS_TABLE} WHERE rowid IN ({ph})",
                tuple(part),
            )
        self._fts_touch_meta()

    def fts_match_keywords(self, keywords, *, max_id: int = 0) -> set:
        """
        对已小写入库的 chunk_fts 做子串检索，返回 chunk id 集合。
        词长 >= 3：FTS5 MATCH（trigram）；更短：instr（C 层扫描小写正文）。
        max_id>0 时只返回 rowid <= max_id（范围检索）。
        """
        out = set()
        kws = [str(k).casefold().strip() for k in (keywords or []) if k]
        kws = [k for k in kws if k]
        if not kws:
            return out
        if not self.fts_schema_ok():
            self.ensure_fts()
        try:
            cap = int(max_id or 0)
        except (TypeError, ValueError):
            cap = 0
        for kw in kws:
            out.update(self._fts_match_one(kw, max_id=cap))
        return out

    def _fts_match_one(self, kw_cf: str, *, max_id: int = 0) -> set:
        if not kw_cf:
            return set()
        try:
            cap = int(max_id or 0)
        except (TypeError, ValueError):
            cap = 0
        id_clause = " AND rowid <= ?" if cap > 0 else ""
        id_params = (cap,) if cap > 0 else ()
        rows = None
        if len(kw_cf) >= _FTS_MIN_CHARS:
            q = '"' + kw_cf.replace('"', '""') + '"'
            try:
                rows = self.db.execute(
                    f"SELECT rowid AS id FROM {_FTS_TABLE} "
                    f"WHERE {_FTS_TABLE} MATCH ?{id_clause}",
                    (q,) + id_params,
                ) or []
            except Exception as e:
                _log.warning(
                    f"[chunk_fts] MATCH {kw_cf!r} failed, instr fallback: {e}"
                )
                rows = None
        if rows is None:
            rows = self.db.execute(
                f"SELECT rowid AS id FROM {_FTS_TABLE} "
                f"WHERE instr(content, ?) > 0{id_clause}",
                (kw_cf,) + id_params,
            ) or []
        ids = set()
        for row in rows:
            cid = row.get('id') if isinstance(row, dict) else None
            if cid is None:
                continue
            try:
                ids.add(int(cid))
            except (TypeError, ValueError):
                continue
        return ids

    def _fts_ensure_schema(self):
        self.db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE} "
            f"USING fts5(content, tokenize='{_FTS_TOKENIZER}')"
        )
        self.db.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_FTS_META} (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                n_indexed INTEGER NOT NULL DEFAULT 0,
                tokenizer TEXT,
                built_at TEXT
            )
            """
        )

    def _fts_rebuild(self):
        t0 = time.perf_counter()
        self.fts_drop()
        self._fts_ensure_schema()
        after_id = 0
        batch = 2000
        n_ins = 0
        while True:
            rows = self.db.execute(
                f"SELECT id, content FROM {self.table} "
                f"WHERE id > ? ORDER BY id ASC LIMIT ?",
                (after_id, batch),
            ) or []
            if not rows:
                break
            pairs = []
            for row in rows:
                try:
                    cid = int(row['id'])
                except (TypeError, ValueError, KeyError):
                    continue
                pairs.append((cid, (row.get('content') or '').casefold()))
                after_id = cid
            if pairs:
                self.db.execute_batch(
                    f"INSERT INTO {_FTS_TABLE}(rowid, content) VALUES (?, ?)",
                    pairs,
                )
                n_ins += len(pairs)
                if n_ins % 50000 < batch:
                    _log.info(f"[chunk_fts] rebuild indexed={n_ins}")
        self._fts_touch_meta()
        _log.info(
            f"[chunk_fts] rebuild done n={n_ins} dt={time.perf_counter() - t0:.3f}s"
        )

    def _fts_meta_n(self) -> int:
        try:
            rows = self.db.execute(
                f"SELECT n_indexed FROM {_FTS_META} WHERE id = 1"
            ) or []
            if not rows:
                return -1
            return int((rows[0] or {}).get('n_indexed') or -1)
        except Exception:
            return -1

    def _fts_touch_meta(self):
        n = self._count_table(_FTS_TABLE) if self.fts_schema_ok() else 0
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        self.db.execute(
            f"INSERT INTO {_FTS_META}(id, n_indexed, tokenizer, built_at) "
            f"VALUES (1, ?, ?, ?) "
            f"ON CONFLICT(id) DO UPDATE SET "
            f"n_indexed=excluded.n_indexed, "
            f"tokenizer=excluded.tokenizer, "
            f"built_at=excluded.built_at",
            (n, _FTS_TOKENIZER, now),
        )

    def _count_table(self, table: str) -> int:
        try:
            rows = self.db.execute(f"SELECT COUNT(*) AS cnt FROM {table}") or []
            return int((rows[0] or {}).get('cnt') or 0)
        except Exception:
            return -1

class HyperedgeDB(BaseDB):
    def __init__(self,db_path):
        name='hyperedge'
        create_table_sql = \
            f"""
            CREATE TABLE IF NOT EXISTS {name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT,
                doc_id INTEGER,
                chunk_id INTEGER,
                name TEXT,
                content TEXT,
                extra TEXT,
                tokens INT,
                embedding_content TEXT,
                embedding_status TEXT DEFAULT 'undone'
            );
            """
        super().__init__(db_path,name,create_table_sql)

class NodeDB(BaseDB):
    def __init__(self,db_path):
        name='node'
        create_table_sql = \
            f"""
            CREATE TABLE IF NOT EXISTS {name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT,
                doc_id INTEGER,
                chunk_id INTEGER,
                hyperedge_id INTEGER,
                name TEXT,
                content TEXT,
                extra TEXT,
                tokens INT,
                embedding_content TEXT,
                embedding_status TEXT DEFAULT 'undone'
            );
            """
        super().__init__(db_path,name,create_table_sql)

class EdgeDB(BaseDB):
    def __init__(self,db_path):
        name='edge'
        create_table_sql = \
            f"""
            CREATE TABLE IF NOT EXISTS {name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT,
                doc_id INTEGER,
                chunk_id INTEGER,
                hyperedge_id INTEGER,
                src_node_id INTEGER,
                dst_node_id INTEGER,
                name TEXT,
                content TEXT,
                extra TEXT,
                tokens INT,
                embedding_content TEXT,
                embedding_status TEXT DEFAULT 'undone'
            );
            """
        super().__init__(db_path,name,create_table_sql)

class DocVDB(BaseVDB):
    def __init__(self, vdb_path, dim, shard_max_vectors=None, **index_kwargs):
        super().__init__(
            vdb_path, 'doc', dim,
            shard_max_vectors=shard_max_vectors, **index_kwargs,
        )

class ChunkVDB(BaseVDB):
    def __init__(self, vdb_path, dim, shard_max_vectors=None, **index_kwargs):
        super().__init__(
            vdb_path, 'chunk', dim,
            shard_max_vectors=shard_max_vectors, **index_kwargs,
        )

class HyperedgeVDB(BaseVDB):
    def __init__(self, vdb_path, dim, shard_max_vectors=None, **index_kwargs):
        super().__init__(
            vdb_path, 'hyperedge', dim,
            shard_max_vectors=shard_max_vectors, **index_kwargs,
        )

class NodeVDB(BaseVDB):
    def __init__(self, vdb_path, dim, shard_max_vectors=None, **index_kwargs):
        super().__init__(
            vdb_path, 'node', dim,
            shard_max_vectors=shard_max_vectors, **index_kwargs,
        )

class EdgeVDB(BaseVDB):
    def __init__(self, vdb_path, dim, shard_max_vectors=None, **index_kwargs):
        super().__init__(
            vdb_path, 'edge', dim,
            shard_max_vectors=shard_max_vectors, **index_kwargs,
        )
