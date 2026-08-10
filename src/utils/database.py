import faiss
import os
import sqlite3
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
    def __init__(self,vdb_path: str,vdb_name: str,vdb_dim: int):
        self.vdb_path = Path(vdb_path)
        self.vdb_name = vdb_name
        self.vdb_dim = vdb_dim
        self.load()
        
        self.buffer=[]

    def load(self):
        self.vdb_file_path = self.vdb_path / f'{self.vdb_name}.vdb'
        self.vdb = FassiVDB(self.vdb_file_path,self.vdb_dim)

    def save(self):
        self.vdb.save()

    def clear(self):
        self.vdb.clear()
        self.load()

    def buffer_clear(self):
        self.buffer=[]

    def add(self,data_list: list):
        if data_list == []:
            return
        ids = [data['id'] for data in data_list]
        # 兼容：embedding 可为 list/ndarray，或历史 JSON 字符串
        vecs = []
        for data in data_list:
            emb = data['embedding']
            if isinstance(emb, str):
                emb = json.loads(emb)
            vecs.append(emb)
        vectors = np.asarray(vecs, dtype=np.float32)
        # 幂等：同 id 先删再加，避免中断重跑 / 误重嵌时 FAISS 出现重复 id
        self.vdb.remove(ids)
        self.vdb.add(ids, vectors)
    
    def search(self,vector: list,topk: int=10):        
        vector = np.array(vector).reshape(1, -1)
        distances,ids = self.vdb.search(vector, topk)
        if ids is None:
            return []
        pairs = [(float(d), int(i)) for d, i in zip(distances[0], ids[0]) if i != -1]

        res=[{'distance':p[0],'id':p[1]} for p in pairs]
        return res

    def remove(self, ids):
        """Remove vectors by id list from the FAISS index and persist."""
        if not ids:
            return 0
        n = self.vdb.remove(ids)
        self.save()
        return n

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
    def __init__(self,vdb_path,vdb_dim):
        self.vdb_path = Path(vdb_path)
        self.vdb_dim = vdb_dim
        self._lock = faiss_lock_for(self.vdb_path)
        self.load()

    def load(self):
        if self.vdb_path.exists():
            with self._lock:
                self.vdb = faiss.read_index(str(self.vdb_path))
        elif self.vdb_dim is not None:
            self.vdb = faiss.IndexIDMap(faiss.IndexFlatL2(self.vdb_dim))
            self.save()
        else:
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
        if self.vdb is None:
            return
        with self._lock:
            file_path = Path(self.vdb_path)
            if file_path.exists():
                file_path.unlink()
        self.load()

    def add(self,ids,items):
        if self.vdb is None:
            return
        with self._lock:
            self.vdb.add_with_ids(items,np.array(ids,dtype=np.int64))

    def remove(self, ids):
        """Remove vectors by ids. Returns number of vectors removed."""
        if self.vdb is None or not ids:
            return 0
        id_array = np.array(list(ids), dtype=np.int64)
        with self._lock:
            selector = faiss.IDSelectorBatch(id_array)
            removed = self.vdb.remove_ids(selector)
        return int(removed)
    
    def search(self,item,topk):
        if self.vdb is None:
            return None,None
        with self._lock:
            distances, ids = self.vdb.search(item, topk)

        return distances,ids