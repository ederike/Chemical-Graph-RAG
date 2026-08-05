import faiss
import sqlite3
from pathlib import Path
import threading
import numpy as np
import json
lock = threading.Lock()

class BaseDB:
    def __init__(self,db_path: str,db_name: str,create_table_sql: str):
        self.db_path = Path(db_path)
        self.table = db_name
        self.create_table_sql = create_table_sql
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

    def add(self,data_list: list):
        if data_list == []:
            return

        keys = list(data_list[0].keys())
        keys = [k for k in keys if k in self.table_columns]
        columns = ','.join(keys)
        placeholders = ','.join('?' * len(keys))
        sql = f'INSERT OR IGNORE INTO {self.table} ({columns}) VALUES ({placeholders})'
        values_list = [tuple(data[k] for k in keys) for data in data_list]
        self.db.execute_batch(sql, values_list)

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
        self.load()
    
    def load(self):
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True)
        if not self.db_path.exists():
            self.db_path.touch()

        with lock:
            self.conn = sqlite3.connect(self.db_path,check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            self.cursor.execute(self.create_table_sql)
            self.conn.commit()

    def save(self):
        ...
    
    def clear(self):
        with lock:
            self.cursor.execute(f"DROP TABLE IF EXISTS {self.table}")
            self.conn.commit()
        self.load()

    def execute(self,SQL,values=()):
        with lock:
            self.cursor.execute(SQL,values)
            self.conn.commit()
            res=self.cursor.fetchall()
        res=[dict(row) for row in res]
        return res

    def execute_batch(self,SQL,values_list):
        with lock:
            self.cursor.executemany(SQL,values_list)
            self.conn.commit()


class FassiVDB:
    def __init__(self,vdb_path,vdb_dim):
        self.vdb_path = Path(vdb_path)
        self.vdb_dim = vdb_dim
        self.load()

    def load(self):
        if self.vdb_path.exists():
            self.vdb = faiss.read_index(str(self.vdb_path))
        elif self.vdb_dim is not None:
            self.vdb = faiss.IndexIDMap(faiss.IndexFlatL2(self.vdb_dim))
            self.save()
        else:
            self.vdb = None
        
    def save(self):
        if self.vdb is None:
            return
        with lock:
            if not self.vdb_path.parent.exists():
                self.vdb_path.parent.mkdir(parents=True)
            faiss.write_index(self.vdb, str(self.vdb_path))

    def clear(self):
        if self.vdb is None:
            return
        file_path = Path(self.vdb_path)
        if file_path.exists():
            file_path.unlink()
        self.load()

    def add(self,ids,items):
        if self.vdb is None:
            return
        with lock:
            self.vdb.add_with_ids(items,np.array(ids,dtype=np.int64))

    def remove(self, ids):
        """Remove vectors by ids. Returns number of vectors removed."""
        if self.vdb is None or not ids:
            return 0
        id_array = np.array(list(ids), dtype=np.int64)
        with lock:
            # IndexIDMap supports remove_ids
            selector = faiss.IDSelectorBatch(id_array)
            removed = self.vdb.remove_ids(selector)
        return int(removed)
    
    def search(self,item,topk):
        if self.vdb is None:
            return None,None
        with lock:
            distances, ids = self.vdb.search(item, topk)

        return distances,ids