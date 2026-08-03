from functools import wraps
import concurrent.futures
import time
from .database import BaseDB
from pathlib import Path
import hashlib
import json
import inspect


def hash_str(s):
    return hashlib.md5(s.encode()).hexdigest()

class CacheDB(BaseDB):
    def __init__(self,cache_path,cache_name):
        cache_path=Path(cache_path)
        db_path= cache_path / f'{cache_name}.db'
        table='cache'
        create_table_sql = \
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash TEXT UNIQUE,
                input TEXT,
                output TEXT  
            );
            """
        super().__init__(db_path,table,create_table_sql)

    def search_cache(self,input_hash):
        res_json = self.db.execute(f'SELECT output FROM {self.table} WHERE hash = ?',(input_hash,))
        if len(res_json) == 0:
            return None
        return res_json[0]['output']
    
    def update_cache(self,input_json,output_json,input_hash):
        self.db.execute(f'INSERT OR IGNORE INTO {self.table} (hash,input,output) VALUES (?,?,?)',(input_hash,input_json,output_json))

    def delete_by_hash(self, input_hash: str) -> int:
        rows = self.db.execute(f'SELECT id FROM {self.table} WHERE hash = ?', (input_hash,))
        if not rows:
            return 0
        self.db.execute(f'DELETE FROM {self.table} WHERE hash = ?', (input_hash,))
        return len(rows)

    def delete_by_name(self, name: str) -> int:
        """
        Delete cache rows whose input JSON has "name" == name
        (used by PDF recognition cache: input stores {name, file_hash}).
        """
        rows = self.db.execute(f'SELECT hash, input FROM {self.table}')
        deleted = 0
        for row in rows:
            try:
                inp = json.loads(row['input']) if row.get('input') else {}
            except Exception:
                continue
            if isinstance(inp, dict) and inp.get('name') == name:
                self.db.execute(f'DELETE FROM {self.table} WHERE hash = ?', (row['hash'],))
                deleted += 1
        return deleted

    def delete_all(self) -> int:
        rows = self.db.execute(f'SELECT id FROM {self.table}')
        self.db.execute(f'DELETE FROM {self.table}')
        return len(rows)

class Cache:
    def __init__(self,cache_dir='cache',cache_name='cache'):
        self.cache_dir = cache_dir
        self.cache_name = cache_name
        self.cache_db = CacheDB(self.cache_dir,self.cache_name)

    @staticmethod
    def input_hash_from_bound(prompt, model_args) -> str:
        """与 @Cache 装饰的 generate(prompt, model_args) 键一致（不含 kwargs）。"""
        inp = {'prompt': prompt, 'model_args': model_args}
        inp_json = json.dumps(inp, ensure_ascii=False, indent=2)
        return hashlib.md5(inp_json.encode('utf-8')).hexdigest()

    def drop_for_generate(self, prompt, model_args) -> int:
        """删除某次 LLM generate 对应的缓存条目。"""
        h = self.input_hash_from_bound(prompt, model_args)
        return self.cache_db.delete_by_hash(h)

    @staticmethod
    def _should_store(out) -> bool:
        """
        失败结果不进缓存：
          - status 明确非 1（API/调用失败）
          - 调用方设置 _cache_store=False（如抽取 JSON 校验失败）
        """
        if not isinstance(out, dict):
            return True
        if out.get('_cache_store') is False:
            return False
        st = out.get('status')
        if st is not None and st != 1:
            return False
        return True
    
    def __call__(self,func):
        @wraps(func)
        def wrapper(*args,**kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind(*args,**kwargs)
            bound_args.apply_defaults()

            inp=bound_args.arguments.copy()
            inp.pop('self',None)
            inp.pop('cls',None)
            inp_kwargs=inp.pop('kwargs',{})

            use_cache=inp_kwargs.get('use_cache',True)
            if not use_cache:
                out = func(*args,**kwargs)
                if isinstance(out, dict):
                    out = dict(out)
                    out['_cache_hit'] = False
                return out
            
            inp_json = json.dumps(inp,ensure_ascii=False,indent=2)
            inp_hash = hashlib.md5(inp_json.encode('utf-8')).hexdigest()

            out_json = self.cache_db.search_cache(inp_hash)

            if out_json is not None:
                out = json.loads(out_json)['output']
                # Mark cache hit so metrics can exclude time/tokens from totals.
                # Do not persist this flag into the cache store.
                if isinstance(out, dict):
                    out = dict(out)
                    out['_cache_hit'] = True
            else:
                out = func(*args,**kwargs)
                if isinstance(out, dict):
                    out = dict(out)
                    out['_cache_hit'] = False
                # 仅缓存成功/可复用结果；失败不进库，避免永久毒化
                if self._should_store(out):
                    to_store = out
                    if isinstance(to_store, dict):
                        to_store = {
                            k: v for k, v in to_store.items()
                            if not str(k).startswith('_')
                        }
                    payload = {'output': to_store}
                    out_json = json.dumps(payload, ensure_ascii=False, indent=2)
                    self.cache_db.update_cache(inp_json, out_json, inp_hash)

            return out
    
        return wrapper

class Retry:
    def __init__(self,max_attempt=5,wait=0.1,timeout=10000):
        self.max_attempt = max_attempt
        self.wait = wait
        self.timeout = timeout

    def __call__(self,func):
        @wraps(func)
        def wrapper(*args,**kwargs):
            
            for attempt in range(1,self.max_attempt+1):
                kwargs_with_attempt = {**kwargs, 'attempt': attempt,'max_attempt':self.max_attempt}
                try:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(func, *args, **kwargs_with_attempt)
                        result = future.result(timeout=self.timeout)
                    return result
                except Exception as e:
                    time.sleep(self.wait)

            return None
        
        return wrapper
