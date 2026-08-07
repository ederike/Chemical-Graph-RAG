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


# 构建流水线进度条：不显示 elapsed/remaining，只保留速率 + postfix
TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]'

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

class NonRetryableError(Exception):
    """明确不可恢复的错误：@Retry 遇到后不再重试，直接失败。"""


def _normalize_wait_schedule(wait) -> list:
    """
    将 wait 规范为非负秒数列表（指数退避 schedule）。

    支持：
      - 标量 float/int → [wait]
      - list/tuple → [w0, w1, ...]（如 [10, 30, 60]）
      - 逗号分隔字符串 → 同上
    第 n 次失败后的休眠取 schedule[min(n-1, len-1)]。
    """
    if wait is None:
        return [0.0]
    if isinstance(wait, (list, tuple)):
        raw = list(wait)
    elif isinstance(wait, str):
        s = wait.strip()
        if not s:
            return [0.0]
        if ',' in s:
            raw = [p.strip() for p in s.split(',') if p.strip()]
        else:
            raw = [s]
    else:
        raw = [wait]

    out = []
    for x in raw:
        try:
            out.append(max(0.0, float(x)))
        except (TypeError, ValueError):
            continue
    return out if out else [0.0]


class Retry:
    """
    方法级重试装饰器。

    参数优先级（高 → 低）：
      1. 运行时从实例 self.config 读取 config_attr（如 'doc.recognition.retry'）
      2. 装饰器构造时传入的 max_attempt / wait / timeout 默认值

    config_attr 指向带 max_attempt / wait / timeout 的对象或 dict。

    wait 支持标量或退避序列：
      wait=0.1          → 每次失败固定等 0.1s
      wait=[10, 30, 60] → 第 1/2/3 次失败后分别等 10/30/60s（超出序列则用末项）
    """

    def __init__(
        self,
        max_attempt: int = 5,
        wait=0.1,
        timeout: float = 10000,
        config_attr: str = None,
    ):
        self.max_attempt = max_attempt
        self.wait = wait
        self.timeout = timeout
        self.config_attr = config_attr

    def _resolve_params(self, args):
        max_attempt = self.max_attempt
        wait = self.wait
        timeout = self.timeout
        if not self.config_attr or not args:
            return (
                max(1, int(max_attempt)),
                _normalize_wait_schedule(wait),
                max(0.0, float(timeout)),
            )

        cfg_root = getattr(args[0], 'config', None)
        if cfg_root is None:
            return (
                max(1, int(max_attempt)),
                _normalize_wait_schedule(wait),
                max(0.0, float(timeout)),
            )

        obj = cfg_root
        try:
            for part in self.config_attr.split('.'):
                obj = getattr(obj, part)
        except AttributeError:
            return (
                max(1, int(max_attempt)),
                _normalize_wait_schedule(wait),
                max(0.0, float(timeout)),
            )

        if obj is None:
            return (
                max(1, int(max_attempt)),
                _normalize_wait_schedule(wait),
                max(0.0, float(timeout)),
            )

        if isinstance(obj, dict):
            max_attempt = int(obj.get('max_attempt', max_attempt))
            wait = obj.get('wait', wait)
            timeout = float(obj.get('timeout', timeout))
        else:
            max_attempt = int(getattr(obj, 'max_attempt', max_attempt))
            wait = getattr(obj, 'wait', wait)
            timeout = float(getattr(obj, 'timeout', timeout))

        return (
            max(1, int(max_attempt)),
            _normalize_wait_schedule(wait),
            max(0.0, float(timeout)),
        )

    @staticmethod
    def _backoff_seconds(schedule: list, attempt: int) -> float:
        """第 attempt 次失败后的休眠秒数（attempt 从 1 起）。"""
        if not schedule:
            return 0.0
        idx = min(max(0, attempt - 1), len(schedule) - 1)
        return float(schedule[idx])

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            max_attempt, wait_schedule, timeout = self._resolve_params(args)
            # 尽量从 self.logger 打日志（Doc/Extract 等实例方法）
            logger = None
            if args:
                logger = getattr(args[0], 'logger', None)

            last_err = None
            for attempt in range(1, max_attempt + 1):
                kwargs_with_attempt = {
                    **kwargs,
                    'attempt': attempt,
                    'max_attempt': max_attempt,
                }
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = None
                try:
                    future = executor.submit(func, *args, **kwargs_with_attempt)
                    result = future.result(timeout=timeout)
                    return result
                except NonRetryableError as e:
                    last_err = e
                    if logger is not None:
                        logger.error(
                            f"[Retry] {func.__qualname__} non-retryable: "
                            f"{type(e).__name__}: {e}"
                        )
                    return None
                except Exception as e:
                    last_err = e
                    err_msg = f"{type(e).__name__}: {e}"
                    if logger is not None:
                        # 每次失败都打出原因（此前静默吞掉，日志只剩 Failed after retries）
                        level = logger.warning if attempt >= max_attempt else logger.info
                        level(
                            f"[Retry] {func.__qualname__} "
                            f"attempt {attempt}/{max_attempt} failed "
                            f"(timeout={timeout}s): {err_msg}"
                        )
                    if attempt < max_attempt:
                        delay = self._backoff_seconds(wait_schedule, attempt)
                        if delay > 0:
                            if logger is not None:
                                logger.info(
                                    f"[Retry] {func.__qualname__} backoff "
                                    f"{delay:.1f}s before attempt "
                                    f"{attempt + 1}/{max_attempt}"
                                )
                            time.sleep(delay)
                finally:
                    # 超时后不要 wait=True 堵死下一次重试（后台请求仍可能跑完）
                    try:
                        executor.shutdown(wait=False, cancel_futures=True)
                    except TypeError:
                        # Python <3.9 无 cancel_futures
                        executor.shutdown(wait=False)

            if logger is not None and last_err is not None:
                logger.error(
                    f"[Retry] {func.__qualname__} exhausted "
                    f"{max_attempt} attempts; last_error={type(last_err).__name__}: {last_err}"
                )
            return None

        return wrapper
