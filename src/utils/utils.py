from functools import wraps
import hashlib
import inspect
import json
import random
import re
import threading
import time
from pathlib import Path

from .database import BaseDB

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


# 进程级：服务端过载时的共享冷却，避免 N 个并行任务同时重打模型
_retry_cooldown_lock = threading.Lock()
_retry_cooldown_until = 0.0  # time.monotonic()
_retry_cooldown_announced_until = 0.0
# 日志去重：(func_name, err_fingerprint, attempt) -> (last_log_mono, suppressed_count)
_retry_log_lock = threading.Lock()
_retry_log_state = {}


def _is_server_overload_error(err: BaseException) -> bool:
    """HTTP 5xx / EngineCore 等服务端故障，适合共享冷却。"""
    s = f'{type(err).__name__}: {err}'
    keys = (
        'Error code: 500',
        'Error code: 502',
        'Error code: 503',
        'Error code: 504',
        'InternalServerError',
        'EngineCore',
        'Service Unavailable',
        'Bad Gateway',
        'Gateway Timeout',
    )
    return any(k in s for k in keys)


def _err_fingerprint(err: BaseException) -> str:
    """用于日志去重：去掉文件名 / attempt / pages 等易变字段。"""
    s = str(err)
    s = re.sub(r'for\s+\S+\.pdf', 'for <file>.pdf', s, flags=re.I)
    s = re.sub(r'attempt\s+\d+\s*/\s*\d+', 'attempt ?/?', s, flags=re.I)
    s = re.sub(r'pages=\d+', 'pages=?', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if len(s) > 160:
        s = s[:160]
    return f'{type(err).__name__}|{s}'


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
        """第 attempt 次失败后的基础休眠秒数（attempt 从 1 起）。"""
        if not schedule:
            return 0.0
        idx = min(max(0, attempt - 1), len(schedule) - 1)
        return float(schedule[idx])

    @staticmethod
    def _with_jitter(delay: float) -> float:
        """±25% 抖动，打散并行任务的同时重试（防雷群）。"""
        if delay <= 0:
            return 0.0
        return max(0.0, delay * (0.75 + random.random() * 0.5))

    @staticmethod
    def _wait_shared_cooldown(logger, func_name: str):
        """若处于服务端过载冷却期，阻塞到冷却结束。"""
        global _retry_cooldown_until, _retry_cooldown_announced_until
        while True:
            with _retry_cooldown_lock:
                until = _retry_cooldown_until
                announced_until = _retry_cooldown_announced_until
            now = time.monotonic()
            remain = until - now
            if remain <= 0:
                return
            # 整段冷却只公告一次，避免 N 个 worker 各打一行
            if logger is not None and announced_until < until:
                with _retry_cooldown_lock:
                    if _retry_cooldown_announced_until < until:
                        _retry_cooldown_announced_until = until
                        logger.info(
                            f"[Retry] {func_name} shared server cooldown "
                            f"~{remain:.1f}s (all parallel workers pause)"
                        )
            time.sleep(min(max(remain, 0.05), 5.0))

    @staticmethod
    def _extend_shared_cooldown(seconds: float):
        """延长全局冷却（取较大值，不缩短已有冷却）。"""
        global _retry_cooldown_until
        if seconds <= 0:
            return
        with _retry_cooldown_lock:
            _retry_cooldown_until = max(
                _retry_cooldown_until,
                time.monotonic() + seconds,
            )

    @staticmethod
    def _log_failure(logger, func_name: str, attempt: int, max_attempt: int, err: BaseException):
        """
        并行时同一类错误只详打一条，其余压缩为计数，避免日志雪崩。
        """
        if logger is None:
            return
        level = logger.warning if attempt >= max_attempt else logger.info
        fp = _err_fingerprint(err)
        key = (func_name, fp, attempt)
        now = time.monotonic()
        with _retry_log_lock:
            last_t, suppressed = _retry_log_state.get(key, (0.0, 0))
            # 2s 内同类失败：累计，不刷屏
            if now - last_t < 2.0 and last_t > 0:
                _retry_log_state[key] = (last_t, suppressed + 1)
                if suppressed + 1 in (1, 2, 5, 10, 20, 50) or (suppressed + 1) % 50 == 0:
                    level(
                        f"[Retry] {func_name} attempt {attempt}/{max_attempt} "
                        f"same error x{suppressed + 1} (concurrent workers; "
                        f"detail suppressed): {type(err).__name__}: "
                        f"{str(err)[:160]}"
                    )
                return
            # 若有上一波被抑制的，先补一行
            if suppressed > 0:
                level(
                    f"[Retry] {func_name} attempt {attempt}/{max_attempt}: "
                    f"+{suppressed} similar concurrent failures suppressed"
                )
            _retry_log_state[key] = (now, 0)

        # 首条完整日志（不再误标 http_timeout）
        overload = ' [server 5xx]' if _is_server_overload_error(err) else ''
        level(
            f"[Retry] {func_name} attempt {attempt}/{max_attempt} failed"
            f"{overload}: {type(err).__name__}: {err}"
        )

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            max_attempt, wait_schedule, timeout = self._resolve_params(args)
            # 单次调用超时由底层 HTTP/LLM timeout 执行；此处 timeout 仅作配置透传/日志参考
            logger = None
            if args:
                logger = getattr(args[0], 'logger', None)
            func_name = getattr(func, '__qualname__', repr(func))

            last_err = None
            for attempt in range(1, max_attempt + 1):
                # 并行识别等场景：先等共享冷却，再打模型
                self._wait_shared_cooldown(logger, func_name)

                kwargs_with_attempt = {
                    **kwargs,
                    'attempt': attempt,
                    'max_attempt': max_attempt,
                }
                try:
                    return func(*args, **kwargs_with_attempt)
                except NonRetryableError:
                    # 不可重试：原样抛出，由调用方处理（勿吞成 None，避免误报「重试耗尽」）
                    raise
                except Exception as e:
                    last_err = e
                    self._log_failure(logger, func_name, attempt, max_attempt, e)

                    if attempt >= max_attempt:
                        break

                    base = self._backoff_seconds(wait_schedule, attempt)
                    delay = self._with_jitter(base)
                    if delay <= 0:
                        continue

                    # 服务端 5xx：只靠共享冷却（全员一起停），不再每人 sleep + 刷 backoff 日志
                    if _is_server_overload_error(e):
                        self._extend_shared_cooldown(max(delay, base))
                        continue

                    if logger is not None:
                        logger.info(
                            f"[Retry] {func_name} backoff {delay:.1f}s "
                            f"before attempt {attempt + 1}/{max_attempt}"
                        )
                    time.sleep(delay)

            if logger is not None and last_err is not None:
                logger.error(
                    f"[Retry] {func_name} exhausted "
                    f"{max_attempt} attempts; last_error="
                    f"{type(last_err).__name__}: {last_err}"
                )
            return None

        return wrapper
