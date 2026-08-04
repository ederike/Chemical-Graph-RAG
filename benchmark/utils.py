"""benchmark 通用工具：JSON 解析、LLM 调用、文档抽样、进度条。"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# 进度条 / 失败输出（避免刷屏中间日志）
# ------------------------------------------------------------------
def progress_iter(
    iterable: Iterable,
    *,
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "it",
):
    """
    带进度条的迭代；无 tqdm 时退化为简单计数 stderr。

    注意：不能与 yield 写在同一函数里（否则 return tqdm 会变成空生成器）。
    """
    if _tqdm is not None:
        return _tqdm(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            file=sys.stderr,
            dynamic_ncols=True,
            leave=True,
            mininterval=0.2,
        )
    return _simple_progress(iterable, total=total, desc=desc)


def _simple_progress(
    iterable: Iterable,
    *,
    total: Optional[int] = None,
    desc: str = "",
):
    items = list(iterable) if total is None else iterable
    n = total if total is not None else len(items)  # type: ignore[arg-type]
    for i, x in enumerate(items, start=1):  # type: ignore[arg-type]
        print(f"\r{desc}: {i}/{n}", end="", file=sys.stderr, flush=True)
        yield x
    print(file=sys.stderr)


def fail_print(msg: str) -> None:
    """打印失败项（tqdm.write 避免打断进度条）。"""
    text = f"[FAIL] {msg}"
    if _tqdm is not None:
        _tqdm.write(text, file=sys.stderr)
    else:
        print(text, file=sys.stderr)


@contextmanager
def quiet_loggers(*names: str, level: int = logging.ERROR):
    """临时提高若干 logger 级别，抑制中间 DEBUG/INFO/WARNING。"""
    saved = []
    for name in names:
        lg = logging.getLogger(name)
        saved.append((lg, lg.level, list(lg.handlers), lg.propagate))
        lg.setLevel(level)
        # 避免 DHMF 等 console handler 仍输出 INFO
        for h in lg.handlers:
            try:
                h.setLevel(level)
            except Exception:
                pass
    try:
        yield
    finally:
        for lg, lvl, _handlers, prop in saved:
            lg.setLevel(lvl)
            for h in lg.handlers:
                try:
                    # 恢复时不强行改 handler level（可能原本就是 DEBUG）
                    pass
                except Exception:
                    pass
            lg.propagate = prop


def resolve_path(path: str | Path, base: Optional[Path] = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    root = base or project_root()
    return (root / p).resolve()


def load_docs_from_db(db_path: str | Path) -> List[Dict[str, Any]]:
    """读取 main.db 的 doc 表，返回 id/name/content 列表。"""
    db_path = resolve_path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"数据库不存在: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, name, content FROM doc WHERE content IS NOT NULL AND TRIM(content) != ''"
        ).fetchall()
    finally:
        conn.close()
    docs = []
    for r in rows:
        docs.append({
            "doc_id": r["id"],
            "name": r["name"] or f"doc_{r['id']}",
            "content": r["content"] or "",
        })
    if not docs:
        raise RuntimeError(f"doc 表无可用 content: {db_path}")
    return docs


def truncate_text(text: str, max_chars: Optional[int]) -> str:
    """
    按字符数截断文本。

    max_chars:
      - None / -1 / <=0 : 不做限制，原样返回
      - >0             : 超过则截断并加省略标记
    """
    if max_chars is None or int(max_chars) <= 0:
        return text
    limit = int(max_chars)
    if len(text) <= limit:
        return text
    # 预留省略标记长度，避免 limit 很小时切片异常
    keep = max(0, limit - 20)
    return text[:keep].rstrip() + "\n…(内容已截断)"


def format_docs_block(
    docs: Sequence[Dict[str, Any]],
    max_chars_per_doc: Optional[int] = 4000,
) -> str:
    """max_chars_per_doc=-1 时全文不截断。"""
    parts = []
    for i, d in enumerate(docs, start=1):
        content = truncate_text(d.get("content") or "", max_chars_per_doc)
        parts.append(
            f"【文档 {i}】文件名: {d.get('name')}\n"
            f"doc_id: {d.get('doc_id')}\n"
            f"内容:\n{content}"
        )
    return "\n\n".join(parts)


def extract_json_object(text: str) -> Optional[dict]:
    """从 LLM 输出中尽量解析出 JSON 对象。"""
    if not text:
        return None
    raw = str(text).strip()
    # 去掉 ```json ... ```
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    # 直接 parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 截取首个 {...}
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def call_llm(
    llm,
    *,
    system: str,
    user: str,
    model_args: dict,
    use_cache: bool = False,
) -> Dict[str, Any]:
    """
    调用 LLM.generate，返回统一结构：
      status, answer, usage_*, latency_s
    """
    t0 = time.perf_counter()
    resp = llm.generate(
        prompt={"system": system or "", "user": user},
        model_args=dict(model_args or {}),
        use_cache=use_cache,
    )
    latency = time.perf_counter() - t0
    if not isinstance(resp, dict):
        return {
            "status": 0,
            "answer": str(resp),
            "usage_prompt_tokens": None,
            "usage_completion_tokens": None,
            "usage_total_tokens": None,
            "latency_s": latency,
        }
    out = dict(resp)
    out["latency_s"] = latency
    return out


def parse_hop_spec(hop_counts: Dict[Any, Any]) -> Dict[int, int]:
    """
    规范化跳数配置。
    支持 {"1": 50, "2": 30} 或 {1: 50, 2: 30}
    """
    out: Dict[int, int] = {}
    for k, v in (hop_counts or {}).items():
        hop = int(k)
        n = int(v)
        if hop < 1:
            raise ValueError(f"跳数必须 >= 1，得到 {hop}")
        if n < 0:
            raise ValueError(f"问题数量不能为负: hop={hop} n={n}")
        if n > 0:
            out[hop] = n
    if not out:
        raise ValueError("hop_counts 为空，请至少配置一种跳数的问题数量")
    return dict(sorted(out.items()))


def mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(x) for x in values if x is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def safe_div(a: float, b: float) -> Optional[float]:
    if b == 0:
        return None
    return a / b


def normalize_filename(name: str) -> str:
    """归一化文件名便于召回比对（小写、去路径、去两端空格）。"""
    if not name:
        return ""
    s = str(name).strip().replace("\\", "/")
    s = s.split("/")[-1]
    return s.lower()


def match_doc_names(expected: Sequence[str], retrieved: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    返回 (hit, miss, retrieved_normalized_unique)。
    hit/miss 使用 expected 原始名；比对用归一化名。
    """
    ret_norm = {}
    for r in retrieved or []:
        n = normalize_filename(r)
        if n and n not in ret_norm:
            ret_norm[n] = r
    hit, miss = [], []
    for e in expected or []:
        if normalize_filename(e) in ret_norm:
            hit.append(e)
        else:
            miss.append(e)
    return hit, miss, list(ret_norm.values())
