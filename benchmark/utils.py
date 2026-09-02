"""benchmark 通用工具：JSON 解析、LLM 调用、文档抽样、进度条。"""

from __future__ import annotations

import hashlib
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

def load_docs_from_db(
    db_path: str | Path,
    *,
    max_doc_id: int = 0,
) -> List[Dict[str, Any]]:
    """读取 main.db 的 doc 表，返回 id/name/content 列表。

    max_doc_id>0 时只取 id <= max_doc_id（例如 TDS 范围）。
    """
    db_path = resolve_path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"数据库不存在: {db_path}")
    try:
        cap = int(max_doc_id or 0)
    except (TypeError, ValueError):
        cap = 0
    sql = (
        "SELECT id, name, content FROM doc "
        "WHERE content IS NOT NULL AND TRIM(content) != ''"
    )
    params: list = []
    if cap > 0:
        sql += " AND id <= ?"
        params.append(cap)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, tuple(params)).fetchall()
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
    *,
    numbered: bool = True,
) -> str:
    """max_chars_per_doc=-1 时全文不截断。

    numbered=True：保留 Document i / filename / doc_id（评测裁判对照用）。
    numbered=False：出题用，文档之间只用无序号分隔，避免模型把「文档1」写进题目。
    """
    parts = []
    for i, d in enumerate(docs, start=1):
        content = truncate_text(d.get("content") or "", max_chars_per_doc)
        if numbered:
            parts.append(
                f"## Document {i}\n"
                f"filename: {d.get('name')}\n"
                f"doc_id: {d.get('doc_id')}\n"
                f"content:\n{content}"
            )
        else:
            parts.append(f"----- 产品资料 -----\n{content}")
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


EMPTY_ANSWER_PLACEHOLDER = "（无回答）"


def fill_prompt(template: str, mapping: Dict[str, Any]) -> str:
    """替换 {key}，并把模板里为 .format 转义的 {{ / }} 还原成单括号。

    回答正文可能含花括号，所以不用 str.format。
    """
    text = str(template or "")
    sentinels: Dict[str, str] = {}
    for i, key in enumerate(sorted(mapping, key=len, reverse=True)):
        token = f"@@PROMPT_SLOT_{i}@@"
        val = mapping[key]
        sentinels[token] = "" if val is None else str(val)
        text = text.replace("{" + key + "}", token)
    text = text.replace("{{", "{").replace("}}", "}")
    for token, val in sentinels.items():
        text = text.replace(token, val)
    return text


def pairwise_system_order(
    qid: Any,
    first: str = "hypergraph",
    second: str = "llm_only",
) -> Tuple[str, str]:
    """按题号确定性打乱 A/B 顺序，减轻位置偏差。"""
    raw = str(qid if qid is not None else "")
    bit = hashlib.md5(raw.encode("utf-8")).digest()[0] & 1
    if bit:
        return second, first
    return first, second


def extract_pair_side(obj: dict, side: str) -> dict:
    """从对比裁判 JSON 中抽出回答 A 或 B 的字段（可能为空 dict）。"""
    if not isinstance(obj, dict):
        return {}
    side = str(side).strip().upper()
    if side not in ("A", "B"):
        return {}
    low = side.lower()
    nested_keys = (
        f"answer_{low}",
        f"answer{low}",
        f"Answer_{side}",
        f"回答{side}",
        f"回答{low}",
        side,
        low,
    )
    for k in nested_keys:
        v = obj.get(k)
        if isinstance(v, dict):
            return v
    answers = obj.get("answers")
    if isinstance(answers, dict):
        for k in nested_keys:
            v = answers.get(k)
            if isinstance(v, dict):
                return v
    if isinstance(answers, list):
        for item in answers:
            if not isinstance(item, dict):
                continue
            lab = str(
                item.get("id") or item.get("label") or item.get("name") or ""
            ).strip()
            if lab.upper() in (side, f"ANSWER_{side}", f"ANSWER{side}", f"回答{side}"):
                return item
    flat: Dict[str, Any] = {}
    mapping = {
        "judgment": ("judgment", "llm_acc"),
        "score": ("score",),
        "reason": ("reason", "comment"),
        "dimension_scores": ("dimension_scores", "dimensions"),
    }
    for out_key, names in mapping.items():
        found = False
        for name in names:
            for k in (
                f"{name}_{low}",
                f"{low}_{name}",
                f"{name}{side}",
                f"{name}_{side}",
            ):
                if k in obj and obj[k] is not None:
                    flat[out_key] = obj[k]
                    found = True
                    break
            if found:
                break
    return flat


def extract_named_system_sides(obj: dict) -> Optional[Dict[str, dict]]:
    """若模型直接用 hypergraph / llm_only 作键，则按系统名取出。"""
    if not isinstance(obj, dict):
        return None
    hg = obj.get("hypergraph") or obj.get("超图") or obj.get("rag")
    lo = (
        obj.get("llm_only")
        or obj.get("纯LLM")
        or obj.get("llm")
        or obj.get("baseline")
    )
    if isinstance(hg, dict) and isinstance(lo, dict):
        return {"hypergraph": hg, "llm_only": lo}
    return None


def normalize_pairwise_better(raw: Any) -> Optional[str]:
    """把 better/winner 归一为 A / B / tie。"""
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = (
            raw.get("better")
            or raw.get("winner")
            or raw.get("prefer")
            or raw.get("choice")
        )
        if raw is None:
            return None
    s = str(raw).strip()
    if not s:
        return None
    compact = (
        s.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("回答", "")
        .replace("answer", "")
    )
    tie_keys = (
        "tie",
        "equal",
        "same",
        "neither",
        "both",
        "平",
        "平局",
        "相同",
        "一样",
        "并列",
        "都不",
        "都好",
        "相当",
    )
    if compact in tie_keys or s in ("平", "平局"):
        return "tie"
    if compact in ("a", "甲") or s.upper() == "A":
        return "A"
    if compact in ("b", "乙") or s.upper() == "B":
        return "B"
    if any(k in s for k in ("平局", "相同", "一样", "并列", "相当")):
        return "tie"
    has_a = bool(re.search(r"(回答\s*A|\bA\b|甲)", s, re.I))
    has_b = bool(re.search(r"(回答\s*B|\bB\b|乙)", s, re.I))
    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"
    return None


def resolve_pairwise_winner(
    raw: Any,
    *,
    a_system: str,
    b_system: str,
) -> Optional[str]:
    """把裁判的 better 映射为 hypergraph / llm_only / tie。"""
    ab = normalize_pairwise_better(raw)
    if ab == "A":
        return a_system
    if ab == "B":
        return b_system
    if ab == "tie":
        return "tie"
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = raw.get("better") or raw.get("winner") or raw.get("prefer")
        if raw is None:
            return None
    s = str(raw).strip()
    compact = s.lower().replace(" ", "").replace("_", "")
    if compact in ("hypergraph", "rag", "graph") or s in ("超图",):
        return "hypergraph"
    if compact in ("llmonly", "llm", "baseline") or s in ("纯LLM", "纯llm"):
        return "llm_only"
    return None


def pairwise_better_raw(obj: dict) -> Any:
    if not isinstance(obj, dict):
        return None
    if obj.get("better") is not None:
        return obj.get("better")
    if obj.get("winner") is not None:
        return obj.get("winner")
    if obj.get("prefer") is not None:
        return obj.get("prefer")
    cmp_ = obj.get("comparison")
    if isinstance(cmp_, dict):
        return (
            cmp_.get("better")
            or cmp_.get("winner")
            or cmp_.get("prefer")
        )
    if isinstance(cmp_, str):
        return cmp_
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


def pin_retrieve_for_eval(dhmf) -> bool:
    """评测前常驻检索索引。成功返回 True，供 finally 里 unpin。"""
    if dhmf is None or not hasattr(dhmf, "pin_retrieve_indexes"):
        return False
    print("[eval] pin_retrieve_indexes …", file=sys.stderr, flush=True)
    dhmf.pin_retrieve_indexes()
    print("[eval] pin_retrieve_indexes done", file=sys.stderr, flush=True)
    return True


def unpin_retrieve_for_eval(dhmf, pinned: bool) -> None:
    if not pinned or dhmf is None or not hasattr(dhmf, "unpin_retrieve_indexes"):
        return
    print("[eval] unpin_retrieve_indexes …", file=sys.stderr, flush=True)
    try:
        dhmf.unpin_retrieve_indexes()
        print("[eval] unpin_retrieve_indexes done", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[eval] unpin_retrieve_indexes failed: {e}", file=sys.stderr, flush=True)
