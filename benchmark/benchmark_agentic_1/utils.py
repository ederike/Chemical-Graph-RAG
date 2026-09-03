"""benchmark_agentic_1 通用工具：路径、文档抽样、LLM 调用、JSON 解析。"""

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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None

EMPTY_ANSWER_PLACEHOLDER = "（无回答）"

SYSTEM_AGENTIC = "agentic"
SYSTEM_LLM_ONLY = "llm_only"
JUDGMENT_LABELS = ("正确", "错误")

# retrieve_items 分阶段耗时（与 src.module.retrieve.RETRIEVE_TIMING_KEYS 对齐，本模块不 import 那边）
RETRIEVE_STAGE_KEYS = (
    "precompute_s",
    "rewrite_s",
    "embed_s",
    "chunk_s",
    "node_s",
    "keyword_s",
    "expand_s",
    "rerank_s",
    "total_s",
)


def project_root() -> Path:
    """仓库根目录：benchmark/benchmark_agentic_1/ 上两级。"""
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(path: Union[str, Path], base: Optional[Path] = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    root = base or project_root()
    return (root / p).resolve()


def progress_iter(
    iterable: Iterable,
    *,
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "it",
):
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
    text = f"[FAIL] {msg}"
    if _tqdm is not None:
        _tqdm.write(text, file=sys.stderr)
    else:
        print(text, file=sys.stderr)


@contextmanager
def quiet_loggers(*names: str, level: int = logging.ERROR):
    saved = []
    for name in names:
        lg = logging.getLogger(name)
        saved.append((lg, lg.level, list(lg.handlers)))
        lg.setLevel(level)
        for h in lg.handlers:
            try:
                h.setLevel(level)
            except Exception:
                pass
    try:
        yield
    finally:
        for lg, lvl, _handlers in saved:
            lg.setLevel(lvl)


def atomic_write_json(data: dict, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return path


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_hop_spec(hop_counts: Dict[Any, Any]) -> Dict[int, int]:
    """规范化跳数配置。支持 {"1": 50, "2": 30} 或 {1: 50, 2: 30}。"""
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


def parse_doc_id_range(raw: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    解析 doc 表 id 闭区间。

      null / 空     → (None, None)  不限制
      N             → (None, N)     id <= N
      [a, b]        → (a, b)        a <= id <= b
      {min: a, max: b}
    a/b 为 0 或负数视为该端不限制。
    """
    if raw is None or raw == "" or raw is False:
        return None, None
    if isinstance(raw, dict):
        lo = raw.get("min", raw.get("lo", raw.get("start")))
        hi = raw.get("max", raw.get("hi", raw.get("end")))
        return _opt_pos_int(lo), _opt_pos_int(hi)
    if isinstance(raw, (list, tuple)):
        vals = [x for x in raw if x is not None and x != ""]
        if len(vals) >= 2:
            lo, hi = _opt_pos_int(vals[0]), _opt_pos_int(vals[1])
        elif len(vals) == 1:
            lo, hi = None, _opt_pos_int(vals[0])
        else:
            lo, hi = None, None
        if lo is not None and hi is not None and lo > hi:
            lo, hi = hi, lo
        return lo, hi
    return None, _opt_pos_int(raw)


def _opt_pos_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    return n


def _connect_doc_db(db_path: Union[str, Path]) -> sqlite3.Connection:
    db_path = resolve_path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"数据库不存在: {db_path}")
    conn = sqlite3.connect(str(db_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    return conn


def _doc_range_clause(
    doc_id_min: Optional[int], doc_id_max: Optional[int]
) -> Tuple[str, list]:
    sql = (
        " WHERE content IS NOT NULL AND TRIM(content) != ''"
    )
    params: list = []
    lo = _opt_pos_int(doc_id_min)
    hi = _opt_pos_int(doc_id_max)
    if lo is not None:
        sql += " AND id >= ?"
        params.append(lo)
    if hi is not None:
        sql += " AND id <= ?"
        params.append(hi)
    return sql, params


def load_doc_index_from_db(
    db_path: Union[str, Path],
    *,
    doc_id_min: Optional[int] = None,
    doc_id_max: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """只读 id/name，不出 content，供抽样用。"""
    where, params = _doc_range_clause(doc_id_min, doc_id_max)
    sql = "SELECT id, name FROM doc" + where + " ORDER BY id"
    conn = _connect_doc_db(db_path)
    try:
        rows = conn.execute(sql, tuple(params)).fetchall()
    finally:
        conn.close()
    docs = [
        {"doc_id": r["id"], "name": r["name"] or f"doc_{r['id']}"}
        for r in rows
    ]
    if not docs:
        lo, hi = _opt_pos_int(doc_id_min), _opt_pos_int(doc_id_max)
        raise RuntimeError(
            f"doc 表无可用 content: {resolve_path(db_path)} id∈[{lo or '*'}, {hi or '*'}]"
        )
    return docs


def load_docs_by_ids(
    db_path: Union[str, Path],
    doc_ids: Sequence[Any],
) -> List[Dict[str, Any]]:
    """按 id 列表取完整 content；保持传入顺序。"""
    ids: List[int] = []
    seen = set()
    for x in doc_ids or []:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if i in seen:
            continue
        seen.add(i)
        ids.append(i)
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    sql = f"SELECT id, name, content FROM doc WHERE id IN ({placeholders})"
    conn = _connect_doc_db(db_path)
    try:
        rows = conn.execute(sql, tuple(ids)).fetchall()
    finally:
        conn.close()
    by_id = {
        r["id"]: {
            "doc_id": r["id"],
            "name": r["name"] or f"doc_{r['id']}",
            "content": r["content"] or "",
        }
        for r in rows
    }
    return [by_id[i] for i in ids if i in by_id]


def load_docs_from_db(
    db_path: Union[str, Path],
    *,
    doc_id_min: Optional[int] = None,
    doc_id_max: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """读取范围内全部 doc 的 content（范围应尽量小）。"""
    index = load_doc_index_from_db(
        db_path, doc_id_min=doc_id_min, doc_id_max=doc_id_max
    )
    return load_docs_by_ids(db_path, [d["doc_id"] for d in index])


def truncate_text(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None or int(max_chars) <= 0:
        return text
    limit = int(max_chars)
    if len(text) <= limit:
        return text
    keep = max(0, limit - 20)
    return text[:keep].rstrip() + "\n…(内容已截断)"


def format_docs_block(
    docs: Sequence[Dict[str, Any]],
    max_chars_per_doc: Optional[int] = 4000,
    *,
    numbered: bool = True,
) -> str:
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
    if not text:
        return None
    raw = str(text).strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def fill_prompt(template: str, mapping: Dict[str, Any]) -> str:
    """替换 {key}，并把模板里为 .format 转义的 {{ / }} 还原成单括号。"""
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
    first: str = "agentic",
    second: str = "llm_only",
) -> Tuple[str, str]:
    raw = str(qid if qid is not None else "")
    bit = hashlib.md5(raw.encode("utf-8")).digest()[0] & 1
    if bit:
        return second, first
    return first, second


def extract_pair_side(obj: dict, side: str) -> dict:
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
        "reason": ("reason", "comment"),
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
    if not isinstance(obj, dict):
        return None
    ag = (
        obj.get("agentic")
        or obj.get("hypergraph")
        or obj.get("超图")
        or obj.get("rag")
    )
    lo = (
        obj.get("llm_only")
        or obj.get("纯LLM")
        or obj.get("llm")
        or obj.get("baseline")
    )
    if isinstance(ag, dict) and isinstance(lo, dict):
        return {"agentic": ag, "llm_only": lo}
    return None


def normalize_pairwise_better(raw: Any) -> Optional[str]:
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
        "tie", "equal", "same", "neither", "both",
        "平", "平局", "相同", "一样", "并列", "都不", "都好", "相当",
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
    if compact in ("agentic", "hypergraph", "rag", "graph") or s in ("超图", "Agentic"):
        return "agentic"
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
        return cmp_.get("better") or cmp_.get("winner") or cmp_.get("prefer")
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
    if not str(out.get("answer") or "").strip() and out.get("reasoning_content"):
        out["answer"] = out["reasoning_content"]
    return out


def ratio_to_percent(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if abs(x) > 1.0 + 1e-9:
        return x
    return x * 100.0


def llm_response_has_text(resp: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(resp, dict):
        return False
    if str(resp.get("answer") or "").strip():
        return True
    return bool(str(resp.get("reasoning_content") or "").strip())


def is_llm_timeout_error(resp: Any) -> bool:
    if isinstance(resp, dict):
        if resp.get("status") == 1:
            return False
        text = " ".join(
            str(resp.get(k) or "")
            for k in ("answer", "error", "message", "query_error")
        )
    else:
        text = str(resp or "")
    s = text.lower()
    return any(
        k in s
        for k in (
            "timeout", "timed out", "time out", "deadline exceeded",
            "readtimeout", "connecttimeout",
        )
    )


def llm_attempt_should_retry(
    resp: Any,
    *,
    attempt: int,
    max_retries: int,
    timeout_s: Optional[float] = None,
) -> bool:
    if attempt >= max_retries:
        return False
    if is_llm_timeout_error(resp):
        return False
    if timeout_s:
        try:
            lat = float((resp or {}).get("latency_s") or 0) if isinstance(resp, dict) else 0.0
        except (TypeError, ValueError):
            lat = 0.0
        try:
            cap = float(timeout_s)
        except (TypeError, ValueError):
            cap = 0.0
        if cap > 0 and lat >= 0.8 * cap:
            return False
    return True


def llm_client_timeout(llm, default: float = 300.0) -> float:
    try:
        return float(getattr(llm, "timeout", default) or default)
    except (TypeError, ValueError):
        return default


def mean(values: Sequence[Any]) -> Optional[float]:
    vals = []
    for x in values:
        if x is None:
            continue
        try:
            vals.append(float(x))
        except (TypeError, ValueError):
            pass
    if not vals:
        return None
    return sum(vals) / len(vals)


def safe_div(a: float, b: float) -> Optional[float]:
    if b == 0:
        return None
    return a / b


def normalize_filename(name: str) -> str:
    if not name:
        return ""
    s = str(name).strip().replace("\\", "/")
    s = s.split("/")[-1]
    return s.lower()


def match_doc_names(
    expected: Sequence[str], retrieved: Sequence[str]
) -> Tuple[List[str], List[str], List[str]]:
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
    """评测前常驻检索索引；范围只读 config.search_range。"""
    if dhmf is None or not hasattr(dhmf, "pin_retrieve_indexes"):
        return False
    rng = getattr(getattr(dhmf, "config", None), "search_range", None)
    chunk_r = getattr(rng, "chunk_max_vectors", 0) if rng is not None else 0
    node_r = getattr(rng, "node_max_vectors", 0) if rng is not None else 0
    print(
        f"[eval] pin_retrieve_indexes (search_range) "
        f"chunk={chunk_r!r} node={node_r!r} …",
        file=sys.stderr,
        flush=True,
    )
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


def count_tool_calls(turns: Sequence[Any], name: str) -> int:
    n = 0
    for t in turns or []:
        if not isinstance(t, dict):
            continue
        for call in t.get("calls") or []:
            if isinstance(call, dict) and str(call.get("name") or "") == name:
                n += 1
    return n


def empty_retrieve_timing() -> Dict[str, float]:
    return {k: 0.0 for k in RETRIEVE_STAGE_KEYS}


def as_timing_dict(raw: Any) -> Dict[str, float]:
    out = empty_retrieve_timing()
    if not isinstance(raw, dict):
        return out
    for k in RETRIEVE_STAGE_KEYS:
        try:
            out[k] = float(raw.get(k) or 0.0)
        except (TypeError, ValueError):
            out[k] = 0.0
    return out
