"""从专利问答 CSV 读取测试集。"""

from __future__ import annotations

import csv
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from benchmark.utils import resolve_path

_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")
_NUMBERED_SPLIT_RE = re.compile(r"(?:^|[；;])\s*\d+\s*[.、．]\s*")
_TOPIC_RE = re.compile(r"配方优化测试_QA_(.+)_(CN\d+[A-Z]?)$", re.I)

DIM_KEYS = ("核心机理", "配方可用性", "工艺与协同", "负向分")


def cell_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v != v:
            return ""
        if v.is_integer():
            return str(int(v))
        return str(v).rstrip("0").rstrip(".")
    return str(v).strip()


def classify_dimension(text: str) -> str:
    t = cell_str(text)
    if "负向" in t or "错误推荐" in t:
        return "负向分"
    if t.startswith("核心机理") or "核心机理" in t:
        return "核心机理"
    if t.startswith("是否命中") and "机理" in t:
        return "核心机理"
    if "配方" in t or "实施例" in t or "权利要求" in t:
        return "配方可用性"
    if "工艺" in t or "协同" in t or "工序" in t:
        return "工艺与协同"
    return "其他"


def dimension_key(index: int, text: str) -> str:
    classified = classify_dimension(text)
    if classified != "其他":
        return classified
    if 1 <= index <= len(DIM_KEYS):
        return DIM_KEYS[index - 1]
    return f"维度{index}"


def split_score_dimensions(text: Any) -> List[str]:
    """把 CSV 的 score_dimensions 拆成 4 条清单。

    优先按「1. 2. 3.」切开；否则半角分号（避免切开括号里的全角分号）；
    再退回全角/半角分号。
    """
    raw = cell_str(text)
    if not raw:
        return []
    if _NUMBERED_SPLIT_RE.search(raw):
        parts = [
            p.strip(" ；;、")
            for p in _NUMBERED_SPLIT_RE.split(raw)
            if p.strip(" ；;、")
        ]
        if len(parts) >= 3:
            return parts
    if ";" in raw:
        half = [p.strip() for p in raw.split(";") if p.strip()]
        if len(half) >= 3:
            return half
    return [p.strip() for p in re.split(r"[；;]+", raw) if p.strip()]


def topic_from_md(md_file: str) -> str:
    stem = Path(cell_str(md_file)).stem
    m = _TOPIC_RE.match(stem)
    if m:
        return m.group(1)
    return stem


def _decode_csv(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    last_err: Optional[Exception] = None
    for enc in _ENCODINGS:
        try:
            text = data.decode(enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        header = text.splitlines()[0] if text else ""
        if "question" in header.lower() or "问题" in header:
            return text, enc
    if last_err is not None:
        raise last_err
    return data.decode("gb18030"), "gb18030"


def load_csv_dataset(
    csv_path: Union[str, Path],
    *,
    n_limit: Optional[int] = None,
    id_list: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    path = Path(csv_path)
    if not path.is_absolute():
        path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"测试集 CSV 不存在: {path}")

    text, encoding = _decode_csv(path)
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError(f"CSV 没有表头: {path}")

    wanted = None
    if id_list:
        wanted = {str(x).strip() for x in id_list if str(x).strip()}

    questions: List[Dict[str, Any]] = []
    for row in reader:
        qid = cell_str(row.get("id"))
        question = cell_str(row.get("question"))
        if not qid and not question:
            continue
        if wanted is not None and qid not in wanted:
            continue
        dims_raw = cell_str(row.get("score_dimensions"))
        dims = split_score_dimensions(dims_raw)
        md_file = cell_str(row.get("md_file"))
        patent = cell_str(row.get("source_patent"))
        topic = topic_from_md(md_file)
        item = {
            "id": qid or f"row_{len(questions) + 1}",
            "raw_id": qid,
            "question": question,
            "expected_answer": cell_str(row.get("expected_answer")),
            "score_dimensions": dims,
            "score_dimensions_raw": dims_raw,
            "source_patent": patent,
            "source_file": cell_str(row.get("source_file")),
            "source_ref": cell_str(row.get("source_ref")),
            "md_file": md_file,
            "topic": topic,
            "knowledge_source": cell_str(row.get("source_ref")),
            "note": md_file,
            "reasoning_path": cell_str(row.get("source_ref")),
            "categories": {
                "来源专利": patent,
                "主题": topic,
                "题型": "配方优化",
            },
        }
        questions.append(item)
        if n_limit is not None and len(questions) >= n_limit:
            break

    if not questions:
        raise ValueError(f"CSV 没有可用题目: {path}")

    n_dims = [len(q["score_dimensions"]) for q in questions]
    return {
        "meta": {
            "schema": "patent_csv_qa",
            "source_csv": str(path),
            "encoding": encoding,
            "n_questions": len(questions),
            "n_dimensions_min": min(n_dims) if n_dims else 0,
            "n_dimensions_max": max(n_dims) if n_dims else 0,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "questions": questions,
    }
