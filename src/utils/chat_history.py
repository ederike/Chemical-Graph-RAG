"""Same-conversation Q&A for the next turn (pronouns / follow-ups)."""
from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional

_FOLLOW = re.compile(
    r"(它|他|她|该产品|该型号|这个|那个|这款|那款|上面|刚才|还有|同样|继续|呢[？?]?$)"
)


def normalize_history(
    history: Optional[Iterable[Any]],
    *,
    max_turns: int = 8,
    max_answer_chars: int = 1200,
) -> List[dict]:
    out: List[dict] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        q = str(item.get("query") or item.get("user") or "").strip()
        a = str(item.get("answer") or item.get("assistant") or "").strip()
        if not q:
            continue
        if len(a) > max_answer_chars:
            a = a[:max_answer_chars] + "…"
        out.append({"query": q, "answer": a})
    if max_turns > 0:
        out = out[-max_turns:]
    return out


def attach_history(query: str, history: Optional[Iterable[Any]] = None) -> str:
    """User-facing prompt: prior turns + current question."""
    q = (query or "").strip()
    items = normalize_history(history)
    if not items:
        return q
    lines = [
        "此前同一对话中的问答（用来理解指代和续问；检索与作答以「当前问题」为准，不要把旧问当成新的检索目标）：",
    ]
    for i, t in enumerate(items, 1):
        lines.append(f"【上轮 {i}】用户：{t['query']}")
        if t["answer"]:
            lines.append(f"助手：{t['answer']}")
    lines.append("")
    lines.append("当前问题：")
    lines.append(q)
    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    n = len(text or "")
    if n <= 0:
        return 0
    return max(1, n // 2)


def estimate_next_prompt_tokens(
    history: Optional[Iterable[Any]] = None,
    query: str = "",
    *,
    overhead: int = 8000,
) -> int:
    """Rough next-prompt size: history + current question + system/tools overhead.

    If a prior turn reported last_prompt_tokens / usage_prompt_tokens, use that
    as a floor and add the new user text plus the last assistant answer.
    """
    items = normalize_history(history)
    body = attach_history(query, items) if (items or query) else ""
    char_est = estimate_tokens(body) + (overhead if body or query else 0)
    last_pt = 0
    last_ans = items[-1]["answer"] if items else ""
    for raw in reversed(list(history or [])):
        if not isinstance(raw, dict):
            continue
        res = raw.get("result") if isinstance(raw.get("result"), dict) else {}
        for key in ("last_prompt_tokens", "usage_prompt_tokens"):
            try:
                n = int(raw.get(key) or (res.get(key) if res else 0) or 0)
            except (TypeError, ValueError):
                n = 0
            if n > last_pt:
                last_pt = n
        if last_pt:
            break
    if last_pt > 0:
        extra = estimate_tokens(query) + estimate_tokens(last_ans)
        return max(char_est, last_pt + extra)
    return char_est


def expand_retrieve_query(query: str, history: Optional[Iterable[Any]] = None) -> str:
    """Keep embedding query short; only splice last questions on follow-ups."""
    q = (query or "").strip()
    items = normalize_history(history, max_turns=3, max_answer_chars=80)
    if not items:
        return q
    if len(q) <= 18 or _FOLLOW.search(q):
        prev = " ".join(t["query"] for t in items[-2:])
        return f"{prev} {q}".strip()
    return q
