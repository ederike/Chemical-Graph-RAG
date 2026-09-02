"""
Tool-calling 检索问答入口（由 DHMF.agentic_query 薄封装调用）。

    respond = dhmf.agentic_query("……", pretty=True)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Union

from .loop import AgenticContext, build_agentic_llm, run_agentic_loop
from .tools import ToolContext

if TYPE_CHECKING:
    from ..DHMF import DHMF


def _format_pretty(respond: dict, *, query: str) -> str:
    from ..DHMF import DHMF

    base = DHMF.format_query_response(respond, query=query)
    turns: List[dict] = list(respond.get("turns") or [])
    lines = ["", "## Agentic Trace", "-" * 60]
    proto = respond.get("protocol") or ""
    if proto:
        lines.append(f"protocol: {proto}")
    if not turns:
        lines.append("(none)")
    else:
        for t in turns:
            n = t.get("turn")
            kind = t.get("kind") or ""
            tag = "forced" if t.get("forced") else kind
            lines.append(f"### Turn {n}  ·  {tag}")
            thought = (t.get("thought") or "").strip()
            if thought:
                preview = thought if len(thought) <= 600 else thought[:600] + "…"
                lines.append("thought:")
                lines += [f"  {al}" for al in preview.splitlines()]
            for call in t.get("calls") or []:
                name = call.get("name") or ""
                args = call.get("arguments") or {}
                lines.append(f"tool: {name}  args={args}")
                result = str(call.get("result") or "")
                if len(result) > 800:
                    result = result[:800] + "…"
                if result:
                    lines.append("result:")
                    lines += [f"  {al}" for al in result.splitlines()[:40]]
            ans = (t.get("answer") or "").strip()
            if ans and kind == "answer":
                preview = ans if len(ans) <= 400 else ans[:400] + "…"
                lines.append("answer:")
                lines += [f"  {al}" for al in preview.splitlines()]
            lines.append("")

    body = "\n".join(lines).rstrip()
    if base.endswith("=" * 60):
        return base[: -len("=" * 60)].rstrip() + "\n" + body + "\n" + ("=" * 60)
    return base + "\n" + body


def run_agentic_query(
    dhmf: "DHMF",
    query: str,
    *,
    pretty: bool = False,
) -> Union[dict, str]:
    """
    同一条对话上的工具循环：模型边想边调 search / read_doc / graph_neighbors。
    配置只读 config.agentic，不读 retrieve / agent。
    """
    config = dhmf.config
    agentic_cfg = getattr(config, "agentic", None)
    if agentic_cfg is None:
        from ..utils.config import AgenticConfig
        agentic_cfg = AgenticConfig()

    logger = dhmf.logger
    llm = build_agentic_llm(config)
    tools = ToolContext(dhmf=dhmf, cfg=agentic_cfg, logger=logger)
    ctx = AgenticContext(cfg=agentic_cfg, llm=llm, tools=tools, logger=logger)

    t0 = time.perf_counter()
    q = (query or "").strip()
    logger.info(f"[agentic_query] start log_trace={bool(agentic_cfg.log_trace)} q={q!r}")
    try:
        respond: Dict[str, Any] = run_agentic_loop(ctx, q)
    except Exception as e:
        logger.exception(f"[agentic_query] failed: {e}")
        respond = {
            "status": 0,
            "answer": f"Agentic 执行失败: {e}",
            "turns": [],
            "retrieval_sources": [],
            "retrieval_doc_ids": [],
        }

    respond["latency_s"] = time.perf_counter() - t0
    respond.setdefault("retrieval_sources", [])
    respond.setdefault("retrieval_doc_ids", [])
    logger.info(
        f"[agentic_query] done status={respond.get('status')} "
        f"turns={len(respond.get('turns') or [])} "
        f"latency={respond['latency_s']:.3f}s "
        f"retrieve={float(respond.get('retrieve_latency_s') or 0):.3f}s"
    )
    if pretty:
        return _format_pretty(respond, query=q)
    return respond
