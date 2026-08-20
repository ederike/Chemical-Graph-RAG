"""
多跳 Agent 执行核心（由 DHMF.agent_query 薄封装调用）。

    respond = dhmf.agent_query("……", pretty=True)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Union

from .graph import build_agent_graph
from .nodes import AgentContext
from .skill import QuerySkill, build_agent_llm
from .state import (
    AgentState,
    PlanStep,
    StepResult,
    aggregate_usage,
    is_llm_step,
    merge_sources,
)

if TYPE_CHECKING:
    from ..DHMF import DHMF

def _format_pretty(
    respond: dict,
    *,
    query: str,
    plan: List[PlanStep],
    results: Dict[str, StepResult],
) -> str:
    from ..DHMF import DHMF

    base  = DHMF.format_query_response(respond, query=query)
    lines = ['', '## Multi-hop Steps', '-' * 60]

    if not plan:
        lines.append('(none)')
    else:
        for s in plan:
            sid      = s['id']
            r        = results.get(sid) or {}
            deps_s   = ','.join(s.get('depends_on') or []) or 'none'
            planned  = (s.get('question') or '').strip()
            resolved = (r.get('resolved_question') or '').strip()
            ans      = (r.get('answer') or '').strip()
            src      = r.get('sources') or []
            kind     = '纯LLM' if is_llm_step(s) else '检索'

            lines.append(f'### Step {sid}  ·  {kind}  ·  deps: {deps_s}')
            lines.append(f'planned:  {planned}')
            if resolved and resolved != planned:
                lines.append(f'resolved: {resolved}')
            if src:
                lines.append(f'sources:  {" / ".join(src)}')
            if ans:
                preview = ans if len(ans) <= 400 else ans[:400] + '…'
                lines.append('answer:')
                lines += [f'  {al}' for al in preview.splitlines()]
            lines.append('')

    body = '\n'.join(lines).rstrip()
    if base.endswith('=' * 60):
        return base[: -len('=' * 60)].rstrip() + '\n' + body + '\n' + ('=' * 60)
    return base + '\n' + body

def run_agent_query(
    dhmf: "DHMF",
    query: str,
    *,
    pretty: bool = False,
) -> Union[dict, str]:
    """
    规划检索依赖图（并行加入纯 LLM 步）→ 按就绪序执行 → 汇总对照检索与纯 LLM。
    状态仅在 LangGraph 内存 state 中传递；线程安全（无共享记事本）。
    返回 dict 字段与 DHMF.query 对齐，并含 plan / steps。
    """
    config    = dhmf.config
    agent_cfg = getattr(config, 'agent', None)
    if agent_cfg is None:
        from ..utils.config import AgentConfig
        agent_cfg = AgentConfig()

    logger   = dhmf.logger
    llm      = build_agent_llm(config)
    skill    = QuerySkill(dhmf, agent_cfg, llm, logger=logger)
    ctx      = AgentContext(cfg=agent_cfg, llm=llm, skill=skill, logger=logger)
    graph    = build_agent_graph(ctx)

    t0 = time.perf_counter()
    init: AgentState = {
        'query': (query or '').strip(),
        'plan': [], 'results': {},
        'final_answer': '', 'final_status': 0,
        'respond': {}, 'error': '',
    }

    try:
        final = graph.invoke(init)
    except Exception as e:
        logger.exception(f'[agent_query] graph failed: {e}')
        err = {
            'status': 0, 'answer': f'Agent 执行失败: {e}',
            'latency_s': time.perf_counter() - t0,
            'plan': [], 'steps': [],
            'retrieval_sources': [], 'retrieval_doc_ids': [],
        }
        return _format_pretty(err, query=query or '', plan=[], results={}) if pretty else err

    plan: List[PlanStep]           = list(final.get('plan') or [])
    results: Dict[str, StepResult] = dict(final.get('results') or {})
    respond: Dict[str, Any]        = dict(final.get('respond') or {})
    if not respond:
        respond = {
            'status': int(final.get('final_status') or 0),
            'answer': final.get('final_answer') or '',
        }

    respond = aggregate_usage(plan, results, respond)
    sources = merge_sources(plan, results)
    if sources:
        respond['retrieval_sources'] = sources
    respond.setdefault('retrieval_sources', [])
    respond.setdefault('retrieval_doc_ids', [])
    respond['latency_s'] = time.perf_counter() - t0
    respond.setdefault('status', int(final.get('final_status') or 0))
    if not respond.get('answer'):
        respond['answer'] = final.get('final_answer') or ''
    respond['plan']  = plan
    respond['steps'] = [results.get(s['id'], {'id': s['id']}) for s in plan]

    logger.info(
        f'[agent_query] done steps={len(plan)} status={respond.get("status")} '
        f'latency={respond["latency_s"]:.3f}s'
    )
    return _format_pretty(respond, query=query or '', plan=plan, results=results) if pretty else respond
