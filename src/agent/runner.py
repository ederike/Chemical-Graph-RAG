"""
多跳 Agent 执行核心（由 DHMF.agent_query 薄封装调用）。

    respond = dhmf.agent_query("……", pretty=True)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Union

from .graph import build_agent_graph
from .notebook import Notebook
from .nodes import AgentContext
from .skill import QuerySkill, build_agent_llm
from .state import (
    AgentState,
    PlanStep,
    StepResult,
    aggregate_usage,
    merge_sources,
)

if TYPE_CHECKING:
    from ..DHMF import DHMF


# ── pretty ───────────────────────────────────────────────────────────────────

def _format_pretty(
    respond: dict,
    *,
    query: str,
    plan: List[PlanStep],
    results: Dict[str, StepResult],
) -> str:
    from ..DHMF import DHMF

    base  = DHMF.format_query_response(respond, query=query)
    lines = ['', '【多跳步骤】', '-' * 60]

    if not plan:
        lines.append('（无）')
    else:
        for s in plan:
            sid      = s['id']
            r        = results.get(sid) or {}
            deps_s   = ','.join(s.get('depends_on') or []) or '无'
            planned  = (s.get('question') or '').strip()
            resolved = (r.get('resolved_question') or '').strip()
            ans      = (r.get('answer') or '').strip()
            src      = r.get('sources') or []

            lines.append(f'{sid}. [deps={deps_s}] {planned}')
            if resolved and resolved != planned:
                lines.append(f'   实际查询: {resolved}')
            if src:
                lines.append(f'   来源: {" / ".join(src)}')
            if ans:
                preview = ans if len(ans) <= 400 else ans[:400] + '…'
                lines += [f'   | {al}' for al in preview.splitlines()]
            lines.append('')

    body = '\n'.join(lines).rstrip()
    if base.endswith('=' * 60):
        return base[: -len('=' * 60)].rstrip() + '\n' + body + '\n' + ('=' * 60)
    return base + '\n' + body


# ── entry ────────────────────────────────────────────────────────────────────

def run_agent_query(
    dhmf: "DHMF",
    query: str,
    *,
    pretty: bool = False,
) -> Union[dict, str]:
    """
    规划依赖步骤图 → 按就绪序 query_skill → 汇总 → 清空记事本。
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
    nb_name  = (getattr(agent_cfg, 'notebook_path', None) or 'agent_scratchpad.md').strip() or 'agent_scratchpad.md'
    notebook = Notebook(Path(config.settings.working_path) / nb_name)
    ctx      = AgentContext(cfg=agent_cfg, llm=llm, skill=skill, notebook=notebook, logger=logger)
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
    finally:
        try:
            notebook.clear()
        except Exception as ce:
            logger.warning(f'[agent_query] notebook clear failed: {ce}')

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
