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
from .nodes import AgentContext, _add_usage
from .skill import QuerySkill, build_agent_llm
from .state import AgentState, PlanStep, StepResult

if TYPE_CHECKING:
    from ..DHMF import DHMF


def _aggregate_usage(plan: List[PlanStep], results: Dict[str, StepResult], respond: dict) -> dict:
    """把各 hop + 汇总的 token 累加进 respond（就地）。"""
    pt = ct = tt = None
    r_lat = 0.0
    for s in plan:
        r = results.get(s['id']) or {}
        pt = _add_usage(pt, r.get('usage_prompt_tokens'))
        ct = _add_usage(ct, r.get('usage_completion_tokens'))
        tt = _add_usage(tt, r.get('usage_total_tokens'))
        r_lat += float(r.get('retrieve_latency_s') or 0.0)

    if isinstance(respond, dict):
        pt = _add_usage(pt, respond.get('usage_prompt_tokens'))
        ct = _add_usage(ct, respond.get('usage_completion_tokens'))
        tt = _add_usage(tt, respond.get('usage_total_tokens'))
        respond['usage_prompt_tokens'] = pt
        respond['usage_completion_tokens'] = ct
        respond['usage_total_tokens'] = tt
        if respond.get('retrieve_latency_s') is None:
            respond['retrieve_latency_s'] = r_lat
    return respond


def _merge_sources_and_doc_ids(
    plan: List[PlanStep],
    results: Dict[str, StepResult],
    respond: dict,
) -> dict:
    """合并各 hop 来源；doc_ids 若 skill 未写入 StepResult 则仅保留 respond 已有。"""
    sources, seen = [], set()
    for s in plan:
        for src in (results.get(s['id']) or {}).get('sources') or []:
            if src and src not in seen:
                seen.add(src)
                sources.append(src)
    if sources:
        # respond 侧可能已有 synthesize 合并结果；以 hop 全量为准
        respond['retrieval_sources'] = sources
    respond.setdefault('retrieval_sources', [])
    respond.setdefault('retrieval_doc_ids', [])
    return respond


def _format_pretty(
    respond: dict,
    *,
    query: str,
    plan: List[PlanStep],
    results: Dict[str, StepResult],
) -> str:
    """与 DHMF.format_query_response 风格对齐，并附加步骤摘要。"""
    from ..DHMF import DHMF

    base = DHMF.format_query_response(respond, query=query)
    lines = [
        '',
        '【多跳步骤】',
        '-' * 60,
    ]
    if not plan:
        lines.append('（无）')
    else:
        for s in plan:
            sid = s['id']
            r = results.get(sid) or {}
            deps = s.get('depends_on') or []
            deps_s = ','.join(deps) if deps else '无'
            planned = (s.get('question') or '').strip()
            resolved = (r.get('resolved_question') or '').strip()
            ans = (r.get('answer') or '').strip()
            src = r.get('sources') or []
            lines.append(f'{sid}. [deps={deps_s}] {planned}')
            if resolved and resolved != planned:
                lines.append(f'   实际查询: {resolved}')
            if src:
                lines.append(f'   来源: {" / ".join(src)}')
            if ans:
                preview = ans if len(ans) <= 400 else ans[:400] + '…'
                for al in preview.splitlines():
                    lines.append(f'   | {al}')
            lines.append('')

    if base.endswith('=' * 60):
        return (
            base[: -len('=' * 60)].rstrip()
            + '\n'
            + '\n'.join(lines).rstrip()
            + '\n'
            + ('=' * 60)
        )
    return base + '\n' + '\n'.join(lines)


def run_agent_query(
    dhmf: "DHMF",
    query: str,
    *,
    pretty: bool = False,
) -> Union[dict, str]:
    """
    多跳 Agent 执行体（供 DHMF.agent_query 调用）。

    流程：规划依赖步骤图 → 按就绪序 query_skill → 汇总 → 清空记事本。
    """
    config = dhmf.config
    agent_cfg = getattr(config, 'agent', None)
    if agent_cfg is None:
        from ..utils.config import AgentConfig
        agent_cfg = AgentConfig()

    logger = dhmf.logger
    llm = build_agent_llm(config)
    skill = QuerySkill(dhmf, agent_cfg, llm, logger=logger)

    nb_name = (getattr(agent_cfg, 'notebook_path', None) or 'agent_scratchpad.md').strip()
    if not nb_name:
        nb_name = 'agent_scratchpad.md'
    notebook = Notebook(Path(config.settings.working_path) / nb_name)

    ctx = AgentContext(
        cfg=agent_cfg,
        llm=llm,
        skill=skill,
        notebook=notebook,
        logger=logger,
    )
    graph = build_agent_graph(ctx)

    t0 = time.perf_counter()
    init: AgentState = {
        'query': (query or '').strip(),
        'plan': [],
        'results': {},
        'final_answer': '',
        'final_status': 0,
        'respond': {},
        'error': '',
    }

    final: AgentState
    try:
        final = graph.invoke(init)
    except Exception as e:
        logger.exception(f'[agent_query] graph failed: {e}')
        err = {
            'status': 0,
            'answer': f'Agent 执行失败: {e}',
            'latency_s': time.perf_counter() - t0,
            'plan': [],
            'steps': [],
            'retrieval_sources': [],
            'retrieval_doc_ids': [],
        }
        if pretty:
            return _format_pretty(err, query=query or '', plan=[], results={})
        return err
    finally:
        try:
            notebook.clear()
        except Exception as ce:
            logger.warning(f'[agent_query] notebook clear failed: {ce}')

    plan: List[PlanStep] = list(final.get('plan') or [])
    results: Dict[str, StepResult] = dict(final.get('results') or {})
    respond: Dict[str, Any] = dict(final.get('respond') or {})
    if not respond:
        respond = {
            'status': int(final.get('final_status') or 0),
            'answer': final.get('final_answer') or '',
        }

    respond = _aggregate_usage(plan, results, respond)
    respond = _merge_sources_and_doc_ids(plan, results, respond)
    respond['latency_s'] = time.perf_counter() - t0
    respond.setdefault('status', int(final.get('final_status') or 0))
    if not respond.get('answer'):
        respond['answer'] = final.get('final_answer') or ''

    respond['plan'] = plan
    respond['steps'] = [results.get(s['id'], {'id': s['id']}) for s in plan]

    logger.info(
        f'[agent_query] done steps={len(plan)} status={respond.get("status")} '
        f'latency={respond["latency_s"]:.3f}s'
    )

    if pretty:
        return _format_pretty(respond, query=query or '', plan=plan, results=results)
    return respond
