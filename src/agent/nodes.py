"""
LangGraph 节点：plan → execute ⟲ → synthesize。

依赖图 = 规划检索步 + 自动并行的纯 LLM 步；单跳/多跳都进 synthesize。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..utils.OpenAIAPI import LLM
from ..utils.config import AgentConfig
from .prompts import Agent_PROMPT
from .skill import QuerySkill
from .state import (
    AgentState,
    PlanStep,
    StepResult,
    answer_of,
    first_line,
    format_llm_answer,
    format_prior,
    format_step_blocks,
    inject_llm_step,
    is_llm_step,
    merge_sources,
    normalize_plan,
    parse_plan,
    pending_steps,
    plan_done,
    ready_steps,
    retrieve_plan,
    step_result_from_skill,
    sum_retrieve_latency,
    sum_retrieve_timing,
)

@dataclass
class AgentContext:
    cfg: AgentConfig
    llm: LLM
    skill: QuerySkill
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def chat(self, system: str, user: str) -> Any:
        """统一 LLM 调用：agent.model_args + use_cache。"""
        return self.llm.generate(
            prompt={'system': system, 'user': user},
            model_args=dict(self.cfg.model_args or {}),
            use_cache=bool(self.cfg.use_cache),
        )

    def chat_text(self, system: str, user: str) -> str:
        return answer_of(self.chat(system, user))


def _fill_prompt(template: str, **kwargs) -> str:
    """按占位符替换，避免 .format 把正文里的 {braces} 当成字段。"""
    text = template or ''
    for key, val in kwargs.items():
        text = text.replace('{' + key + '}', str(val if val is not None else ''))
    return text

def plan_node(ctx: AgentContext, state: AgentState) -> dict:
    query = (state.get('query') or '').strip()
    ctx.logger.info(f'[agent.plan] query={query!r}')

    raw = ctx.chat_text(
        Agent_PROMPT.get('PLAN_SYSTEM', ''),
        Agent_PROMPT['PLAN_USER'].format(query=query, max_steps=int(ctx.cfg.max_steps)),
    )
    steps = normalize_plan(parse_plan(raw), max_steps=int(ctx.cfg.max_steps), fallback=query)
    steps = inject_llm_step(steps, query)
    n_ret = len(retrieve_plan(steps))
    ctx.logger.info(
        f'[agent.plan] retrieve={n_ret} + llm '
        + ' | '.join(
            f"{s['id']}:{s.get('kind', 'retrieve')}:{s['question'][:40]}"
            for s in steps
        )
    )
    return {'plan': steps, 'results': {}, 'error': ''}

def _resolve_question(
    ctx: AgentContext,
    *,
    query: str,
    step: PlanStep,
    results: Dict[str, StepResult],
) -> str:
    planned = (step.get('question') or '').strip()
    deps = step.get('depends_on') or []
    if not deps:
        return planned

    prior = format_prior(results, deps)
    try:
        resp = ctx.chat(
            Agent_PROMPT.get('RESOLVE_SYSTEM', ''),
            _fill_prompt(
                Agent_PROMPT['RESOLVE_USER'],
                query=query,
                prior_results=prior,
                step_question=planned,
            ),
        )
        if isinstance(resp, dict) and resp.get('status') == 1:
            text = first_line(answer_of(resp))
            if text:
                return text
    except Exception as e:
        ctx.logger.warning(f'[agent.execute] resolve failed step={step.get("id")}: {e}')
    return f'{planned}\n（已知：{prior[:500]}）'

def execute_node(ctx: AgentContext, state: AgentState) -> dict:
    query   = state.get('query') or ''
    plan    = list(state.get('plan') or [])
    results = dict(state.get('results') or {})

    ready = ready_steps(plan, results)
    if not ready:
        pending = pending_steps(plan, results)
        if not pending:
            return {'results': results}
        ctx.logger.warning(f'[agent.execute] deadlock, force-run {[s["id"] for s in pending]}')
        ready = pending

    n_retrieve = len(retrieve_plan(plan))
    for step in ready:
        sid = step['id']
        t0 = time.perf_counter()
        if is_llm_step(step):
            ctx.logger.info(f'[agent.execute] step={sid} kind=llm q={query!r}')
            resp = ctx.chat(
                Agent_PROMPT.get('LLM_DIRECT_SYSTEM', ''),
                _fill_prompt(
                    Agent_PROMPT.get('LLM_DIRECT_USER', ''),
                    question=query,
                    query=query,
                ),
            )
            resolved = query
        else:
            resolved = _resolve_question(ctx, query=query, step=step, results=results)
            ctx.logger.info(f'[agent.execute] step={sid} q={resolved!r}')
            resp = ctx.skill.run(
                resolved,
                original_query=query,
                step_id=sid,
                n_steps=n_retrieve,
                planned_question=step.get('question') or '',
                depends_on=list(step.get('depends_on') or []),
            )
        results[sid] = step_result_from_skill(
            sid=sid,
            planned=step.get('question') or '',
            resolved=resolved,
            resp=resp,
            latency_s=time.perf_counter() - t0,
        )

    return {'results': results}

def route_after_execute(state: AgentState) -> str:
    plan = state.get('plan') or []
    results = state.get('results') or {}
    if not plan_done(plan, results):
        return 'execute'
    return 'synthesize'

def synthesize_node(ctx: AgentContext, state: AgentState) -> dict:
    query   = state.get('query') or ''
    plan: List[PlanStep] = list(state.get('plan') or [])
    results: Dict[str, StepResult] = dict(state.get('results') or {})

    resp = ctx.chat(
        Agent_PROMPT.get('SYNTH_SYSTEM', ''),
        _fill_prompt(
            Agent_PROMPT.get('SYNTH_USER', ''),
            query=query,
            step_results=format_step_blocks(plan, results),
            llm_answer=format_llm_answer(plan, results),
        ),
    )
    if not isinstance(resp, dict):
        resp = {'status': 0, 'answer': str(resp)}

    answer = str(resp.get('answer') or '').strip()
    resp['retrieval_sources']  = merge_sources(plan, results)
    resp['retrieve_latency_s'] = sum_retrieve_latency(plan, results)
    resp['retrieve_timing']    = sum_retrieve_timing(plan, results)

    return {
        'final_answer': answer,
        'final_status': int(resp.get('status') or 0),
        'respond': resp,
    }
