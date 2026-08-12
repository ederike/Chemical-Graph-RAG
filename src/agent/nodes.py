"""
LangGraph 节点：plan → execute ⟲ → synthesize。

依赖经 AgentContext 注入；节点只做编排，状态在 LangGraph state 中传递。
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
    format_prior,
    format_step_blocks,
    merge_sources,
    normalize_plan,
    parse_plan,
    pending_steps,
    plan_done,
    ready_steps,
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

def plan_node(ctx: AgentContext, state: AgentState) -> dict:
    query = (state.get('query') or '').strip()
    ctx.logger.info(f'[agent.plan] query={query!r}')

    raw = ctx.chat_text(
        Agent_PROMPT.get('PLAN_SYSTEM', ''),
        Agent_PROMPT['PLAN_USER'].format(query=query, max_steps=int(ctx.cfg.max_steps)),
    )
    steps = normalize_plan(parse_plan(raw), max_steps=int(ctx.cfg.max_steps), fallback=query)
    ctx.logger.info(
        f'[agent.plan] steps={len(steps)} '
        + ' | '.join(f"{s['id']}:{s['question'][:40]}" for s in steps)
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
            Agent_PROMPT['RESOLVE_USER'].format(
                query=query, prior_results=prior, step_question=planned,
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

    for step in ready:
        sid      = step['id']
        resolved = _resolve_question(ctx, query=query, step=step, results=results)
        ctx.logger.info(f'[agent.execute] step={sid} q={resolved!r}')

        t0 = time.perf_counter()
        resp = ctx.skill.run(resolved)
        results[sid] = step_result_from_skill(
            sid=sid,
            planned=step.get('question') or '',
            resolved=resolved,
            resp=resp,
            latency_s=time.perf_counter() - t0,
        )

    return {'results': results}

def route_after_execute(state: AgentState) -> str:
    return 'synthesize' if plan_done(state.get('plan') or [], state.get('results') or {}) else 'execute'

def synthesize_node(ctx: AgentContext, state: AgentState) -> dict:
    query   = state.get('query') or ''
    plan: List[PlanStep] = list(state.get('plan') or [])
    results: Dict[str, StepResult] = dict(state.get('results') or {})

    # 单步：直接采用 hop 答案，避免二次汇总丢细节
    if len(plan) == 1 and plan[0]['id'] in results:
        r = results[plan[0]['id']]
        answer = (r.get('answer') or '').strip()
        respond = {
            'status': int(r.get('status') or 0),
            'answer': answer,
            'usage_prompt_tokens':     r.get('usage_prompt_tokens'),
            'usage_completion_tokens': r.get('usage_completion_tokens'),
            'usage_total_tokens':      r.get('usage_total_tokens'),
            'retrieve_latency_s':      r.get('retrieve_latency_s'),
            'retrieve_timing':         dict(r.get('retrieve_timing') or {}),
            'retrieval_sources':       list(r.get('sources') or []),
        }
        return {'final_answer': answer, 'final_status': respond['status'], 'respond': respond}

    resp = ctx.chat(
        Agent_PROMPT.get('SYNTH_SYSTEM', ''),
        Agent_PROMPT['SYNTH_USER'].format(
            query=query, step_results=format_step_blocks(plan, results),
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
