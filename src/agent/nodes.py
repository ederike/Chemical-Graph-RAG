"""
LangGraph 节点：plan → execute → synthesize。

每个节点是纯函数式可调用对象，依赖通过 AgentContext 注入，保持图构建与业务解耦。
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..utils.OpenAIAPI import LLM
from ..utils.config import AgentConfig
from .notebook import Notebook
from .prompts import Agent_PROMPT
from .skill import QuerySkill
from .state import AgentState, PlanStep, StepResult


# ======================================================================
# 共享上下文（图编译时注入，节点只读）
# ======================================================================
@dataclass
class AgentContext:
    cfg: AgentConfig
    llm: LLM
    skill: QuerySkill
    notebook: Notebook
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))


# ======================================================================
# JSON / plan 解析
# ======================================================================
def _strip_code_fence(text: str) -> str:
    t = (text or '').strip()
    if t.startswith('```'):
        lines = t.splitlines()
        if lines and lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        t = '\n'.join(lines).strip()
    return t


def _parse_plan_json(raw: str) -> List[PlanStep]:
    text = _strip_code_fence(raw)
    # 尝试直接 loads；失败则截取首个 {...}
    data = None
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
    if not isinstance(data, dict):
        return []

    steps_raw = data.get('steps') or data.get('plan') or []
    if not isinstance(steps_raw, list):
        return []

    steps: List[PlanStep] = []
    for i, item in enumerate(steps_raw, start=1):
        if not isinstance(item, dict):
            continue
        sid = str(item.get('id') or i).strip() or str(i)
        q = (item.get('question') or item.get('query') or '').strip()
        if not q:
            continue
        deps_raw = item.get('depends_on') or item.get('deps') or []
        if not isinstance(deps_raw, list):
            deps_raw = []
        deps = [str(d).strip() for d in deps_raw if str(d).strip()]
        steps.append(PlanStep(id=sid, question=q, depends_on=deps))
    return steps


def _normalize_plan(steps: List[PlanStep], *, max_steps: int, fallback_query: str) -> List[PlanStep]:
    """去重 id、截断 max_steps、去掉指向不存在步骤的依赖；空则回退单步。"""
    if not steps:
        return [PlanStep(id='1', question=fallback_query.strip(), depends_on=[])]

    seen = set()
    out: List[PlanStep] = []
    for s in steps:
        sid = str(s.get('id') or '').strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append(PlanStep(
            id=sid,
            question=(s.get('question') or '').strip(),
            depends_on=list(s.get('depends_on') or []),
        ))
        if len(out) >= max_steps:
            break

    ids = {s['id'] for s in out}
    for s in out:
        s['depends_on'] = [d for d in (s.get('depends_on') or []) if d in ids and d != s['id']]

    if not out:
        return [PlanStep(id='1', question=fallback_query.strip(), depends_on=[])]
    return out


def _usage_tuple(resp: Optional[dict]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if not isinstance(resp, dict):
        return None, None, None
    return (
        resp.get('usage_prompt_tokens'),
        resp.get('usage_completion_tokens'),
        resp.get('usage_total_tokens'),
    )


def _add_usage(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None and b is None:
        return None
    return int(a or 0) + int(b or 0)


# ======================================================================
# plan
# ======================================================================
def plan_node(ctx: AgentContext, state: AgentState) -> dict:
    query = (state.get('query') or '').strip()
    ctx.logger.info(f'[agent.plan] query={query!r}')

    user = Agent_PROMPT['PLAN_USER'].format(
        query=query, max_steps=int(ctx.cfg.max_steps)
    )
    resp = ctx.llm.generate(
        prompt={
            'system': Agent_PROMPT.get('PLAN_SYSTEM', ''),
            'user': user,
        },
        model_args=dict(ctx.cfg.model_args or {}),
        use_cache=bool(ctx.cfg.use_cache),
    )
    raw = ''
    if isinstance(resp, dict):
        raw = str(resp.get('answer') or '')
    else:
        raw = str(resp)

    steps = _normalize_plan(
        _parse_plan_json(raw),
        max_steps=int(ctx.cfg.max_steps),
        fallback_query=query,
    )
    ctx.logger.info(
        f'[agent.plan] steps={len(steps)} '
        + ' | '.join(f"{s['id']}:{s['question'][:40]}" for s in steps)
    )

    ctx.notebook.sync(query=query, plan=steps, results={})
    return {
        'plan': steps,
        'results': {},
        'error': '',
    }


# ======================================================================
# execute：每轮执行「依赖已满足」的就绪步骤（一批），再由路由决定是否继续
# ======================================================================
def _ready_steps(plan: List[PlanStep], results: Dict[str, StepResult]) -> List[PlanStep]:
    done = set(results.keys())
    ready = []
    for s in plan:
        sid = s['id']
        if sid in done:
            continue
        deps = s.get('depends_on') or []
        if all(d in done for d in deps):
            ready.append(s)
    return ready


def _format_prior(results: Dict[str, StepResult], dep_ids: List[str]) -> str:
    lines = []
    for did in dep_ids:
        r = results.get(did)
        if not r:
            continue
        ans = (r.get('answer') or '').strip()
        q = (r.get('resolved_question') or r.get('planned_question') or '').strip()
        lines.append(f'[步骤 {did}] 问：{q}\n答：{ans}')
    return '\n\n'.join(lines) if lines else '（无）'


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

    prior = _format_prior(results, deps)
    user = Agent_PROMPT['RESOLVE_USER'].format(
        query=query,
        prior_results=prior,
        step_question=planned,
    )
    try:
        resp = ctx.llm.generate(
            prompt={
                'system': Agent_PROMPT.get('RESOLVE_SYSTEM', ''),
                'user': user,
            },
            model_args=dict(ctx.cfg.model_args or {}),
            use_cache=bool(ctx.cfg.use_cache),
        )
        if isinstance(resp, dict) and resp.get('status') == 1:
            text = (resp.get('answer') or '').strip().splitlines()[0].strip()
            if text:
                return text
    except Exception as e:
        ctx.logger.warning(f'[agent.execute] resolve failed step={step.get("id")}: {e}')
    # 回退：把先验结论拼进问题前缀
    return f'{planned}\n（已知：{prior[:500]}）'


def execute_node(ctx: AgentContext, state: AgentState) -> dict:
    query = state.get('query') or ''
    plan: List[PlanStep] = list(state.get('plan') or [])
    results: Dict[str, StepResult] = dict(state.get('results') or {})

    ready = _ready_steps(plan, results)
    if not ready:
        # 死锁：仍有未完成步骤但无就绪 → 强制把剩余当无依赖执行
        pending = [s for s in plan if s['id'] not in results]
        if pending:
            ctx.logger.warning(
                f'[agent.execute] dependency deadlock, force-run pending='
                f'{[s["id"] for s in pending]}'
            )
            ready = pending
        else:
            return {'results': results}

    for step in ready:
        sid = step['id']
        resolved = _resolve_question(ctx, query=query, step=step, results=results)
        ctx.logger.info(f'[agent.execute] step={sid} q={resolved!r}')

        t0 = time.perf_counter()
        resp = ctx.skill.run(resolved)
        latency = time.perf_counter() - t0

        answer = ''
        status = 0
        if isinstance(resp, dict):
            answer = str(resp.get('answer') or '')
            status = int(resp.get('status') or 0)
            sources = list(resp.get('retrieval_sources') or [])
            pt, ct, tt = _usage_tuple(resp)
            r_lat = float(resp.get('retrieve_latency_s') or 0.0)
        else:
            answer = str(resp)
            sources = []
            pt = ct = tt = None
            r_lat = 0.0

        results[sid] = StepResult(
            id=sid,
            planned_question=step.get('question') or '',
            resolved_question=resolved,
            answer=answer,
            sources=sources,
            status=status,
            retrieve_latency_s=r_lat,
            latency_s=latency,
            usage_prompt_tokens=pt,
            usage_completion_tokens=ct,
            usage_total_tokens=tt,
        )

        ctx.notebook.sync(query=query, plan=plan, results=results)

    return {'results': results}


def route_after_execute(state: AgentState) -> str:
    """全部完成 → synthesize；否则继续 execute。"""
    plan = state.get('plan') or []
    results = state.get('results') or {}
    if not plan:
        return 'synthesize'
    if all(s['id'] in results for s in plan):
        return 'synthesize'
    return 'execute'


# ======================================================================
# synthesize
# ======================================================================
def _format_step_results_for_synth(plan: List[PlanStep], results: Dict[str, StepResult]) -> str:
    blocks = []
    for s in plan:
        sid = s['id']
        r = results.get(sid) or {}
        q = (r.get('resolved_question') or s.get('question') or '').strip()
        ans = (r.get('answer') or '').strip() or '（无结论）'
        src = r.get('sources') or []
        src_s = ' / '.join(src) if src else '（无）'
        blocks.append(
            f'### 步骤 {sid}\n'
            f'问：{q}\n'
            f'答：{ans}\n'
            f'来源：{src_s}'
        )
    return '\n\n'.join(blocks) if blocks else '（无子步骤结论）'


def synthesize_node(ctx: AgentContext, state: AgentState) -> dict:
    query = state.get('query') or ''
    plan: List[PlanStep] = list(state.get('plan') or [])
    results: Dict[str, StepResult] = dict(state.get('results') or {})

    # 单步时直接采用 hop 答案，避免二次汇总损失细节
    if len(plan) == 1 and plan[0]['id'] in results:
        r = results[plan[0]['id']]
        answer = (r.get('answer') or '').strip()
        respond = {
            'status': int(r.get('status') or 0),
            'answer': answer,
            'usage_prompt_tokens': r.get('usage_prompt_tokens'),
            'usage_completion_tokens': r.get('usage_completion_tokens'),
            'usage_total_tokens': r.get('usage_total_tokens'),
            'retrieve_latency_s': r.get('retrieve_latency_s'),
            'retrieval_sources': list(r.get('sources') or []),
        }
        ctx.notebook.sync(query=query, plan=plan, results=results, final_answer=answer)
        return {
            'final_answer': answer,
            'final_status': respond['status'],
            'respond': respond,
        }

    step_text = _format_step_results_for_synth(plan, results)
    user = Agent_PROMPT['SYNTH_USER'].format(
        query=query, step_results=step_text
    )
    resp = ctx.llm.generate(
        prompt={
            'system': Agent_PROMPT.get('SYNTH_SYSTEM', ''),
            'user': user,
        },
        model_args=dict(ctx.cfg.model_args or {}),
        use_cache=bool(ctx.cfg.use_cache),
    )
    if not isinstance(resp, dict):
        resp = {'status': 0, 'answer': str(resp)}

    answer = str(resp.get('answer') or '').strip()
    # 合并子步骤来源
    sources, seen = [], set()
    for s in plan:
        for src in (results.get(s['id']) or {}).get('sources') or []:
            if src not in seen:
                seen.add(src)
                sources.append(src)
    resp['retrieval_sources'] = sources

    # 汇总检索延迟
    r_lat = 0.0
    for s in plan:
        r_lat += float((results.get(s['id']) or {}).get('retrieve_latency_s') or 0.0)
    resp['retrieve_latency_s'] = r_lat

    ctx.notebook.sync(query=query, plan=plan, results=results, final_answer=answer)
    return {
        'final_answer': answer,
        'final_status': int(resp.get('status') or 0),
        'respond': resp,
    }
