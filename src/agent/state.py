"""Agent 运行时类型与纯数据工具。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TypedDict

class PlanStep(TypedDict, total=False):
    id: str
    question: str
    depends_on: List[str]

class StepResult(TypedDict, total=False):
    id: str
    planned_question: str
    resolved_question: str
    answer: str
    sources: List[str]
    status: int
    retrieve_latency_s: float
    latency_s: float
    usage_prompt_tokens: Optional[int]
    usage_completion_tokens: Optional[int]
    usage_total_tokens: Optional[int]

class AgentState(TypedDict, total=False):
    """LangGraph 状态：plan → execute ⟲ → synthesize。"""
    query: str
    plan: List[PlanStep]
    results: Dict[str, StepResult]
    final_answer: str
    final_status: int
    respond: Dict[str, Any]
    error: str

def add_usage(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None and b is None:
        return None
    return int(a or 0) + int(b or 0)

def usage_of(resp: Optional[dict]) -> tuple:
    """→ (prompt, completion, total)；非 dict 则全 None。"""
    if not isinstance(resp, dict):
        return None, None, None
    return (
        resp.get('usage_prompt_tokens'),
        resp.get('usage_completion_tokens'),
        resp.get('usage_total_tokens'),
    )

def answer_of(resp: Any) -> str:
    if isinstance(resp, dict):
        return str(resp.get('answer') or '').strip()
    return str(resp or '').strip()

def first_line(text: str) -> str:
    return (text or '').strip().splitlines()[0].strip() if (text or '').strip() else ''

def merge_sources(plan: List[PlanStep], results: Dict[str, StepResult]) -> List[str]:
    out, seen = [], set()
    for s in plan:
        for src in (results.get(s['id']) or {}).get('sources') or []:
            if src and src not in seen:
                seen.add(src)
                out.append(src)
    return out

def sum_retrieve_latency(plan: List[PlanStep], results: Dict[str, StepResult]) -> float:
    return sum(float((results.get(s['id']) or {}).get('retrieve_latency_s') or 0.0) for s in plan)

def aggregate_usage(plan: List[PlanStep], results: Dict[str, StepResult], respond: dict) -> dict:
    """累加各 hop + respond 的 token / 检索延迟（就地写 respond）。"""
    pt = ct = tt = None
    r_lat = 0.0
    for s in plan:
        r = results.get(s['id']) or {}
        pt = add_usage(pt, r.get('usage_prompt_tokens'))
        ct = add_usage(ct, r.get('usage_completion_tokens'))
        tt = add_usage(tt, r.get('usage_total_tokens'))
        r_lat += float(r.get('retrieve_latency_s') or 0.0)

    pt = add_usage(pt, respond.get('usage_prompt_tokens'))
    ct = add_usage(ct, respond.get('usage_completion_tokens'))
    tt = add_usage(tt, respond.get('usage_total_tokens'))
    respond['usage_prompt_tokens'] = pt
    respond['usage_completion_tokens'] = ct
    respond['usage_total_tokens'] = tt
    if respond.get('retrieve_latency_s') is None:
        respond['retrieve_latency_s'] = r_lat
    return respond

def _strip_fence(text: str) -> str:
    t = (text or '').strip()
    if not t.startswith('```'):
        return t
    lines = t.splitlines()
    if lines and lines[0].startswith('```'):
        lines = lines[1:]
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    return '\n'.join(lines).strip()

def _loads_obj(raw: str) -> Optional[dict]:
    text = _strip_fence(raw)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        m = re.search(r'\{[\s\S]*\}', text)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

def parse_plan(raw: str) -> List[PlanStep]:
    data = _loads_obj(raw)
    if not data:
        return []
    items = data.get('steps') or data.get('plan') or []
    if not isinstance(items, list):
        return []

    steps: List[PlanStep] = []
    for i, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        q = (item.get('question') or item.get('query') or '').strip()
        if not q:
            continue
        sid = str(item.get('id') or i).strip() or str(i)
        deps_raw = item.get('depends_on') or item.get('deps') or []
        deps = [str(d).strip() for d in (deps_raw if isinstance(deps_raw, list) else []) if str(d).strip()]
        steps.append(PlanStep(id=sid, question=q, depends_on=deps))
    return steps

def normalize_plan(steps: List[PlanStep], *, max_steps: int, fallback: str) -> List[PlanStep]:
    """去重 id、截断、清洗悬空依赖；空则回退单步。"""
    fb = [PlanStep(id='1', question=(fallback or '').strip(), depends_on=[])]
    if not steps:
        return fb

    seen, out = set(), []
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
    return out or fb

def ready_steps(plan: List[PlanStep], results: Dict[str, StepResult]) -> List[PlanStep]:
    done = set(results)
    return [
        s for s in plan
        if s['id'] not in done and all(d in done for d in (s.get('depends_on') or []))
    ]

def pending_steps(plan: List[PlanStep], results: Dict[str, StepResult]) -> List[PlanStep]:
    return [s for s in plan if s['id'] not in results]

def plan_done(plan: List[PlanStep], results: Dict[str, StepResult]) -> bool:
    return (not plan) or all(s['id'] in results for s in plan)

def format_prior(results: Dict[str, StepResult], dep_ids: List[str]) -> str:
    lines = []
    for did in dep_ids:
        r = results.get(did)
        if not r:
            continue
        q = (r.get('resolved_question') or r.get('planned_question') or '').strip()
        a = (r.get('answer') or '').strip()
        lines.append(f'[步骤 {did}] 问：{q}\n答：{a}')
    return '\n\n'.join(lines) if lines else '（无）'

def format_step_blocks(plan: List[PlanStep], results: Dict[str, StepResult]) -> str:
    blocks = []
    for s in plan:
        sid = s['id']
        r = results.get(sid) or {}
        q = (r.get('resolved_question') or s.get('question') or '').strip()
        a = (r.get('answer') or '').strip() or '（无结论）'
        src = ' / '.join(r.get('sources') or []) or '（无）'
        blocks.append(f'### 步骤 {sid}\n问：{q}\n答：{a}\n来源：{src}')
    return '\n\n'.join(blocks) if blocks else '（无子步骤结论）'

def step_result_from_skill(
    *,
    sid: str,
    planned: str,
    resolved: str,
    resp: Any,
    latency_s: float,
) -> StepResult:
    if isinstance(resp, dict):
        answer  = str(resp.get('answer') or '')
        status  = int(resp.get('status') or 0)
        sources = list(resp.get('retrieval_sources') or [])
        pt, ct, tt = usage_of(resp)
        r_lat   = float(resp.get('retrieve_latency_s') or 0.0)
    else:
        answer, status, sources = str(resp), 0, []
        pt = ct = tt = None
        r_lat = 0.0

    return StepResult(
        id=sid,
        planned_question=planned,
        resolved_question=resolved,
        answer=answer,
        sources=sources,
        status=status,
        retrieve_latency_s=r_lat,
        latency_s=latency_s,
        usage_prompt_tokens=pt,
        usage_completion_tokens=ct,
        usage_total_tokens=tt,
    )
