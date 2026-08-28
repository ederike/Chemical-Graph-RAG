"""Agent 运行时类型与纯数据工具。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TypedDict

LLM_STEP_ID = 'llm'
LLM_STEP_KIND = 'llm'
RETRIEVE_STEP_KIND = 'retrieve'
DIRECT_STEP_ID = 'direct'
DIRECT_STEP_KIND = 'direct'


class PlanStep(TypedDict, total=False):
    id: str
    question: str
    depends_on: List[str]
    kind: str  # retrieve | llm | direct

class StepResult(TypedDict, total=False):
    id: str
    planned_question: str
    resolved_question: str
    answer: str
    sources: List[str]
    status: int
    retrieve_latency_s: float
    retrieve_timing: Dict[str, float]
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


def _empty_retrieve_timing() -> Dict[str, float]:
    return {
        'precompute_s': 0.0,
        'rewrite_s': 0.0,
        'embed_s': 0.0,
        'chunk_s': 0.0,
        'node_s': 0.0,
        'keyword_s': 0.0,
        'expand_s': 0.0,
        'rerank_s': 0.0,
        'total_s': 0.0,
    }


def sum_retrieve_timing(
    plan: List[PlanStep], results: Dict[str, StepResult]
) -> Dict[str, float]:
    """多 hop 累加各步 retrieve_timing。"""
    out = _empty_retrieve_timing()
    for s in plan:
        t = (results.get(s['id']) or {}).get('retrieve_timing') or {}
        if not isinstance(t, dict):
            continue
        for k in out:
            try:
                out[k] += float(t.get(k) or 0.0)
            except (TypeError, ValueError):
                pass
    return out

def aggregate_usage(plan: List[PlanStep], results: Dict[str, StepResult], respond: dict) -> dict:
    """累加各 hop + respond 的 token / 检索延迟（就地写 respond）。"""
    pt = ct = tt = None
    r_lat = 0.0
    r_timing = _empty_retrieve_timing()
    for s in plan:
        r = results.get(s['id']) or {}
        pt = add_usage(pt, r.get('usage_prompt_tokens'))
        ct = add_usage(ct, r.get('usage_completion_tokens'))
        tt = add_usage(tt, r.get('usage_total_tokens'))
        r_lat += float(r.get('retrieve_latency_s') or 0.0)
        t = r.get('retrieve_timing') or {}
        if isinstance(t, dict):
            for k in r_timing:
                try:
                    r_timing[k] += float(t.get(k) or 0.0)
                except (TypeError, ValueError):
                    pass

    pt = add_usage(pt, respond.get('usage_prompt_tokens'))
    ct = add_usage(ct, respond.get('usage_completion_tokens'))
    tt = add_usage(tt, respond.get('usage_total_tokens'))
    respond['usage_prompt_tokens'] = pt
    respond['usage_completion_tokens'] = ct
    respond['usage_total_tokens'] = tt
    if respond.get('retrieve_latency_s') is None:
        respond['retrieve_latency_s'] = r_lat
    if respond.get('retrieve_timing') is None:
        respond['retrieve_timing'] = r_timing
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

def is_llm_step(step: Optional[PlanStep]) -> bool:
    if not step:
        return False
    return (step.get('kind') == LLM_STEP_KIND) or (str(step.get('id') or '') == LLM_STEP_ID)


def is_direct_step(step: Optional[PlanStep]) -> bool:
    if not step:
        return False
    kind = step.get('kind')
    if kind == DIRECT_STEP_KIND:
        return True
    if kind:
        return False
    return str(step.get('id') or '') == DIRECT_STEP_ID


def retrieve_plan(plan: List[PlanStep]) -> List[PlanStep]:
    """规划出的检索子步（不含纯 LLM、不含原问题直检）。"""
    return [
        s for s in (plan or [])
        if not is_llm_step(s) and not is_direct_step(s)
    ]


def inject_llm_step(plan: List[PlanStep], query: str) -> List[PlanStep]:
    """在依赖图中并行加入纯 LLM 步（depends_on=[]，不计入 max_steps）。"""
    q = (query or '').strip()
    out: List[PlanStep] = []
    renames: Dict[str, str] = {}
    for s in plan or []:
        if is_llm_step(s):
            continue
        sid = str(s.get('id') or '').strip() or str(len(out) + 1)
        if sid == LLM_STEP_ID:
            new_id = 'r-llm'
            renames[sid] = new_id
            sid = new_id
        existing_kind = s.get('kind')
        kind = (
            DIRECT_STEP_KIND
            if existing_kind == DIRECT_STEP_KIND
            else RETRIEVE_STEP_KIND
        )
        out.append(PlanStep(
            id=sid,
            question=(s.get('question') or '').strip(),
            depends_on=list(s.get('depends_on') or []),
            kind=kind,
        ))
    if renames:
        for s in out:
            s['depends_on'] = [
                renames.get(d, d) for d in (s.get('depends_on') or [])
            ]
    out.append(PlanStep(
        id=LLM_STEP_ID,
        question=q,
        depends_on=[],
        kind=LLM_STEP_KIND,
    ))
    return out


def inject_direct_retrieve_step(plan: List[PlanStep], query: str) -> List[PlanStep]:
    """
    并行加入「用原始总问题检索并作答一次」步（depends_on=[]，不计入 max_steps）。
    走 QuerySkill.run（retrieve_items + 一次生成），不嵌套 agent_query。

    仅应在多跳（规划检索子步 > 1）时调用；单跳本身就是原问题直检。
    """
    q = (query or '').strip()
    if any(is_direct_step(s) for s in (plan or [])):
        return list(plan or [])

    out: List[PlanStep] = []
    renames: Dict[str, str] = {}
    for s in plan or []:
        sid = str(s.get('id') or '').strip() or str(len(out) + 1)
        if sid == DIRECT_STEP_ID and not is_direct_step(s):
            new_id = 'r-direct'
            renames[sid] = new_id
            sid = new_id
        out.append(PlanStep(
            id=sid,
            question=(s.get('question') or '').strip(),
            depends_on=list(s.get('depends_on') or []),
            kind=s.get('kind') or RETRIEVE_STEP_KIND,
        ))
    if renames:
        for s in out:
            s['depends_on'] = [
                renames.get(d, d) for d in (s.get('depends_on') or [])
            ]
    out.append(PlanStep(
        id=DIRECT_STEP_ID,
        question=q,
        depends_on=[],
        kind=DIRECT_STEP_KIND,
    ))
    return out


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
    for s in retrieve_plan(plan):
        sid = s['id']
        r = results.get(sid) or {}
        q = (r.get('resolved_question') or s.get('question') or '').strip()
        a = (r.get('answer') or '').strip() or '（无结论）'
        src = ' / '.join(r.get('sources') or []) or '（无）'
        blocks.append(f'### 步骤 {sid}\n问：{q}\n答：{a}\n来源：{src}')
    return '\n\n'.join(blocks) if blocks else '（无检索子步骤结论）'


def format_llm_answer(plan: List[PlanStep], results: Dict[str, StepResult]) -> str:
    for s in plan or []:
        if not is_llm_step(s):
            continue
        r = results.get(s['id']) or {}
        a = (r.get('answer') or '').strip()
        return a or '（纯 LLM 无结论）'
    return '（无纯 LLM 回答）'


def format_direct_answer(plan: List[PlanStep], results: Dict[str, StepResult]) -> str:
    for s in plan or []:
        if not is_direct_step(s):
            continue
        r = results.get(s['id']) or {}
        a = (r.get('answer') or '').strip()
        src = ' / '.join(r.get('sources') or []) or '（无）'
        if a:
            return f'来源：{src}\n{a}'
        return '（原问题直检无结论）'
    return '（无。本题未做原问题直检，忽略本段。）'

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
        r_timing = resp.get('retrieve_timing') or {}
        if not isinstance(r_timing, dict):
            r_timing = {}
    else:
        answer, status, sources = str(resp), 0, []
        pt = ct = tt = None
        r_lat = 0.0
        r_timing = {}

    return StepResult(
        id=sid,
        planned_question=planned,
        resolved_question=resolved,
        answer=answer,
        sources=sources,
        status=status,
        retrieve_latency_s=r_lat,
        retrieve_timing=dict(r_timing),
        latency_s=latency_s,
        usage_prompt_tokens=pt,
        usage_completion_tokens=ct,
        usage_total_tokens=tt,
    )
