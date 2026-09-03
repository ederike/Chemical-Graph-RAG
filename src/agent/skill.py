"""
query_skill：单跳精准查询。

语义 ≈ 一次 DHMF.query（retrieve_items + LLM 作答），
但 LLM / 改写全部走 agent 配置，与 retrieve 的模型配置解耦。
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from ..utils.OpenAIAPI import LLM
from ..utils.config import AgentConfig, resolve_credentials, resolve_llm_timeout
from .prompts import Agent_PROMPT
from .state import DIRECT_STEP_KIND, answer_of, first_line

if TYPE_CHECKING:
    from ..DHMF import DHMF

def build_agent_llm(config) -> LLM:
    """从 config.agent（回退 settings）构造 LLM。"""
    agent_cfg = getattr(config, 'agent', None)
    api_key, base_url = resolve_credentials(config, agent_cfg)
    return LLM(api_key, base_url, timeout=resolve_llm_timeout(agent_cfg))

class QuerySkill:
    """双路检索 + agent LLM 作答，封装为可调用 skill。"""

    def __init__(
        self,
        dhmf: "DHMF",
        agent_cfg: AgentConfig,
        llm: LLM,
        logger: Optional[logging.Logger] = None,
    ):
        self.dhmf   = dhmf
        self.cfg    = agent_cfg
        self.llm    = llm
        self.logger = logger or logging.getLogger(__name__)

    def __call__(self, question: str, **kwargs) -> Dict[str, Any]:
        return self.run(question, **kwargs)

    def run(
        self,
        question: str,
        *,
        original_query: Optional[str] = None,
        step_id: Optional[str] = None,
        n_steps: Optional[int] = None,
        planned_question: Optional[str] = None,
        depends_on: Optional[Sequence[str]] = None,
        kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        q = (question or '').strip()
        t_all = time.perf_counter()
        if not q:
            return {
                'status': 0, 'answer': '（空查询）',
                'retrieval_sources': [], 'retrieve_latency_s': 0.0,
                'retrieve_timing': {}, 'latency_s': 0.0,
            }

        search_q = self._rewrite(q) if self.cfg.enable_query_rewrite else q

        t0 = time.perf_counter()
        items, retrieval_text, retrieve_timing = self._retrieve(search_q or q)
        retrieve_latency_s = time.perf_counter() - t0
        sources, doc_ids = self._collect_refs(items)

        respond = self._answer(
            q,
            retrieval_text,
            original_query=original_query,
            step_id=step_id,
            n_steps=n_steps,
            planned_question=planned_question,
            depends_on=depends_on,
            kind=kind,
        )
        respond['retrieve_latency_s'] = retrieve_latency_s
        respond['retrieve_timing']    = retrieve_timing
        respond['latency_s']          = time.perf_counter() - t_all
        respond['retrieval_sources']  = sources
        respond['retrieval_doc_ids']  = doc_ids
        respond['search_query']       = search_q or q
        return respond

    def _chat(self, system: str, user: str) -> Any:
        return self.llm.generate(
            prompt={'system': system, 'user': user},
            model_args=dict(self.cfg.model_args or {}),
            use_cache=bool(self.cfg.use_cache),
        )

    def _rewrite(self, query: str) -> str:
        try:
            resp = self._chat(
                Agent_PROMPT.get('REWRITE_SYSTEM', ''),
                Agent_PROMPT['REWRITE_USER'].format(query=query),
            )
            if not isinstance(resp, dict) or resp.get('status') != 1:
                return query
            return first_line(answer_of(resp)) or query
        except Exception as e:
            self.logger.warning(f'[agent.skill] rewrite failed: {e}')
            return query

    def _retrieve(self, search_q: str):
        retrieve = self.dhmf.retrieve_module
        kw = {
            # 参数级关闭 retrieve 侧改写，避免改共享 config（多线程安全）
            'enable_query_rewrite': False,
        }
        if self.cfg.chunk_candidate_k is not None:
            kw['chunk_candidate_k'] = int(self.cfg.chunk_candidate_k)
        if self.cfg.node_candidate_k is not None:
            kw['node_candidate_k'] = int(self.cfg.node_candidate_k)

        items = retrieve.retrieve_items(search_q, **kw)
        timing = {}
        try:
            timing = dict(retrieve.get_last_timing() or {})
        except Exception:
            timing = {}
        return items, retrieve._format_retrieved_chunks(items), timing

    @staticmethod
    def _collect_refs(items) -> tuple:
        sources, doc_ids, seen_src, seen_did = [], [], set(), set()
        for it in items or []:
            src = it.get('source')
            if src and src not in seen_src:
                seen_src.add(src)
                sources.append(str(src))
            did = it.get('doc_id')
            if did is not None and did not in seen_did:
                seen_did.add(did)
                doc_ids.append(did)
        return sources, doc_ids

    @staticmethod
    def _format_step_context(
        *,
        step_id: Optional[str],
        n_steps: Optional[int],
        depends_on: Optional[Sequence[str]],
        kind: Optional[str] = None,
    ) -> str:
        sid = str(step_id or '1').strip() or '1'
        if kind == DIRECT_STEP_KIND:
            return (
                '这是用用户原始总问题做的一次完整检索作答，'
                '结果只作为最终汇总的对照参考。'
                '请依据语料完整回答原始总问题，不要按规划子步裁剪信息。'
            )
        try:
            n = max(1, int(n_steps or 1))
        except (TypeError, ValueError):
            n = 1
        deps = [str(d).strip() for d in (depends_on or []) if str(d).strip()]
        if n <= 1:
            return (
                f'步骤 {sid} / 共 1 步。这是针对原始总问题的唯一步骤，'
                f'请完整回答当前单跳问题，并覆盖原始问题需要的信息。'
            )
        deps_s = '、'.join(deps) if deps else '无（可与其它步骤并行）'
        return (
            f'步骤 {sid} / 共 {n} 步。\n'
            f'依赖步骤：{deps_s}\n'
            f'你正在处理规划中的第 {sid} 步：只负责本步主体与约束，'
            f'同时对照原始总问题，保留对后续步骤或最终汇总有用的细节。'
        )

    def _answer(
        self,
        query: str,
        retrieval_text: str,
        *,
        original_query: Optional[str] = None,
        step_id: Optional[str] = None,
        n_steps: Optional[int] = None,
        planned_question: Optional[str] = None,
        depends_on: Optional[Sequence[str]] = None,
        kind: Optional[str] = None,
    ) -> dict:
        original = (original_query or query or "").strip()
        planned = (planned_question or query or "").strip()
        user = Agent_PROMPT["QUERY_SKILL_ANSWER_USER"]
        for key, val in (
            ("retrieval_result", str(retrieval_text or "")),
            (
                "step_context",
                self._format_step_context(
                    step_id=step_id,
                    n_steps=n_steps,
                    depends_on=depends_on,
                    kind=kind,
                ),
            ),
            ("original_query", original),
            ("planned_question", planned),
            ("query", query or ""),
        ):
            user = user.replace("{" + key + "}", str(val))
        resp = self._chat(
            Agent_PROMPT.get("QUERY_SKILL_ANSWER_SYSTEM", ""),
            user,
        )
        return resp if isinstance(resp, dict) else {"status": 0, "answer": str(resp)}
