"""
query_skill：单跳精准查询。

语义 ≈ 一次 DHMF.query（retrive_items + LLM 作答），
但 LLM / 改写全部走 agent 配置，与 retrieve 的模型配置解耦。
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..utils.OpenAIAPI import LLM
from ..utils.config import AgentConfig, resolve_credentials
from .prompts import Agent_PROMPT
from .state import answer_of, first_line

if TYPE_CHECKING:
    from ..DHMF import DHMF


@contextmanager
def _suppress_retrieve_rewrite(retrieve_module):
    """临时关闭 retrieve 侧 rewrite，避免混用其模型配置。"""
    rcfg = retrieve_module.config.retrieve
    prev = bool(getattr(rcfg, 'enable_query_rewrite', True))
    rcfg.enable_query_rewrite = False
    try:
        yield
    finally:
        rcfg.enable_query_rewrite = prev


def build_agent_llm(config) -> LLM:
    """从 config.agent（回退 settings）构造 LLM。"""
    agent_cfg = getattr(config, 'agent', None)
    api_key, base_url = resolve_credentials(config, agent_cfg)
    return LLM(api_key, base_url)


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

    def __call__(self, question: str) -> Dict[str, Any]:
        return self.run(question)

    # ── public ───────────────────────────────────────────────────────────

    def run(self, question: str) -> Dict[str, Any]:
        q = (question or '').strip()
        t_all = time.perf_counter()
        if not q:
            return {
                'status': 0, 'answer': '（空查询）',
                'retrieval_sources': [], 'retrieve_latency_s': 0.0, 'latency_s': 0.0,
            }

        search_q = self._rewrite(q) if self.cfg.enable_query_rewrite else q

        t0 = time.perf_counter()
        items, retrieval_text = self._retrieve(search_q or q)
        retrieve_latency_s = time.perf_counter() - t0
        sources, doc_ids = self._collect_refs(items)

        respond = self._answer(q, retrieval_text)
        respond['retrieve_latency_s'] = retrieve_latency_s
        respond['latency_s']          = time.perf_counter() - t_all
        respond['retrieval_sources']  = sources
        respond['retrieval_doc_ids']  = doc_ids
        respond['search_query']       = search_q or q
        return respond

    # ── internals ────────────────────────────────────────────────────────

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
        kw = {}
        if self.cfg.chunk_candidate_k is not None:
            kw['chunk_candidate_k'] = int(self.cfg.chunk_candidate_k)
        if self.cfg.node_candidate_k is not None:
            kw['node_candidate_k'] = int(self.cfg.node_candidate_k)

        with _suppress_retrieve_rewrite(retrieve):
            items = retrieve.retrive_items(search_q, **kw)
        return items, retrieve._format_retrieved_chunks(items)

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

    def _answer(self, query: str, retrieval_text: str) -> dict:
        resp = self._chat(
            Agent_PROMPT.get('QUERY_SKILL_ANSWER_SYSTEM', ''),
            Agent_PROMPT['QUERY_SKILL_ANSWER_USER'].format(
                retrieval_result=str(retrieval_text or ''),
                query=query,
            ),
        )
        return resp if isinstance(resp, dict) else {'status': 0, 'answer': str(resp)}
