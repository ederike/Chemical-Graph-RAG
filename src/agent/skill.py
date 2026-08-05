"""
query_skill：单跳精准查询。

语义等价于一次 DHMF.query（retrive_items + LLM 作答），
但 LLM / 改写全部使用 agent 配置，不读取 retrieve 的模型配置。
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..utils.OpenAIAPI import LLM
from ..utils.config import AgentConfig, resolve_credentials
from .prompts import Agent_PROMPT

if TYPE_CHECKING:
    from ..DHMF import DHMF


@contextmanager
def _suppress_retrieve_rewrite(retrieve_module):
    """暂时关闭 retrieve 侧查询改写，避免混用 retrieve 的 rewrite 模型配置。"""
    rcfg = retrieve_module.config.retrieve
    prev = bool(getattr(rcfg, 'enable_query_rewrite', True))
    rcfg.enable_query_rewrite = False
    try:
        yield
    finally:
        rcfg.enable_query_rewrite = prev


class QuerySkill:
    """把双路检索 + agent LLM 作答封装成可调用 skill。"""

    def __init__(
        self,
        dhmf: "DHMF",
        agent_cfg: AgentConfig,
        llm: LLM,
        logger: Optional[logging.Logger] = None,
    ):
        self.dhmf = dhmf
        self.cfg = agent_cfg
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)

    def __call__(self, question: str) -> Dict[str, Any]:
        return self.run(question)

    def run(self, question: str) -> Dict[str, Any]:
        q = (question or '').strip()
        t_all = time.perf_counter()

        if not q:
            return {
                'status': 0,
                'answer': '（空查询）',
                'retrieval_sources': [],
                'retrieve_latency_s': 0.0,
                'latency_s': 0.0,
            }

        # 0) 可选：agent 侧改写（与 retrieve.rewrite 配置无关）
        search_q = q
        if bool(self.cfg.enable_query_rewrite):
            search_q = self._rewrite(q) or q

        # 1) 检索：仅复用 retrieve 的向量/双路逻辑；关闭其 LLM 改写
        t0 = time.perf_counter()
        retrieve = self.dhmf.retrieve_module
        kw = {}
        if self.cfg.chunk_candidate_k is not None:
            kw['chunk_candidate_k'] = int(self.cfg.chunk_candidate_k)
        if self.cfg.node_candidate_k is not None:
            kw['node_candidate_k'] = int(self.cfg.node_candidate_k)

        with _suppress_retrieve_rewrite(retrieve):
            items = retrieve.retrive_items(search_q, **kw)
        retrieval_text = retrieve._format_retrieved_chunks(items)
        retrieve_latency_s = time.perf_counter() - t0

        sources, doc_ids = [], []
        seen_src, seen_did = set(), set()
        for it in items or []:
            src = it.get('source')
            if src and src not in seen_src:
                seen_src.add(src)
                sources.append(str(src))
            did = it.get('doc_id')
            if did is not None and did not in seen_did:
                seen_did.add(did)
                doc_ids.append(did)

        # 2) 作答：Agent_PROMPT 专用提示词 + agent.model_args（不用 retrieve/query_answer）
        user_prompt = Agent_PROMPT['QUERY_SKILL_ANSWER_USER'].format(
            retrieval_result=str(retrieval_text or ''),
            query=q,
        )
        respond = self.llm.generate(
            prompt={
                'system': Agent_PROMPT.get('QUERY_SKILL_ANSWER_SYSTEM', ''),
                'user': user_prompt,
            },
            model_args=dict(self.cfg.model_args or {}),
            use_cache=bool(self.cfg.use_cache),
        )
        if not isinstance(respond, dict):
            respond = {'status': 0, 'answer': str(respond)}

        respond['retrieve_latency_s'] = retrieve_latency_s
        respond['latency_s'] = time.perf_counter() - t_all
        respond['retrieval_sources'] = sources
        respond['retrieval_doc_ids'] = doc_ids
        respond['search_query'] = search_q
        return respond

    def _rewrite(self, query: str) -> str:
        try:
            resp = self.llm.generate(
                prompt={
                    'system': Agent_PROMPT.get('REWRITE_SYSTEM', ''),
                    'user': Agent_PROMPT['REWRITE_USER'].format(query=query),
                },
                model_args=dict(self.cfg.model_args or {}),
                use_cache=bool(self.cfg.use_cache),
            )
            if not isinstance(resp, dict) or resp.get('status') != 1:
                return query
            text = (resp.get('answer') or '').strip().splitlines()[0].strip()
            return text or query
        except Exception as e:
            self.logger.warning(f'[agent.skill] rewrite failed: {e}')
            return query


def build_agent_llm(config) -> LLM:
    """从 config.agent（回退 settings）构造 LLM。"""
    agent_cfg = getattr(config, 'agent', None)
    api_key, base_url = resolve_credentials(config, agent_cfg)
    return LLM(api_key, base_url)
