"""agentic_query + 纯 LLM 对照 + 成对 Judge。"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import merge_llm_only_model_args
from .prompts import Benchmark_PROMPT
from .report import build_eval_document
from .utils import (
    EMPTY_ANSWER_PLACEHOLDER,
    JUDGMENT_LABELS,
    RETRIEVE_STAGE_KEYS,
    SYSTEM_AGENTIC,
    SYSTEM_LLM_ONLY,
    as_timing_dict,
    call_llm,
    count_tool_calls,
    extract_json_object,
    extract_named_system_sides,
    extract_pair_side,
    fail_print,
    fill_prompt,
    format_docs_block,
    llm_attempt_should_retry,
    llm_client_timeout,
    llm_response_has_text,
    match_doc_names,
    pairwise_better_raw,
    pairwise_system_order,
    progress_iter,
    quiet_loggers,
    resolve_pairwise_winner,
    safe_div,
    truncate_text,
)

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None

OnProgress = Optional[Callable[[Dict[str, Any], int, int], None]]


def _empty_system_block() -> Dict[str, Any]:
    return {
        "answer": "",
        "raw_answer": "",
        "query_status": 0,
        "query_error": None,
        "retrieval_sources": [],
        "retrieval_doc_ids": [],
        "llm_acc": None,
        "judge_reason": "",
        "judge_status": None,
        "judge_error": None,
        "metrics": {},
    }


def _answer_model_args_from_dhmf(dhmf, override: Optional[dict] = None) -> dict:
    base: Dict[str, Any] = {}
    try:
        if dhmf is not None:
            base = dict(getattr(dhmf.config.retrieve, "model_args", None) or {})
    except Exception:
        pass
    return merge_llm_only_model_args(base, override)


def normalize_judgment(raw: str) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s in JUDGMENT_LABELS:
        return s
    low = s.lower().strip()
    neg_keys = (
        "不正确", "全部错误", "完全错误", "错误", "错",
        "wrong", "incorrect", "false", "no",
    )
    pos_keys = (
        "全部正确", "完全正确", "正确", "对",
        "correct", "right", "true", "yes",
    )
    for k in neg_keys:
        if k.lower() == low or k in s:
            return "错误"
    for k in pos_keys:
        if k.lower() == low or k in s:
            return "正确"
    return None


class QueryEvaluator:
    """
    每题固定两路：
      1) DHMF.agentic_query
      2) 纯 LLM（不检索）
    两路都答完后用 JUDGE_USER 一次对比打分。
    """

    def __init__(
        self,
        dhmf,
        judge_llm=None,
        *,
        judge_model_args: Optional[dict] = None,
        answer_llm=None,
        answer_model_args: Optional[dict] = None,
        use_cache: bool = False,
        max_judge_retries: int = 3,
        max_llm_only_retries: int = 3,
        llm_only_use_cache: Optional[bool] = None,
        max_source_chars: int = -1,
        sleep_between: float = 0.0,
        enable_doc_recall: bool = True,
        enable_llm_only: bool = True,
        num_thread: int = 1,
    ):
        self.dhmf = dhmf
        self.judge_llm = judge_llm or getattr(dhmf, "llmmodel", None)
        if self.judge_llm is None:
            raise ValueError("judge_llm 未提供且 DHMF 无 llmmodel")
        self.answer_llm = (
            answer_llm or getattr(dhmf, "llmmodel", None) or self.judge_llm
        )

        self.judge_model_args = dict(judge_model_args or {})
        self.judge_model_args.setdefault("temperature", 0.0)
        self.judge_model_args.setdefault("enable_thinking", False)
        self.judge_model_args.setdefault("response_format", {"type": "json_object"})
        if not self.judge_model_args.get("model"):
            try:
                ma = getattr(dhmf.config.retrieve, "model_args", None) or {}
                if ma.get("model"):
                    self.judge_model_args["model"] = ma["model"]
            except Exception:
                pass

        self.answer_model_args = _answer_model_args_from_dhmf(dhmf, answer_model_args)
        self.enable_llm_only = bool(enable_llm_only)
        self.use_cache = bool(use_cache)
        self.llm_only_use_cache = (
            self.use_cache if llm_only_use_cache is None else bool(llm_only_use_cache)
        )
        self.max_judge_retries = max(1, int(max_judge_retries))
        self.max_llm_only_retries = max(1, int(max_llm_only_retries))
        self.max_source_chars = int(max_source_chars)
        self.sleep_between = float(sleep_between)
        self.enable_doc_recall = bool(enable_doc_recall)
        try:
            nt = int(num_thread)
        except (TypeError, ValueError):
            nt = 1
        self.num_thread = max(1, nt)

    def _run_llm_only(self, question: str) -> Dict[str, Any]:
        user = Benchmark_PROMPT["PURE_LLM_USER"].replace("{question}", question or "")
        last: Dict[str, Any] = {}
        timeout_s = llm_client_timeout(self.answer_llm)
        for attempt in range(1, self.max_llm_only_retries + 1):
            last = call_llm(
                self.answer_llm,
                system=Benchmark_PROMPT.get("PURE_LLM_SYSTEM", ""),
                user=user,
                model_args=self.answer_model_args,
                use_cache=self.llm_only_use_cache,
            )
            if isinstance(last, dict) and last.get("status") == 1 and llm_response_has_text(last):
                return last
            if not llm_attempt_should_retry(
                last,
                attempt=attempt,
                max_retries=self.max_llm_only_retries,
                timeout_s=timeout_s,
            ):
                break
        return last if isinstance(last, dict) else {"status": 0, "answer": str(last)}

    @staticmethod
    def _apply_judge(block: Dict[str, Any], judge: Dict[str, Any]) -> None:
        block["llm_acc"] = judge.get("llm_acc")
        block["judge_reason"] = judge.get("judge_reason")
        block["judge_status"] = judge.get("judge_status")
        block["judge_error"] = judge.get("judge_error")
        metrics = block.setdefault("metrics", {})
        metrics["judge_latency_s"] = judge.get("judge_latency_s")
        ju = judge.get("judge_usage") or {}
        metrics["judge_prompt_tokens"] = ju.get("prompt_tokens")
        metrics["judge_completion_tokens"] = ju.get("completion_tokens")
        metrics["judge_total_tokens"] = ju.get("total_tokens")

    def _extract_answer_text(self, respond: Any, *, parse_sections: bool = True) -> str:
        if not isinstance(respond, dict):
            return str(respond or "")
        raw = respond.get("answer")
        if not str(raw or "").strip():
            raw = respond.get("reasoning_content") or ""
        raw = str(raw or "")
        if not parse_sections:
            return raw.strip()
        try:
            from src.DHMF import DHMF
            parsed = DHMF.parse_query_answer(raw)
            return (parsed.get("answer") or parsed.get("raw") or "").strip()
        except Exception:
            return raw.strip()

    @staticmethod
    def _format_agentic_process(respond: Dict[str, Any]) -> str:
        turns = list(respond.get("turns") or [])
        lines: List[str] = ["## Agentic Trace", "-" * 48]
        proto = respond.get("protocol") or ""
        if proto:
            lines.append(f"protocol: {proto}")
        tp = respond.get("trace_path") or ""
        if tp:
            lines.append(f"trace:    {tp}")
        lp = respond.get("last_prompt_tokens")
        if lp is not None:
            lines.append(f"last_prompt_tokens: {lp}")
        if not turns:
            lines.append("(no turns)")
        else:
            for t in turns:
                if not isinstance(t, dict):
                    continue
                n = t.get("turn")
                kind = t.get("kind") or ""
                tag = "forced" if t.get("forced") else kind
                lines.append(f"### Turn {n}  ·  {tag}")
                thought = (t.get("thought") or "").strip()
                if thought:
                    lines.append("thought:")
                    for al in thought.splitlines() or [thought]:
                        lines.append(f"  {al}")
                for call in t.get("calls") or []:
                    if not isinstance(call, dict):
                        continue
                    name = call.get("name") or ""
                    args = call.get("arguments") or {}
                    lines.append(f"tool: {name}  args={args}")
                    result = str(call.get("result") or "")
                    if len(result) > 1200:
                        result = result[:1200] + "…"
                    if result:
                        lines.append("result:")
                        for al in result.splitlines()[:50]:
                            lines.append(f"  {al}")
                ans = (t.get("answer") or "").strip()
                if ans:
                    lines.append("answer:")
                    for al in ans.splitlines() or [ans]:
                        lines.append(f"  {al}")
                reason = (t.get("force_reason") or "").strip()
                if reason:
                    lines.append(f"force_reason: {reason}")
                lines.append("")

        final = str(respond.get("answer") or "").strip()
        if final:
            lines.append("## Final Answer")
            lines.append("-" * 48)
            lines.append(final)
        return "\n".join(lines).rstrip()

    def _compute_recall(
        self,
        expected_names: Sequence[str],
        retrieved_names: Sequence[str],
    ) -> Dict[str, Any]:
        hit, miss, ret_unique = match_doc_names(expected_names, retrieved_names)
        n_exp = len(list(expected_names or []))
        n_hit = len(hit)
        recall = safe_div(float(n_hit), float(n_exp)) if n_exp else None
        return {
            "expected_docs": list(expected_names or []),
            "retrieved_docs": list(retrieved_names or []),
            "hit_docs": hit,
            "miss_docs": miss,
            "n_expected": n_exp,
            "n_hit": n_hit,
            "n_retrieved": len(ret_unique),
            "recall": recall,
        }

    def _source_block(self, source_docs: Sequence[dict]) -> str:
        full_docs = []
        for d in source_docs or []:
            full_docs.append({
                "doc_id": d.get("doc_id"),
                "name": d.get("name"),
                "content": truncate_text(d.get("content") or "", self.max_source_chars),
            })
        return format_docs_block(full_docs, max_chars_per_doc=self.max_source_chars)

    @staticmethod
    def _block_answer(block: Optional[Dict[str, Any]]) -> str:
        if not isinstance(block, dict):
            return ""
        return (block.get("answer") or block.get("raw_answer") or "").strip()

    @classmethod
    def _system_attempted(cls, block: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(block, dict) or not block:
            return False
        if cls._block_answer(block):
            return True
        if block.get("query_error"):
            return True
        if block.get("query_status") == 1:
            return True
        metrics = block.get("metrics") or {}
        return any(
            metrics.get(k) is not None
            for k in ("query_latency_s", "wall_latency_s")
        )

    @staticmethod
    def _parse_side_judgment(obj: dict) -> Optional[Dict[str, Any]]:
        if not isinstance(obj, dict) or not obj:
            return None
        judgment = normalize_judgment(
            obj.get("judgment") or obj.get("llm_acc") or ""
        )
        if judgment not in JUDGMENT_LABELS:
            return None
        return {
            "llm_acc": judgment,
            "judge_reason": (obj.get("reason") or "").strip(),
            "judge_status": 1,
            "judge_error": None,
        }

    @staticmethod
    def _infer_winner(
        sides: Dict[str, Dict[str, Any]],
        a_sys: str,
        b_sys: str,
    ) -> Optional[str]:
        rank = {"正确": 1, "错误": 0}
        ra = rank.get((sides.get(a_sys) or {}).get("llm_acc"))
        rb = rank.get((sides.get(b_sys) or {}).get("llm_acc"))
        if ra is None or rb is None:
            return None
        if ra > rb:
            return a_sys
        if rb > ra:
            return b_sys
        return "tie"

    def _judge_one(
        self,
        *,
        question: str,
        ground_truth_answer: str,
        source_docs: Sequence[dict],
        rag_answer: str,
    ) -> Dict[str, Any]:
        user = fill_prompt(
            Benchmark_PROMPT["JUDGE_USER_SINGLE"],
            {
                "question": question or "",
                "ground_truth_answer": ground_truth_answer or "",
                "source_block": self._source_block(source_docs),
                "rag_answer": rag_answer or "",
            },
        )
        last_err = None
        timeout_s = llm_client_timeout(self.judge_llm)
        for attempt in range(1, self.max_judge_retries + 1):
            resp = call_llm(
                self.judge_llm,
                system=Benchmark_PROMPT.get("JUDGE_SYSTEM", ""),
                user=user,
                model_args=self.judge_model_args,
                use_cache=self.use_cache,
            )
            usage = {
                "prompt_tokens": resp.get("usage_prompt_tokens"),
                "completion_tokens": resp.get("usage_completion_tokens"),
                "total_tokens": resp.get("usage_total_tokens"),
            }
            if resp.get("status") != 1:
                last_err = (
                    f"judge llm status={resp.get('status')}: "
                    f"{str(resp.get('answer') or '')[:120]}"
                )
                if not llm_attempt_should_retry(
                    resp,
                    attempt=attempt,
                    max_retries=self.max_judge_retries,
                    timeout_s=timeout_s,
                ):
                    break
                continue
            obj = extract_json_object(resp.get("answer") or "")
            if not obj:
                last_err = "judge json parse failed"
                if not llm_attempt_should_retry(
                    resp,
                    attempt=attempt,
                    max_retries=self.max_judge_retries,
                    timeout_s=timeout_s,
                ):
                    break
                continue
            parsed = self._parse_side_judgment(obj)
            if not parsed:
                last_err = f"invalid judgment={obj.get('judgment')!r}"
                if not llm_attempt_should_retry(
                    resp,
                    attempt=attempt,
                    max_retries=self.max_judge_retries,
                    timeout_s=timeout_s,
                ):
                    break
                continue
            parsed["judge_latency_s"] = resp.get("latency_s")
            parsed["judge_usage"] = usage
            return parsed
        return {
            "llm_acc": None,
            "judge_reason": "",
            "judge_status": 0,
            "judge_error": last_err or "judge failed",
            "judge_latency_s": None,
            "judge_usage": {},
        }

    def _judge_pair(
        self,
        *,
        qid: Any,
        question: str,
        ground_truth_answer: str,
        source_docs: Sequence[dict],
        answers: Dict[str, str],
    ) -> Dict[str, Any]:
        a_sys, b_sys = pairwise_system_order(qid, SYSTEM_AGENTIC, SYSTEM_LLM_ONLY)
        ans_a = (answers.get(a_sys) or "").strip() or EMPTY_ANSWER_PLACEHOLDER
        ans_b = (answers.get(b_sys) or "").strip() or EMPTY_ANSWER_PLACEHOLDER
        user = fill_prompt(
            Benchmark_PROMPT["JUDGE_USER"],
            {
                "question": question or "",
                "ground_truth_answer": ground_truth_answer or "",
                "source_block": self._source_block(source_docs),
                "answer_a": ans_a,
                "answer_b": ans_b,
            },
        )
        last_err = None
        timeout_s = llm_client_timeout(self.judge_llm)
        for attempt in range(1, self.max_judge_retries + 1):
            resp = call_llm(
                self.judge_llm,
                system=Benchmark_PROMPT.get("JUDGE_SYSTEM", ""),
                user=user,
                model_args=self.judge_model_args,
                use_cache=self.use_cache,
            )
            usage = {
                "prompt_tokens": resp.get("usage_prompt_tokens"),
                "completion_tokens": resp.get("usage_completion_tokens"),
                "total_tokens": resp.get("usage_total_tokens"),
            }
            if resp.get("status") != 1:
                last_err = (
                    f"judge llm status={resp.get('status')}: "
                    f"{str(resp.get('answer') or '')[:120]}"
                )
                if not llm_attempt_should_retry(
                    resp,
                    attempt=attempt,
                    max_retries=self.max_judge_retries,
                    timeout_s=timeout_s,
                ):
                    break
                continue
            obj = extract_json_object(resp.get("answer") or "")
            if not obj:
                last_err = "judge json parse failed"
                if not llm_attempt_should_retry(
                    resp,
                    attempt=attempt,
                    max_retries=self.max_judge_retries,
                    timeout_s=timeout_s,
                ):
                    break
                continue

            named = extract_named_system_sides(obj)
            if named:
                parsed_a = self._parse_side_judgment(named.get(a_sys) or {})
                parsed_b = self._parse_side_judgment(named.get(b_sys) or {})
            else:
                parsed_a = self._parse_side_judgment(extract_pair_side(obj, "A"))
                parsed_b = self._parse_side_judgment(extract_pair_side(obj, "B"))
            if not parsed_a or not parsed_b:
                last_err = (
                    f"invalid pair judgment "
                    f"A={None if not parsed_a else parsed_a.get('llm_acc')} "
                    f"B={None if not parsed_b else parsed_b.get('llm_acc')}"
                )
                if not llm_attempt_should_retry(
                    resp,
                    attempt=attempt,
                    max_retries=self.max_judge_retries,
                    timeout_s=timeout_s,
                ):
                    break
                continue

            sides = {a_sys: parsed_a, b_sys: parsed_b}
            winner = resolve_pairwise_winner(
                pairwise_better_raw(obj),
                a_system=a_sys,
                b_system=b_sys,
            ) or self._infer_winner(sides, a_sys, b_sys)
            reason = ""
            if isinstance(obj, dict):
                reason = str(obj.get("reason") or "").strip()
                cmp_ = obj.get("comparison")
                if not reason and isinstance(cmp_, dict):
                    reason = str(cmp_.get("reason") or "").strip()
            return {
                "sides": sides,
                "winner": winner,
                "reason": reason,
                "order": [a_sys, b_sys],
                "judge_status": 1,
                "judge_error": None,
                "judge_latency_s": resp.get("latency_s"),
                "judge_usage": usage,
            }

        return {
            "sides": {},
            "winner": None,
            "reason": "",
            "order": [a_sys, b_sys],
            "judge_status": 0,
            "judge_error": last_err or "judge failed",
            "judge_latency_s": None,
            "judge_usage": {},
        }

    def _apply_pair_judge(
        self,
        result: Dict[str, Any],
        judge: Dict[str, Any],
    ) -> None:
        sides = judge.get("sides") or {}
        usage = judge.get("judge_usage") or {}
        shared = {
            "judge_status": judge.get("judge_status"),
            "judge_error": judge.get("judge_error"),
            "judge_latency_s": judge.get("judge_latency_s"),
            "judge_usage": usage,
        }
        for sys in (SYSTEM_AGENTIC, SYSTEM_LLM_ONLY):
            block = result.get(sys)
            if not isinstance(block, dict):
                continue
            payload = dict(sides.get(sys) or {})
            payload.update(shared)
            if "llm_acc" not in payload:
                payload["llm_acc"] = None
                payload["judge_reason"] = payload.get("judge_reason") or ""
            self._apply_judge(block, payload)
        result["comparison"] = {
            "winner": judge.get("winner"),
            "reason": judge.get("reason") or "",
            "order": list(judge.get("order") or []),
            "judge_status": judge.get("judge_status"),
            "judge_error": judge.get("judge_error"),
            "metrics": {
                "judge_latency_s": judge.get("judge_latency_s"),
                "judge_prompt_tokens": usage.get("prompt_tokens"),
                "judge_completion_tokens": usage.get("completion_tokens"),
                "judge_total_tokens": usage.get("total_tokens"),
            },
        }

    def _skip_judge_block(self, block: Optional[Dict[str, Any]], err: str) -> None:
        if not isinstance(block, dict):
            return
        block["judge_status"] = 0
        block["judge_error"] = err

    def _judge_result(
        self,
        result: Dict[str, Any],
        *,
        source_docs: Sequence[dict],
    ) -> None:
        question = result.get("question") or ""
        gt = result.get("ground_truth_answer") or ""
        ag = result.get(SYSTEM_AGENTIC) or {}
        lo = result.get(SYSTEM_LLM_ONLY) or {}
        ag_ans = self._block_answer(ag)
        lo_ans = self._block_answer(lo)

        lo_present = self._system_attempted(lo)
        if self.enable_llm_only and lo_present:
            if not ag_ans and not lo_ans:
                self._skip_judge_block(
                    ag,
                    "skip judge: query failed" if ag.get("query_error") else "skip judge: empty answer",
                )
                self._skip_judge_block(
                    lo,
                    "skip judge: query failed" if lo.get("query_error") else "skip judge: empty answer",
                )
                return
            self._apply_pair_judge(
                result,
                self._judge_pair(
                    qid=result.get("id"),
                    question=question,
                    ground_truth_answer=gt,
                    source_docs=source_docs,
                    answers={
                        SYSTEM_AGENTIC: ag_ans,
                        SYSTEM_LLM_ONLY: lo_ans,
                    },
                ),
            )
            return

        if ag_ans:
            self._apply_judge(
                ag,
                self._judge_one(
                    question=question,
                    ground_truth_answer=gt,
                    source_docs=source_docs,
                    rag_answer=ag_ans,
                ),
            )
        elif ag.get("query_error"):
            self._skip_judge_block(ag, "skip judge: query failed")
        else:
            self._skip_judge_block(ag, "skip judge: empty answer")

    @staticmethod
    def _collect_agentic_metrics(respond: Dict[str, Any], wall_s: float) -> Dict[str, Any]:
        """把 agentic_query 返回的 retrieve_timing / 工具次数摊成可汇总字段。"""
        pt = respond.get("usage_prompt_tokens")
        ct = respond.get("usage_completion_tokens")
        tt = respond.get("usage_total_tokens")
        if tt is None and pt is not None and ct is not None:
            try:
                tt = int(pt) + int(ct)
            except Exception:
                tt = None

        turns = list(respond.get("turns") or [])
        n_search = count_tool_calls(turns, "search")
        n_read = count_tool_calls(turns, "read_doc")
        n_neighbors = count_tool_calls(turns, "graph_neighbors")
        timing = as_timing_dict(respond.get("retrieve_timing"))
        retrieve_s = respond.get("retrieve_latency_s")
        try:
            retrieve_s = float(retrieve_s) if retrieve_s is not None else None
        except (TypeError, ValueError):
            retrieve_s = None

        def _per_search(total: Optional[float]) -> Optional[float]:
            if total is None or n_search <= 0:
                return None
            return float(total) / n_search

        metrics: Dict[str, Any] = {
            "query_latency_s": respond.get("latency_s"),
            "retrieve_latency_s": retrieve_s,
            "wall_latency_s": wall_s,
            "n_turns": len(turns),
            "n_search": n_search,
            "n_read_doc": n_read,
            "n_graph_neighbors": n_neighbors,
            "protocol": respond.get("protocol"),
            "trace_path": respond.get("trace_path"),
            "last_prompt_tokens": respond.get("last_prompt_tokens"),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,
            "retrieve_timing": dict(timing),
            "mean_search_s": _per_search(retrieve_s),
        }
        for k in RETRIEVE_STAGE_KEYS:
            metrics[k] = timing.get(k)
            metrics[f"mean_{k}"] = _per_search(timing.get(k))
        # 别名，汇总表更好读
        metrics["precompute_latency_s"] = timing.get("precompute_s")
        metrics["rewrite_latency_s"] = timing.get("rewrite_s")
        metrics["embed_latency_s"] = timing.get("embed_s")
        metrics["chunk_latency_s"] = timing.get("chunk_s")
        metrics["node_latency_s"] = timing.get("node_s")
        metrics["keyword_latency_s"] = timing.get("keyword_s")
        metrics["expand_latency_s"] = timing.get("expand_s")
        metrics["rerank_latency_s"] = timing.get("rerank_s")
        metrics["mean_precompute_s"] = metrics["mean_precompute_s"]
        metrics["mean_rewrite_s"] = metrics["mean_rewrite_s"]
        metrics["mean_chunk_s"] = metrics["mean_chunk_s"]
        metrics["mean_node_s"] = metrics["mean_node_s"]
        metrics["mean_keyword_s"] = metrics["mean_keyword_s"]
        metrics["mean_rerank_s"] = metrics["mean_rerank_s"]
        return metrics

    def _fill_agentic(self, question: str) -> Dict[str, Any]:
        block = _empty_system_block()
        t0 = time.perf_counter()
        quiet_names = ["DHMF"]
        try:
            if getattr(self.dhmf, "logger", None) is not None:
                quiet_names.append(self.dhmf.logger.name)
        except Exception:
            pass
        try:
            with quiet_loggers(*quiet_names, level=40):
                respond = self.dhmf.agentic_query(question, pretty=False)
        except Exception as e:
            block["query_error"] = str(e)
            block["metrics"]["query_latency_s"] = time.perf_counter() - t0
            block["metrics"]["wall_latency_s"] = time.perf_counter() - t0
            return block

        wall_s = time.perf_counter() - t0
        if not isinstance(respond, dict):
            block["query_error"] = f"unexpected respond type: {type(respond)}"
            block["answer"] = str(respond)
            block["raw_answer"] = str(respond)
            block["metrics"]["wall_latency_s"] = wall_s
            return block

        block["query_status"] = respond.get("status", 0)
        block["answer"] = self._extract_answer_text(respond)
        block["raw_answer"] = self._format_agentic_process(respond)
        block["retrieval_sources"] = list(respond.get("retrieval_sources") or [])
        block["retrieval_doc_ids"] = list(respond.get("retrieval_doc_ids") or [])
        if block["query_status"] != 1:
            block["query_error"] = (
                f"query status={block['query_status']}: "
                f"{str(block['raw_answer'])[:200]}"
            )
        block["metrics"] = self._collect_agentic_metrics(respond, wall_s)
        return block

    def _fill_llm_only(self, question: str) -> Dict[str, Any]:
        block = _empty_system_block()
        t0 = time.perf_counter()
        try:
            respond = self._run_llm_only(question)
        except Exception as e:
            block["query_error"] = str(e)
            block["metrics"]["query_latency_s"] = time.perf_counter() - t0
            block["metrics"]["wall_latency_s"] = time.perf_counter() - t0
            return block

        wall_s = time.perf_counter() - t0
        if not isinstance(respond, dict):
            block["query_error"] = f"unexpected respond type: {type(respond)}"
            block["answer"] = str(respond)
            block["raw_answer"] = str(respond)
            block["metrics"]["wall_latency_s"] = wall_s
            return block

        block["query_status"] = respond.get("status", 0)
        block["answer"] = self._extract_answer_text(respond, parse_sections=False)
        block["raw_answer"] = respond.get("answer") or block["answer"]
        if block["query_status"] != 1:
            block["query_error"] = (
                f"llm_only status={block['query_status']}: "
                f"{str(block['raw_answer'])[:200]}"
            )
        elif not str(block["answer"] or "").strip():
            block["query_status"] = 0
            block["query_error"] = "llm_only empty answer"
        pt = respond.get("usage_prompt_tokens")
        ct = respond.get("usage_completion_tokens")
        tt = respond.get("usage_total_tokens")
        if tt is None and pt is not None and ct is not None:
            try:
                tt = int(pt) + int(ct)
            except Exception:
                tt = None
        block["metrics"] = {
            "query_latency_s": respond.get("latency_s"),
            "retrieve_latency_s": None,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,
            "wall_latency_s": wall_s,
        }
        return block

    def evaluate_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        qid = item.get("id") or "unknown"
        question = item.get("question") or ""
        expected_names = list(item.get("source_names") or [])
        if not expected_names and item.get("source_docs"):
            expected_names = [
                d.get("name") for d in item["source_docs"] if d.get("name")
            ]

        result: Dict[str, Any] = {
            "id": qid,
            "hop": item.get("hop"),
            "question": question,
            "ground_truth_answer": item.get("ground_truth_answer") or "",
            "explanation": item.get("explanation") or "",
            "source_names": expected_names,
            "recall": None,
            SYSTEM_AGENTIC: _empty_system_block(),
        }
        if self.enable_llm_only:
            result[SYSTEM_LLM_ONLY] = _empty_system_block()

        try:
            if not question.strip():
                result[SYSTEM_AGENTIC]["query_error"] = "empty question"
                if self.enable_llm_only:
                    result[SYSTEM_LLM_ONLY]["query_error"] = "empty question"
                return result

            result[SYSTEM_AGENTIC] = self._fill_agentic(question)
            ag = result[SYSTEM_AGENTIC]
            if self.enable_doc_recall:
                result["recall"] = self._compute_recall(
                    expected_names, ag.get("retrieval_sources") or []
                )

            if self.enable_llm_only:
                result[SYSTEM_LLM_ONLY] = self._fill_llm_only(question)

            self._judge_result(result, source_docs=item.get("source_docs") or [])
            return result
        finally:
            if self.sleep_between > 0:
                time.sleep(self.sleep_between)

    @staticmethod
    def _system_failed(block: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(block, dict) or not block:
            return True
        if block.get("query_error"):
            return True
        if block.get("query_status") == 0:
            return True
        if block.get("judge_status") == 0 or block.get("judge_error"):
            return True
        return False

    def _agentic_attempted(self, r: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(r, dict):
            return False
        return self._system_attempted(r.get(SYSTEM_AGENTIC))

    def _is_item_complete(self, r: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(r, dict) or r.get("id") is None:
            return False
        if not self._agentic_attempted(r):
            return False
        if not self.enable_llm_only:
            return True
        return self._system_attempted(r.get(SYSTEM_LLM_ONLY))

    def _backfill_llm_only(
        self,
        result: Dict[str, Any],
        item: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        r = dict(result)
        question = r.get("question") or (item or {}).get("question") or ""
        r[SYSTEM_LLM_ONLY] = self._fill_llm_only(question)
        source_docs = (item or {}).get("source_docs") or []
        self._judge_result(r, source_docs=source_docs)
        return r

    @classmethod
    def _is_eval_failure(cls, r: Dict[str, Any]) -> bool:
        ag = r.get(SYSTEM_AGENTIC)
        if isinstance(ag, dict) and (
            ag.get("query_status") is not None
            or ag.get("query_error")
            or ag.get("llm_acc") is not None
            or ag.get("answer")
        ):
            if not r.get("question"):
                return True
            return cls._system_failed(ag)
        if not r.get("question"):
            return True
        return False

    def _log_eval_failure(self, r: Dict[str, Any]) -> None:
        parts = [f"评测 {r.get('id')}"]
        ag = r.get(SYSTEM_AGENTIC) or {}
        lo = r.get(SYSTEM_LLM_ONLY) or {}
        qerr = ag.get("query_error")
        jerr = (r.get("comparison") or {}).get("judge_error") or ag.get("judge_error")
        if qerr:
            parts.append(f"agentic_query={str(qerr)[:120]}")
        if jerr:
            parts.append(f"judge={str(jerr)[:120]}")
        if lo.get("query_error"):
            parts.append(f"llm_only_query={str(lo['query_error'])[:120]}")
        fail_print(" | ".join(parts))

    def _count_ok(self, results: Sequence[Optional[Dict[str, Any]]], system: str) -> int:
        n = 0
        for x in results:
            if not isinstance(x, dict):
                continue
            block = x.get(system) or {}
            if isinstance(block, dict) and block.get("llm_acc") == "正确":
                n += 1
        return n

    def evaluate_all(
        self,
        dataset: Dict[str, Any],
        *,
        existing_results: Optional[Sequence[Dict[str, Any]]] = None,
        on_progress: OnProgress = None,
        extra_meta: Optional[dict] = None,
    ) -> Dict[str, Any]:
        questions = list(dataset.get("questions") or [])
        items = [
            q for q in questions if q.get("gen_status", 1) == 1 and q.get("question")
        ]
        skipped = len(questions) - len(items)
        if skipped:
            fail_print(f"评测跳过 {skipped} 条生成失败/空问题的样本")

        complete: Dict[str, Dict[str, Any]] = {}
        backfill: Dict[str, Dict[str, Any]] = {}
        for r in existing_results or []:
            if not isinstance(r, dict) or r.get("id") is None:
                continue
            qid = str(r["id"])
            if self._is_item_complete(r):
                complete[qid] = r
            elif self.enable_llm_only and self._agentic_attempted(r):
                backfill[qid] = r

        total = len(items)
        results: List[Optional[Dict[str, Any]]] = [None] * total
        todo: List[tuple] = []
        n_resumed = 0
        for idx, item in enumerate(items):
            qid = str(item.get("id") or f"idx_{idx}")
            if qid in complete:
                results[idx] = complete[qid]
                n_resumed += 1
            elif qid in backfill:
                todo.append((idx, item, backfill[qid]))
            else:
                todo.append((idx, item, None))
        n_todo = len(todo)
        n_backfill = sum(1 for t in todo if t[2] is not None)
        if n_resumed or n_backfill:
            print(
                f"[resume] 已完成 {n_resumed}，补纯LLM {n_backfill}，"
                f"新跑 {n_todo - n_backfill}",
                file=sys.stderr,
            )

        def _run_item(item: Dict[str, Any], existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            if existing is not None:
                return self._backfill_llm_only(existing, item)
            return self.evaluate_one(item)

        workers = min(self.num_thread, max(1, n_todo)) if n_todo else 1
        t0 = time.perf_counter()
        created_at = datetime.now().isoformat(timespec="seconds")
        fail_n = sum(
            1 for r in results if r is not None and self._is_eval_failure(r)
        )
        done_n = sum(1 for r in results if r is not None)

        def _on_item(r: Dict[str, Any]) -> None:
            nonlocal fail_n, done_n
            done_n += 1
            if self._is_eval_failure(r):
                fail_n += 1
                self._log_eval_failure(r)

        def _pack(done: bool) -> Dict[str, Any]:
            done_results = [r for r in results if r is not None]
            return build_eval_document(
                dataset=dataset,
                results=done_results,
                skipped=skipped,
                t0=t0,
                created_at=created_at,
                done=done,
                enable_doc_recall=self.enable_doc_recall,
                enable_llm_only=self.enable_llm_only,
                extra_meta={
                    "query_mode": "agentic",
                    "judge_model": (self.judge_model_args or {}).get("model"),
                    "n_total_planned": total,
                    "num_thread": workers,
                    "n_resumed": n_resumed,
                    **(extra_meta or {}),
                },
            )

        def _emit_progress() -> None:
            if on_progress is None:
                return
            mid = _pack(done=False)
            try:
                on_progress(mid, sum(1 for r in results if r is not None), total)
            except Exception as e:
                fail_print(f"on_progress 保存失败: {e}")

        if n_todo == 0:
            return _pack(done=True)

        if workers <= 1:
            pbar = progress_iter(todo, total=n_todo, desc="评测问答", unit="题")
            for idx, item, existing in pbar:
                r = _run_item(item, existing)
                results[idx] = r
                _on_item(r)
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(
                        fail=fail_n,
                        ag=self._count_ok(results, SYSTEM_AGENTIC),
                        llm=self._count_ok(results, SYSTEM_LLM_ONLY),
                        thr=1,
                        refresh=False,
                    )
                _emit_progress()
        else:
            if _tqdm is not None:
                pbar = _tqdm(
                    total=n_todo,
                    desc=f"评测问答×{workers}",
                    unit="题",
                    file=sys.stderr,
                    dynamic_ncols=True,
                    leave=True,
                    mininterval=0.2,
                )
            else:
                pbar = None

            with ThreadPoolExecutor(max_workers=workers) as pool:
                fut_to_idx = {
                    pool.submit(_run_item, item, existing): idx
                    for idx, item, existing in todo
                }
                for fut in as_completed(fut_to_idx):
                    idx = fut_to_idx[fut]
                    try:
                        r = fut.result()
                    except Exception as e:
                        item = items[idx]
                        ag = _empty_system_block()
                        ag["query_error"] = f"worker exception: {e}"
                        lo = _empty_system_block()
                        lo["query_error"] = f"worker exception: {e}"
                        r = {
                            "id": item.get("id") or f"idx_{idx}",
                            "hop": item.get("hop"),
                            "question": item.get("question") or "",
                            "ground_truth_answer": item.get("ground_truth_answer") or "",
                            "explanation": item.get("explanation") or "",
                            "source_names": list(item.get("source_names") or []),
                            "recall": None,
                            SYSTEM_AGENTIC: ag,
                            SYSTEM_LLM_ONLY: lo,
                        }
                    results[idx] = r
                    _on_item(r)
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(
                            fail=fail_n,
                            ag=self._count_ok(results, SYSTEM_AGENTIC),
                            llm=self._count_ok(results, SYSTEM_LLM_ONLY),
                            thr=workers,
                            refresh=False,
                        )
                    else:
                        print(
                            f"\r评测问答×{workers}: {done_n}/{total} "
                            f"ok={done_n - fail_n} fail={fail_n}",
                            end="",
                            file=sys.stderr,
                            flush=True,
                        )
                    _emit_progress()
            if pbar is not None:
                pbar.close()
            elif total:
                print(file=sys.stderr)

        return _pack(done=True)
