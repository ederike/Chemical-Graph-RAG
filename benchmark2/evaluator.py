"""Excel 测试集：DHMF RAG 回答 + LLM 评判。"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from benchmark.config import merge_llm_only_model_args
from benchmark.evaluator import QueryEvaluator
from benchmark.utils import (
    EMPTY_ANSWER_PLACEHOLDER,
    call_llm,
    extract_json_object,
    extract_named_system_sides,
    extract_pair_side,
    fail_print,
    fill_prompt,
    llm_attempt_should_retry,
    llm_client_timeout,
    llm_response_has_text,
    pairwise_better_raw,
    pairwise_system_order,
    progress_iter,
    quiet_loggers,
    resolve_pairwise_winner,
)

from .prompts import Benchmark2_PROMPT

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None

JUDGMENT_LABELS = ("正确", "部分正确", "错误")
SCORE_MAX = 3


def normalize_score(raw: Any) -> Optional[int]:
    if raw is None or raw == "":
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        s = str(raw).strip()
        digits = "".join(ch for ch in s if ch.isdigit() or ch == ".")
        if not digits:
            return None
        try:
            v = float(digits)
        except ValueError:
            return None
    iv = int(round(v))
    if iv < 0:
        iv = 0
    if iv > SCORE_MAX:
        iv = SCORE_MAX
    return iv


def judgment_from_score(score: Optional[int]) -> Optional[str]:
    if score is None:
        return None
    if score >= 2:
        return "正确"
    if score == 1:
        return "部分正确"
    return "错误"


def normalize_judgment(raw: Any, *, score: Optional[int] = None) -> Optional[str]:
    if raw is not None:
        s = str(raw).strip()
        if s in JUDGMENT_LABELS:
            return s
        low = s.lower()
        if any(k in s for k in ("部分正确", "部分对", "不完全", "partial")):
            return "部分正确"
        if any(
            k in s
            for k in ("不正确", "全部错误", "完全错误", "错误", "wrong", "incorrect")
        ):
            return "错误"
        if any(
            k in s
            for k in ("全部正确", "完全正确", "正确", "correct", "right")
        ) or low in ("true", "yes"):
            return "正确"
    return judgment_from_score(score)


def format_score_dimensions(dims: Sequence[str], raw: str = "") -> str:
    items = [str(d).strip() for d in (dims or []) if str(d).strip()]
    if items:
        lines = [f"{i}. {d}" for i, d in enumerate(items, 1)]
        if raw and raw.strip() and raw.strip() not in items:
            lines.append(f"（原文：{raw.strip()}）")
        return "\n".join(lines)
    return (raw or "").strip() or "（本题未给出独立评分维度，按预期要点整体评判）"


SYSTEM_HYPERGRAPH = "hypergraph"
SYSTEM_LLM_ONLY = "llm_only"

# 已写入 hypergraph / llm_only 后不再在顶层重复的字段
_MIRRORED_TOP_KEYS = (
    "rag_answer",
    "rag_raw_answer",
    "query_status",
    "query_error",
    "retrieval_sources",
    "retrieval_doc_ids",
    "llm_acc",
    "score",
    "judge_reason",
    "dimension_scores",
    "judge_status",
    "judge_error",
    "metrics",
    "llm_only_answer",
)


def _empty_system_block() -> Dict[str, Any]:
    return {
        "answer": "",
        "raw_answer": "",
        "query_status": 0,
        "query_error": None,
        "retrieval_sources": [],
        "retrieval_doc_ids": [],
        "llm_acc": None,
        "score": None,
        "judge_reason": "",
        "dimension_scores": [],
        "judge_status": None,
        "judge_error": None,
        "metrics": {},
    }


def _answer_model_args_from_dhmf(dhmf, override: Optional[dict] = None) -> dict:
    """纯 LLM 生成参数：继承 retrieve 的 model；思考开关以 yaml/override 为准。"""
    base: Dict[str, Any] = {}
    try:
        if dhmf is not None:
            base = dict(getattr(dhmf.config.retrieve, "model_args", None) or {})
    except Exception:
        pass
    return merge_llm_only_model_args(base, override)


class ExcelQueryEvaluator:
    """
    对 Excel 测试集逐题：
      1) 超图：按 query_mode 调用 DHMF.query / agent_query / agentic_query
      2) 纯 LLM：同一模型、不检索，直接根据问题作答
      3) 两路回答一次送入裁判，按同一套验收标准对比打分
    """

    def __init__(
        self,
        dhmf,
        judge_llm=None,
        *,
        judge_model_args: Optional[dict] = None,
        answer_llm=None,
        answer_model_args: Optional[dict] = None,
        query_mode: str = "agent",
        use_cache: bool = False,
        max_judge_retries: int = 3,
        max_llm_only_retries: int = 3,
        llm_only_use_cache: Optional[bool] = None,
        sleep_between: float = 0.0,
        num_thread: int = 1,
        enable_llm_only: bool = True,
        prompts: Optional[dict] = None,
    ):
        self.dhmf = dhmf
        self.prompts = prompts or Benchmark2_PROMPT
        self.judge_llm = judge_llm or (
            getattr(dhmf, "llmmodel", None) if dhmf is not None else None
        )
        if self.judge_llm is None:
            raise ValueError("judge_llm 未提供且 DHMF 无 llmmodel")
        self.answer_llm = answer_llm or (
            getattr(dhmf, "llmmodel", None) if dhmf is not None else None
        ) or self.judge_llm

        self.judge_model_args = dict(judge_model_args or {})
        self.judge_model_args.setdefault("temperature", 0.0)
        self.judge_model_args.setdefault("enable_thinking", False)
        self.judge_model_args.setdefault(
            "response_format", {"type": "json_object"}
        )
        if not self.judge_model_args.get("model"):
            try:
                ma = getattr(dhmf.config.retrieve, "model_args", None) or {}
                if ma.get("model"):
                    self.judge_model_args["model"] = ma["model"]
            except Exception:
                pass

        self.answer_model_args = _answer_model_args_from_dhmf(
            dhmf, answer_model_args
        )
        self.enable_llm_only = bool(enable_llm_only)

        self.query_mode = QueryEvaluator._normalize_query_mode(query_mode)
        self.use_cache = bool(use_cache)
        self.llm_only_use_cache = (
            self.use_cache if llm_only_use_cache is None else bool(llm_only_use_cache)
        )
        self.max_judge_retries = max(1, int(max_judge_retries))
        self.max_llm_only_retries = max(1, int(max_llm_only_retries))
        self.sleep_between = float(sleep_between)
        try:
            nt = int(num_thread)
        except (TypeError, ValueError):
            nt = 1
        self.num_thread = max(1, nt)

    def _run_query(self, question: str) -> Any:
        if self.query_mode == "agentic":
            return self.dhmf.agentic_query(question, pretty=False)
        if self.query_mode == "agent":
            return self.dhmf.agent_query(question, pretty=False)
        return self.dhmf.query(question, mode="dual_path", pretty=False)

    def _run_llm_only(self, question: str) -> Dict[str, Any]:
        """不检索，直接用同一套生成模型回答。"""
        user = self.prompts["PURE_LLM_USER"].replace(
            "{question}", question or ""
        )
        last: Dict[str, Any] = {}
        timeout_s = llm_client_timeout(self.answer_llm)
        for attempt in range(1, self.max_llm_only_retries + 1):
            last = call_llm(
                self.answer_llm,
                system=self.prompts.get("PURE_LLM_SYSTEM", ""),
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
        return last if isinstance(last, dict) else {
            "status": 0,
            "answer": str(last),
        }

    @staticmethod
    def _apply_judge(block: Dict[str, Any], judge: Dict[str, Any]) -> None:
        block["llm_acc"] = judge.get("llm_acc")
        block["score"] = judge.get("score")
        block["judge_reason"] = judge.get("judge_reason")
        block["dimension_scores"] = judge.get("dimension_scores") or []
        block["judge_status"] = judge.get("judge_status")
        block["judge_error"] = judge.get("judge_error")
        metrics = block.setdefault("metrics", {})
        metrics["judge_latency_s"] = judge.get("judge_latency_s")
        ju = judge.get("judge_usage") or {}
        metrics["judge_prompt_tokens"] = ju.get("prompt_tokens")
        metrics["judge_completion_tokens"] = ju.get("completion_tokens")
        metrics["judge_total_tokens"] = ju.get("total_tokens")

    @staticmethod
    def _drop_mirrored_system_fields(result: Dict[str, Any]) -> Dict[str, Any]:
        """两路回答只留在 hypergraph / llm_only，去掉顶层副本。"""
        hg = result.get(SYSTEM_HYPERGRAPH)
        if isinstance(hg, dict) and hg:
            for k in _MIRRORED_TOP_KEYS:
                result.pop(k, None)
        return result

    def _extract_answer_text(self, respond: Any, *, parse_sections: bool = True) -> str:
        return QueryEvaluator._extract_answer_text(
            self, respond, parse_sections=parse_sections
        )

    def _format_agent_process(self, respond: Dict[str, Any]) -> str:
        return QueryEvaluator._format_agent_process(respond)

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

    def _parse_side_judgment(
        self,
        obj: dict,
        *,
        score_dimensions: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(obj, dict) or not obj:
            return None
        score = normalize_score(obj.get("score"))
        judgment = normalize_judgment(
            obj.get("judgment") or obj.get("llm_acc"),
            score=score,
        )
        if judgment not in JUDGMENT_LABELS:
            return None
        if score is None:
            score = {"正确": 2, "部分正确": 1, "错误": 0}[judgment]

        dim_scores = []
        raw_dims = obj.get("dimension_scores") or obj.get("dimensions") or []
        if isinstance(raw_dims, list):
            for d in raw_dims:
                if not isinstance(d, dict):
                    continue
                name = str(d.get("dimension") or d.get("name") or "").strip()
                if not name:
                    continue
                ds = normalize_score(d.get("score"))
                passed = d.get("pass")
                if passed is None and ds is not None:
                    passed = ds >= 2
                dim_scores.append(
                    {
                        "dimension": name,
                        "score": ds,
                        "pass": bool(passed) if passed is not None else None,
                        "comment": str(d.get("comment") or "").strip(),
                    }
                )
        if not dim_scores:
            for name in score_dimensions or []:
                dim_scores.append(
                    {
                        "dimension": name,
                        "score": score,
                        "pass": score >= 2 if score is not None else None,
                        "comment": "",
                    }
                )
        return {
            "llm_acc": judgment,
            "score": score,
            "judge_reason": (obj.get("reason") or "").strip(),
            "dimension_scores": dim_scores,
            "judge_status": 1,
            "judge_error": None,
        }

    @staticmethod
    def _infer_winner(
        sides: Dict[str, Dict[str, Any]],
        a_sys: str,
        b_sys: str,
    ) -> Optional[str]:
        sa = (sides.get(a_sys) or {}).get("score")
        sb = (sides.get(b_sys) or {}).get("score")
        if sa is None or sb is None:
            return None
        try:
            fa, fb = float(sa), float(sb)
        except (TypeError, ValueError):
            return None
        if fa > fb:
            return a_sys
        if fb > fa:
            return b_sys
        return "tie"

    def _judge_one(
        self,
        *,
        question: str,
        expected_answer: str,
        score_dimensions: Sequence[str],
        score_dimensions_raw: str,
        rag_answer: str,
    ) -> Dict[str, Any]:
        """单路 0-3 打分（仅 enable_llm_only=false 时使用）。"""
        dims_block = format_score_dimensions(
            score_dimensions, raw=score_dimensions_raw or ""
        )
        user = fill_prompt(
            self.prompts["JUDGE_USER_SINGLE"],
            {
                "question": question or "",
                "expected_answer": expected_answer or "",
                "score_dimensions": dims_block,
                "answer": rag_answer or "",
            },
        )

        last_err = None
        timeout_s = llm_client_timeout(self.judge_llm)
        for attempt in range(1, self.max_judge_retries + 1):
            resp = call_llm(
                self.judge_llm,
                system=self.prompts.get("JUDGE_SYSTEM", ""),
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
            parsed = self._parse_side_judgment(
                obj, score_dimensions=score_dimensions
            )
            if not parsed:
                last_err = (
                    f"invalid judgment={obj.get('judgment')!r} "
                    f"score={obj.get('score')!r}"
                )
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
            "score": None,
            "judge_reason": "",
            "dimension_scores": [],
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
        expected_answer: str,
        score_dimensions: Sequence[str],
        score_dimensions_raw: str,
        answers: Dict[str, str],
    ) -> Dict[str, Any]:
        """两路回答一次送入裁判，对比后分别打 0-3 分。"""
        a_sys, b_sys = pairwise_system_order(qid, SYSTEM_HYPERGRAPH, SYSTEM_LLM_ONLY)
        dims_block = format_score_dimensions(
            score_dimensions, raw=score_dimensions_raw or ""
        )
        ans_a = (answers.get(a_sys) or "").strip() or EMPTY_ANSWER_PLACEHOLDER
        ans_b = (answers.get(b_sys) or "").strip() or EMPTY_ANSWER_PLACEHOLDER
        user = fill_prompt(
            self.prompts["JUDGE_USER"],
            {
                "question": question or "",
                "expected_answer": expected_answer or "",
                "score_dimensions": dims_block,
                "answer_a": ans_a,
                "answer_b": ans_b,
            },
        )

        last_err = None
        timeout_s = llm_client_timeout(self.judge_llm)
        for attempt in range(1, self.max_judge_retries + 1):
            resp = call_llm(
                self.judge_llm,
                system=self.prompts.get("JUDGE_SYSTEM", ""),
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
                parsed_a = self._parse_side_judgment(
                    named.get(a_sys) or {}, score_dimensions=score_dimensions
                )
                parsed_b = self._parse_side_judgment(
                    named.get(b_sys) or {}, score_dimensions=score_dimensions
                )
            else:
                parsed_a = self._parse_side_judgment(
                    extract_pair_side(obj, "A"), score_dimensions=score_dimensions
                )
                parsed_b = self._parse_side_judgment(
                    extract_pair_side(obj, "B"), score_dimensions=score_dimensions
                )
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
        for sys in (SYSTEM_HYPERGRAPH, SYSTEM_LLM_ONLY):
            block = result.get(sys)
            if not isinstance(block, dict):
                continue
            payload = dict(sides.get(sys) or {})
            payload.update(shared)
            payload.setdefault("llm_acc", None)
            payload.setdefault("score", None)
            payload.setdefault("judge_reason", "")
            payload.setdefault("dimension_scores", [])
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

    def _judge_result(self, result: Dict[str, Any]) -> None:
        question = result.get("question") or ""
        expected = result.get("expected_answer") or ""
        dims = list(result.get("score_dimensions") or [])
        dims_raw = result.get("score_dimensions_raw") or ""
        hg = result.get(SYSTEM_HYPERGRAPH) or {}
        lo = result.get(SYSTEM_LLM_ONLY) or {}
        hg_ans = self._block_answer(hg)
        lo_ans = self._block_answer(lo)

        lo_present = self._system_attempted(lo)
        if self.enable_llm_only and lo_present:
            if not hg_ans and not lo_ans:
                self._skip_judge_block(
                    hg,
                    "skip judge: query failed" if hg.get("query_error") else "skip judge: empty answer",
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
                    expected_answer=expected,
                    score_dimensions=dims,
                    score_dimensions_raw=dims_raw,
                    answers={
                        SYSTEM_HYPERGRAPH: hg_ans,
                        SYSTEM_LLM_ONLY: lo_ans,
                    },
                ),
            )
            return

        if hg_ans:
            self._apply_judge(
                hg,
                self._judge_one(
                    question=question,
                    expected_answer=expected,
                    score_dimensions=dims,
                    score_dimensions_raw=dims_raw,
                    rag_answer=hg_ans,
                ),
            )
        elif hg.get("query_error"):
            self._skip_judge_block(hg, "skip judge: query failed")
        else:
            self._skip_judge_block(hg, "skip judge: empty answer")

    def _fill_hypergraph(self, question: str) -> Dict[str, Any]:
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
                respond = self._run_query(question)
        except Exception as e:
            block["query_error"] = str(e)
            block["metrics"]["query_latency_s"] = time.perf_counter() - t0
            block["metrics"]["wall_latency_s"] = time.perf_counter() - t0
            return block

        if not isinstance(respond, dict):
            block["query_error"] = f"unexpected respond type: {type(respond)}"
            block["answer"] = str(respond)
            block["raw_answer"] = str(respond)
            block["metrics"]["wall_latency_s"] = time.perf_counter() - t0
            return block

        block["query_status"] = respond.get("status", 0)
        block["answer"] = self._extract_answer_text(respond)
        if (
            self.query_mode in ("agent", "agentic")
            or respond.get("plan")
            or respond.get("steps")
            or respond.get("turns")
        ):
            block["raw_answer"] = self._format_agent_process(respond)
        else:
            block["raw_answer"] = respond.get("answer") or ""
        block["retrieval_sources"] = list(respond.get("retrieval_sources") or [])
        block["retrieval_doc_ids"] = list(respond.get("retrieval_doc_ids") or [])

        if block["query_status"] != 1:
            block["query_error"] = (
                f"query status={block['query_status']}: "
                f"{str(block['raw_answer'])[:200]}"
            )

        pt = respond.get("usage_prompt_tokens")
        ct = respond.get("usage_completion_tokens")
        tt = respond.get("usage_total_tokens")
        if tt is None and pt is not None and ct is not None:
            try:
                tt = int(pt) + int(ct)
            except Exception:
                tt = None
        retrieve_timing = respond.get("retrieve_timing") or {}
        if not isinstance(retrieve_timing, dict):
            retrieve_timing = {}
        block["metrics"] = {
            "query_latency_s": respond.get("latency_s"),
            "retrieve_latency_s": respond.get("retrieve_latency_s"),
            "retrieve_timing": dict(retrieve_timing),
            "precompute_latency_s": retrieve_timing.get("precompute_s"),
            "rewrite_latency_s": retrieve_timing.get("rewrite_s"),
            "embed_latency_s": retrieve_timing.get("embed_s"),
            "chunk_latency_s": retrieve_timing.get("chunk_s"),
            "node_latency_s": retrieve_timing.get("node_s"),
            "keyword_latency_s": retrieve_timing.get("keyword_s"),
            "expand_latency_s": retrieve_timing.get("expand_s"),
            "rerank_latency_s": retrieve_timing.get("rerank_s"),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,
            "wall_latency_s": time.perf_counter() - t0,
            "last_prompt_tokens": respond.get("last_prompt_tokens"),
            "n_turns": len(respond.get("turns") or respond.get("steps") or []),
            "protocol": respond.get("protocol"),
            "trace_path": respond.get("trace_path"),
        }
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

        if not isinstance(respond, dict):
            block["query_error"] = f"unexpected respond type: {type(respond)}"
            block["answer"] = str(respond)
            block["raw_answer"] = str(respond)
            block["metrics"]["wall_latency_s"] = time.perf_counter() - t0
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
            "wall_latency_s": time.perf_counter() - t0,
        }
        return block

    def evaluate_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        qid = item.get("id") or "unknown"
        question = item.get("question") or ""
        expected = item.get("expected_answer") or ""
        dims = list(item.get("score_dimensions") or [])
        dims_raw = item.get("score_dimensions_raw") or ""

        result: Dict[str, Any] = {
            "id": qid,
            "raw_id": item.get("raw_id"),
            "question": question,
            "expected_answer": expected,
            "score_dimensions": dims,
            "score_dimensions_raw": dims_raw,
            "categories": dict(item.get("categories") or {}),
            "reasoning_path": item.get("reasoning_path") or "",
            "knowledge_source": item.get("knowledge_source") or "",
            "note": item.get("note") or "",
            SYSTEM_HYPERGRAPH: _empty_system_block(),
        }
        if self.enable_llm_only:
            result[SYSTEM_LLM_ONLY] = _empty_system_block()

        try:
            if not str(question).strip():
                result[SYSTEM_HYPERGRAPH]["query_error"] = "empty question"
                if self.enable_llm_only:
                    result[SYSTEM_LLM_ONLY]["query_error"] = "empty question"
                return self._drop_mirrored_system_fields(result)

            result[SYSTEM_HYPERGRAPH] = self._fill_hypergraph(question)
            if self.enable_llm_only:
                result[SYSTEM_LLM_ONLY] = self._fill_llm_only(question)
            self._judge_result(result)
            return self._drop_mirrored_system_fields(result)
        finally:
            if self.sleep_between > 0:
                time.sleep(self.sleep_between)

    def rejudge_one(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """只重跑裁判，不重新检索/生成。两路回答一次对比打分。"""
        r = self._drop_mirrored_system_fields(dict(result))
        self._judge_result(r)
        return r

    def rejudge_all(
        self,
        eval_data: Dict[str, Any],
        *,
        on_progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        results_in = list(eval_data.get("results") or [])
        total = len(results_in)
        if total == 0:
            raise ValueError("evals 没有 results，无法重判")

        dataset = {
            "meta": (eval_data.get("meta") or {}).get("dataset_meta"),
            "stats": eval_data.get("stats"),
            "questions": results_in,
        }
        workers = min(self.num_thread, max(1, total))
        t0 = time.perf_counter()
        created_at = datetime.now().isoformat(timespec="seconds")
        out: List[Optional[Dict[str, Any]]] = [None] * total
        done_n = 0

        def _emit(done: bool) -> None:
            if on_progress is None:
                return
            done_results = [x for x in out if x is not None]
            mid = self._make_report(
                dataset=dataset,
                results=done_results,
                t0=t0,
                created_at=created_at,
                done=done,
            )
            mid["meta"]["n_total_planned"] = total
            mid["meta"]["num_thread"] = workers
            mid["meta"]["rejudge"] = True
            src_meta = eval_data.get("meta") if isinstance(eval_data.get("meta"), dict) else {}
            for k in ("dataset_meta", "query_mode", "judge_model", "answer_model", "enable_llm_only"):
                if src_meta.get(k) is not None:
                    mid["meta"].setdefault(k, src_meta[k])
            try:
                on_progress(mid, len(done_results), total)
            except Exception as e:
                fail_print(f"on_progress 保存失败: {e}")

        if workers <= 1:
            pbar = progress_iter(list(enumerate(results_in)), total=total, desc="重判", unit="题")
            for idx, row in pbar:
                out[idx] = self.rejudge_one(row)
                done_n += 1
                _emit(False)
        else:
            if _tqdm is not None:
                pbar = _tqdm(
                    total=total,
                    desc=f"重判×{workers}",
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
                    pool.submit(self.rejudge_one, row): idx
                    for idx, row in enumerate(results_in)
                }
                for fut in as_completed(fut_to_idx):
                    idx = fut_to_idx[fut]
                    try:
                        out[idx] = fut.result()
                    except Exception as e:
                        row = dict(results_in[idx])
                        fail_print(f"重判 {row.get('id')}: {e}")
                        out[idx] = row
                    done_n += 1
                    if pbar is not None:
                        pbar.update(1)
                    _emit(False)
            if pbar is not None:
                pbar.close()

        final_rows = [x for x in out if x is not None]
        report = self._make_report(
            dataset=dataset,
            results=final_rows,
            t0=t0,
            created_at=created_at,
            done=True,
        )
        src_meta = eval_data.get("meta") if isinstance(eval_data.get("meta"), dict) else {}
        report["meta"].update(
            {
                "n_total_planned": total,
                "num_thread": workers,
                "rejudge": True,
                "dataset_meta": src_meta.get("dataset_meta") or report["meta"].get("dataset_meta"),
                "query_mode": src_meta.get("query_mode") or report["meta"].get("query_mode"),
                "enable_llm_only": src_meta.get("enable_llm_only")
                if src_meta.get("enable_llm_only") is not None
                else report["meta"].get("enable_llm_only"),
            }
        )
        if eval_data.get("stats"):
            report["stats"] = eval_data["stats"]
        return report

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

    @staticmethod
    def _is_eval_failure(r: Dict[str, Any]) -> bool:
        """单系统行（投影后 / 旧顶层格式）是否流水线失败。"""
        if r.get("query_error"):
            return True
        if r.get("query_status") == 0:
            return True
        if r.get("judge_status") == 0 or r.get("judge_error"):
            return True
        if not r.get("question"):
            return True
        return False

    def _is_item_failure(self, r: Dict[str, Any]) -> bool:
        if not r.get("question"):
            return True
        hg = r.get(SYSTEM_HYPERGRAPH)
        if isinstance(hg, dict) and hg:
            hg_fail = self._system_failed(hg)
        else:
            hg_fail = self._is_eval_failure(r)
        if not self.enable_llm_only:
            return hg_fail
        return hg_fail or self._system_failed(r.get(SYSTEM_LLM_ONLY))

    def _hypergraph_attempted(self, r: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(r, dict):
            return False
        if self._system_attempted(r.get(SYSTEM_HYPERGRAPH)):
            return True
        return r.get("query_status") is not None or bool(r.get("rag_answer"))

    def _is_item_complete(self, r: Optional[Dict[str, Any]]) -> bool:
        """断点续跑：两路都跑过才算完成。空的 llm_only 占位不算。"""
        if not isinstance(r, dict) or r.get("id") is None:
            return False
        if not self._hypergraph_attempted(r):
            return False
        if not self.enable_llm_only:
            return True
        return self._system_attempted(r.get(SYSTEM_LLM_ONLY))

    def _backfill_llm_only(
        self,
        result: Dict[str, Any],
        item: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """保留已有超图结果，只补纯 LLM 并重新对比评判。"""
        r = self._drop_mirrored_system_fields(dict(result))
        question = r.get("question") or (item or {}).get("question") or ""
        r[SYSTEM_LLM_ONLY] = self._fill_llm_only(question)
        self._judge_result(r)
        return self._drop_mirrored_system_fields(r)

    def _make_report(
        self,
        *,
        dataset: Dict[str, Any],
        results: List[Dict[str, Any]],
        t0: float,
        created_at: str,
        done: bool = False,
        n_resumed: int = 0,
    ) -> Dict[str, Any]:
        from .report import build_summary

        stats = dataset.get("stats") if isinstance(dataset.get("stats"), dict) else None
        summary = build_summary(
            results,
            stats=stats,
            enable_llm_only=self.enable_llm_only,
        )
        return {
            "meta": {
                "created_at": created_at,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "done": bool(done),
                "dataset_meta": dataset.get("meta"),
                "query_mode": self.query_mode,
                "judge_model": (self.judge_model_args or {}).get("model"),
                "judge_mode": "pairwise" if self.enable_llm_only else "single",
                "answer_model": (self.answer_model_args or {}).get("model"),
                "enable_llm_only": self.enable_llm_only,
                "n_questions": len(results),
                "n_total_planned": None,
                "n_resumed": n_resumed,
                "elapsed_s": round(time.perf_counter() - t0, 3),
            },
            "results": results,
            "summary": summary,
        }

    def _log_eval_failure(self, r: Dict[str, Any]) -> None:
        parts = [f"评测 {r.get('id')}"]
        hg = r.get(SYSTEM_HYPERGRAPH) or {}
        lo = r.get(SYSTEM_LLM_ONLY) or {}
        qerr = hg.get("query_error") or r.get("query_error")
        jerr = (
            (r.get("comparison") or {}).get("judge_error")
            or hg.get("judge_error")
            or r.get("judge_error")
        )
        if qerr:
            parts.append(f"hypergraph_query={str(qerr)[:120]}")
        if jerr:
            parts.append(f"hypergraph_judge={str(jerr)[:120]}")
        if lo.get("query_error"):
            parts.append(f"llm_only_query={str(lo['query_error'])[:120]}")
        if lo.get("judge_error"):
            parts.append(f"llm_only_judge={str(lo['judge_error'])[:120]}")
        fail_print(" | ".join(parts))

    def _count_ok(self, results: Sequence[Optional[Dict[str, Any]]], system: str) -> int:
        n = 0
        for x in results:
            if x is None:
                continue
            if system == SYSTEM_HYPERGRAPH:
                block = x.get(SYSTEM_HYPERGRAPH)
                lab = (block or {}).get("llm_acc") if isinstance(block, dict) else x.get("llm_acc")
            else:
                block = x.get(SYSTEM_LLM_ONLY) or {}
                lab = block.get("llm_acc")
            if lab == "正确":
                n += 1
        return n

    def evaluate_all(
        self,
        dataset: Dict[str, Any],
        *,
        existing_results: Optional[Sequence[Dict[str, Any]]] = None,
        on_progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        questions = list(dataset.get("questions") or [])
        items = [q for q in questions if (q.get("question") or "").strip()]
        skipped = len(questions) - len(items)
        if skipped:
            fail_print(f"评测跳过 {skipped} 条空问题")

        by_id: Dict[str, Dict[str, Any]] = {}
        backfill: Dict[str, Dict[str, Any]] = {}
        for r in existing_results or []:
            if not isinstance(r, dict) or r.get("id") is None:
                continue
            qid = str(r["id"])
            if self._is_item_complete(r):
                by_id[qid] = r
            elif self.enable_llm_only and self._hypergraph_attempted(r):
                backfill[qid] = r
        n_resumed = 0
        todo: List[tuple] = []
        for idx, item in enumerate(items):
            qid = str(item.get("id") or f"idx_{idx}")
            if qid in by_id:
                n_resumed += 1
            elif qid in backfill:
                todo.append((idx, item, backfill[qid]))
            else:
                todo.append((idx, item, None))

        total = len(items)
        n_todo = len(todo)
        n_backfill = sum(1 for t in todo if t[2] is not None)
        if n_resumed or n_backfill:
            print(
                f"[resume] 已完成 {n_resumed}，补纯LLM {n_backfill}，"
                f"新跑 {n_todo - n_backfill}",
                file=sys.stderr,
            )
        workers = min(self.num_thread, max(1, n_todo)) if n_todo else 1
        t0 = time.perf_counter()
        created_at = datetime.now().isoformat(timespec="seconds")
        results: List[Optional[Dict[str, Any]]] = [None] * total
        for idx, item in enumerate(items):
            qid = str(item.get("id") or f"idx_{idx}")
            if qid in by_id:
                results[idx] = self._drop_mirrored_system_fields(dict(by_id[qid]))

        def _run_item(item: Dict[str, Any], existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            if existing is not None:
                return self._backfill_llm_only(existing, item)
            return self.evaluate_one(item)

        fail_n = sum(
            1
            for r in results
            if r is not None and self._is_item_failure(r)
        )
        done_n = sum(1 for r in results if r is not None)

        def _on_item(r: Dict[str, Any]) -> None:
            nonlocal fail_n, done_n
            done_n += 1
            if self._is_item_failure(r):
                fail_n += 1
                self._log_eval_failure(r)

        def _emit_progress() -> None:
            if on_progress is None:
                return
            done_results = [r for r in results if r is not None]
            mid = self._make_report(
                dataset=dataset,
                results=done_results,
                t0=t0,
                created_at=created_at,
                done=False,
                n_resumed=n_resumed,
            )
            mid["meta"]["n_total_planned"] = total
            mid["meta"]["num_thread"] = workers
            try:
                on_progress(mid, len(done_results), total)
            except Exception as e:
                fail_print(f"on_progress 保存失败: {e}")

        if n_todo == 0:
            out_results = [r for r in results if r is not None]
            report = self._make_report(
                dataset=dataset,
                results=out_results,
                t0=t0,
                created_at=created_at,
                done=True,
                n_resumed=n_resumed,
            )
            report["meta"]["n_total_planned"] = total
            report["meta"]["num_thread"] = workers
            return report

        if workers <= 1:
            pbar = progress_iter(todo, total=n_todo, desc="评测问答", unit="题")
            for idx, item, existing in pbar:
                r = _run_item(item, existing)
                results[idx] = r
                _on_item(r)
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(
                        fail=fail_n,
                        hg=self._count_ok(results, SYSTEM_HYPERGRAPH),
                        llm=self._count_ok(results, SYSTEM_LLM_ONLY),
                        resume=n_resumed,
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
                        hg = _empty_system_block()
                        hg["query_error"] = f"worker exception: {e}"
                        lo = _empty_system_block()
                        lo["query_error"] = f"worker exception: {e}"
                        r = self._drop_mirrored_system_fields({
                            "id": item.get("id") or f"idx_{idx}",
                            "question": item.get("question") or "",
                            "expected_answer": item.get("expected_answer") or "",
                            "categories": dict(item.get("categories") or {}),
                            SYSTEM_HYPERGRAPH: hg,
                            SYSTEM_LLM_ONLY: lo,
                        })
                    results[idx] = r
                    _on_item(r)
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(
                            fail=fail_n,
                            hg=self._count_ok(results, SYSTEM_HYPERGRAPH),
                            llm=self._count_ok(results, SYSTEM_LLM_ONLY),
                            resume=n_resumed,
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
            elif n_todo:
                print(file=sys.stderr)

        out_results: List[Dict[str, Any]] = [r for r in results if r is not None]
        report = self._make_report(
            dataset=dataset,
            results=out_results,
            t0=t0,
            created_at=created_at,
            done=True,
            n_resumed=n_resumed,
        )
        report["meta"]["n_total_planned"] = total
        report["meta"]["num_thread"] = workers
        return report
