"""agentic_query + 纯 LLM 对照 + Excel 维度成对 Judge。

指标抽取对齐 benchmark_agentic_1；裁判提示词与 0-3 分制对齐 Excel 测试集。
本包独立实现，不 import 其他评测流。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import merge_llm_only_model_args
from .prompts import PROMPT
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
    return max(0, min(SCORE_MAX, iv))


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
        if any(k in s for k in ("部分正确", "部分对", "不完全", "partial")):
            return "部分正确"
        if any(k in s for k in ("不正确", "全部错误", "完全错误", "错误", "wrong", "incorrect")):
            return "错误"
        if any(k in s for k in ("全部正确", "完全正确", "正确", "correct", "right")) or str(raw).strip().lower() in ("true", "yes"):
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
    base: Dict[str, Any] = {}
    try:
        if dhmf is not None:
            base = dict(getattr(dhmf.config.retrieve, "model_args", None) or {})
    except Exception:
        pass
    return merge_llm_only_model_args(base, override)


def _parse_dimension_scores(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        sc = normalize_score(item.get("score"))
        out.append({
            "dimension": str(item.get("dimension") or item.get("name") or "").strip(),
            "score": sc,
            "pass": bool(item.get("pass")) if item.get("pass") is not None else (sc is not None and sc >= 2),
            "comment": str(item.get("comment") or item.get("reason") or "").strip(),
        })
    return out

class ExcelAgenticEvaluator:
    """
    每题固定两路：
      1) DHMF.agentic_query
      2) 纯 LLM（不检索）
    两路都答完后用 JUDGE_USER 一次对比打分（0-3 + 维度）。
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
        enable_doc_recall: bool = False,
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
        user = PROMPT["PURE_LLM_USER"].replace("{question}", question or "")
        last: Dict[str, Any] = {}
        timeout_s = llm_client_timeout(self.answer_llm)
        for attempt in range(1, self.max_llm_only_retries + 1):
            last = call_llm(
                self.answer_llm,
                system=PROMPT.get("PURE_LLM_SYSTEM", ""),
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

    def _apply_judge(self, block: Dict[str, Any], judge: Dict[str, Any]) -> None:
        block["llm_acc"] = judge.get("llm_acc")
        block["score"] = judge.get("score")
        block["judge_reason"] = judge.get("judge_reason")
        block["dimension_scores"] = list(judge.get("dimension_scores") or [])
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
        return any(metrics.get(k) is not None for k in ("query_latency_s", "wall_latency_s"))

    @staticmethod
    def _parse_side_judgment(obj: dict) -> Optional[Dict[str, Any]]:
        if not isinstance(obj, dict) or not obj:
            return None
        score = normalize_score(obj.get("score"))
        judgment = normalize_judgment(
            obj.get("judgment") or obj.get("llm_acc"),
            score=score,
        )
        if judgment not in JUDGMENT_LABELS and score is None:
            return None
        if judgment not in JUDGMENT_LABELS:
            judgment = judgment_from_score(score)
        if judgment not in JUDGMENT_LABELS:
            return None
        if score is None:
            score = {"正确": 3, "部分正确": 1, "错误": 0}.get(judgment)
        return {
            "llm_acc": judgment,
            "score": score,
            "judge_reason": (obj.get("reason") or "").strip(),
            "dimension_scores": _parse_dimension_scores(obj.get("dimension_scores")),
            "judge_status": 1,
            "judge_error": None,
        }

    @staticmethod
    def _infer_winner(
        sides: Dict[str, Dict[str, Any]],
        a_sys: str,
        b_sys: str,
    ) -> Optional[str]:
        def _rank(side: dict) -> Optional[float]:
            if not side:
                return None
            sc = side.get("score")
            if sc is not None:
                return float(sc)
            return {"正确": 3.0, "部分正确": 1.0, "错误": 0.0}.get(side.get("llm_acc") or "")

        ra = _rank(sides.get(a_sys) or {})
        rb = _rank(sides.get(b_sys) or {})
        if ra is None or rb is None:
            return None
        if ra > rb:
            return a_sys
        if rb > ra:
            return b_sys
        return "tie"

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
        a_sys, b_sys = pairwise_system_order(qid, SYSTEM_AGENTIC, SYSTEM_LLM_ONLY)
        ans_a = (answers.get(a_sys) or "").strip() or EMPTY_ANSWER_PLACEHOLDER
        ans_b = (answers.get(b_sys) or "").strip() or EMPTY_ANSWER_PLACEHOLDER
        user = fill_prompt(
            PROMPT["JUDGE_USER"],
            {
                "question": question or "",
                "expected_answer": expected_answer or "",
                "score_dimensions": format_score_dimensions(
                    score_dimensions, score_dimensions_raw or ""
                ),
                "answer_a": ans_a,
                "answer_b": ans_b,
            },
        )
        last_err = None
        timeout_s = llm_client_timeout(self.judge_llm)
        for attempt in range(1, self.max_judge_retries + 1):
            resp = call_llm(
                self.judge_llm,
                system=PROMPT.get("JUDGE_SYSTEM", ""),
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
                    resp, attempt=attempt, max_retries=self.max_judge_retries, timeout_s=timeout_s,
                ):
                    break
                continue
            obj = extract_json_object(resp.get("answer") or "")
            if not obj:
                last_err = "judge json parse failed"
                if not llm_attempt_should_retry(
                    resp, attempt=attempt, max_retries=self.max_judge_retries, timeout_s=timeout_s,
                ):
                    break
                continue

            parsed_a = self._parse_side_judgment(extract_pair_side(obj, "A"))
            parsed_b = self._parse_side_judgment(extract_pair_side(obj, "B"))
            if not parsed_a or not parsed_b:
                last_err = (
                    f"invalid pair judgment "
                    f"A={None if not parsed_a else parsed_a.get('llm_acc')} "
                    f"B={None if not parsed_b else parsed_b.get('llm_acc')}"
                )
                if not llm_attempt_should_retry(
                    resp, attempt=attempt, max_retries=self.max_judge_retries, timeout_s=timeout_s,
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

    def _judge_one(
        self,
        *,
        question: str,
        expected_answer: str,
        score_dimensions: Sequence[str],
        score_dimensions_raw: str,
        answer: str,
    ) -> Dict[str, Any]:
        user = fill_prompt(
            PROMPT["JUDGE_USER_SINGLE"],
            {
                "question": question or "",
                "expected_answer": expected_answer or "",
                "score_dimensions": format_score_dimensions(
                    score_dimensions, score_dimensions_raw or ""
                ),
                "answer": answer or "",
            },
        )
        last_err = None
        timeout_s = llm_client_timeout(self.judge_llm)
        for attempt in range(1, self.max_judge_retries + 1):
            resp = call_llm(
                self.judge_llm,
                system=PROMPT.get("JUDGE_SYSTEM", ""),
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
                last_err = f"judge llm status={resp.get('status')}"
                if not llm_attempt_should_retry(
                    resp, attempt=attempt, max_retries=self.max_judge_retries, timeout_s=timeout_s,
                ):
                    break
                continue
            obj = extract_json_object(resp.get("answer") or "")
            parsed = self._parse_side_judgment(obj or {})
            if not parsed:
                last_err = "judge json/side parse failed"
                if not llm_attempt_should_retry(
                    resp, attempt=attempt, max_retries=self.max_judge_retries, timeout_s=timeout_s,
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

    def _apply_pair_judge(self, result: Dict[str, Any], judge: Dict[str, Any]) -> None:
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

    def _judge_result(self, result: Dict[str, Any]) -> None:
        question = result.get("question") or ""
        expected = result.get("expected_answer") or result.get("ground_truth_answer") or ""
        dims = result.get("score_dimensions") or []
        dims_raw = result.get("score_dimensions_raw") or ""
        ag = result.get(SYSTEM_AGENTIC) or {}
        lo = result.get(SYSTEM_LLM_ONLY) or {}
        ag_ans = self._block_answer(ag)
        lo_ans = self._block_answer(lo)
        lo_present = self._system_attempted(lo)

        if self.enable_llm_only and lo_present:
            if not ag_ans and not lo_ans:
                self._skip_judge_block(ag, "skip judge: empty answer")
                self._skip_judge_block(lo, "skip judge: empty answer")
                return
            self._apply_pair_judge(
                result,
                self._judge_pair(
                    qid=result.get("id"),
                    question=question,
                    expected_answer=expected,
                    score_dimensions=dims,
                    score_dimensions_raw=dims_raw,
                    answers={SYSTEM_AGENTIC: ag_ans, SYSTEM_LLM_ONLY: lo_ans},
                ),
            )
            return

        if ag_ans:
            self._apply_judge(
                ag,
                self._judge_one(
                    question=question,
                    expected_answer=expected,
                    score_dimensions=dims,
                    score_dimensions_raw=dims_raw,
                    answer=ag_ans,
                ),
            )
        elif ag.get("query_error"):
            self._skip_judge_block(ag, "skip judge: query failed")
        else:
            self._skip_judge_block(ag, "skip judge: empty answer")

    @staticmethod
    def _collect_agentic_metrics(respond: Dict[str, Any], wall_s: float) -> Dict[str, Any]:
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
        metrics["precompute_latency_s"] = timing.get("precompute_s")
        metrics["rewrite_latency_s"] = timing.get("rewrite_s")
        metrics["embed_latency_s"] = timing.get("embed_s")
        metrics["chunk_latency_s"] = timing.get("chunk_s")
        metrics["node_latency_s"] = timing.get("node_s")
        metrics["keyword_latency_s"] = timing.get("keyword_s")
        metrics["expand_latency_s"] = timing.get("expand_s")
        metrics["rerank_latency_s"] = timing.get("rerank_s")
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
            expected_names = [d.get("name") for d in item["source_docs"] if d.get("name")]

        result: Dict[str, Any] = {
            "id": qid,
            "dataset_id": item.get("dataset_id"),
            "row": item.get("row"),
            "raw_id": item.get("raw_id"),
            "hop": item.get("hop") or (item.get("categories") or {}).get("推理跳数"),
            "question": question,
            "expected_answer": item.get("expected_answer") or "",
            "ground_truth_answer": item.get("expected_answer") or "",
            "score_dimensions": list(item.get("score_dimensions") or []),
            "score_dimensions_raw": item.get("score_dimensions_raw") or "",
            "reasoning_path": item.get("reasoning_path") or "",
            "knowledge_source": item.get("knowledge_source") or "",
            "note": item.get("note") or "",
            "categories": dict(item.get("categories") or {}),
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

            self._judge_result(result)
            return result
        finally:
            if self.sleep_between > 0:
                time.sleep(self.sleep_between)

    def rejudge_one(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """仅重跑评判，保留已有两路回答与 metrics。"""
        r = dict(result)
        for sys in (SYSTEM_AGENTIC, SYSTEM_LLM_ONLY):
            block = r.get(sys)
            if isinstance(block, dict):
                block = dict(block)
                block["llm_acc"] = None
                block["score"] = None
                block["judge_reason"] = ""
                block["dimension_scores"] = []
                block["judge_status"] = None
                block["judge_error"] = None
                r[sys] = block
        r.pop("comparison", None)
        self._judge_result(r)
        return r

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

    def _merge_resume(
        self,
        items: Sequence[Dict[str, Any]],
        existing: Sequence[Dict[str, Any]],
    ) -> tuple:
        by_id = {}
        for r in existing or []:
            if isinstance(r, dict) and r.get("id") is not None:
                by_id[r["id"]] = r
        pending = []
        kept = []
        for it in items:
            qid = it.get("id")
            old = by_id.get(qid)
            if old is not None and self._is_item_complete(old):
                kept.append(old)
            else:
                pending.append(it)
        return kept, pending

    def evaluate(
        self,
        items: Sequence[Dict[str, Any]],
        *,
        existing_results: Optional[Sequence[Dict[str, Any]]] = None,
        resume: bool = True,
        on_progress: OnProgress = None,
        dataset_meta: Optional[dict] = None,
        config_meta: Optional[dict] = None,
    ) -> Dict[str, Any]:
        items = list(items or [])
        kept: List[Dict[str, Any]] = []
        pending = items
        if resume and existing_results:
            kept, pending = self._merge_resume(items, existing_results)
            print(
                f"[evaluate] resume: keep={len(kept)} pending={len(pending)}",
                flush=True,
            )

        results: List[Dict[str, Any]] = list(kept)
        total = len(pending)
        if total == 0:
            return build_eval_document(
                results,
                dataset_meta=dataset_meta,
                config_meta=config_meta,
                enable_llm_only=self.enable_llm_only,
                enable_doc_recall=self.enable_doc_recall,
            )

        def _work(it: Dict[str, Any]) -> Dict[str, Any]:
            return self.evaluate_one(it)

        done = 0
        if self.num_thread <= 1:
            for it in progress_iter(pending, desc="evaluate"):
                r = _work(it)
                if self._is_eval_failure(r):
                    self._log_eval_failure(r)
                results.append(r)
                done += 1
                if on_progress:
                    on_progress(r, done, total)
        else:
            with ThreadPoolExecutor(max_workers=self.num_thread) as ex:
                futs = {ex.submit(_work, it): it for it in pending}
                for fut in progress_iter(as_completed(futs), total=total, desc="evaluate"):
                    r = fut.result()
                    if self._is_eval_failure(r):
                        self._log_eval_failure(r)
                    results.append(r)
                    done += 1
                    if on_progress:
                        on_progress(r, done, total)

        # stable order by original item ids
        order = {it.get("id"): i for i, it in enumerate(items)}
        results.sort(key=lambda r: order.get(r.get("id"), 10**9))

        return build_eval_document(
            results,
            dataset_meta=dataset_meta,
            config_meta=config_meta,
            enable_llm_only=self.enable_llm_only,
            enable_doc_recall=self.enable_doc_recall,
        )

    def rejudge(
        self,
        results: Sequence[Dict[str, Any]],
        *,
        dataset_meta: Optional[dict] = None,
        config_meta: Optional[dict] = None,
        on_progress: OnProgress = None,
    ) -> Dict[str, Any]:
        rows = list(results or [])
        out: List[Dict[str, Any]] = []
        total = len(rows)

        def _work(r: Dict[str, Any]) -> Dict[str, Any]:
            return self.rejudge_one(r)

        if self.num_thread <= 1:
            for i, r in enumerate(progress_iter(rows, desc="rejudge"), 1):
                nr = _work(r)
                out.append(nr)
                if on_progress:
                    on_progress(nr, i, total)
        else:
            with ThreadPoolExecutor(max_workers=self.num_thread) as ex:
                futs = {ex.submit(_work, r): r for r in rows}
                done = 0
                for fut in progress_iter(as_completed(futs), total=total, desc="rejudge"):
                    nr = fut.result()
                    out.append(nr)
                    done += 1
                    if on_progress:
                        on_progress(nr, done, total)
            order = {r.get("id"): i for i, r in enumerate(rows)}
            out.sort(key=lambda r: order.get(r.get("id"), 10**9))

        return build_eval_document(
            out,
            dataset_meta=dataset_meta,
            config_meta=config_meta,
            enable_llm_only=self.enable_llm_only,
            enable_doc_recall=self.enable_doc_recall,
        )

