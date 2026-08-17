"""Excel 测试集：DHMF RAG 回答 + LLM 从宽评判。"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from benchmark.evaluator import QueryEvaluator
from benchmark.utils import (
    call_llm,
    extract_json_object,
    fail_print,
    progress_iter,
    quiet_loggers,
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
    """纯 LLM 生成参数：继承 retrieve.model_args，去掉 json_object。"""
    base: Dict[str, Any] = {}
    try:
        base = dict(getattr(dhmf.config.retrieve, "model_args", None) or {})
    except Exception:
        pass
    over = dict(override or {})
    if not over.get("model"):
        over.pop("model", None)
    merged = {**base, **over}
    merged.pop("response_format", None)
    merged.setdefault("temperature", 0.2)
    merged.setdefault("enable_thinking", False)
    return merged


class ExcelQueryEvaluator:
    """
    对 Excel 测试集逐题：
      1) 超图+LLM：按 query_mode 调用 DHMF.query / agent_query
      2) 纯 LLM：同一模型、不检索，直接根据问题作答
      3) 两路回答各自用同一套从宽标准打分
    """

    def __init__(
        self,
        dhmf,
        judge_llm=None,
        *,
        judge_model_args: Optional[dict] = None,
        answer_model_args: Optional[dict] = None,
        query_mode: str = "agent",
        use_cache: bool = False,
        max_judge_retries: int = 3,
        sleep_between: float = 0.0,
        num_thread: int = 1,
        enable_llm_only: bool = True,
    ):
        self.dhmf = dhmf
        self.judge_llm = judge_llm or getattr(dhmf, "llmmodel", None)
        if self.judge_llm is None:
            raise ValueError("judge_llm 未提供且 DHMF 无 llmmodel")
        self.answer_llm = getattr(dhmf, "llmmodel", None) or self.judge_llm

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
        self.max_judge_retries = max(1, int(max_judge_retries))
        self.sleep_between = float(sleep_between)
        try:
            nt = int(num_thread)
        except (TypeError, ValueError):
            nt = 1
        self.num_thread = max(1, nt)

    def _run_query(self, question: str) -> Any:
        if self.query_mode == "agent":
            return self.dhmf.agent_query(question, pretty=False)
        return self.dhmf.query(question, mode="dual_path", pretty=False)

    def _run_llm_only(self, question: str) -> Dict[str, Any]:
        """不检索，直接用同一套生成模型回答。"""
        user = Benchmark2_PROMPT["PURE_LLM_USER"].replace(
            "{question}", question or ""
        )
        return call_llm(
            self.answer_llm,
            system=Benchmark2_PROMPT.get("PURE_LLM_SYSTEM", ""),
            user=user,
            model_args=self.answer_model_args,
            use_cache=self.use_cache,
        )

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

    def _mirror_hypergraph(self, result: Dict[str, Any]) -> None:
        """顶层字段与 hypergraph 同步，便于浏览 / 兼容旧 summary。"""
        hg = result.get(SYSTEM_HYPERGRAPH) or {}
        result["rag_answer"] = hg.get("answer") or ""
        result["rag_raw_answer"] = hg.get("raw_answer") or ""
        result["query_status"] = hg.get("query_status", 0)
        result["query_error"] = hg.get("query_error")
        result["retrieval_sources"] = list(hg.get("retrieval_sources") or [])
        result["retrieval_doc_ids"] = list(hg.get("retrieval_doc_ids") or [])
        result["llm_acc"] = hg.get("llm_acc")
        result["score"] = hg.get("score")
        result["judge_reason"] = hg.get("judge_reason") or ""
        result["dimension_scores"] = list(hg.get("dimension_scores") or [])
        result["judge_status"] = hg.get("judge_status")
        result["judge_error"] = hg.get("judge_error")
        result["metrics"] = dict(hg.get("metrics") or {})
        lo = result.get(SYSTEM_LLM_ONLY) or {}
        result["llm_only_answer"] = lo.get("answer") or ""

    def _extract_answer_text(self, respond: Any) -> str:
        return QueryEvaluator._extract_answer_text(self, respond)

    def _format_agent_process(self, respond: Dict[str, Any]) -> str:
        return QueryEvaluator._format_agent_process(respond)

    def _judge_one(
        self,
        *,
        question: str,
        expected_answer: str,
        score_dimensions: Sequence[str],
        score_dimensions_raw: str,
        rag_answer: str,
    ) -> Dict[str, Any]:
        dims_block = format_score_dimensions(
            score_dimensions, raw=score_dimensions_raw or ""
        )
        user = Benchmark2_PROMPT["JUDGE_USER"]
        for key, val in (
            ("question", question or ""),
            ("expected_answer", expected_answer or ""),
            ("score_dimensions", dims_block),
            ("answer", rag_answer or ""),
        ):
            user = user.replace("{" + key + "}", str(val))

        last_err = None
        for _attempt in range(1, self.max_judge_retries + 1):
            resp = call_llm(
                self.judge_llm,
                system=Benchmark2_PROMPT.get("JUDGE_SYSTEM", ""),
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
                continue
            obj = extract_json_object(resp.get("answer") or "")
            if not obj:
                last_err = "judge json parse failed"
                continue
            score = normalize_score(obj.get("score"))
            judgment = normalize_judgment(
                obj.get("judgment") or obj.get("llm_acc"),
                score=score,
            )
            if judgment not in JUDGMENT_LABELS:
                last_err = f"invalid judgment={obj.get('judgment')!r} score={score!r}"
                continue
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
                "judge_latency_s": resp.get("latency_s"),
                "judge_usage": usage,
            }

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
        if self.query_mode == "agent" or respond.get("plan") or respond.get("steps"):
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
        block["answer"] = self._extract_answer_text(respond)
        block["raw_answer"] = respond.get("answer") or block["answer"]
        if block["query_status"] != 1:
            block["query_error"] = (
                f"llm_only status={block['query_status']}: "
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
            SYSTEM_LLM_ONLY: _empty_system_block(),
        }

        try:
            if not str(question).strip():
                result[SYSTEM_HYPERGRAPH]["query_error"] = "empty question"
                result[SYSTEM_LLM_ONLY]["query_error"] = "empty question"
                self._mirror_hypergraph(result)
                return result

            result[SYSTEM_HYPERGRAPH] = self._fill_hypergraph(question)
            hg = result[SYSTEM_HYPERGRAPH]
            hg_ans = (hg.get("answer") or hg.get("raw_answer") or "").strip()
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
                hg["judge_status"] = 0
                hg["judge_error"] = "skip judge: query failed"

            if self.enable_llm_only:
                result[SYSTEM_LLM_ONLY] = self._fill_llm_only(question)
                lo = result[SYSTEM_LLM_ONLY]
                lo_ans = (lo.get("answer") or lo.get("raw_answer") or "").strip()
                if lo_ans:
                    self._apply_judge(
                        lo,
                        self._judge_one(
                            question=question,
                            expected_answer=expected,
                            score_dimensions=dims,
                            score_dimensions_raw=dims_raw,
                            rag_answer=lo_ans,
                        ),
                    )
                elif lo.get("query_error"):
                    lo["judge_status"] = 0
                    lo["judge_error"] = "skip judge: query failed"

            self._mirror_hypergraph(result)
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

    def _is_item_complete(self, r: Optional[Dict[str, Any]]) -> bool:
        """断点续跑：两路都跑过才算完成。"""
        if not isinstance(r, dict) or r.get("id") is None:
            return False
        hg = r.get(SYSTEM_HYPERGRAPH)
        has_hg = isinstance(hg, dict) and (
            hg.get("query_status") is not None or hg.get("query_error")
        )
        if not has_hg:
            # 旧格式只有顶层 rag 字段
            has_hg = r.get("query_status") is not None or bool(r.get("rag_answer"))
        if not has_hg:
            return False
        if not self.enable_llm_only:
            return True
        lo = r.get(SYSTEM_LLM_ONLY)
        return isinstance(lo, dict) and (
            lo.get("query_status") is not None or lo.get("query_error") is not None
        )

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
        jerr = hg.get("judge_error") or r.get("judge_error")
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
        for r in existing_results or []:
            if self._is_item_complete(r):
                by_id[str(r["id"])] = r
        n_resumed = 0
        todo: List[tuple] = []
        for idx, item in enumerate(items):
            qid = str(item.get("id") or f"idx_{idx}")
            if qid in by_id:
                n_resumed += 1
            else:
                todo.append((idx, item))

        total = len(items)
        n_todo = len(todo)
        workers = min(self.num_thread, max(1, n_todo)) if n_todo else 1
        t0 = time.perf_counter()
        created_at = datetime.now().isoformat(timespec="seconds")
        results: List[Optional[Dict[str, Any]]] = [None] * total
        for idx, item in enumerate(items):
            qid = str(item.get("id") or f"idx_{idx}")
            if qid in by_id:
                results[idx] = by_id[qid]

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
            for idx, item in pbar:
                r = self.evaluate_one(item)
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
                    pool.submit(self.evaluate_one, item): idx for idx, item in todo
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
                        r = {
                            "id": item.get("id") or f"idx_{idx}",
                            "question": item.get("question") or "",
                            "expected_answer": item.get("expected_answer") or "",
                            "categories": dict(item.get("categories") or {}),
                            SYSTEM_HYPERGRAPH: hg,
                            SYSTEM_LLM_ONLY: lo,
                            "rag_answer": "",
                            "query_status": 0,
                            "query_error": f"worker exception: {e}",
                            "llm_acc": None,
                            "score": None,
                            "metrics": {},
                        }
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
