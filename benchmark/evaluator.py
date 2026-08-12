"""DHMF RAG 回答 + LLM 评判 + 召回/耗时/token 统计。"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .prompts import Benchmark_PROMPT
from .utils import (
    call_llm,
    extract_json_object,
    fail_print,
    format_docs_block,
    match_doc_names,
    mean,
    progress_iter,
    quiet_loggers,
    safe_div,
    truncate_text,
)

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None

# llm-acc 仅二分类
JUDGMENT_LABELS = ("正确", "错误")

def normalize_judgment(raw: str) -> Optional[str]:
    """将模型输出归一为 正确 / 错误。"""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s in JUDGMENT_LABELS:
        return s
    low = s.lower().strip()
    # 负面优先，避免「不正确」被判成正确
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
    对测试集 JSON 中的问题：
      1) 按 query_mode 调用 DHMF.query（dual_path）或 DHMF.agent_query（agent）
      2) 用 LLM 二分类评判 llm-acc（正确/错误）
      3) 文档召回率 = 命中 gold 文档数 / gold 文档数
      4) 汇总时延、token 等综合统计
    """

    def __init__(
        self,
        dhmf,
        judge_llm=None,
        *,
        judge_model_args: Optional[dict] = None,
        query_mode: str = "dual_path",
        use_cache: bool = False,
        max_judge_retries: int = 3,
        max_source_chars: int = -1,
        sleep_between: float = 0.0,
        enable_doc_recall: bool = True,
        num_thread: int = 1,
    ):
        self.dhmf = dhmf
        self.judge_llm = judge_llm or getattr(dhmf, "llmmodel", None)
        if self.judge_llm is None:
            raise ValueError("judge_llm 未提供且 DHMF 无 llmmodel")

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

        self.query_mode = self._normalize_query_mode(query_mode)
        self.use_cache = bool(use_cache)
        self.max_judge_retries = max(1, int(max_judge_retries))
        # -1 = 送入完整文档不截断
        self.max_source_chars = int(max_source_chars)
        self.sleep_between = float(sleep_between)
        self.enable_doc_recall = bool(enable_doc_recall)
        try:
            nt = int(num_thread)
        except (TypeError, ValueError):
            nt = 1
        self.num_thread = max(1, nt)

    @staticmethod
    def _normalize_query_mode(mode: Any) -> str:
        """
        规范化查询方式：
          dual_path / query  →  DHMF.query(mode='dual_path')
          agent / agent_query →  DHMF.agent_query()
        """
        s = str(mode or "dual_path").strip().lower().replace("-", "_")
        if s in ("agent", "agent_query", "multi_hop", "multihop"):
            return "agent"
        if s in ("dual_path", "dualpath", "query", "rag"):
            return "dual_path"
        raise ValueError(
            f"Unknown benchmark query_mode={mode!r}. "
            f"Supported: 'dual_path' | 'agent'"
        )

    def _run_query(self, question: str) -> Any:
        """按 query_mode 调用 DHMF.query 或 DHMF.agent_query。"""
        if self.query_mode == "agent":
            return self.dhmf.agent_query(question, pretty=False)
        return self.dhmf.query(question, mode="dual_path", pretty=False)

    def _extract_answer_text(self, respond: Any) -> str:
        if not isinstance(respond, dict):
            return str(respond or "")
        raw = respond.get("answer") or ""
        try:
            from src.DHMF import DHMF
            parsed = DHMF.parse_query_answer(str(raw))
            return (parsed.get("answer") or parsed.get("raw") or "").strip()
        except Exception:
            return str(raw).strip()

    @staticmethod
    def _format_agent_process(respond: Dict[str, Any]) -> str:
        """
        将 agent 多跳规划与各子步骤回答格式化为可读文本。

        用于 rag_raw_answer：与最终回答 rag_answer 分开，记录完整过程。
        无 plan/steps 时回退为模型原始 answer。
        """
        plan = list(respond.get("plan") or [])
        steps = list(respond.get("steps") or [])
        if not plan and not steps:
            return str(respond.get("answer") or "")

        # steps 按 id 索引，便于按 plan 顺序展开
        by_id: Dict[str, Any] = {}
        for st in steps:
            if isinstance(st, dict) and st.get("id") is not None:
                by_id[str(st["id"])] = st

        lines: List[str] = [
            "## Multi-hop Plan & Execution",
            "-" * 48,
        ]
        if not plan:
            lines.append("(no plan steps)")
        else:
            for s in plan:
                if not isinstance(s, dict):
                    continue
                sid = str(s.get("id") or "?")
                deps = s.get("depends_on") or []
                deps_s = ", ".join(str(d) for d in deps) if deps else "none"
                planned = (s.get("question") or "").strip()
                r = by_id.get(sid) or {}
                resolved = (r.get("resolved_question") or "").strip()
                ans = (r.get("answer") or "").strip()
                src = r.get("sources") or []
                status = r.get("status")

                lines.append(f"### Step {sid}  ·  deps: {deps_s}")
                lines.append(f"planned:  {planned}")
                if resolved and resolved != planned:
                    lines.append(f"resolved: {resolved}")
                if status is not None:
                    lines.append(f"status:   {status}")
                if src:
                    lines.append(f"sources:  {' / '.join(str(x) for x in src)}")
                lines.append("answer:")
                if ans:
                    for al in ans.splitlines() or [ans]:
                        lines.append(f"  {al}")
                else:
                    lines.append("  (empty)")
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
        """
        文档召回率 = 在检索结果中命中的 gold 文档数 / gold 文档总数。

        例：gold 2 篇，检索 5 篇里只命中 1 篇 → recall=0.5。
        不计算 precision 等其它指标。
        """
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

    def _judge_one(
        self,
        *,
        question: str,
        ground_truth_answer: str,
        source_docs: Sequence[dict],
        rag_answer: str,
    ) -> Dict[str, Any]:
        """
        将 问题 + 标准答案 + 全部相关完整文档 + RAG 回答 发给评测模型。
        二分类：正确 / 错误（答到要点即可，不必字面一致）。
        """
        # 完整文档；max_source_chars=-1 时不截断
        full_docs = []
        for d in source_docs or []:
            full_docs.append({
                "doc_id": d.get("doc_id"),
                "name": d.get("name"),
                "content": truncate_text(
                    d.get("content") or "", self.max_source_chars
                ),
            })
        source_block = format_docs_block(
            full_docs, max_chars_per_doc=self.max_source_chars
        )
        user = Benchmark_PROMPT['JUDGE_USER'].format(
            question=question or "",
            ground_truth_answer=ground_truth_answer or "",
            source_block=source_block,
            rag_answer=rag_answer or "",
        )

        last_err = None
        for _attempt in range(1, self.max_judge_retries + 1):
            resp = call_llm(
                self.judge_llm,
                system=Benchmark_PROMPT.get('JUDGE_SYSTEM', ''),
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
            judgment = normalize_judgment(
                obj.get("judgment") or obj.get("llm_acc") or ""
            )
            if judgment not in JUDGMENT_LABELS:
                last_err = f"invalid judgment={obj.get('judgment')!r}"
                continue
            return {
                "llm_acc": judgment,
                "judge_reason": (obj.get("reason") or "").strip(),
                "judge_status": 1,
                "judge_error": None,
                "judge_latency_s": resp.get("latency_s"),
                "judge_usage": usage,
            }

        return {
            "llm_acc": None,
            "judge_reason": "",
            "judge_status": 0,
            "judge_error": last_err or "judge failed",
            "judge_latency_s": None,
            "judge_usage": {},
        }

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
            "rag_answer": "",
            # agent 模式：多跳规划 + 各子问题回答；dual_path：模型原始 answer
            "rag_raw_answer": "",
            "query_status": 0,
            "query_error": None,
            "retrieval_sources": [],
            "retrieval_doc_ids": [],
            "recall": None,  # enable_doc_recall=False 时保持 None
            "llm_acc": None,
            "judge_reason": "",
            "metrics": {},
        }

        try:
            if not question.strip():
                result["query_error"] = "empty question"
                return result

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
                result["query_error"] = str(e)
                result["metrics"]["total_latency_s"] = time.perf_counter() - t0
                return result

            if not isinstance(respond, dict):
                result["query_error"] = f"unexpected respond type: {type(respond)}"
                result["rag_answer"] = str(respond)
                return result

            result["query_status"] = respond.get("status", 0)
            result["rag_answer"] = self._extract_answer_text(respond)
            # agent：完整多步过程；dual_path：原始 answer（与 rag_answer 可能相同）
            if self.query_mode == "agent" or respond.get("plan") or respond.get("steps"):
                result["rag_raw_answer"] = self._format_agent_process(respond)
            else:
                result["rag_raw_answer"] = respond.get("answer") or ""
            result["retrieval_sources"] = list(respond.get("retrieval_sources") or [])
            result["retrieval_doc_ids"] = list(respond.get("retrieval_doc_ids") or [])

            if result["query_status"] != 1:
                result["query_error"] = (
                    result["query_error"]
                    or f"query status={result['query_status']}: "
                    f"{str(result['rag_raw_answer'])[:200]}"
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
            # 扁平字段便于汇总；与 retrieve_timing 同源
            result["metrics"] = {
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

            # 文档召回：命中 gold 数 / gold 总数（可关闭）
            if self.enable_doc_recall:
                result["recall"] = self._compute_recall(
                    expected_names, result["retrieval_sources"]
                )
            else:
                result["recall"] = None

            # LLM 评判：问题 + 标准答案 + 全部完整文档 + RAG 回答
            judge = self._judge_one(
                question=question,
                ground_truth_answer=result["ground_truth_answer"],
                source_docs=item.get("source_docs") or [],
                rag_answer=result["rag_answer"] or result["rag_raw_answer"],
            )
            result["llm_acc"] = judge.get("llm_acc")
            result["judge_reason"] = judge.get("judge_reason")
            result["judge_status"] = judge.get("judge_status")
            result["judge_error"] = judge.get("judge_error")
            result["metrics"]["judge_latency_s"] = judge.get("judge_latency_s")
            ju = judge.get("judge_usage") or {}
            result["metrics"]["judge_prompt_tokens"] = ju.get("prompt_tokens")
            result["metrics"]["judge_completion_tokens"] = ju.get("completion_tokens")
            result["metrics"]["judge_total_tokens"] = ju.get("total_tokens")
            return result
        finally:
            if self.sleep_between > 0:
                time.sleep(self.sleep_between)

    @staticmethod
    def _is_eval_failure(r: Dict[str, Any]) -> bool:
        """流水线失败（非「答案判错」）。"""
        if r.get("query_error"):
            return True
        if r.get("query_status") == 0:
            return True
        if r.get("judge_status") == 0 or r.get("judge_error"):
            return True
        if not r.get("question"):
            return True
        return False

    @staticmethod
    def _slim_dataset_meta(meta: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """问题集 meta 精简：只保留出题侧关键字段。"""
        if not meta or not isinstance(meta, dict):
            return None
        keep = (
            "created_at",
            "db_path",
            "hop_counts",
            "seed",
            "num_thread",
            "model",
            "total",
            "success",
            "failed",
            "elapsed_s",
        )
        out = {k: meta[k] for k in keep if k in meta}
        return out or None

    def _make_report(
        self,
        *,
        dataset: Dict[str, Any],
        results: List[Dict[str, Any]],
        skipped: int,
        t0: float,
        created_at: str,
        done: bool = False,
    ) -> Dict[str, Any]:
        """组装评测报告（可中途调用，summary 随 results 增量更新）。"""
        summary = self.build_summary(
            results, enable_doc_recall=self.enable_doc_recall
        )
        return {
            "meta": {
                "created_at": created_at,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "done": bool(done),
                "dataset_meta": self._slim_dataset_meta(dataset.get("meta")),
                "query_mode": self.query_mode,
                "judge_model": (self.judge_model_args or {}).get("model"),
                "enable_doc_recall": self.enable_doc_recall,
                "n_questions": len(results),
                "n_total_planned": None,  # evaluate_all 写入
                "n_skipped_gen_fail": skipped,
                "elapsed_s": round(time.perf_counter() - t0, 3),
            },
            "results": results,
            "summary": summary,
        }

    def _log_eval_failure(self, r: Dict[str, Any]) -> None:
        parts = [f"评测 {r.get('id')}"]
        if r.get("query_error"):
            parts.append(f"query={r['query_error'][:160]}")
        if r.get("judge_error"):
            parts.append(f"judge={r['judge_error'][:160]}")
        if r.get("query_status") == 0 and not r.get("query_error"):
            parts.append("query_status=0")
        fail_print(" | ".join(parts))

    def evaluate_all(
        self,
        dataset: Dict[str, Any],
        *,
        on_progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        批量评测（num_thread>1 时并行 query+judge）。

        on_progress:
          可选回调 ``on_progress(report, index, total)``。
          每评完一题调用一次（含增量 summary），便于实时落盘。
          多线程时按完成顺序触发；results 始终按原题序。
        """
        questions = list(dataset.get("questions") or [])
        items = [
            q for q in questions if q.get("gen_status", 1) == 1 and q.get("question")
        ]
        skipped = len(questions) - len(items)
        if skipped:
            fail_print(f"评测跳过 {skipped} 条生成失败/空问题的样本")

        total = len(items)
        workers = min(self.num_thread, max(1, total))
        t0 = time.perf_counter()
        created_at = datetime.now().isoformat(timespec="seconds")
        results: List[Optional[Dict[str, Any]]] = [None] * total
        fail_n = 0
        done_n = 0

        def _on_item(r: Dict[str, Any]) -> None:
            nonlocal fail_n, done_n
            done_n += 1
            if self._is_eval_failure(r):
                fail_n += 1
                self._log_eval_failure(r)

        def _emit_progress() -> None:
            if on_progress is None:
                return
            done_results = [r for r in results if r is not None]
            mid = self._make_report(
                dataset=dataset,
                results=done_results,
                skipped=skipped,
                t0=t0,
                created_at=created_at,
                done=False,
            )
            mid["meta"]["n_total_planned"] = total
            mid["meta"]["num_thread"] = workers
            try:
                on_progress(mid, len(done_results), total)
            except Exception as e:
                fail_print(f"on_progress 保存失败: {e}")

        if workers <= 1:
            pbar = progress_iter(items, total=total, desc="评测问答", unit="题")
            for idx, item in enumerate(pbar):
                r = self.evaluate_one(item)
                results[idx] = r
                _on_item(r)
                if hasattr(pbar, "set_postfix"):
                    ok_n = sum(
                        1 for x in results if x is not None and x.get("llm_acc") == "正确"
                    )
                    pbar.set_postfix(fail=fail_n, ok=ok_n, thr=1, refresh=False)
                _emit_progress()
        else:
            if _tqdm is not None:
                pbar = _tqdm(
                    total=total,
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
                    pool.submit(self.evaluate_one, item): idx
                    for idx, item in enumerate(items)
                }
                for fut in as_completed(fut_to_idx):
                    idx = fut_to_idx[fut]
                    try:
                        r = fut.result()
                    except Exception as e:
                        item = items[idx]
                        r = {
                            "id": item.get("id") or f"idx_{idx}",
                            "hop": item.get("hop"),
                            "question": item.get("question") or "",
                            "ground_truth_answer": item.get("ground_truth_answer") or "",
                            "explanation": item.get("explanation") or "",
                            "source_names": list(item.get("source_names") or []),
                            "rag_answer": "",
                            "rag_raw_answer": "",
                            "query_status": 0,
                            "query_error": f"worker exception: {e}",
                            "retrieval_sources": [],
                            "retrieval_doc_ids": [],
                            "recall": None,
                            "llm_acc": None,
                            "judge_reason": "",
                            "metrics": {},
                        }
                    results[idx] = r
                    _on_item(r)
                    if pbar is not None:
                        ok_n = sum(
                            1
                            for x in results
                            if x is not None and x.get("llm_acc") == "正确"
                        )
                        pbar.update(1)
                        pbar.set_postfix(
                            fail=fail_n, ok=ok_n, thr=workers, refresh=False
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

        out_results: List[Dict[str, Any]] = [r for r in results if r is not None]
        report = self._make_report(
            dataset=dataset,
            results=out_results,
            skipped=skipped,
            t0=t0,
            created_at=created_at,
            done=True,
        )
        report["meta"]["n_total_planned"] = total
        report["meta"]["num_thread"] = workers
        return report

    @staticmethod
    def _sum_tokens(results: Sequence[dict], key: str) -> Optional[int]:
        total = 0
        any_v = False
        for r in results:
            m = r.get("metrics") or {}
            v = m.get(key)
            if v is not None:
                any_v = True
                try:
                    total += int(v)
                except Exception:
                    pass
        return total if any_v else None

    @classmethod
    def build_report_document(
        cls,
        eval_data: Dict[str, Any],
        *,
        source_path: Optional[str] = None,
        enable_doc_recall: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        从 evaluate 写出的 JSON（含 results）汇总为独立 report 文档。

        不依赖 DHMF / LLM，可单独跑 report 步骤。
        """
        results = list(eval_data.get("results") or [])
        src_meta = dict(eval_data.get("meta") or {})

        if enable_doc_recall is None:
            if "enable_doc_recall" in src_meta:
                enable_doc_recall = bool(src_meta.get("enable_doc_recall"))
            elif isinstance(eval_data.get("summary"), dict) and (
                "enable_doc_recall" in (eval_data.get("summary") or {})
            ):
                enable_doc_recall = bool(
                    (eval_data.get("summary") or {}).get("enable_doc_recall")
                )
            else:
                enable_doc_recall = True
        enable_doc_recall = bool(enable_doc_recall)

        # 借用空实例调用实例方法（build_summary 仅依赖 enable_doc_recall / 静态工具）
        helper = cls.__new__(cls)
        helper.enable_doc_recall = enable_doc_recall
        summary = helper.build_summary(results, enable_doc_recall=enable_doc_recall)

        # 从评测 meta / 问题集 meta 抽扁平字段；逐题明细不写入 report（见 eval results）
        ds_meta = src_meta.get("dataset_meta")
        if not isinstance(ds_meta, dict):
            ds_meta = {}
        slim_ds = cls._slim_dataset_meta(ds_meta) or {}
        cfg_src = src_meta.get("config")
        if not isinstance(cfg_src, dict):
            cfg_src = {}

        return {
            "meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source_path": source_path,
                "config_file": cfg_src.get("config_file"),
                "dhmf_config_path": cfg_src.get("dhmf_config_path"),
                "query_mode": src_meta.get("query_mode") or cfg_src.get("query_mode"),
                "judge_model": src_meta.get("judge_model") or cfg_src.get("judge_model"),
                "enable_doc_recall": enable_doc_recall,
                "db_path": slim_ds.get("db_path") or cfg_src.get("db_path"),
                "hop_counts": slim_ds.get("hop_counts") or cfg_src.get("hop_counts"),
                "seed": slim_ds.get("seed", cfg_src.get("seed")),
                "n_results": len(results),
                "n_questions": src_meta.get("n_questions", len(results)),
                "n_total_planned": src_meta.get("n_total_planned"),
                "n_skipped_gen_fail": src_meta.get("n_skipped_gen_fail"),
                "eval_elapsed_s": src_meta.get("elapsed_s"),
                "eval_done": src_meta.get("done"),
            },
            "summary": summary,
        }

    def build_summary(
        self,
        results: Sequence[Dict[str, Any]],
        *,
        enable_doc_recall: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if enable_doc_recall is None:
            enable_doc_recall = self.enable_doc_recall
        enable_doc_recall = bool(enable_doc_recall)

        n = len(results)
        acc_counts = {lab: 0 for lab in JUDGMENT_LABELS}
        acc_counts["未知"] = 0
        for r in results:
            lab = r.get("llm_acc")
            if lab in JUDGMENT_LABELS:
                acc_counts[lab] += 1
            else:
                acc_counts["未知"] += 1

        n_correct = acc_counts["正确"]
        n_wrong = acc_counts["错误"]
        judged = n_correct + n_wrong

        recalls = []
        if enable_doc_recall:
            for r in results:
                rec = r.get("recall") or {}
                if isinstance(rec, dict) and rec.get("recall") is not None:
                    recalls.append(float(rec["recall"]))

        def _metric_lats(key: str, rows=None) -> list:
            rows = results if rows is None else rows
            out = []
            for r in rows:
                v = (r.get("metrics") or {}).get(key)
                if v is not None:
                    try:
                        out.append(float(v))
                    except (TypeError, ValueError):
                        pass
            return out

        # 检索分阶段字段：metrics 扁平键 → latency 汇总键
        retrieve_stage_keys = (
            ("precompute_latency_s", "mean_precompute_s", "sum_precompute_s"),
            ("rewrite_latency_s", "mean_rewrite_s", "sum_rewrite_s"),
            ("embed_latency_s", "mean_embed_s", "sum_embed_s"),
            ("chunk_latency_s", "mean_chunk_s", "sum_chunk_s"),
            ("node_latency_s", "mean_node_s", "sum_node_s"),
            ("keyword_latency_s", "mean_keyword_s", "sum_keyword_s"),
            ("expand_latency_s", "mean_expand_s", "sum_expand_s"),
            ("rerank_latency_s", "mean_rerank_s", "sum_rerank_s"),
        )

        q_lats = _metric_lats("query_latency_s")
        r_lats = _metric_lats("retrieve_latency_s")
        w_lats = _metric_lats("wall_latency_s")
        stage_lats = {mk: _metric_lats(src) for src, mk, _sk in retrieve_stage_keys}

        by_hop: Dict[str, Any] = {}
        hop_groups: Dict[Any, list] = defaultdict(list)
        for r in results:
            hop_groups[r.get("hop")].append(r)
        for hop, group in sorted(
            hop_groups.items(), key=lambda x: (x[0] is None, x[0] or 0)
        ):
            g_acc = {lab: 0 for lab in JUDGMENT_LABELS}
            for r in group:
                if r.get("llm_acc") in g_acc:
                    g_acc[r["llm_acc"]] += 1
            g_n = len(group)
            g_judged = sum(g_acc.values())
            hop_item: Dict[str, Any] = {
                "n": g_n,
                "llm_acc_counts": g_acc,
                "accuracy": safe_div(g_acc["正确"], g_judged) if g_judged else None,
                "mean_query_latency_s": mean(_metric_lats("query_latency_s", group)),
                "mean_retrieve_latency_s": mean(_metric_lats("retrieve_latency_s", group)),
            }
            for src, mk, _sk in retrieve_stage_keys:
                hop_item[mk] = mean(_metric_lats(src, group))
            if enable_doc_recall:
                g_recalls = [
                    float((r.get("recall") or {}).get("recall"))
                    for r in group
                    if isinstance(r.get("recall"), dict)
                    and (r.get("recall") or {}).get("recall") is not None
                ]
                hop_item["mean_doc_recall"] = mean(g_recalls)
            by_hop[str(hop)] = hop_item

        n_query_fail = sum(1 for r in results if self._is_eval_failure(r) and (
            r.get("query_error") or r.get("query_status") == 0
        ))
        n_judge_fail = sum(
            1 for r in results
            if r.get("judge_status") == 0 or r.get("judge_error")
        )

        latency_summary: Dict[str, Any] = {
            "mean_query_s": mean(q_lats),
            "mean_retrieve_s": mean(r_lats),
            "mean_wall_s": mean(w_lats),
            "sum_query_s": sum(q_lats) if q_lats else None,
            "sum_retrieve_s": sum(r_lats) if r_lats else None,
            "sum_wall_s": sum(w_lats) if w_lats else None,
        }
        for src, mk, sk in retrieve_stage_keys:
            vals = stage_lats[mk]
            latency_summary[mk] = mean(vals)
            latency_summary[sk] = sum(vals) if vals else None

        summary: Dict[str, Any] = {
            "n_total": n,
            "enable_doc_recall": enable_doc_recall,
            "pipeline": {
                "n_query_fail": n_query_fail,
                "n_judge_fail": n_judge_fail,
                "n_ok": n - sum(1 for r in results if self._is_eval_failure(r)),
            },
            "llm_acc": {
                "counts": acc_counts,
                "accuracy": safe_div(n_correct, judged) if judged else None,
                "error_rate": safe_div(n_wrong, judged) if judged else None,
                "n_judged": judged,
                "n_correct": n_correct,
                "n_wrong": n_wrong,
            },
            "latency": latency_summary,
            "tokens": {
                "sum_prompt": self._sum_tokens(results, "prompt_tokens"),
                "sum_completion": self._sum_tokens(results, "completion_tokens"),
                "sum_total": self._sum_tokens(results, "total_tokens"),
                "mean_prompt": mean(
                    [
                        (r.get("metrics") or {}).get("prompt_tokens")
                        for r in results
                        if (r.get("metrics") or {}).get("prompt_tokens") is not None
                    ]
                ),
                "mean_completion": mean(
                    [
                        (r.get("metrics") or {}).get("completion_tokens")
                        for r in results
                        if (r.get("metrics") or {}).get("completion_tokens") is not None
                    ]
                ),
                "sum_judge_prompt": self._sum_tokens(results, "judge_prompt_tokens"),
                "sum_judge_completion": self._sum_tokens(
                    results, "judge_completion_tokens"
                ),
                "sum_judge_total": self._sum_tokens(results, "judge_total_tokens"),
            },
            "by_hop": by_hop,
        }
        if enable_doc_recall:
            summary["doc_recall"] = {
                # 各题 (命中gold数/gold总数) 的平均
                "mean_recall": mean(recalls),
                # 召回 × 答案准确率 列联表
                "vs_accuracy": self._build_recall_vs_accuracy(results),
            }
        return summary

    @staticmethod
    def _build_recall_vs_accuracy(
        results: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        文档召回 × LLM 答案准确率 对照表。

        行（召回，基于 per-question recall.recall）：
          - hit:     recall == 1.0（全部 gold 出现在 retrieval_sources）
          - partial: 0 < recall < 1
          - miss:    recall == 0.0
          - unknown: 无有效 recall 字段
        列（答案）：
          - 正确 / 错误 / 未知（llm_acc）

        另附 hit×错误 下 query 成功/失败拆分，便于区分「检索对但答错」与「流水线失败」。
        """
        # cells[recall_bucket][acc_label] = count
        buckets = ("hit", "partial", "miss", "unknown")
        acc_labels = ("正确", "错误", "未知")
        cells: Dict[str, Dict[str, int]] = {
            b: {lab: 0 for lab in acc_labels} for b in buckets
        }
        n_with_recall = 0
        hit_wrong_query_ok = 0
        hit_wrong_query_fail = 0

        for r in results:
            rec_obj = r.get("recall") if isinstance(r.get("recall"), dict) else None
            rec_val = rec_obj.get("recall") if rec_obj else None
            if rec_val is None:
                bucket = "unknown"
            else:
                n_with_recall += 1
                v = float(rec_val)
                if v >= 1.0:
                    bucket = "hit"
                elif v <= 0.0:
                    bucket = "miss"
                else:
                    bucket = "partial"

            lab = r.get("llm_acc")
            if lab not in ("正确", "错误"):
                lab = "未知"
            cells[bucket][lab] += 1

            if bucket == "hit" and lab == "错误":
                q_fail = bool(r.get("query_error")) or r.get("query_status") == 0
                if q_fail:
                    hit_wrong_query_fail += 1
                else:
                    hit_wrong_query_ok += 1

        row_totals = {b: sum(cells[b].values()) for b in buckets}
        col_totals = {
            lab: sum(cells[b][lab] for b in buckets) for lab in acc_labels
        }

        # 便于阅读的二维 counts（只保留实际可能用到的格子；partial/unknown 仍写入）
        table = {
            "hit_correct": cells["hit"]["正确"],
            "hit_wrong": cells["hit"]["错误"],
            "partial_correct": cells["partial"]["正确"],
            "partial_wrong": cells["partial"]["错误"],
            "miss_correct": cells["miss"]["正确"],
            "miss_wrong": cells["miss"]["错误"],
            "unknown_correct": cells["unknown"]["正确"],
            "unknown_wrong": cells["unknown"]["错误"],
            "hit_unknown": cells["hit"]["未知"],
            "partial_unknown": cells["partial"]["未知"],
            "miss_unknown": cells["miss"]["未知"],
            "unknown_unknown": cells["unknown"]["未知"],
        }

        matrix = [
            {"recall": b, "accuracy": lab, "n": cells[b][lab]}
            for b in buckets
            for lab in acc_labels
            if cells[b][lab] > 0 or b in ("hit", "miss") and lab in ("正确", "错误")
        ]

        return {
            "definition": {
                "hit": "recall == 1.0：全部 gold 文档出现在 retrieval_sources",
                "partial": "0 < recall < 1：部分 gold 命中（多跳时出现）",
                "miss": "recall == 0.0：gold 均未出现在 retrieval_sources",
                "unknown": "该题无有效 recall 字段",
                "correct": "llm_acc == 正确",
                "wrong": "llm_acc == 错误",
            },
            "table": table,
            "matrix": matrix,
            "row_totals": row_totals,
            "col_totals": col_totals,
            "n_with_recall": n_with_recall,
            "n_total": len(results),
            "hit_wrong_breakdown": {
                "query_ok": hit_wrong_query_ok,
                "query_fail": hit_wrong_query_fail,
                "note": "命中 gold 且答案错误时，按 query 是否失败再拆分",
            },
        }
