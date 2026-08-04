"""DHMF RAG 回答 + LLM 评判 + 召回/耗时/token 统计。"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .prompts import JUDGE_SYSTEM, JUDGE_USER
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
      1) 调用 DHMF.query 作答
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

        self.query_mode = query_mode
        self.use_cache = bool(use_cache)
        self.max_judge_retries = max(1, int(max_judge_retries))
        # -1 = 送入完整文档不截断
        self.max_source_chars = int(max_source_chars)
        self.sleep_between = float(sleep_between)

    # ------------------------------------------------------------------
    # single item
    # ------------------------------------------------------------------
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
        user = JUDGE_USER.format(
            question=question or "",
            ground_truth_answer=ground_truth_answer or "",
            source_block=source_block,
            rag_answer=rag_answer or "",
        )

        last_err = None
        for _attempt in range(1, self.max_judge_retries + 1):
            resp = call_llm(
                self.judge_llm,
                system=JUDGE_SYSTEM,
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
            "rag_raw_answer": "",
            "query_status": 0,
            "query_error": None,
            "retrieval_sources": [],
            "retrieval_doc_ids": [],
            "recall": {},
            "llm_acc": None,
            "judge_reason": "",
            "metrics": {},
        }

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
                respond = self.dhmf.query(
                    question, mode=self.query_mode, pretty=False
                )
        except Exception as e:
            result["query_error"] = str(e)
            result["metrics"]["total_latency_s"] = time.perf_counter() - t0
            return result

        if not isinstance(respond, dict):
            result["query_error"] = f"unexpected respond type: {type(respond)}"
            result["rag_answer"] = str(respond)
            return result

        result["query_status"] = respond.get("status", 0)
        result["rag_raw_answer"] = respond.get("answer") or ""
        result["rag_answer"] = self._extract_answer_text(respond)
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

        result["metrics"] = {
            "query_latency_s": respond.get("latency_s"),
            "retrieve_latency_s": respond.get("retrieve_latency_s"),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,
            "wall_latency_s": time.perf_counter() - t0,
        }

        # 文档召回：命中 gold 数 / gold 总数
        result["recall"] = self._compute_recall(
            expected_names, result["retrieval_sources"]
        )

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

    # ------------------------------------------------------------------
    # batch + summary
    # ------------------------------------------------------------------
    def evaluate_all(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        questions = list(dataset.get("questions") or [])
        items = [
            q for q in questions if q.get("gen_status", 1) == 1 and q.get("question")
        ]
        skipped = len(questions) - len(items)
        if skipped:
            fail_print(f"评测跳过 {skipped} 条生成失败/空问题的样本")

        results: List[Dict[str, Any]] = []
        fail_n = 0
        t0 = time.perf_counter()
        pbar = progress_iter(items, total=len(items), desc="评测问答", unit="题")
        for item in pbar:
            r = self.evaluate_one(item)
            results.append(r)
            if self._is_eval_failure(r):
                fail_n += 1
                parts = [f"评测 {r.get('id')}"]
                if r.get("query_error"):
                    parts.append(f"query={r['query_error'][:160]}")
                if r.get("judge_error"):
                    parts.append(f"judge={r['judge_error'][:160]}")
                if r.get("query_status") == 0 and not r.get("query_error"):
                    parts.append("query_status=0")
                fail_print(" | ".join(parts))
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(fail=fail_n, refresh=False)
            if self.sleep_between > 0:
                time.sleep(self.sleep_between)

        summary = self.build_summary(results)
        report = {
            "meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "dataset_meta": dataset.get("meta"),
                "query_mode": self.query_mode,
                "judge_model": (self.judge_model_args or {}).get("model"),
                "n_questions": len(results),
                "n_skipped_gen_fail": skipped,
                "elapsed_s": round(time.perf_counter() - t0, 3),
            },
            "results": results,
            "summary": summary,
            "summary_table": self.format_summary_table(summary),
        }
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

    def build_summary(self, results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
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
        for r in results:
            rec = r.get("recall") or {}
            if rec.get("recall") is not None:
                recalls.append(float(rec["recall"]))

        q_lats = [
            (r.get("metrics") or {}).get("query_latency_s")
            for r in results
            if (r.get("metrics") or {}).get("query_latency_s") is not None
        ]
        r_lats = [
            (r.get("metrics") or {}).get("retrieve_latency_s")
            for r in results
            if (r.get("metrics") or {}).get("retrieve_latency_s") is not None
        ]

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
            g_recalls = [
                float((r.get("recall") or {}).get("recall"))
                for r in group
                if (r.get("recall") or {}).get("recall") is not None
            ]
            by_hop[str(hop)] = {
                "n": g_n,
                "llm_acc_counts": g_acc,
                "accuracy": safe_div(g_acc["正确"], g_judged) if g_judged else None,
                "mean_doc_recall": mean(g_recalls),
                "mean_query_latency_s": mean(
                    [
                        (r.get("metrics") or {}).get("query_latency_s")
                        for r in group
                        if (r.get("metrics") or {}).get("query_latency_s") is not None
                    ]
                ),
            }

        summary = {
            "n_total": n,
            "llm_acc": {
                "counts": acc_counts,
                "accuracy": safe_div(n_correct, judged) if judged else None,
                "error_rate": safe_div(n_wrong, judged) if judged else None,
                "n_judged": judged,
            },
            "doc_recall": {
                # 唯一指标：各题 (命中gold数/gold总数) 的平均
                "mean_recall": mean(recalls),
            },
            "latency": {
                "mean_query_s": mean(q_lats),
                "mean_retrieve_s": mean(r_lats),
                "sum_query_s": sum(q_lats) if q_lats else None,
            },
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
        return summary

    @staticmethod
    def format_summary_table(summary: Dict[str, Any]) -> str:
        """生成可读综合统计表（纯文本）。"""
        lines = []
        lines.append("=" * 72)
        lines.append("综合评测统计表")
        lines.append("=" * 72)

        n = summary.get("n_total") or 0
        acc = summary.get("llm_acc") or {}
        counts = acc.get("counts") or {}
        lines.append(f"样本数: {n}")
        lines.append("")
        lines.append("【LLM 准确率 llm-acc】（二分类：正确 / 错误）")
        lines.append(f"  正确: {counts.get('正确', 0)}")
        lines.append(f"  错误: {counts.get('错误', 0)}")
        if counts.get("未知"):
            lines.append(f"  未知/失败: {counts.get('未知', 0)}")
        lines.append(f"  准确率 accuracy: {_pct(acc.get('accuracy'))}")
        lines.append(f"  错误率: {_pct(acc.get('error_rate'))}")

        rec = summary.get("doc_recall") or {}
        lines.append("")
        lines.append(
            "【文档召回率】命中 gold 文档数 / gold 文档总数（按题平均）"
        )
        lines.append(f"  平均召回率 mean_recall: {_pct(rec.get('mean_recall'))}")

        lat = summary.get("latency") or {}
        lines.append("")
        lines.append("【时延】")
        lines.append(f"  平均 query 时延: {_sec(lat.get('mean_query_s'))}")
        lines.append(f"  平均 retrieve 时延: {_sec(lat.get('mean_retrieve_s'))}")
        lines.append(f"  query 总时延: {_sec(lat.get('sum_query_s'))}")

        tok = summary.get("tokens") or {}
        lines.append("")
        lines.append("【Token 消耗（RAG 回答）】")
        lines.append(f"  prompt 合计: {tok.get('sum_prompt')}")
        lines.append(f"  completion 合计: {tok.get('sum_completion')}")
        lines.append(f"  total 合计: {tok.get('sum_total')}")
        lines.append(f"  平均 prompt: {_num(tok.get('mean_prompt'))}")
        lines.append(f"  平均 completion: {_num(tok.get('mean_completion'))}")
        lines.append("【Token 消耗（评判 LLM）】")
        lines.append(f"  judge prompt 合计: {tok.get('sum_judge_prompt')}")
        lines.append(f"  judge completion 合计: {tok.get('sum_judge_completion')}")
        lines.append(f"  judge total 合计: {tok.get('sum_judge_total')}")

        by_hop = summary.get("by_hop") or {}
        if by_hop:
            lines.append("")
            lines.append("【分跳数统计】")
            lines.append(
                f"{'hop':>4} | {'n':>4} | {'准确率':>8} | "
                f"{'文档召回':>8} | {'均时延s':>8}"
            )
            lines.append("-" * 72)
            for hop, g in by_hop.items():
                lines.append(
                    f"{str(hop):>4} | {g.get('n', 0):>4} | "
                    f"{_pct(g.get('accuracy')):>8} | "
                    f"{_pct(g.get('mean_doc_recall')):>8} | "
                    f"{_num(g.get('mean_query_latency_s')):>8}"
                )

        lines.append("=" * 72)
        return "\n".join(lines)


def _pct(x) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return str(x)


def _sec(x) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.3f}s"
    except Exception:
        return str(x)


def _num(x) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)
