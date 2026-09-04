"""评测结果汇总：延迟分解、agentic turns/search、分类统计、两路对比。"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .utils import (
    JUDGMENT_LABELS,
    RETRIEVE_STAGE_KEYS,
    SYSTEM_AGENTIC,
    SYSTEM_LLM_ONLY,
    mean,
    safe_div,
)

STAGE_LABELS = {
    "precompute_s": "precompute",
    "rewrite_s": "rewrite",
    "embed_s": "embed",
    "chunk_s": "chunk",
    "node_s": "node",
    "keyword_s": "keyword",
    "expand_s": "expand",
    "rerank_s": "rerank",
    "total_s": "retrieve_total",
}


def _metric_vals(rows: Sequence[dict], key: str) -> List[float]:
    out: List[float] = []
    for r in rows:
        m = r.get("metrics") or {}
        v = m.get(key)
        if v is None:
            v = r.get(key)
        try:
            if v is not None:
                out.append(float(v))
        except (TypeError, ValueError):
            pass
    return out


def _sum_tokens(rows: Sequence[dict], key: str) -> Optional[int]:
    vals = _metric_vals(rows, key)
    if not vals:
        return None
    return int(sum(vals))


def _system_attempted(block: Optional[dict]) -> bool:
    if not isinstance(block, dict) or not block:
        return False
    if (block.get("answer") or block.get("raw_answer") or "").strip():
        return True
    if block.get("query_error") or block.get("query_status") == 1:
        return True
    metrics = block.get("metrics") or {}
    return any(metrics.get(k) is not None for k in ("query_latency_s", "wall_latency_s"))


def _system_failed(block: Optional[dict]) -> bool:
    if not isinstance(block, dict) or not block:
        return True
    if block.get("query_error") or block.get("query_status") == 0:
        return True
    if block.get("judge_status") == 0 or block.get("judge_error"):
        return True
    return False


def project_system_row(result: Dict[str, Any], system: str) -> Dict[str, Any]:
    block = result.get(system) if isinstance(result.get(system), dict) else {}
    row = {
        "id": result.get("id"),
        "dataset_id": result.get("dataset_id"),
        "hop": result.get("hop"),
        "question": result.get("question"),
        "categories": result.get("categories") or {},
        "llm_acc": block.get("llm_acc"),
        "score": block.get("score"),
        "judge_reason": block.get("judge_reason"),
        "dimension_scores": block.get("dimension_scores") or [],
        "query_status": block.get("query_status"),
        "query_error": block.get("query_error"),
        "judge_status": block.get("judge_status"),
        "judge_error": block.get("judge_error"),
        "answer": block.get("answer"),
        "metrics": dict(block.get("metrics") or {}),
        "recall": result.get("recall"),
    }
    return row


def _latency_block(rows: Sequence[dict], *, is_agentic: bool) -> Dict[str, Any]:
    q_lats = _metric_vals(rows, "query_latency_s")
    r_lats = _metric_vals(rows, "retrieve_latency_s")
    w_lats = _metric_vals(rows, "wall_latency_s")
    search_lats = _metric_vals(rows, "mean_search_s")
    n_search = _metric_vals(rows, "n_search")
    n_turns = _metric_vals(rows, "n_turns")
    n_read = _metric_vals(rows, "n_read_doc")
    n_neighbors = _metric_vals(rows, "n_graph_neighbors")

    out: Dict[str, Any] = {
        "mean_query_s": mean(q_lats),
        "mean_retrieve_s": mean(r_lats),
        "mean_wall_s": mean(w_lats),
        "sum_query_s": sum(q_lats) if q_lats else None,
        "sum_retrieve_s": sum(r_lats) if r_lats else None,
        "sum_wall_s": sum(w_lats) if w_lats else None,
        "n_with_query_latency": len(q_lats),
        "mean_embed_s": mean(_metric_vals(rows, "embed_latency_s") or _metric_vals(rows, "embed_s")),
        "mean_rerank_s": mean(_metric_vals(rows, "rerank_latency_s") or _metric_vals(rows, "rerank_s")),
    }
    if not is_agentic:
        return out

    out.update({
        "mean_search_s": mean(search_lats),
        "mean_n_search": mean(n_search),
        "mean_n_turns": mean(n_turns),
        "mean_n_read_doc": mean(n_read),
        "mean_n_graph_neighbors": mean(n_neighbors),
        "sum_n_search": int(sum(n_search)) if n_search else None,
        "sum_n_turns": int(sum(n_turns)) if n_turns else None,
        "mean_turn_s": safe_div(mean(w_lats) or 0.0, mean(n_turns) or 0.0)
        if n_turns and mean(n_turns)
        else None,
    })

    stages: Dict[str, Any] = {}
    for k in RETRIEVE_STAGE_KEYS:
        totals = _metric_vals(rows, k)
        per_search = _metric_vals(rows, f"mean_{k}")
        stages[k] = {
            "label": STAGE_LABELS.get(k, k),
            "mean_per_question_s": mean(totals),
            "sum_s": sum(totals) if totals else None,
            "mean_per_search_s": mean(per_search),
            "n": len(totals),
        }
        out[f"mean_{k}"] = mean(totals)
        out[f"sum_{k}"] = sum(totals) if totals else None
        out[f"mean_{k}_per_search"] = mean(per_search)
    out["stages"] = stages
    return out


def _acc_block(rows: Sequence[dict]) -> Dict[str, Any]:
    acc_counts = {lab: 0 for lab in JUDGMENT_LABELS}
    acc_counts["未知"] = 0
    scores: List[float] = []
    for r in rows:
        lab = r.get("llm_acc")
        if lab in JUDGMENT_LABELS:
            acc_counts[lab] += 1
        else:
            acc_counts["未知"] += 1
        if r.get("score") is not None:
            try:
                scores.append(float(r["score"]))
            except (TypeError, ValueError):
                pass
    n_correct = acc_counts["正确"]
    n_partial = acc_counts.get("部分正确", 0)
    n_wrong = acc_counts["错误"]
    judged = n_correct + n_partial + n_wrong
    return {
        "counts": acc_counts,
        "accuracy": safe_div(n_correct, judged) if judged else None,
        "partial_rate": safe_div(n_partial, judged) if judged else None,
        "error_rate": safe_div(n_wrong, judged) if judged else None,
        "n_judged": judged,
        "n_correct": n_correct,
        "n_partial": n_partial,
        "n_wrong": n_wrong,
        "mean_score": mean(scores),
    }


def _by_category(rows: Sequence[dict], fields: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for field in fields:
        groups: Dict[str, list] = defaultdict(list)
        for r in rows:
            cats = r.get("categories") or {}
            label = str(cats.get(field) or "NA")
            groups[label].append(r)
        field_out = {}
        for label, group in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0])):
            field_out[label] = {
                "n": len(group),
                "llm_acc": _acc_block(group),
                "mean_query_latency_s": mean(_metric_vals(group, "query_latency_s")),
                "mean_wall_latency_s": mean(_metric_vals(group, "wall_latency_s")),
                "mean_n_turns": mean(_metric_vals(group, "n_turns")),
                "mean_n_search": mean(_metric_vals(group, "n_search")),
                "mean_score": mean(
                    [float(r["score"]) for r in group if r.get("score") is not None]
                ),
            }
        out[field] = field_out
    return out


def _summarize_one(
    rows: Sequence[Dict[str, Any]],
    *,
    enable_doc_recall: bool,
    is_agentic: bool,
    category_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    n = len(rows)
    n_query_fail = sum(1 for r in rows if r.get("query_error") or r.get("query_status") == 0)
    n_judge_fail = sum(1 for r in rows if r.get("judge_status") == 0 or r.get("judge_error"))

    # hop groups
    by_hop: Dict[str, Any] = {}
    hop_groups: Dict[Any, list] = defaultdict(list)
    for r in rows:
        hop_groups[r.get("hop")].append(r)
    for hop, group in sorted(hop_groups.items(), key=lambda x: (x[0] is None, str(x[0]) or "")):
        hop_item = {
            "n": len(group),
            "llm_acc": _acc_block(group),
            "mean_query_latency_s": mean(_metric_vals(group, "query_latency_s")),
            "mean_retrieve_latency_s": mean(_metric_vals(group, "retrieve_latency_s")),
            "mean_wall_latency_s": mean(_metric_vals(group, "wall_latency_s")),
            "mean_search_s": mean(_metric_vals(group, "mean_search_s")),
            "mean_n_search": mean(_metric_vals(group, "n_search")),
            "mean_n_turns": mean(_metric_vals(group, "n_turns")),
        }
        by_hop[str(hop)] = hop_item

    fields = list(category_fields or [])
    if not fields:
        # discover from data
        seen = []
        for r in rows:
            for k in (r.get("categories") or {}).keys():
                if k not in seen:
                    seen.append(k)
        fields = seen

    summary: Dict[str, Any] = {
        "n_total": n,
        "enable_doc_recall": enable_doc_recall,
        "pipeline": {
            "n_query_fail": n_query_fail,
            "n_judge_fail": n_judge_fail,
            "n_ok": n - sum(1 for r in rows if _system_failed(r)),
        },
        "llm_acc": _acc_block(rows),
        "latency": _latency_block(rows, is_agentic=is_agentic),
        "tokens": {
            "sum_prompt": _sum_tokens(rows, "prompt_tokens"),
            "sum_completion": _sum_tokens(rows, "completion_tokens"),
            "sum_total": _sum_tokens(rows, "total_tokens"),
            "mean_prompt": mean(_metric_vals(rows, "prompt_tokens")),
            "mean_completion": mean(_metric_vals(rows, "completion_tokens")),
            "sum_judge_prompt": _sum_tokens(rows, "judge_prompt_tokens"),
            "sum_judge_completion": _sum_tokens(rows, "judge_completion_tokens"),
            "sum_judge_total": _sum_tokens(rows, "judge_total_tokens"),
        },
        "by_hop": by_hop,
        "by_category": _by_category(rows, fields),
    }
    if enable_doc_recall:
        recalls = []
        for r in rows:
            rec = r.get("recall") or {}
            if isinstance(rec, dict) and rec.get("recall") is not None:
                recalls.append(float(rec["recall"]))
        summary["doc_recall"] = {"mean_recall": mean(recalls)}
    return summary


def _compare_systems(
    results: Sequence[Dict[str, Any]],
    ag_rows: Sequence[Dict[str, Any]],
    lo_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    by_id_lo = {r.get("id"): r for r in lo_rows}
    by_id_raw = {r.get("id"): r for r in results}
    n_paired = 0
    n_both = 0
    n_ag_only = 0
    n_lo_only = 0
    n_neither = 0
    n_ag_win = 0
    n_lo_win = 0
    n_tie = 0
    n_pairwise = 0
    crosstab: Dict[str, int] = defaultdict(int)
    score_delta: List[float] = []

    for ag in ag_rows:
        lo = by_id_lo.get(ag.get("id"))
        if lo is None:
            continue
        aj = ag.get("llm_acc") if ag.get("llm_acc") in JUDGMENT_LABELS else None
        lj = lo.get("llm_acc") if lo.get("llm_acc") in JUDGMENT_LABELS else None
        if aj is None or lj is None:
            continue
        n_paired += 1
        crosstab[f"Agentic{aj}_纯LLM{lj}"] += 1
        if aj == "正确" and lj == "正确":
            n_both += 1
        elif aj == "正确":
            n_ag_only += 1
        elif lj == "正确":
            n_lo_only += 1
        else:
            n_neither += 1
        if ag.get("score") is not None and lo.get("score") is not None:
            try:
                score_delta.append(float(ag["score"]) - float(lo["score"]))
            except (TypeError, ValueError):
                pass
        raw = by_id_raw.get(ag.get("id")) or {}
        winner = (raw.get("comparison") or {}).get("winner")
        if winner in (SYSTEM_AGENTIC, SYSTEM_LLM_ONLY, "tie"):
            n_pairwise += 1
            if winner == SYSTEM_AGENTIC:
                n_ag_win += 1
            elif winner == SYSTEM_LLM_ONLY:
                n_lo_win += 1
            else:
                n_tie += 1

    return {
        "n_paired": n_paired,
        "both_correct": n_both,
        "agentic_only_correct": n_ag_only,
        "llm_only_only_correct": n_lo_only,
        "neither_correct": n_neither,
        "crosstab": dict(crosstab),
        "pairwise": {
            "n": n_pairwise,
            "agentic_win": n_ag_win,
            "llm_only_win": n_lo_win,
            "tie": n_tie,
            "agentic_win_rate": safe_div(n_ag_win, n_pairwise) if n_pairwise else None,
            "llm_only_win_rate": safe_div(n_lo_win, n_pairwise) if n_pairwise else None,
            "tie_rate": safe_div(n_tie, n_pairwise) if n_pairwise else None,
        },
        "mean_score_delta_agentic_minus_llm": mean(score_delta),
    }


def build_summary(
    results: Sequence[Dict[str, Any]],
    *,
    enable_llm_only: bool = True,
    enable_doc_recall: bool = False,
    category_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    ag_rows = [project_system_row(r, SYSTEM_AGENTIC) for r in results]
    systems = {
        SYSTEM_AGENTIC: _summarize_one(
            ag_rows,
            enable_doc_recall=enable_doc_recall,
            is_agentic=True,
            category_fields=category_fields,
        )
    }
    comparison = None
    if enable_llm_only:
        lo_rows = [
            project_system_row(r, SYSTEM_LLM_ONLY)
            for r in results
            if _system_attempted(r.get(SYSTEM_LLM_ONLY))
        ]
        systems[SYSTEM_LLM_ONLY] = _summarize_one(
            lo_rows,
            enable_doc_recall=False,
            is_agentic=False,
            category_fields=category_fields,
        )
        comparison = _compare_systems(results, ag_rows, lo_rows)

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_results": len(results),
        "systems": systems,
        "comparison": comparison,
        "system_order": [SYSTEM_AGENTIC] + ([SYSTEM_LLM_ONLY] if enable_llm_only else []),
    }


def build_eval_document(
    results: Sequence[Dict[str, Any]],
    *,
    dataset_meta: Optional[dict] = None,
    config_meta: Optional[dict] = None,
    enable_llm_only: bool = True,
    enable_doc_recall: bool = False,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "meta": {
            "dataset": dataset_meta,
            "config": config_meta,
            "enable_llm_only": enable_llm_only,
            "enable_doc_recall": enable_doc_recall,
            "n_results": len(results),
        },
        "results": list(results),
    }


def build_report_document(
    eval_data: Dict[str, Any],
    *,
    stats: Optional[Dict[str, Any]] = None,
    category_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    results = list(eval_data.get("results") or [])
    meta = eval_data.get("meta") or {}
    enable_llm_only = bool(meta.get("enable_llm_only", True))
    enable_doc_recall = bool(meta.get("enable_doc_recall", False))

    fields = list(category_fields or [])
    if not fields and stats:
        cf = (stats.get("category_fields") or {})
        fields = list(cf.get("primary") or cf.get("all") or [])
    if not fields:
        ds = meta.get("dataset") or {}
        fields = list(ds.get("category_fields") or [])

    summary = build_summary(
        results,
        enable_llm_only=enable_llm_only,
        enable_doc_recall=enable_doc_recall,
        category_fields=fields,
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "meta": {
            "eval_meta": meta,
            "stats_meta": (stats or {}).get("meta") if stats else None,
            "category_fields": fields,
        },
        "stats": stats,
        "summary": summary,
        "n_results": len(results),
    }


def format_summary_text(summary: Dict[str, Any]) -> str:
    lines = ["=== benchmark_agentic_2 summary ==="]
    systems = summary.get("systems") or {}
    for key, label in ((SYSTEM_AGENTIC, "Agentic"), (SYSTEM_LLM_ONLY, "纯LLM")):
        s = systems.get(key)
        if not s:
            continue
        acc = s.get("llm_acc") or {}
        lat = s.get("latency") or {}
        lines.append(f"[{label}] n={s.get('n_total')}  accuracy={acc.get('accuracy')}  mean_score={acc.get('mean_score')}")
        lines.append(
            f"  latency: query={lat.get('mean_query_s')} wall={lat.get('mean_wall_s')} "
            f"retrieve={lat.get('mean_retrieve_s')}"
        )
        if key == SYSTEM_AGENTIC:
            lines.append(
                f"  agentic: mean_turns={lat.get('mean_n_turns')} mean_search={lat.get('mean_n_search')} "
                f"mean_search_s={lat.get('mean_search_s')} mean_turn_s={lat.get('mean_turn_s')}"
            )
            lines.append(
                f"  stages: embed={lat.get('mean_embed_s')} rerank={lat.get('mean_rerank_s')}"
            )
    cmp_ = summary.get("comparison") or {}
    pw = cmp_.get("pairwise") or {}
    if pw:
        lines.append(
            f"[对比] agentic_win={pw.get('agentic_win')} llm_win={pw.get('llm_only_win')} "
            f"tie={pw.get('tie')}  win_rate_ag={pw.get('agentic_win_rate')}"
        )
    return "\n".join(lines)
