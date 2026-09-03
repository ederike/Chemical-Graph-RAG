"""从逐题 evals 汇总正确率、召回、延迟（含三路检索分阶段）。"""

from __future__ import annotations

import time
from collections import defaultdict
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

# 汇总表里要单独点名的检索阶段
STAGE_LABELS = {
    "precompute_s": "预加载",
    "rewrite_s": "改写",
    "embed_s": "向量化",
    "chunk_s": "chunk路",
    "node_s": "node路",
    "keyword_s": "keyword路",
    "expand_s": "扩展",
    "rerank_s": "重排",
    "total_s": "检索内部合计",
}


def _metric_vals(rows: Sequence[dict], key: str) -> List[float]:
    out: List[float] = []
    for r in rows:
        v = (r.get("metrics") or {}).get(key)
        if v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            pass
    return out


def _sum_tokens(rows: Sequence[dict], key: str) -> Optional[int]:
    total = 0
    any_v = False
    for r in rows:
        v = (r.get("metrics") or {}).get(key)
        if v is None:
            continue
        any_v = True
        try:
            total += int(v)
        except Exception:
            pass
    return total if any_v else None


def _system_attempted(block: Optional[dict]) -> bool:
    if not isinstance(block, dict) or not block:
        return False
    if (block.get("answer") or block.get("raw_answer") or "").strip():
        return True
    if block.get("query_error"):
        return True
    if block.get("query_status") == 1:
        return True
    metrics = block.get("metrics") or {}
    return any(metrics.get(k) is not None for k in ("query_latency_s", "wall_latency_s"))


def _system_failed(block: Optional[dict]) -> bool:
    if not isinstance(block, dict) or not block:
        return True
    if block.get("query_error"):
        return True
    if block.get("query_status") == 0:
        return True
    if block.get("judge_status") == 0 or block.get("judge_error"):
        return True
    return False


def project_system_row(result: Dict[str, Any], system: str) -> Dict[str, Any]:
    block = result.get(system)
    if not isinstance(block, dict):
        block = {}
    return {
        "id": result.get("id"),
        "hop": result.get("hop"),
        "question": result.get("question"),
        "llm_acc": block.get("llm_acc"),
        "query_status": block.get("query_status"),
        "query_error": block.get("query_error"),
        "judge_status": block.get("judge_status"),
        "judge_error": block.get("judge_error"),
        "metrics": block.get("metrics") or {},
        "recall": result.get("recall") if system == SYSTEM_AGENTIC else None,
    }


def _slim_dataset_meta(meta: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not meta or not isinstance(meta, dict):
        return None
    keep = (
        "created_at",
        "db_path",
        "hop_counts",
        "seed",
        "doc_id_min",
        "doc_id_max",
        "num_thread",
        "model",
        "total",
        "success",
        "failed",
        "elapsed_s",
        "n_source_docs",
        "question_gen_prompt",
    )
    out = {k: meta[k] for k in keep if k in meta}
    return out or None


def _build_recall_vs_accuracy(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
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
    col_totals = {lab: sum(cells[b][lab] for b in buckets) for lab in acc_labels}
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
        if cells[b][lab] > 0 or (b in ("hit", "miss") and lab in ("正确", "错误"))
    ]
    return {
        "definition": {
            "hit": "recall == 1.0：全部 gold 文档出现在 retrieval_sources",
            "partial": "0 < recall < 1：部分 gold 命中",
            "miss": "recall == 0.0：gold 均未出现在 retrieval_sources",
            "unknown": "该题无有效 recall 字段",
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
        },
    }


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


def _summarize_one(
    rows: Sequence[Dict[str, Any]],
    *,
    enable_doc_recall: bool,
    is_agentic: bool,
) -> Dict[str, Any]:
    n = len(rows)
    acc_counts = {lab: 0 for lab in JUDGMENT_LABELS}
    acc_counts["未知"] = 0
    for r in rows:
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
        for r in rows:
            rec = r.get("recall") or {}
            if isinstance(rec, dict) and rec.get("recall") is not None:
                recalls.append(float(rec["recall"]))

    by_hop: Dict[str, Any] = {}
    hop_groups: Dict[Any, list] = defaultdict(list)
    for r in rows:
        hop_groups[r.get("hop")].append(r)
    for hop, group in sorted(hop_groups.items(), key=lambda x: (x[0] is None, x[0] or 0)):
        g_acc = {lab: 0 for lab in JUDGMENT_LABELS}
        for r in group:
            if r.get("llm_acc") in g_acc:
                g_acc[r["llm_acc"]] += 1
        g_judged = sum(g_acc.values())
        hop_item: Dict[str, Any] = {
            "n": len(group),
            "llm_acc_counts": g_acc,
            "accuracy": safe_div(g_acc["正确"], g_judged) if g_judged else None,
            "mean_query_latency_s": mean(_metric_vals(group, "query_latency_s")),
            "mean_retrieve_latency_s": mean(_metric_vals(group, "retrieve_latency_s")),
            "mean_wall_latency_s": mean(_metric_vals(group, "wall_latency_s")),
            "mean_search_s": mean(_metric_vals(group, "mean_search_s")),
            "mean_n_search": mean(_metric_vals(group, "n_search")),
            "mean_n_turns": mean(_metric_vals(group, "n_turns")),
        }
        for k in RETRIEVE_STAGE_KEYS:
            hop_item[f"mean_{k}"] = mean(_metric_vals(group, k))
            hop_item[f"mean_{k}_per_search"] = mean(_metric_vals(group, f"mean_{k}"))
        if enable_doc_recall:
            g_recalls = [
                float((r.get("recall") or {}).get("recall"))
                for r in group
                if isinstance(r.get("recall"), dict)
                and (r.get("recall") or {}).get("recall") is not None
            ]
            hop_item["mean_doc_recall"] = mean(g_recalls)
        by_hop[str(hop)] = hop_item

    n_query_fail = sum(
        1 for r in rows
        if r.get("query_error") or r.get("query_status") == 0
    )
    n_judge_fail = sum(
        1 for r in rows
        if r.get("judge_status") == 0 or r.get("judge_error")
    )

    summary: Dict[str, Any] = {
        "n_total": n,
        "enable_doc_recall": enable_doc_recall,
        "pipeline": {
            "n_query_fail": n_query_fail,
            "n_judge_fail": n_judge_fail,
            "n_ok": n - sum(1 for r in rows if _system_failed(r)),
        },
        "llm_acc": {
            "counts": acc_counts,
            "accuracy": safe_div(n_correct, judged) if judged else None,
            "error_rate": safe_div(n_wrong, judged) if judged else None,
            "n_judged": judged,
            "n_correct": n_correct,
            "n_wrong": n_wrong,
        },
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
    }
    if enable_doc_recall:
        summary["doc_recall"] = {
            "mean_recall": mean(recalls),
            "vs_accuracy": _build_recall_vs_accuracy(rows),
        }
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
        raw = by_id_raw.get(ag.get("id")) or {}
        winner = (raw.get("comparison") or {}).get("winner")
        if winner in (SYSTEM_AGENTIC, SYSTEM_LLM_ONLY, "tie"):
            n_pairwise += 1
        else:
            if aj == "正确" and lj != "正确":
                winner = SYSTEM_AGENTIC
            elif lj == "正确" and aj != "正确":
                winner = SYSTEM_LLM_ONLY
            else:
                winner = "tie"
        if winner == SYSTEM_AGENTIC:
            n_ag_win += 1
        elif winner == SYSTEM_LLM_ONLY:
            n_lo_win += 1
        else:
            n_tie += 1
    return {
        "n_paired": n_paired,
        "n_pairwise_judged": n_pairwise,
        "both_correct": n_both,
        "agentic_only": n_ag_only,
        "llm_only_only": n_lo_only,
        "both_wrong": n_neither,
        "win": {
            "agentic": n_ag_win,
            "llm_only": n_lo_win,
            "tie": n_tie,
        },
        "win_note": "胜负优先取裁判 comparison.winner；缺省则正确>错误",
        "judgment_crosstab": dict(crosstab),
    }


def build_summary(
    results: Sequence[Dict[str, Any]],
    *,
    enable_doc_recall: bool = True,
    enable_llm_only: Optional[bool] = None,
) -> Dict[str, Any]:
    enable_doc_recall = bool(enable_doc_recall)
    if enable_llm_only is None:
        enable_llm_only = any(
            _system_attempted(r.get(SYSTEM_LLM_ONLY)) for r in results
        )

    ag_rows = [project_system_row(r, SYSTEM_AGENTIC) for r in results]
    ag_sum = _summarize_one(
        ag_rows, enable_doc_recall=enable_doc_recall, is_agentic=True
    )
    out: Dict[str, Any] = dict(ag_sum)
    out["systems"] = [SYSTEM_AGENTIC] + ([SYSTEM_LLM_ONLY] if enable_llm_only else [])
    out[SYSTEM_AGENTIC] = ag_sum
    if enable_llm_only:
        lo_rows = [project_system_row(r, SYSTEM_LLM_ONLY) for r in results]
        out[SYSTEM_LLM_ONLY] = _summarize_one(
            lo_rows, enable_doc_recall=False, is_agentic=False
        )
        out["comparison"] = _compare_systems(results, ag_rows, lo_rows)
    return out


def build_eval_document(
    *,
    dataset: Dict[str, Any],
    results: List[Dict[str, Any]],
    skipped: int,
    t0: float,
    created_at: str,
    done: bool = False,
    enable_doc_recall: bool = True,
    enable_llm_only: bool = True,
    extra_meta: Optional[dict] = None,
) -> Dict[str, Any]:
    summary = build_summary(
        results,
        enable_doc_recall=enable_doc_recall,
        enable_llm_only=enable_llm_only,
    )
    meta = {
        "created_at": created_at,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "done": bool(done),
        "dataset_meta": _slim_dataset_meta(dataset.get("meta")),
        "query_mode": "agentic",
        "judge_mode": "pairwise" if enable_llm_only else "single",
        "enable_doc_recall": enable_doc_recall,
        "enable_llm_only": enable_llm_only,
        "n_questions": len(results),
        "n_skipped_gen_fail": skipped,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }
    if extra_meta:
        meta.update(extra_meta)
    return {
        "meta": meta,
        "results": results,
        "summary": summary,
    }


def build_report_document(
    eval_data: Dict[str, Any],
    *,
    source_path: Optional[str] = None,
    enable_doc_recall: Optional[bool] = None,
) -> Dict[str, Any]:
    results = list(eval_data.get("results") or [])
    src_meta = dict(eval_data.get("meta") or {})

    if enable_doc_recall is None:
        if "enable_doc_recall" in src_meta:
            enable_doc_recall = bool(src_meta.get("enable_doc_recall"))
        else:
            enable_doc_recall = True
    enable_doc_recall = bool(enable_doc_recall)

    enable_llm_only = src_meta.get("enable_llm_only")
    if enable_llm_only is None:
        cfg_probe = src_meta.get("config")
        if isinstance(cfg_probe, dict):
            enable_llm_only = cfg_probe.get("enable_llm_only")
    summary = build_summary(
        results,
        enable_doc_recall=enable_doc_recall,
        enable_llm_only=enable_llm_only,
    )

    ds_meta = src_meta.get("dataset_meta")
    if not isinstance(ds_meta, dict):
        ds_meta = {}
    slim_ds = _slim_dataset_meta(ds_meta) or {}
    cfg_src = src_meta.get("config")
    if not isinstance(cfg_src, dict):
        cfg_src = {}

    return {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_path": source_path,
            "config_file": cfg_src.get("config_file"),
            "dhmf_config_path": cfg_src.get("dhmf_config_path"),
            "query_mode": "agentic",
            "judge_model": src_meta.get("judge_model") or cfg_src.get("judge_model"),
            "judge_mode": src_meta.get("judge_mode") or (
                "pairwise" if (
                    src_meta.get("enable_llm_only")
                    if src_meta.get("enable_llm_only") is not None
                    else cfg_src.get("enable_llm_only")
                ) else "single"
            ),
            "enable_doc_recall": enable_doc_recall,
            "enable_llm_only": src_meta.get("enable_llm_only")
            if src_meta.get("enable_llm_only") is not None
            else cfg_src.get("enable_llm_only"),
            "db_path": slim_ds.get("db_path") or cfg_src.get("db_path"),
            "hop_counts": slim_ds.get("hop_counts") or cfg_src.get("hop_counts"),
            "seed": slim_ds.get("seed", cfg_src.get("seed")),
            "doc_id_min": slim_ds.get("doc_id_min", cfg_src.get("doc_id_min")),
            "doc_id_max": slim_ds.get("doc_id_max", cfg_src.get("doc_id_max")),
            "n_results": len(results),
            "n_questions": src_meta.get("n_questions", len(results)),
            "n_total_planned": src_meta.get("n_total_planned"),
            "n_skipped_gen_fail": src_meta.get("n_skipped_gen_fail"),
            "eval_elapsed_s": src_meta.get("elapsed_s"),
            "eval_done": src_meta.get("done"),
        },
        "summary": summary,
    }
