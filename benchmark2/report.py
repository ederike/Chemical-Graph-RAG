"""评测结果汇总：总体指标 + 按测试集统计分类的多维切分。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from benchmark.utils import mean, safe_div

from .evaluator import (
    JUDGMENT_LABELS,
    SYSTEM_HYPERGRAPH,
    SYSTEM_LLM_ONLY,
    ExcelQueryEvaluator,
)

RETRIEVE_STAGE_KEYS = (
    ("precompute_latency_s", "mean_precompute_s", "sum_precompute_s"),
    ("rewrite_latency_s", "mean_rewrite_s", "sum_rewrite_s"),
    ("embed_latency_s", "mean_embed_s", "sum_embed_s"),
    ("chunk_latency_s", "mean_chunk_s", "sum_chunk_s"),
    ("node_latency_s", "mean_node_s", "sum_node_s"),
    ("keyword_latency_s", "mean_keyword_s", "sum_keyword_s"),
    ("expand_latency_s", "mean_expand_s", "sum_expand_s"),
    ("rerank_latency_s", "mean_rerank_s", "sum_rerank_s"),
)


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


def _metric_lats(results: Sequence[dict], key: str) -> List[float]:
    out: List[float] = []
    for r in results:
        v = (r.get("metrics") or {}).get(key)
        if v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            pass
    return out


def _score_hist(results: Sequence[dict]) -> Dict[str, int]:
    hist = {str(i): 0 for i in range(4)}
    for r in results:
        s = r.get("score")
        if s is None:
            continue
        try:
            iv = int(s)
        except (TypeError, ValueError):
            continue
        if 0 <= iv <= 3:
            hist[str(iv)] += 1
    return hist


def _group_metrics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    acc_counts = {lab: 0 for lab in JUDGMENT_LABELS}
    acc_counts["未知"] = 0
    scores: List[float] = []
    for r in results:
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
    n_partial = acc_counts["部分正确"]
    n_wrong = acc_counts["错误"]
    judged = n_correct + n_partial + n_wrong
    n_scored = len(scores)

    q_lats = _metric_lats(results, "query_latency_s")
    r_lats = _metric_lats(results, "retrieve_latency_s")
    w_lats = _metric_lats(results, "wall_latency_s")
    j_lats = _metric_lats(results, "judge_latency_s")

    n_query_fail = sum(
        1
        for r in results
        if ExcelQueryEvaluator._is_eval_failure(r)
        and (r.get("query_error") or r.get("query_status") == 0)
    )
    n_judge_fail = sum(
        1
        for r in results
        if r.get("judge_status") == 0 or r.get("judge_error")
    )
    n_pipeline_fail = sum(
        1 for r in results if ExcelQueryEvaluator._is_eval_failure(r)
    )

    latency: Dict[str, Any] = {
        "mean_query_s": mean(q_lats),
        "mean_retrieve_s": mean(r_lats),
        "mean_wall_s": mean(w_lats),
        "mean_judge_s": mean(j_lats),
        "sum_query_s": sum(q_lats) if q_lats else None,
        "sum_retrieve_s": sum(r_lats) if r_lats else None,
        "sum_wall_s": sum(w_lats) if w_lats else None,
        "sum_judge_s": sum(j_lats) if j_lats else None,
    }
    for src, mk, sk in RETRIEVE_STAGE_KEYS:
        vals = _metric_lats(results, src)
        latency[mk] = mean(vals)
        latency[sk] = sum(vals) if vals else None

    return {
        "n": n,
        "pipeline": {
            "n_query_fail": n_query_fail,
            "n_judge_fail": n_judge_fail,
            "n_ok": n - n_pipeline_fail,
        },
        "judgment": {
            "counts": acc_counts,
            "accuracy": safe_div(n_correct, judged) if judged else None,
            "lenient_accuracy": (
                safe_div(n_correct + n_partial, judged) if judged else None
            ),
            "error_rate": safe_div(n_wrong, judged) if judged else None,
            "partial_rate": safe_div(n_partial, judged) if judged else None,
            "n_judged": judged,
            "n_correct": n_correct,
            "n_partial": n_partial,
            "n_wrong": n_wrong,
        },
        "score": {
            "mean": mean(scores),
            "mean_normalized": (
                safe_div(mean(scores), 3.0) if scores else None
            ),
            "histogram": _score_hist(results),
            "n_scored": n_scored,
        },
        "latency": latency,
        "tokens": {
            "sum_prompt": _sum_tokens(results, "prompt_tokens"),
            "sum_completion": _sum_tokens(results, "completion_tokens"),
            "sum_total": _sum_tokens(results, "total_tokens"),
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
            "sum_judge_prompt": _sum_tokens(results, "judge_prompt_tokens"),
            "sum_judge_completion": _sum_tokens(results, "judge_completion_tokens"),
            "sum_judge_total": _sum_tokens(results, "judge_total_tokens"),
        },
    }


def _dimension_summary(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    n_items = 0
    n_pass = 0
    scores: List[float] = []
    for r in results:
        for d in r.get("dimension_scores") or []:
            if not isinstance(d, dict):
                continue
            name = str(d.get("dimension") or "").strip()
            if not name:
                continue
            n_items += 1
            by_name[name].append(d)
            if d.get("pass") is True:
                n_pass += 1
            elif d.get("pass") is None and d.get("score") is not None:
                try:
                    if float(d["score"]) >= 2:
                        n_pass += 1
                except (TypeError, ValueError):
                    pass
            if d.get("score") is not None:
                try:
                    scores.append(float(d["score"]))
                except (TypeError, ValueError):
                    pass

    by_dimension: Dict[str, Any] = {}
    for name, items in sorted(by_name.items(), key=lambda x: -len(x[1])):
        dscores = []
        n_ok = 0
        for d in items:
            if d.get("score") is not None:
                try:
                    dscores.append(float(d["score"]))
                except (TypeError, ValueError):
                    pass
            passed = d.get("pass")
            if passed is True:
                n_ok += 1
            elif passed is None and d.get("score") is not None:
                try:
                    if float(d["score"]) >= 2:
                        n_ok += 1
                except (TypeError, ValueError):
                    pass
        by_dimension[name] = {
            "n": len(items),
            "n_pass": n_ok,
            "pass_rate": safe_div(n_ok, len(items)),
            "mean_score": mean(dscores),
        }

    return {
        "n_dimension_items": n_items,
        "n_unique": len(by_name),
        "mean_pass_rate": safe_div(n_pass, n_items) if n_items else None,
        "mean_score": mean(scores),
        "by_dimension": by_dimension,
    }


def _sheet_dist_map(stats: Optional[Dict[str, Any]], field: str) -> Dict[str, Any]:
    if not stats:
        return {}
    dist = (stats.get("distributions") or {}).get(field) or {}
    items = dist.get("items") or []
    return {str(it.get("label")): it for it in items if it.get("label") is not None}


def _summarize_field(
    results: Sequence[Dict[str, Any]],
    field: str,
    *,
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        cats = r.get("categories") or {}
        label = cats.get(field)
        if label is None or label == "":
            label = "NA"
        groups[str(label)].append(r)

    sheet_map = _sheet_dist_map(stats, field)
    n_all = len(results)
    out_groups: Dict[str, Any] = {}
    for label, rows in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0])):
        gm = _group_metrics(rows)
        sheet = sheet_map.get(label) or {}
        out_groups[label] = {
            "n": gm["n"],
            "eval_ratio": safe_div(gm["n"], n_all),
            "dataset_count": sheet.get("count"),
            "dataset_ratio": sheet.get("ratio"),
            "coverage": (
                safe_div(gm["n"], sheet["count"])
                if sheet.get("count")
                else None
            ),
            "judgment": gm["judgment"],
            "score": gm["score"],
            "latency": {
                "mean_query_s": gm["latency"]["mean_query_s"],
                "mean_retrieve_s": gm["latency"]["mean_retrieve_s"],
                "mean_wall_s": gm["latency"]["mean_wall_s"],
            },
            "pipeline": gm["pipeline"],
        }

    return {
        "field": field,
        "n_groups": len(out_groups),
        "n_results": n_all,
        "groups": out_groups,
    }


def _resolve_category_fields(
    results: Sequence[Dict[str, Any]],
    stats: Optional[Dict[str, Any]],
) -> Dict[str, List[str]]:
    primary: List[str] = []
    extra: List[str] = []
    if stats:
        cf = stats.get("category_fields") or {}
        if isinstance(cf, dict):
            primary = list(cf.get("primary") or [])
            extra = list(cf.get("extra") or [])
        if not primary:
            primary = list((stats.get("distributions") or {}).keys())

    seen = set(primary) | set(extra)
    inferred: List[str] = []
    for r in results:
        for k in (r.get("categories") or {}).keys():
            if k not in seen:
                inferred.append(k)
                seen.add(k)
    extra = extra + inferred
    if not primary and extra:
        primary, extra = extra, []
    return {"primary": primary, "extra": extra}


def project_system_row(result: Dict[str, Any], system: str) -> Dict[str, Any]:
    """把某一路（超图 / 纯 LLM）抽成与旧 summary 兼容的扁平行。"""
    if system == SYSTEM_HYPERGRAPH:
        block = result.get(SYSTEM_HYPERGRAPH)
        if not isinstance(block, dict) or not block:
            block = {
                "answer": result.get("rag_answer"),
                "raw_answer": result.get("rag_raw_answer"),
                "query_status": result.get("query_status"),
                "query_error": result.get("query_error"),
                "llm_acc": result.get("llm_acc"),
                "score": result.get("score"),
                "judge_reason": result.get("judge_reason"),
                "dimension_scores": result.get("dimension_scores"),
                "judge_status": result.get("judge_status"),
                "judge_error": result.get("judge_error"),
                "metrics": result.get("metrics") or {},
            }
    else:
        block = result.get(SYSTEM_LLM_ONLY)
        if not isinstance(block, dict):
            block = {}
    return {
        "id": result.get("id"),
        "question": result.get("question"),
        "categories": result.get("categories") or {},
        "llm_acc": block.get("llm_acc"),
        "score": block.get("score"),
        "dimension_scores": block.get("dimension_scores") or [],
        "query_status": block.get("query_status"),
        "query_error": block.get("query_error"),
        "judge_status": block.get("judge_status"),
        "judge_error": block.get("judge_error"),
        "metrics": block.get("metrics") or {},
    }


def _build_one_summary(
    rows: Sequence[Dict[str, Any]],
    *,
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    overall = _group_metrics(rows)
    fields = _resolve_category_fields(rows, stats)
    return {
        "n_total": overall["n"],
        "pipeline": overall["pipeline"],
        "judgment": overall["judgment"],
        "score": overall["score"],
        "dimensions": _dimension_summary(rows),
        "latency": overall["latency"],
        "tokens": overall["tokens"],
        "by_category": {
            f: _summarize_field(rows, f, stats=stats) for f in fields["primary"]
        },
        "by_category_extra": {
            f: _summarize_field(rows, f, stats=stats) for f in fields["extra"]
        },
    }


def _compare_systems(
    hg_rows: Sequence[Dict[str, Any]],
    lo_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    by_id_lo = {r.get("id"): r for r in lo_rows}
    n_paired = 0
    n_hg_win = 0
    n_lo_win = 0
    n_tie = 0
    deltas: List[float] = []
    crosstab: Dict[str, int] = defaultdict(int)
    for hg in hg_rows:
        lo = by_id_lo.get(hg.get("id"))
        if lo is None:
            continue
        hs, ls = hg.get("score"), lo.get("score")
        if hs is None or ls is None:
            continue
        try:
            hf, lf = float(hs), float(ls)
        except (TypeError, ValueError):
            continue
        n_paired += 1
        deltas.append(hf - lf)
        if hf > lf:
            n_hg_win += 1
        elif lf > hf:
            n_lo_win += 1
        else:
            n_tie += 1
        hj = hg.get("llm_acc") if hg.get("llm_acc") in JUDGMENT_LABELS else "未知"
        lj = lo.get("llm_acc") if lo.get("llm_acc") in JUDGMENT_LABELS else "未知"
        crosstab[f"超图{hj}_纯LLM{lj}"] += 1

    return {
        "n_paired": n_paired,
        "score_delta_mean": mean(deltas),
        "score_delta_note": "正值表示超图+LLM 均分更高",
        "win": {
            "hypergraph": n_hg_win,
            "llm_only": n_lo_win,
            "tie": n_tie,
        },
        "judgment_crosstab": dict(crosstab),
    }


def build_summary(
    results: Sequence[Dict[str, Any]],
    *,
    stats: Optional[Dict[str, Any]] = None,
    enable_llm_only: Optional[bool] = None,
) -> Dict[str, Any]:
    if enable_llm_only is None:
        enable_llm_only = any(
            isinstance(r.get(SYSTEM_LLM_ONLY), dict)
            and (
                r[SYSTEM_LLM_ONLY].get("query_status") is not None
                or r[SYSTEM_LLM_ONLY].get("answer")
            )
            for r in results
        )

    hg_rows = [project_system_row(r, SYSTEM_HYPERGRAPH) for r in results]
    out: Dict[str, Any] = {
        "systems": [SYSTEM_HYPERGRAPH]
        + ([SYSTEM_LLM_ONLY] if enable_llm_only else []),
        SYSTEM_HYPERGRAPH: _build_one_summary(hg_rows, stats=stats),
    }
    if enable_llm_only:
        lo_rows = [project_system_row(r, SYSTEM_LLM_ONLY) for r in results]
        out[SYSTEM_LLM_ONLY] = _build_one_summary(lo_rows, stats=stats)
        out["comparison"] = _compare_systems(hg_rows, lo_rows)
    return out


def build_report_document(
    eval_data: Dict[str, Any],
    *,
    stats: Optional[Dict[str, Any]] = None,
    source_path: Optional[str] = None,
) -> Dict[str, Any]:
    results = list(eval_data.get("results") or [])
    src_meta = dict(eval_data.get("meta") or {})

    if stats is None:
        ds_meta = src_meta.get("dataset_meta")
        if isinstance(eval_data.get("stats"), dict):
            stats = eval_data["stats"]
        elif isinstance(ds_meta, dict) and isinstance(ds_meta.get("stats"), dict):
            stats = ds_meta["stats"]

    enable_llm_only = src_meta.get("enable_llm_only")
    if enable_llm_only is None:
        cfg_probe = src_meta.get("config")
        if isinstance(cfg_probe, dict):
            enable_llm_only = cfg_probe.get("enable_llm_only")
    summary = build_summary(results, stats=stats, enable_llm_only=enable_llm_only)
    cfg_src = src_meta.get("config")
    if not isinstance(cfg_src, dict):
        cfg_src = {}

    stats_meta = (stats or {}).get("meta") if isinstance(stats, dict) else None

    return {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_path": source_path,
            "config_file": cfg_src.get("config_file"),
            "dhmf_config_path": src_meta.get("dhmf_config_path")
            or cfg_src.get("dhmf_config_path"),
            "query_mode": src_meta.get("query_mode") or cfg_src.get("query_mode"),
            "judge_model": src_meta.get("judge_model") or cfg_src.get("judge_model"),
            "enable_llm_only": (
                enable_llm_only
                if enable_llm_only is not None
                else src_meta.get("enable_llm_only")
            ),
            "excel_path": (
                (stats_meta or {}).get("source_excel")
                or cfg_src.get("excel_path")
            ),
            "n_results": len(results),
            "n_questions": src_meta.get("n_questions", len(results)),
            "n_total_planned": src_meta.get("n_total_planned"),
            "n_resumed": src_meta.get("n_resumed"),
            "eval_elapsed_s": src_meta.get("elapsed_s"),
            "eval_done": src_meta.get("done"),
        },
        "dataset_stats": {
            "meta": stats_meta,
            "category_fields": (stats or {}).get("category_fields"),
            "distributions": (stats or {}).get("distributions"),
        }
        if stats
        else None,
        "summary": summary,
    }


def _format_one_summary(summary: Dict[str, Any]) -> str:
    j = summary.get("judgment") or {}
    s = summary.get("score") or {}
    p = summary.get("pipeline") or {}
    lines = [
        f"n={summary.get('n_total')}  "
        f"pipeline_ok={p.get('n_ok')}  "
        f"query_fail={p.get('n_query_fail')}  "
        f"judge_fail={p.get('n_judge_fail')}",
        f"judgment: {j.get('counts')}  "
        f"acc={j.get('accuracy')}  "
        f"lenient={j.get('lenient_accuracy')}",
        f"score: mean={s.get('mean')}  hist={s.get('histogram')}",
    ]
    by_cat = summary.get("by_category") or {}
    for field, block in by_cat.items():
        lines.append(f"— {field} —")
        groups = (block or {}).get("groups") or {}
        for label, g in groups.items():
            gj = g.get("judgment") or {}
            gs = g.get("score") or {}
            lines.append(
                f"  {label}: n={g.get('n')}  "
                f"acc={gj.get('accuracy')}  "
                f"lenient={gj.get('lenient_accuracy')}  "
                f"mean_score={gs.get('mean')}"
            )
    return "\n".join(lines)


def format_summary_text(summary: Dict[str, Any]) -> str:
    """终端可读的简要汇总。"""
    if not summary:
        return ""
    titles = {
        SYSTEM_HYPERGRAPH: "超图+LLM",
        SYSTEM_LLM_ONLY: "纯LLM",
    }
    if SYSTEM_HYPERGRAPH in summary or SYSTEM_LLM_ONLY in summary:
        parts: List[str] = []
        for key in (SYSTEM_HYPERGRAPH, SYSTEM_LLM_ONLY):
            block = summary.get(key)
            if isinstance(block, dict) and block.get("judgment") is not None:
                parts.append(f"======== {titles.get(key, key)} ========")
                parts.append(_format_one_summary(block))
        cmp_ = summary.get("comparison")
        if isinstance(cmp_, dict) and cmp_:
            win = cmp_.get("win") or {}
            parts.append("======== 对比 ========")
            parts.append(
                f"paired={cmp_.get('n_paired')}  "
                f"delta_mean={cmp_.get('score_delta_mean')}  "
                f"win 超图={win.get('hypergraph')}  "
                f"纯LLM={win.get('llm_only')}  "
                f"平={win.get('tie')}"
            )
            xt = cmp_.get("judgment_crosstab") or {}
            if xt:
                parts.append("crosstab: " + ", ".join(f"{k}={v}" for k, v in xt.items()))
        return "\n".join(parts)
    return _format_one_summary(summary)
