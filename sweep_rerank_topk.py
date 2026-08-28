#!/usr/bin/env python3
"""
rerank_top_k 敏感性评测。

固定使用已有问题集，只改 retrieve.rerank_top_k。
每个 k 的目录只写 evals JSON（模型回答 + 评判），不复制问题集、不写 report/xlsx。

全部跑完后扫描 RESULTS_ROOT 下所有 benchmark_result_topk={k}，
汇总准确率 / 召回率 / 延迟，并画出随 topk 变化的曲线。

改下面「可调参数」，然后::

    python sweep_rerank_topk.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 可调参数
# ---------------------------------------------------------------------------

# 要扫描的 rerank_top_k，按顺序评测
RERANK_TOP_KS: List[int] = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# 固定问题集（已有，不再生成、不复制到各结果目录）
QUESTIONS: str = "benchmark_results/benchmark_result_topk=8/questions_complex.json"

BENCHMARK_CONFIG: str = "benchmark/config.yaml"
RESULTS_ROOT: str = "benchmark_results"

# True：目标目录已有 evals 则跳过该 k
SKIP_EXISTING: bool = False

# True：不评测，只根据 RESULTS_ROOT 里已有 evals 做总结和画图
SUMMARIZE_ONLY: bool = False

# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.config import BenchmarkConfig
from benchmark.evaluator import QueryEvaluator
from benchmark.utils import resolve_path
from benchmark.workflow import TestQueryWorkflow

_TOPK_DIR_RE = re.compile(r"^benchmark_result_topk=(\d+)$")


def _as_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else resolve_path(path)


def _uniq_ks(raw: List[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for k in raw:
        k = int(k)
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def _find_evals_file(folder: Path, preferred: str) -> Optional[Path]:
    cand = folder / preferred
    if cand.is_file():
        return cand
    hits = sorted(folder.glob("evals*.json"))
    return hits[0] if hits else None


def _run_one(
    *,
    config_path: str,
    out_dir: Path,
    questions_path: Path,
    k: int,
    dcfg,
    dhmf,
) -> Path:
    cfg = BenchmarkConfig.from_yaml(
        config_path,
        overrides={
            "run": {"mode": "evaluate"},
            "paths": {
                "output_dir": str(out_dir),
                "eval_questions_path": str(questions_path),
            },
        },
    )
    wf = TestQueryWorkflow(cfg)
    wf.dhmf_config = dcfg
    wf.dhmf = dhmf
    dcfg.retrieve.rerank_top_k = int(k)
    dcfg.retrieve.enable_rerank = True

    print(
        f"[sweep] k={k}  rerank_top_k={dcfg.retrieve.rerank_top_k}  → {out_dir}",
        file=sys.stderr,
    )
    eval_data = wf.evaluate(save=True)
    eval_data.setdefault("meta", {})["rerank_top_k"] = int(k)
    out = wf.cfg.eval_results_file()
    wf.save_json(eval_data, out)
    return out


def _metrics_from_eval(eval_data: Dict[str, Any]) -> Dict[str, Any]:
    """用现有 evaluator 汇总，取超图一路的准确率 / 召回 / 延迟。"""
    report = QueryEvaluator.build_report_document(eval_data)
    summary = report.get("summary") or {}
    hg = summary.get("hypergraph") if isinstance(summary.get("hypergraph"), dict) else summary
    acc = (hg.get("llm_acc") or {}) if isinstance(hg.get("llm_acc"), dict) else {}
    rec = (hg.get("doc_recall") or summary.get("doc_recall") or {})
    if not isinstance(rec, dict):
        rec = {}
    lat = hg.get("latency") if isinstance(hg.get("latency"), dict) else {}
    meta = report.get("meta") or {}
    return {
        "n_results": meta.get("n_results"),
        "eval_done": meta.get("eval_done"),
        "accuracy": acc.get("accuracy"),
        "n_judged": acc.get("n_judged"),
        "n_correct": acc.get("n_correct"),
        "n_wrong": acc.get("n_wrong"),
        "mean_recall": rec.get("mean_recall"),
        "mean_query_s": lat.get("mean_query_s"),
        "mean_retrieve_s": lat.get("mean_retrieve_s"),
        "mean_rerank_s": lat.get("mean_rerank_s"),
        "mean_wall_s": lat.get("mean_wall_s"),
        "by_hop": hg.get("by_hop") or {},
    }


def _discover_runs(results_root: Path, evals_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not results_root.is_dir():
        return rows
    for folder in sorted(results_root.iterdir()):
        if not folder.is_dir():
            continue
        m = _TOPK_DIR_RE.match(folder.name)
        if not m:
            continue
        k = int(m.group(1))
        ev_path = _find_evals_file(folder, evals_name)
        if ev_path is None:
            print(f"[sweep] skip {folder.name}：无 evals JSON", file=sys.stderr)
            continue
        try:
            with open(ev_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        except Exception as e:
            print(f"[sweep] skip {folder.name}：读 {ev_path.name} 失败: {e}", file=sys.stderr)
            continue
        if not isinstance(eval_data, dict) or "results" not in eval_data:
            print(f"[sweep] skip {folder.name}：不是评测结果 JSON", file=sys.stderr)
            continue
        try:
            metrics = _metrics_from_eval(eval_data)
        except Exception as e:
            print(f"[sweep] skip {folder.name}：汇总失败: {e}", file=sys.stderr)
            continue
        rows.append({
            "rerank_top_k": k,
            "dir": folder.name,
            "evals_path": str(ev_path),
            **metrics,
        })
    rows.sort(key=lambda r: r["rerank_top_k"])
    return rows


def _configure_chinese_font() -> None:
    try:
        from matplotlib import font_manager
        import matplotlib.pyplot as plt
    except ImportError:
        return
    font_dirs = [
        ROOT / "scripts" / "assets" / "fonts",
        ROOT / "assets" / "fonts",
    ]
    for font_dir in font_dirs:
        if not font_dir.is_dir():
            continue
        for font_path in sorted(font_dir.glob("NotoSansSC-Regular.*")):
            try:
                font_manager.fontManager.addfont(str(font_path))
                family = font_manager.FontProperties(fname=str(font_path)).get_name()
                plt.rcParams["font.family"] = "sans-serif"
                plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                continue
    plt.rcParams["axes.unicode_minus"] = False


def _xy(rows: List[dict], key: str) -> Tuple[List[int], List[float]]:
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        try:
            ys.append(float(v))
        except (TypeError, ValueError):
            continue
        xs.append(int(r["rerank_top_k"]))
    return xs, ys


def _plot_curves(rows: List[dict], out_dir: Path) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_chinese_font()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}

    def _style(ax) -> None:
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel("rerank_top_k")

    # 准确率 + 召回率
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x_acc, y_acc = _xy(rows, "accuracy")
    x_rec, y_rec = _xy(rows, "mean_recall")
    if x_acc:
        ax.plot(x_acc, y_acc, marker="o", label="准确率")
    if x_rec:
        ax.plot(x_rec, y_rec, marker="s", label="文档召回率")
    _style(ax)
    ax.set_ylabel("比例")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("准确率 / 召回率 vs rerank_top_k")
    ax.legend()
    fig.tight_layout()
    p1 = out_dir / "rerank_topk_acc_recall.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved["acc_recall"] = str(p1)

    # 延迟
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    series = (
        ("mean_query_s", "问答耗时", "o"),
        ("mean_retrieve_s", "检索耗时", "s"),
        ("mean_rerank_s", "rerank 耗时", "^"),
    )
    any_line = False
    for key, label, marker in series:
        xs, ys = _xy(rows, key)
        if not xs:
            continue
        ax.plot(xs, ys, marker=marker, label=label)
        any_line = True
    _style(ax)
    ax.set_ylabel("秒")
    ax.set_title("延迟 vs rerank_top_k")
    if any_line:
        ax.legend()
    fig.tight_layout()
    p2 = out_dir / "rerank_topk_latency.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved["latency"] = str(p2)
    return saved


def _write_summary(rows: List[dict], results_root: Path) -> Path:
    plots = _plot_curves(rows, results_root)
    doc = {
        "meta": {
            "results_root": str(results_root),
            "n_runs": len(rows),
            "rerank_top_ks": [r["rerank_top_k"] for r in rows],
        },
        "runs": rows,
        "plots": plots,
    }
    out = results_root / "rerank_topk_sweep_summary.json"
    out.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[sweep] summary → {out}", file=sys.stderr)
    for name, path in plots.items():
        print(f"[sweep] plot {name} → {path}", file=sys.stderr)
    return out


def _print_table(rows: List[dict]) -> None:
    if not rows:
        print("[sweep] 没有可汇总的 evals", file=sys.stderr)
        return
    hdr = (
        f"{'k':>4}  {'acc':>7}  {'recall':>7}  "
        f"{'query_s':>8}  {'retr_s':>8}  {'rerank_s':>8}  n"
    )
    print(hdr, file=sys.stderr)
    for r in rows:
        def _f(key, w):
            v = r.get(key)
            if v is None:
                return f"{'—':>{w}}"
            return f"{float(v):>{w}.3f}"

        print(
            f"{r['rerank_top_k']:>4}  {_f('accuracy', 7)}  {_f('mean_recall', 7)}  "
            f"{_f('mean_query_s', 8)}  {_f('mean_retrieve_s', 8)}  "
            f"{_f('mean_rerank_s', 8)}  {r.get('n_results') or '—'}",
            file=sys.stderr,
        )


def main() -> int:
    ks = _uniq_ks(RERANK_TOP_KS)
    config_path = str(_as_path(BENCHMARK_CONFIG))
    questions = _as_path(QUESTIONS)
    results_root = _as_path(RESULTS_ROOT)
    base_cfg = BenchmarkConfig.from_yaml(config_path)
    evals_name = base_cfg.eval_results_filename or "evals_complex.json"

    if not SUMMARIZE_ONLY:
        if not questions.is_file():
            print(f"[sweep] 问题集不存在: {questions}", file=sys.stderr)
            return 2
        if not ks:
            print("[sweep] RERANK_TOP_KS 为空", file=sys.stderr)
            return 2

        planned = [(results_root / f"benchmark_result_topk={k}", k) for k in ks]
        print(f"[sweep] config     = {config_path}", file=sys.stderr)
        print(f"[sweep] questions  = {questions}", file=sys.stderr)
        print(f"[sweep] rerank_top_k = {ks}", file=sys.stderr)
        for out_dir, k in planned:
            print(f"[sweep]   k={k} → {out_dir / evals_name}", file=sys.stderr)

        print("[sweep] loading DHMF (once) …", file=sys.stderr)
        bootstrap = TestQueryWorkflow.from_config(config_path)
        dcfg = bootstrap._load_dhmf_config()
        dhmf = bootstrap.setup_dhmf()

        failed = []
        for out_dir, k in planned:
            evals_path = out_dir / evals_name
            if SKIP_EXISTING and evals_path.is_file():
                print(f"[sweep] skip k={k}（已有 {evals_path}）", file=sys.stderr)
                continue
            try:
                _run_one(
                    config_path=config_path,
                    out_dir=out_dir,
                    questions_path=questions,
                    k=k,
                    dcfg=dcfg,
                    dhmf=dhmf,
                )
                print(f"[sweep] done k={k}", file=sys.stderr)
            except KeyboardInterrupt:
                print(f"[sweep] interrupted at k={k}", file=sys.stderr)
                return 130
            except Exception as e:
                print(f"[sweep] failed k={k}: {e}", file=sys.stderr)
                failed.append((k, str(e)))
        if failed:
            print("[sweep] unfinished:", file=sys.stderr)
            for k, msg in failed:
                print(f"  k={k}: {msg}", file=sys.stderr)

    print("[sweep] summarizing all runs in", results_root, file=sys.stderr)
    rows = _discover_runs(results_root, evals_name)
    _print_table(rows)
    if rows:
        _write_summary(rows, results_root)
    print("[sweep] all done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
