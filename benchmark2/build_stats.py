#!/usr/bin/env python3
"""
从测试集 Excel 结构化写出统计 JSON。

默认读 benchmark2/config.yaml 的 dataset.excel_path，
写出 paths.stats_filename（并同时写出 dataset.json 供评测复用）。

也可直接指定::

    python -m benchmark2.build_stats
    python -m benchmark2.build_stats --excel benchmark2/建筑涂料Graph-RAG测试问题集1.xlsx \\
        --out benchmark2_results/stats_set1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from benchmark2.config import DEFAULT_CONFIG_PATH, Benchmark2Config
    from benchmark2.dataset import build_stats_document, load_excel_dataset
    from benchmark.utils import resolve_path

    p = argparse.ArgumentParser(description="Excel 测试集 → 结构化统计 JSON")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="yaml 配置路径")
    p.add_argument("--excel", default=None, help="覆盖 dataset.excel_path")
    p.add_argument("--out", default=None, help="统计 JSON 写出路径")
    p.add_argument(
        "--dataset-out",
        default=None,
        help="完整数据集（含题目）写出路径；默认与配置一致",
    )
    p.add_argument("--questions-sheet", default=None)
    p.add_argument("--stats-sheet", default=None)
    args = p.parse_args(argv)

    cfg = Benchmark2Config.from_yaml(args.config).resolve_paths()
    excel = args.excel or cfg.excel_path
    q_sheet = args.questions_sheet or cfg.questions_sheet
    s_sheet = args.stats_sheet or cfg.stats_sheet

    stats = build_stats_document(
        excel_path=excel,
        questions_sheet=q_sheet,
        stats_sheet=s_sheet,
        design_sheet=cfg.design_sheet,
    )
    out = Path(args.out) if args.out else cfg.stats_file()
    if not out.is_absolute():
        out = resolve_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out}", file=sys.stderr)

    dataset = load_excel_dataset(
        excel_path=excel,
        questions_sheet=q_sheet,
        stats_sheet=s_sheet,
        design_sheet=cfg.design_sheet,
        n_limit=cfg.n_limit,
        id_list=cfg.id_list,
    )
    ds_out = Path(args.dataset_out) if args.dataset_out else cfg.dataset_file()
    if not ds_out.is_absolute():
        ds_out = resolve_path(ds_out)
    ds_out.parent.mkdir(parents=True, exist_ok=True)
    ds_out.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[saved] {ds_out}", file=sys.stderr)

    meta = stats.get("meta") or {}
    print(
        f"schema={meta.get('schema')}  n={meta.get('n_questions')}  "
        f"sheet_q={meta.get('questions_sheet')}  sheet_s={meta.get('stats_sheet')}",
        file=sys.stderr,
    )
    for field, block in (stats.get("distributions") or {}).items():
        items = block.get("items") or []
        labels = ", ".join(
            f"{it['label']}={it['count']}" for it in items[:8]
        )
        print(f"  {field}: {labels}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
