#!/usr/bin/env python3
"""从测试集 Excel 结构化写出统计 JSON。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from benchmark.benchmark_agentic_2.config import DEFAULT_CONFIG_PATH, BenchmarkAgentic2Config
    from benchmark.benchmark_agentic_2.dataset import build_stats_document, load_excel_dataset
    from benchmark.benchmark_agentic_2.utils import resolve_path

    p = argparse.ArgumentParser(description="Excel 测试集 → 结构化统计 JSON")
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="yaml 配置路径")
    p.add_argument("--dataset-id", default=None, help="只构建某个 dataset id")
    p.add_argument("--excel", default=None, help="覆盖单个 excel（配合 --out）")
    p.add_argument("--out", default=None, help="统计 JSON 写出路径")
    p.add_argument("--dataset-out", default=None, help="完整数据集写出路径")
    args = p.parse_args(argv)

    cfg = BenchmarkAgentic2Config.from_yaml(args.config).resolve_paths()

    if args.excel:
        stats = build_stats_document(excel_path=args.excel)
        out = Path(args.out) if args.out else Path("stats.json")
        if not out.is_absolute():
            out = resolve_path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {out}", file=sys.stderr)
        if args.dataset_out:
            dataset = load_excel_dataset(excel_path=args.excel)
            ds_out = Path(args.dataset_out)
            if not ds_out.is_absolute():
                ds_out = resolve_path(ds_out)
            ds_out.parent.mkdir(parents=True, exist_ok=True)
            ds_out.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[saved] {ds_out}", file=sys.stderr)
        return 0

    specs = cfg.enabled_datasets()
    if args.dataset_id:
        specs = [s for s in specs if s.id == args.dataset_id]
    if not specs:
        print("no datasets", file=sys.stderr)
        return 2

    for spec in specs:
        cfg.bind_dataset(spec)
        stats = build_stats_document(
            excel_path=spec.excel_path,
            questions_sheet=spec.questions_sheet,
            stats_sheet=spec.stats_sheet,
            design_sheet=spec.design_sheet,
        )
        stats.setdefault("meta", {})["dataset_id"] = spec.id
        out = cfg.stats_file()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {out}", file=sys.stderr)

        dataset = load_excel_dataset(
            excel_path=spec.excel_path,
            questions_sheet=spec.questions_sheet,
            stats_sheet=spec.stats_sheet,
            design_sheet=spec.design_sheet,
            n_limit=cfg.n_limit,
            id_list=cfg.id_list,
        )
        for q in dataset.get("questions") or []:
            q["dataset_id"] = spec.id
        dataset.setdefault("meta", {})["dataset_id"] = spec.id
        ds_out = cfg.dataset_file()
        ds_out.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {ds_out}", file=sys.stderr)
        meta = stats.get("meta") or {}
        print(
            f"[{spec.id}] schema={meta.get('schema')} n={meta.get('n_questions')}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
