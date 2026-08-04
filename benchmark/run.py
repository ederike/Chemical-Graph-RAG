#!/usr/bin/env python3
"""
benchmark CLI 入口。

示例::

    # 使用默认 benchmark/config.yaml
    python -m benchmark.run generate
    python -m benchmark.run evaluate
    python -m benchmark.run all

    # 指定配置文件
    python -m benchmark.run all --bench-config benchmark/config.yaml

    # CLI 临时覆盖跳数 / 路径（覆盖 yaml）
    python -m benchmark.run generate --hop 1:5 --hop 2:3 \\
        --questions-path benchmark/outputs/q.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_hop(s: str) -> tuple:
    """'1:50' / '1=50' → (1, 50)"""
    for sep in (":", "=", ","):
        if sep in s:
            a, b = s.split(sep, 1)
            return int(a.strip()), int(b.strip())
    raise argparse.ArgumentTypeError(
        f"跳数配置格式应为 hop:count，例如 1:50，得到: {s!r}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="多跳问题生成 + DHMF RAG 评测工作流（配置见 benchmark/config.yaml）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "command",
        choices=["generate", "evaluate", "all"],
        help="generate=只生成问题; evaluate=只评测; all=生成+评测",
    )
    p.add_argument(
        "--bench-config",
        default="benchmark/config.yaml",
        help="benchmark 统一配置 yaml（路径/模型/跳数等）",
    )
    # ---- 可选覆盖 paths / dhmf ----
    p.add_argument(
        "--dhmf-config",
        default=None,
        help="覆盖 dhmf.config_path",
    )
    p.add_argument(
        "--db",
        default=None,
        help="覆盖 paths.db_path",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="覆盖 paths.output_dir",
    )
    p.add_argument(
        "--questions-path",
        default=None,
        help="覆盖 paths.questions_path（生成输出 / 评测输入）",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="同 --questions-path / dataset_path，兼容旧参数名",
    )
    p.add_argument(
        "--report-path",
        "--output",
        dest="report_path",
        default=None,
        help="覆盖 paths.report_path",
    )
    p.add_argument(
        "--summary-path",
        default=None,
        help="覆盖 paths.summary_path",
    )
    # ---- 生成 / 评测行为 ----
    p.add_argument(
        "--hop",
        action="append",
        type=_parse_hop,
        default=None,
        metavar="N:COUNT",
        help="覆盖 generate.hop_counts，可重复。如 --hop 1:50 --hop 2:30",
    )
    p.add_argument("--seed", type=int, default=None, help="覆盖 generate.seed")
    p.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="覆盖 generate.max_chars_per_doc（-1=不截断）",
    )
    p.add_argument(
        "--use-cache",
        action="store_true",
        help="出题与评判均启用 LLM 缓存（覆盖 yaml）",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="强制关闭缓存（覆盖 yaml）",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="覆盖每题休眠秒数（生成与评测）",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=None,
        help="覆盖 max_retries（生成与评测）",
    )
    p.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="覆盖 logging.level",
    )
    # 模型快捷覆盖
    p.add_argument(
        "--gen-model",
        default=None,
        help="覆盖 generate.model_args.model",
    )
    p.add_argument(
        "--judge-model",
        default=None,
        help="覆盖 evaluate.model_args.model",
    )
    p.add_argument(
        "--gen-base-url",
        default=None,
        help="覆盖 generate.base_url",
    )
    p.add_argument(
        "--judge-base-url",
        default=None,
        help="覆盖 evaluate.base_url",
    )
    p.add_argument(
        "--enable-doc-recall",
        action="store_true",
        help="开启文档召回率评测（覆盖 yaml）",
    )
    p.add_argument(
        "--disable-doc-recall",
        action="store_true",
        help="关闭文档召回率评测（覆盖 yaml；总结中也不显示）",
    )
    return p


def _cli_overrides(args) -> dict:
    """把 CLI 转成 BenchmarkConfig 扁平字段。"""
    over = {}
    if args.dhmf_config is not None:
        over["dhmf_config_path"] = args.dhmf_config
    if args.db is not None:
        over["db_path"] = args.db
    if args.output_dir is not None:
        over["output_dir"] = args.output_dir

    qpath = args.questions_path or args.dataset
    if qpath is not None:
        over["questions_path"] = qpath
        over["dataset_path"] = qpath
    if args.report_path is not None:
        over["report_path"] = args.report_path
    if args.summary_path is not None:
        over["summary_path"] = args.summary_path

    if args.hop:
        over["hop_counts"] = {h: c for h, c in args.hop}
    if args.seed is not None:
        over["seed"] = args.seed
    if args.max_chars is not None:
        over["max_chars_per_doc"] = args.max_chars
        over["max_source_chars"] = args.max_chars
    if args.sleep is not None:
        over["gen_sleep_between"] = args.sleep
        over["eval_sleep_between"] = args.sleep
    if args.retries is not None:
        over["gen_max_retries"] = args.retries
        over["eval_max_retries"] = args.retries
    if args.log_level is not None:
        over["log_level"] = args.log_level

    if args.no_cache:
        over["gen_use_cache"] = False
        over["eval_use_cache"] = False
    elif args.use_cache:
        over["gen_use_cache"] = True
        over["eval_use_cache"] = True

    if args.gen_base_url is not None:
        over["gen_base_url"] = args.gen_base_url
    if args.judge_base_url is not None:
        over["judge_base_url"] = args.judge_base_url
    if args.gen_model is not None:
        over["gen_model_args"] = {"model": args.gen_model}
    if args.judge_model is not None:
        over["judge_model_args"] = {"model": args.judge_model}

    if args.disable_doc_recall:
        over["enable_doc_recall"] = False
    elif args.enable_doc_recall:
        over["enable_doc_recall"] = True

    return over


def main(argv=None) -> int:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    args = build_parser().parse_args(argv)
    overrides = _cli_overrides(args)

    from benchmark.config import BenchmarkConfig
    from benchmark.workflow import TestQueryWorkflow

    # gen_model_args 若仅含 model，需 merge 进 yaml 的 model_args
    cfg = BenchmarkConfig.from_yaml(args.bench_config)
    # 单独处理 model_args 合并
    if "gen_model_args" in overrides:
        merged = dict(cfg.gen_model_args)
        merged.update(overrides.pop("gen_model_args"))
        overrides["gen_model_args"] = merged
    if "judge_model_args" in overrides:
        merged = dict(cfg.judge_model_args)
        merged.update(overrides.pop("judge_model_args"))
        overrides["judge_model_args"] = merged
    cfg.apply_flat_overrides(overrides)

    wf = TestQueryWorkflow(cfg)

    if args.command == "generate":
        wf.generate_questions(save=True)
        return 0

    if args.command == "evaluate":
        path = cfg.dataset_path or cfg.questions_path
        if not path and not wf._dataset:
            print(
                "evaluate 需要 config.paths.questions_path / dataset_path "
                "或 --questions-path / --dataset",
                file=sys.stderr,
            )
            return 2
        wf.evaluate(save=True)
        return 0

    if args.command == "all":
        wf.run_all()
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
