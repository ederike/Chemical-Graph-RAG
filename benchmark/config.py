"""
benchmark 配置加载与合并。

优先级（高 → 低）：
  CLI 显式参数 > BenchmarkConfig 实例字段覆盖 > config.yaml > 内置默认
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .utils import parse_hop_spec, project_root, resolve_path

DEFAULT_CONFIG_PATH = "benchmark/config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base) if base else {}
    for k, v in (override or {}).items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _clean_model_args(ma: Optional[dict]) -> dict:
    if not ma:
        return {}
    out = dict(ma)
    m = out.get("model")
    if m is None or (isinstance(m, str) and not m.strip()):
        out.pop("model", None)
    return out


def _log_level(name: Any) -> int:
    if isinstance(name, int):
        return name
    s = str(name or "INFO").upper().strip()
    return getattr(logging, s, logging.INFO)


def _opt_str(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("null", "none", "~"):
        return None
    return s


def _opt_bool(v) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("null", "none", ""):
        return None
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(v)


def _as_bool(v, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(v)


def _normalize_query_mode(mode: Any) -> str:
    s = str(mode or "dual_path").strip().lower().replace("-", "_")
    if s in ("agent", "agent_query", "multi_hop", "multihop"):
        return "agent"
    if s in ("dual_path", "dualpath", "query", "rag"):
        return "dual_path"
    raise ValueError(
        f"Unknown query_mode={mode!r}. Supported: 'dual_path' | 'agent'"
    )


def _normalize_run_mode(mode: Any) -> str:
    s = str(mode or "all").strip().lower()
    if s in ("generate", "gen", "g"):
        return "generate"
    if s in ("evaluate", "eval", "e"):
        return "evaluate"
    if s in ("report", "summary", "summarize", "r"):
        return "report"
    if s in ("all", "both", "full"):
        return "all"
    raise ValueError(
        f"Unknown run.mode={mode!r}. "
        f"Supported: 'generate' | 'evaluate' | 'report' | 'all'"
    )


def resolve_file_path(
    output_dir: str,
    filename: str,
    custom_path: Optional[str] = None,
    *,
    timestamp: Optional[str] = None,
) -> Path:
    """
    路径规则（简洁）：
      - custom_path 非空 → 用该相对/绝对路径（相对项目根解析）
      - 否则 → output_dir / filename
    filename 中的 {timestamp} 可替换。
    """
    custom = _opt_str(custom_path)
    if custom:
        name = custom
        if "{timestamp}" in name:
            ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
            name = name.replace("{timestamp}", ts)
        p = Path(name)
        return p if p.is_absolute() else resolve_path(p)

    base = resolve_path(output_dir or "benchmark_results")
    name = (filename or "out.json").strip() or "out.json"
    if "{timestamp}" in name:
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        name = name.replace("{timestamp}", ts)
    return base / name


@dataclass
class BenchmarkConfig:
    """工作流全部可配置项的扁平视图。"""

    # run：脚本入口按此模式执行（不使用 CLI 传参）
    run_mode: str = "all"  # generate | evaluate | report | all

    # dhmf
    dhmf_config_path: str = "example/a/config_open.yaml"
    query_mode: str = "dual_path"

    # paths：默认目录 + 默认文件名；*_path 非空则覆盖
    db_path: str = "example/a/DB/main.db"
    output_dir: str = "benchmark_results"

    questions_filename: str = "questions.json"
    questions_path: Optional[str] = None  # 覆盖生成写出路径

    eval_questions_filename: Optional[str] = None  # 空 = questions_filename
    eval_questions_path: Optional[str] = None  # 覆盖评测读取路径

    # evaluate 写出：逐题详细结果（可增量）
    eval_results_filename: str = "eval_results.json"
    eval_results_path: Optional[str] = None

    # report 读入：默认 = eval_results；可单独指定其它评测文件
    report_source_filename: Optional[str] = None
    report_source_path: Optional[str] = None

    # report 写出：汇总统计 JSON
    report_filename: str = "report.json"
    report_path: Optional[str] = None

    # generate
    hop_counts: Dict[int, int] = field(default_factory=lambda: {1: 5, 2: 3, 3: 2})
    seed: int = 42
    max_chars_per_doc: int = 4000
    gen_max_retries: int = 3
    gen_sleep_between: float = 0.0
    gen_num_thread: int = 4  # 出题并行线程数；1 = 串行
    gen_use_cache: bool = False
    gen_api_key: Optional[str] = None
    gen_base_url: Optional[str] = None
    gen_timeout: float = 180.0
    gen_model_args: Dict[str, Any] = field(default_factory=dict)

    # evaluate
    enable_doc_recall: bool = True
    max_source_chars: int = -1
    eval_max_retries: int = 3
    eval_sleep_between: float = 0.0
    eval_use_cache: bool = False
    dhmf_retrieve_use_cache: Optional[bool] = None
    judge_api_key: Optional[str] = None
    judge_base_url: Optional[str] = None
    judge_timeout: float = 180.0
    judge_model_args: Dict[str, Any] = field(default_factory=dict)

    # report
    report_print_table: bool = True  # 是否在终端打印文本统计表

    # logging
    log_level: int = logging.INFO

    config_file: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path, None] = None,
        *,
        overrides: Optional[dict] = None,
    ) -> "BenchmarkConfig":
        raw: Dict[str, Any] = {}
        cfg_path: Optional[Path] = None

        if path is not None:
            cfg_path = resolve_path(path)
            if not cfg_path.exists():
                raise FileNotFoundError(f"benchmark 配置不存在: {cfg_path}")
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        else:
            default_p = resolve_path(DEFAULT_CONFIG_PATH)
            if default_p.exists():
                cfg_path = default_p
                with open(cfg_path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}

        if overrides and any(
            k in overrides
            for k in ("run", "dhmf", "paths", "generate", "evaluate", "logging")
        ):
            raw = _deep_merge(raw, overrides)

        run = raw.get("run") or {}
        dhmf = raw.get("dhmf") or {}
        paths = raw.get("paths") or {}
        gen = raw.get("generate") or {}
        ev = raw.get("evaluate") or {}
        log = raw.get("logging") or {}

        hop_counts = parse_hop_spec(gen.get("hop_counts") or {1: 5, 2: 3, 3: 2})
        rep = raw.get("report") or {}

        q_name = _opt_str(paths.get("questions_filename")) or "questions.json"

        # 路径兼容：
        #   新：eval_results_filename（评测明细）+ report_filename（汇总）
        #   旧：仅 report_filename → 当作评测明细；汇总默认 {stem}.report.json
        eval_results_name = _opt_str(paths.get("eval_results_filename"))
        report_out_name = _opt_str(paths.get("report_filename"))
        legacy_only_report = eval_results_name is None and report_out_name is not None
        if eval_results_name is None:
            # 兼容旧字段：report_filename / 默认 eval_results.json
            eval_results_name = report_out_name or "eval_results.json"
        if legacy_only_report:
            # 旧配置把 report_filename 当评测明细
            stem = Path(eval_results_name).stem
            report_out_name = f"{stem}.report.json"
        elif report_out_name is None:
            report_out_name = "report.json"

        cfg = cls(
            run_mode=_normalize_run_mode(run.get("mode") or "all"),
            dhmf_config_path=str(dhmf.get("config_path") or "example/a/config_open.yaml"),
            query_mode=_normalize_query_mode(dhmf.get("query_mode") or "dual_path"),
            db_path=str(paths.get("db_path") or "example/a/DB/main.db"),
            output_dir=str(paths.get("output_dir") or "benchmark_results"),
            questions_filename=q_name,
            questions_path=_opt_str(paths.get("questions_path")),
            eval_questions_filename=_opt_str(paths.get("eval_questions_filename")),
            eval_questions_path=_opt_str(paths.get("eval_questions_path")),
            eval_results_filename=eval_results_name,
            eval_results_path=_opt_str(
                paths.get("eval_results_path")
                # 旧配置把 report_path 当评测明细路径
                or (paths.get("report_path") if legacy_only_report else None)
            ),
            report_source_filename=_opt_str(paths.get("report_source_filename")),
            report_source_path=_opt_str(paths.get("report_source_path")),
            report_filename=report_out_name,
            report_path=_opt_str(
                None if legacy_only_report else paths.get("report_path")
            ),
            hop_counts=hop_counts,
            seed=int(gen.get("seed", 42)),
            max_chars_per_doc=int(gen.get("max_chars_per_doc", 4000)),
            gen_max_retries=int(gen.get("max_retries", 3)),
            gen_sleep_between=float(gen.get("sleep_between", 0.0)),
            gen_num_thread=max(1, int(gen.get("num_thread", gen.get("num_threads", 4)) or 4)),
            gen_use_cache=bool(gen.get("use_cache", False)),
            gen_api_key=_opt_str(gen.get("api_key")),
            gen_base_url=_opt_str(gen.get("base_url")),
            gen_timeout=float(gen.get("timeout", 180)),
            gen_model_args=_clean_model_args(gen.get("model_args")),
            enable_doc_recall=_as_bool(ev.get("enable_doc_recall", True), default=True),
            max_source_chars=int(
                ev.get("max_source_chars", gen.get("max_chars_per_doc", -1))
            ),
            eval_max_retries=int(ev.get("max_retries", 3)),
            eval_sleep_between=float(ev.get("sleep_between", 0.0)),
            eval_use_cache=bool(ev.get("use_cache", False)),
            dhmf_retrieve_use_cache=_opt_bool(ev.get("dhmf_retrieve_use_cache")),
            judge_api_key=_opt_str(ev.get("api_key")),
            judge_base_url=_opt_str(ev.get("base_url")),
            judge_timeout=float(ev.get("timeout", 180)),
            judge_model_args=_clean_model_args(ev.get("model_args")),
            report_print_table=_as_bool(
                rep.get("print_table", ev.get("print_table", True)),
                default=True,
            ),
            log_level=_log_level(log.get("level", "INFO")),
            config_file=str(cfg_path) if cfg_path else None,
            raw=raw,
        )
        return cfg

    def resolve_paths(self) -> "BenchmarkConfig":
        """解析固定路径字段为绝对路径；filename 保持原样。"""
        self.dhmf_config_path = str(resolve_path(self.dhmf_config_path))
        self.db_path = str(resolve_path(self.db_path))
        self.output_dir = str(resolve_path(self.output_dir))
        for attr in (
            "questions_path",
            "eval_questions_path",
            "eval_results_path",
            "report_source_path",
            "report_path",
        ):
            val = getattr(self, attr, None)
            if val:
                p = Path(val)
                if not p.is_absolute():
                    setattr(self, attr, str(resolve_path(p)))
        return self

    # ------------------------------------------------------------------
    # 文件路径：questions / eval_results / report
    # ------------------------------------------------------------------
    def questions_file(self) -> Path:
        """生成问题集写出路径。"""
        return resolve_file_path(
            self.output_dir,
            self.questions_filename or "questions.json",
            self.questions_path,
        )

    def eval_questions_file(self) -> Path:
        """评测读取的问题集路径。"""
        fname = (
            _opt_str(self.eval_questions_filename)
            or self.questions_filename
            or "questions.json"
        )
        return resolve_file_path(
            self.output_dir,
            fname,
            self.eval_questions_path,
        )

    def eval_results_file(self, *, timestamp: Optional[str] = None) -> Path:
        """evaluate 写出：逐题详细结果。"""
        return resolve_file_path(
            self.output_dir,
            self.eval_results_filename or "eval_results.json",
            self.eval_results_path,
            timestamp=timestamp,
        )

    def report_source_file(self) -> Path:
        """report 读入的评测结果路径（默认 = eval_results）。"""
        if self.report_source_path:
            return resolve_file_path(self.output_dir, "", self.report_source_path)
        fname = _opt_str(self.report_source_filename)
        if fname:
            return resolve_file_path(self.output_dir, fname, None)
        return self.eval_results_file()

    def report_file(self, *, timestamp: Optional[str] = None) -> Path:
        """report 写出：汇总统计 JSON。"""
        return resolve_file_path(
            self.output_dir,
            self.report_filename or "report.json",
            self.report_path,
            timestamp=timestamp,
        )

    def to_dict(self) -> dict:
        return {
            "config_file": self.config_file,
            "run_mode": self.run_mode,
            "dhmf_config_path": self.dhmf_config_path,
            "query_mode": self.query_mode,
            "db_path": self.db_path,
            "output_dir": self.output_dir,
            "questions_filename": self.questions_filename,
            "questions_path": self.questions_path,
            "eval_questions_filename": self.eval_questions_filename,
            "eval_questions_path": self.eval_questions_path,
            "eval_results_filename": self.eval_results_filename,
            "eval_results_path": self.eval_results_path,
            "report_source_filename": self.report_source_filename,
            "report_source_path": self.report_source_path,
            "report_filename": self.report_filename,
            "report_path": self.report_path,
            "hop_counts": {str(k): v for k, v in self.hop_counts.items()},
            "seed": self.seed,
            "max_chars_per_doc": self.max_chars_per_doc,
            "enable_doc_recall": self.enable_doc_recall,
            "gen_model_args": self.gen_model_args,
            "judge_model_args": self.judge_model_args,
            "gen_api_key_set": bool(self.gen_api_key),
            "gen_base_url": self.gen_base_url,
            "judge_api_key_set": bool(self.judge_api_key),
            "judge_base_url": self.judge_base_url,
        }
