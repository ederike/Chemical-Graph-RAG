"""
benchmark_agentic_4 配置加载。

优先级：config.yaml 内字段；本包默认路径为本目录 config.yaml。
不读取、不引用 useless/ 或其他 benchmark 包。
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .utils import (
    parse_doc_id_range,
    parse_type_counts,
    resolve_path,
)

DEFAULT_CONFIG_PATH = "benchmark/benchmark_agentic_4/config.yaml"

QUESTION_TYPES = (
    "product_to_formula",
    "application_to_formula",
    "formula_optimize",
    "ingredient_to_formula_and_others",
    "ingredient_interchange",
)


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


def _opt_int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    return n


def merge_llm_only_model_args(
    retrieve_model_args: Optional[dict],
    override: Optional[dict] = None,
) -> dict:
    base = dict(retrieve_model_args or {})
    over = dict(override or {})
    thinking_set = "enable_thinking" in over
    if not over.get("model"):
        over.pop("model", None)
    merged = {**base, **over}
    merged.pop("response_format", None)
    if not thinking_set:
        merged["enable_thinking"] = False
    merged.setdefault("temperature", 0.2)
    return merged


def parse_llm_only_settings(
    ev: Optional[dict],
    *,
    eval_use_cache: bool,
    judge_timeout: float,
) -> dict:
    ev = ev or {}
    lo = ev.get("llm_only")
    if not isinstance(lo, dict):
        lo = {}
    timeout_raw = lo.get("timeout")
    if timeout_raw is None:
        timeout = float(judge_timeout)
    else:
        timeout = float(timeout_raw)
    cache = _opt_bool(lo.get("use_cache"))
    return {
        "llm_only_api_key": _opt_str(lo.get("api_key")),
        "llm_only_base_url": _opt_str(lo.get("base_url")),
        "llm_only_timeout": timeout,
        "llm_only_max_retries": max(1, int(lo.get("max_retries", 3) or 3)),
        "llm_only_use_cache": eval_use_cache if cache is None else cache,
        "llm_only_model_args": _clean_model_args(lo.get("model_args")),
    }


def _normalize_run_mode(mode: Any) -> str:
    s = str(mode or "generate").strip().lower()
    if s in ("generate", "gen", "g"):
        return "generate"
    if s in ("evaluate", "eval", "e"):
        return "evaluate"
    if s in ("report", "summary", "summarize", "r"):
        return "report"
    if s in ("excel", "xlsx", "export", "export_excel"):
        return "excel"
    if s in ("all", "both", "full"):
        return "all"
    raise ValueError(
        f"Unknown run.mode={mode!r}. "
        f"Supported: 'generate' | 'evaluate' | 'report' | 'excel' | 'all'"
    )


def resolve_file_path(
    output_dir: str,
    filename: str,
    custom_path: Optional[str] = None,
    *,
    timestamp: Optional[str] = None,
) -> Path:
    custom = _opt_str(custom_path)
    if custom:
        name = custom
        if "{timestamp}" in name:
            ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
            name = name.replace("{timestamp}", ts)
        p = Path(name)
        return p if p.is_absolute() else resolve_path(p)

    base = resolve_path(output_dir or "benchmark/benchmark_agentic_4/results")
    name = (filename or "out.json").strip() or "out.json"
    if "{timestamp}" in name:
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        name = name.replace("{timestamp}", ts)
    return base / name


@dataclass
class BenchmarkConfig:
    run_mode: str = "generate"

    dhmf_config_path: str = "example/a/config_open.yaml"
    query_mode: str = "agentic"

    db_path: str = "example/a/DB/main.db"
    output_dir: str = "benchmark/benchmark_agentic_4/results"

    questions_filename: str = "questions.json"
    questions_path: Optional[str] = None
    eval_questions_filename: Optional[str] = None
    eval_questions_path: Optional[str] = None
    eval_results_filename: str = "evals.json"
    eval_results_path: Optional[str] = None
    report_source_filename: Optional[str] = None
    report_source_path: Optional[str] = None
    report_filename: str = "report.json"
    report_path: Optional[str] = None
    report_excel_filename: Optional[str] = None
    report_excel_path: Optional[str] = None

    type_counts: Dict[str, int] = field(
        default_factory=lambda: {
            "product_to_formula": 10,
            "application_to_formula": 10,
        }
    )
    seed: int = 42
    max_chars_per_doc: int = 12000
    n_limit: Optional[int] = None
    gen_resume: bool = True
    # 对外配置只用 doc_id_range；min/max 为内部 lo/hi（由 from_yaml 解析填充）
    doc_id_range: Optional[object] = None
    doc_id_min: Optional[int] = None
    doc_id_max: Optional[int] = None
    gen_max_retries: int = 3
    gen_sleep_between: float = 0.0
    gen_num_thread: int = 4
    gen_use_cache: bool = False
    gen_api_key: Optional[str] = None
    gen_base_url: Optional[str] = None
    gen_timeout: float = 180.0
    gen_model_args: Dict[str, Any] = field(default_factory=dict)

    eval_resume: bool = True
    enable_llm_only: bool = True
    enable_doc_recall: bool = True
    max_source_chars: int = -1
    eval_max_retries: int = 3
    eval_sleep_between: float = 0.0
    eval_num_thread: int = 1
    eval_use_cache: bool = False
    dhmf_retrieve_use_cache: Optional[bool] = None
    judge_api_key: Optional[str] = None
    judge_base_url: Optional[str] = None
    judge_timeout: float = 180.0
    judge_model_args: Dict[str, Any] = field(default_factory=dict)
    llm_only_api_key: Optional[str] = None
    llm_only_base_url: Optional[str] = None
    llm_only_timeout: float = 180.0
    llm_only_max_retries: int = 3
    llm_only_use_cache: bool = False
    llm_only_model_args: Dict[str, Any] = field(default_factory=dict)

    log_level: int = logging.INFO
    config_file: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

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
                raise FileNotFoundError(f"benchmark_agentic_4 配置不存在: {cfg_path}")
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

        type_counts = parse_type_counts(
            gen.get("type_counts")
            or {
                "product_to_formula": 10,
                "application_to_formula": 10,
            }
        )
        doc_id_range_raw = gen.get("doc_id_range")
        range_lo, range_hi = parse_doc_id_range(doc_id_range_raw)

        q_name = _opt_str(paths.get("questions_filename")) or "questions.json"
        eval_results_name = (
            _opt_str(paths.get("eval_results_filename")) or "evals.json"
        )
        report_out_name = _opt_str(paths.get("report_filename")) or "report.json"

        eval_use_cache = bool(ev.get("use_cache", False))
        judge_timeout = float(ev.get("timeout", 180))
        llm_only = parse_llm_only_settings(
            ev, eval_use_cache=eval_use_cache, judge_timeout=judge_timeout
        )

        query_mode = str(dhmf.get("query_mode") or "agentic").strip().lower()
        if query_mode not in ("agentic", "agentic_query"):
            raise ValueError(
                f"benchmark_agentic_4 只支持 agentic_query，"
                f"得到 dhmf.query_mode={dhmf.get('query_mode')!r}"
            )

        n_limit = gen.get("n_limit")
        if n_limit is not None and str(n_limit).strip() != "":
            try:
                n_limit_i = int(n_limit)
            except (TypeError, ValueError):
                n_limit_i = None
            if n_limit_i is not None and n_limit_i <= 0:
                n_limit_i = None
        else:
            n_limit_i = None

        cfg = cls(
            run_mode=_normalize_run_mode(run.get("mode") or "generate"),
            dhmf_config_path=str(dhmf.get("config_path") or "example/a/config_open.yaml"),
            query_mode="agentic",
            db_path=str(paths.get("db_path") or "example/a/DB/main.db"),
            output_dir=str(
                paths.get("output_dir") or "benchmark/benchmark_agentic_4/results"
            ),
            questions_filename=q_name,
            questions_path=_opt_str(paths.get("questions_path")),
            eval_questions_filename=_opt_str(paths.get("eval_questions_filename")),
            eval_questions_path=_opt_str(paths.get("eval_questions_path")),
            eval_results_filename=eval_results_name,
            eval_results_path=_opt_str(paths.get("eval_results_path")),
            report_source_filename=_opt_str(paths.get("report_source_filename")),
            report_source_path=_opt_str(paths.get("report_source_path")),
            report_filename=report_out_name,
            report_path=_opt_str(paths.get("report_path")),
            report_excel_filename=_opt_str(paths.get("report_excel_filename")),
            report_excel_path=_opt_str(paths.get("report_excel_path")),
            type_counts=type_counts,
            seed=int(gen.get("seed", 42)),
            max_chars_per_doc=int(gen.get("max_chars_per_doc", 12000)),
            n_limit=n_limit_i,
            gen_resume=_as_bool(gen.get("resume", True), default=True),
            doc_id_range=doc_id_range_raw,
            doc_id_min=range_lo,
            doc_id_max=range_hi,
            gen_max_retries=int(gen.get("max_retries", 3)),
            gen_sleep_between=float(gen.get("sleep_between", 0.0)),
            gen_num_thread=max(1, int(gen.get("num_thread", 4) or 4)),
            gen_use_cache=bool(gen.get("use_cache", False)),
            gen_api_key=_opt_str(gen.get("api_key")),
            gen_base_url=_opt_str(gen.get("base_url")),
            gen_timeout=float(gen.get("timeout", 180)),
            gen_model_args=_clean_model_args(gen.get("model_args")),
            eval_resume=_as_bool(ev.get("resume", True), default=True),
            enable_llm_only=_as_bool(ev.get("enable_llm_only", True), default=True),
            enable_doc_recall=_as_bool(ev.get("enable_doc_recall", True), default=True),
            max_source_chars=int(
                ev.get("max_source_chars", gen.get("max_chars_per_doc", -1))
            ),
            eval_max_retries=int(ev.get("max_retries", 3)),
            eval_sleep_between=float(ev.get("sleep_between", 0.0)),
            eval_num_thread=max(1, int(ev.get("num_thread", 1) or 1)),
            eval_use_cache=eval_use_cache,
            dhmf_retrieve_use_cache=_opt_bool(ev.get("dhmf_retrieve_use_cache")),
            judge_api_key=_opt_str(ev.get("api_key")),
            judge_base_url=_opt_str(ev.get("base_url")),
            judge_timeout=judge_timeout,
            judge_model_args=_clean_model_args(ev.get("model_args")),
            llm_only_api_key=llm_only["llm_only_api_key"],
            llm_only_base_url=llm_only["llm_only_base_url"],
            llm_only_timeout=llm_only["llm_only_timeout"],
            llm_only_max_retries=llm_only["llm_only_max_retries"],
            llm_only_use_cache=llm_only["llm_only_use_cache"],
            llm_only_model_args=llm_only["llm_only_model_args"],
            log_level=_log_level(log.get("level", "INFO")),
            config_file=str(cfg_path) if cfg_path else None,
            raw=raw,
        )
        return cfg

    def resolve_paths(self) -> "BenchmarkConfig":
        self.dhmf_config_path = str(resolve_path(self.dhmf_config_path))
        self.db_path = str(resolve_path(self.db_path))
        self.output_dir = str(resolve_path(self.output_dir))
        for attr in (
            "questions_path",
            "eval_questions_path",
            "eval_results_path",
            "report_source_path",
            "report_path",
            "report_excel_path",
        ):
            val = getattr(self, attr, None)
            if val:
                p = Path(val)
                if not p.is_absolute():
                    setattr(self, attr, str(resolve_path(p)))
        return self

    def questions_file(self) -> Path:
        return resolve_file_path(
            self.output_dir,
            self.questions_filename or "questions.json",
            self.questions_path,
        )

    def eval_questions_file(self) -> Path:
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
        return resolve_file_path(
            self.output_dir,
            self.eval_results_filename or "evals.json",
            self.eval_results_path,
            timestamp=timestamp,
        )

    def report_source_file(self) -> Path:
        if self.report_source_path:
            return resolve_file_path(self.output_dir, "", self.report_source_path)
        fname = _opt_str(self.report_source_filename)
        if fname:
            return resolve_file_path(self.output_dir, fname, None)
        return self.eval_results_file()

    def report_file(self, *, timestamp: Optional[str] = None) -> Path:
        return resolve_file_path(
            self.output_dir,
            self.report_filename or "report.json",
            self.report_path,
            timestamp=timestamp,
        )

    def report_excel_file(self) -> Path:
        if self.report_excel_path:
            return resolve_file_path(self.output_dir, "", self.report_excel_path)
        fname = _opt_str(self.report_excel_filename)
        if not fname:
            json_name = self.report_filename or "report.json"
            fname = str(Path(json_name).with_suffix(".xlsx"))
        return resolve_file_path(self.output_dir, fname, None)

    def to_meta_snapshot(self) -> dict:
        return {
            "config_file": self.config_file,
            "dhmf_config_path": self.dhmf_config_path,
            "query_mode": "agentic",
            "db_path": self.db_path,
            "type_counts": dict(self.type_counts),
            "seed": self.seed,
            "doc_id_range": self.doc_id_range,
            "n_limit": self.n_limit,
            "enable_doc_recall": self.enable_doc_recall,
            "enable_llm_only": self.enable_llm_only,
            "eval_num_thread": self.eval_num_thread,
            "gen_model": (self.gen_model_args or {}).get("model"),
            "judge_model": (self.judge_model_args or {}).get("model"),
            "llm_only_timeout": self.llm_only_timeout,
            "llm_only_model": (self.llm_only_model_args or {}).get("model"),
            "llm_only_enable_thinking": (self.llm_only_model_args or {}).get(
                "enable_thinking"
            ),
        }
