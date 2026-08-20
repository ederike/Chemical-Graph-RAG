"""
benchmark2 配置加载。

优先级：yaml 显式字段 > 内置默认。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from benchmark.config import (
    _as_bool,
    _clean_model_args,
    _deep_merge,
    _log_level,
    _normalize_query_mode,
    _opt_bool,
    _opt_str,
    resolve_file_path,
)
from benchmark.utils import resolve_path

DEFAULT_CONFIG_PATH = "benchmark2/config.yaml"

_RUN_MODE_ALIASES = {
    "stats": "stats",
    "stat": "stats",
    "statistics": "stats",
    "build_stats": "stats",
    "evaluate": "evaluate",
    "eval": "evaluate",
    "e": "evaluate",
    "report": "report",
    "summary": "report",
    "summarize": "report",
    "r": "report",
    "excel": "excel",
    "xlsx": "excel",
    "export": "excel",
    "export_excel": "excel",
    "all": "all",
    "both": "all",
    "full": "all",
}


def _normalize_run_mode(mode: Any) -> str:
    s = str(mode or "all").strip().lower()
    key = _RUN_MODE_ALIASES.get(s)
    if key is None:
        raise ValueError(
            f"Unknown run.mode={mode!r}. "
            "Supported: 'stats' | 'evaluate' | 'report' | 'excel' | 'all'"
        )
    return key


def _opt_int(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    s = str(v).strip().lower()
    if not s or s in ("null", "none", "~"):
        return None
    return int(v)


def _opt_id_list(v) -> Optional[List[str]]:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        out = [str(x).strip() for x in v if str(x).strip()]
        return out or None
    s = str(v).strip()
    if not s or s.lower() in ("null", "none", "~"):
        return None
    return [p.strip() for p in s.split(",") if p.strip()] or None


@dataclass
class Benchmark2Config:
    run_mode: str = "all"

    dhmf_config_path: str = "example/a/config_open.yaml"
    query_mode: str = "agent"

    excel_path: str = "benchmark2/建筑涂料行业超图评测测试集2.xlsx"
    questions_sheet: Optional[str] = None
    stats_sheet: Optional[str] = None
    design_sheet: Optional[str] = None
    n_limit: Optional[int] = None
    id_list: Optional[List[str]] = None

    output_dir: str = "benchmark2_results"

    stats_filename: str = "stats.json"
    stats_path: Optional[str] = None

    dataset_filename: str = "dataset.json"
    dataset_path: Optional[str] = None

    eval_results_filename: str = "evals.json"
    eval_results_path: Optional[str] = None

    report_source_filename: Optional[str] = None
    report_source_path: Optional[str] = None

    report_filename: str = "report.json"
    report_path: Optional[str] = None

    report_excel_filename: Optional[str] = None
    report_excel_path: Optional[str] = None

    eval_resume: bool = True
    enable_llm_only: bool = True
    eval_max_retries: int = 3
    eval_sleep_between: float = 0.0
    eval_num_thread: int = 1
    eval_use_cache: bool = False
    dhmf_retrieve_use_cache: Optional[bool] = None
    judge_api_key: Optional[str] = None
    judge_base_url: Optional[str] = None
    judge_timeout: float = 180.0
    judge_model_args: Dict[str, Any] = field(default_factory=dict)

    log_level: int = logging.INFO

    config_file: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path, None] = None,
        *,
        overrides: Optional[dict] = None,
    ) -> "Benchmark2Config":
        raw: Dict[str, Any] = {}
        cfg_path: Optional[Path] = None

        if path is not None:
            cfg_path = resolve_path(path)
            if not cfg_path.exists():
                raise FileNotFoundError(f"benchmark2 配置不存在: {cfg_path}")
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        else:
            default_p = resolve_path(DEFAULT_CONFIG_PATH)
            if default_p.exists():
                cfg_path = default_p
                with open(cfg_path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}

        if overrides:
            raw = _deep_merge(raw, overrides)

        run = raw.get("run") or {}
        dhmf = raw.get("dhmf") or {}
        ds = raw.get("dataset") or {}
        paths = raw.get("paths") or {}
        ev = raw.get("evaluate") or {}
        log = raw.get("logging") or {}

        default_excel = "benchmark2/建筑涂料行业超图评测测试集2.xlsx"

        cfg = cls(
            run_mode=_normalize_run_mode(run.get("mode") or "all"),
            dhmf_config_path=str(
                dhmf.get("config_path") or "example/a/config_open.yaml"
            ),
            query_mode=_normalize_query_mode(dhmf.get("query_mode") or "agent"),
            excel_path=str(ds.get("excel_path") or default_excel),
            questions_sheet=_opt_str(ds.get("questions_sheet")),
            stats_sheet=_opt_str(ds.get("stats_sheet")),
            design_sheet=_opt_str(ds.get("design_sheet")),
            n_limit=_opt_int(ds.get("n_limit")),
            id_list=_opt_id_list(ds.get("id_list")),
            output_dir=str(paths.get("output_dir") or "benchmark2_results"),
            stats_filename=_opt_str(paths.get("stats_filename")) or "stats.json",
            stats_path=_opt_str(paths.get("stats_path")),
            dataset_filename=_opt_str(paths.get("dataset_filename")) or "dataset.json",
            dataset_path=_opt_str(paths.get("dataset_path")),
            eval_results_filename=(
                _opt_str(paths.get("eval_results_filename")) or "evals.json"
            ),
            eval_results_path=_opt_str(paths.get("eval_results_path")),
            report_source_filename=_opt_str(paths.get("report_source_filename")),
            report_source_path=_opt_str(paths.get("report_source_path")),
            report_filename=_opt_str(paths.get("report_filename")) or "report.json",
            report_path=_opt_str(paths.get("report_path")),
            report_excel_filename=_opt_str(paths.get("report_excel_filename")),
            report_excel_path=_opt_str(paths.get("report_excel_path")),
            eval_resume=_as_bool(ev.get("resume", True), default=True),
            enable_llm_only=_as_bool(ev.get("enable_llm_only", True), default=True),
            eval_max_retries=int(ev.get("max_retries", 3)),
            eval_sleep_between=float(ev.get("sleep_between", 0.0)),
            eval_num_thread=max(1, int(ev.get("num_thread", 1) or 1)),
            eval_use_cache=bool(ev.get("use_cache", False)),
            dhmf_retrieve_use_cache=_opt_bool(ev.get("dhmf_retrieve_use_cache")),
            judge_api_key=_opt_str(ev.get("api_key")),
            judge_base_url=_opt_str(ev.get("base_url")),
            judge_timeout=float(ev.get("timeout", 180)),
            judge_model_args=_clean_model_args(ev.get("model_args")),
            log_level=_log_level(log.get("level", "INFO")),
            config_file=str(cfg_path) if cfg_path else None,
            raw=raw,
        )
        return cfg

    def resolve_paths(self) -> "Benchmark2Config":
        self.dhmf_config_path = str(resolve_path(self.dhmf_config_path))
        self.excel_path = str(resolve_path(self.excel_path))
        self.output_dir = str(resolve_path(self.output_dir))
        for attr in (
            "stats_path",
            "dataset_path",
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

    def stats_file(self) -> Path:
        return resolve_file_path(
            self.output_dir,
            self.stats_filename or "stats.json",
            self.stats_path,
        )

    def dataset_file(self) -> Path:
        return resolve_file_path(
            self.output_dir,
            self.dataset_filename or "dataset.json",
            self.dataset_path,
        )

    def eval_results_file(self) -> Path:
        return resolve_file_path(
            self.output_dir,
            self.eval_results_filename or "evals.json",
            self.eval_results_path,
        )

    def report_source_file(self) -> Path:
        if self.report_source_path:
            return resolve_file_path(self.output_dir, "", self.report_source_path)
        fname = _opt_str(self.report_source_filename)
        if fname:
            return resolve_file_path(self.output_dir, fname, None)
        return self.eval_results_file()

    def report_file(self) -> Path:
        return resolve_file_path(
            self.output_dir,
            self.report_filename or "report.json",
            self.report_path,
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
            "query_mode": self.query_mode,
            "excel_path": self.excel_path,
            "n_limit": self.n_limit,
            "eval_num_thread": self.eval_num_thread,
            "enable_llm_only": self.enable_llm_only,
            "judge_model": (self.judge_model_args or {}).get("model"),
        }
