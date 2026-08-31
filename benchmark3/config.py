"""benchmark3 配置加载。"""

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
    parse_llm_only_settings,
    resolve_file_path,
)
from benchmark.utils import resolve_path

DEFAULT_CONFIG_PATH = "benchmark3/config.yaml"

_RUN_MODE_ALIASES = {
    "evaluate": "evaluate",
    "eval": "evaluate",
    "e": "evaluate",
    "rejudge": "rejudge",
    "judge": "rejudge",
    "re-judge": "rejudge",
    "report": "report",
    "summary": "report",
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
            "Supported: 'evaluate' | 'rejudge' | 'report' | 'excel' | 'all'"
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
class Benchmark3Config:
    run_mode: str = "all"

    dhmf_config_path: str = "example/a/config_open.yaml"
    query_mode: str = "agent"

    csv_path: str = "benchmark3/建筑涂料专利集问答测试问题.csv"
    n_limit: Optional[int] = None
    id_list: Optional[List[str]] = None

    output_dir: str = "benchmark3_results"

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
    enable_llm_only: bool = False
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
    ) -> "Benchmark3Config":
        raw: Dict[str, Any] = {}
        cfg_path: Optional[Path] = None

        if path is not None:
            cfg_path = resolve_path(path)
            if not cfg_path.exists():
                raise FileNotFoundError(f"benchmark3 配置不存在: {cfg_path}")
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

        default_csv = "benchmark3/建筑涂料专利集问答测试问题.csv"

        eval_use_cache = bool(ev.get("use_cache", False))
        judge_timeout = float(ev.get("timeout", 180))
        llm_only = parse_llm_only_settings(
            ev, eval_use_cache=eval_use_cache, judge_timeout=judge_timeout
        )

        cfg = cls(
            run_mode=_normalize_run_mode(run.get("mode") or "all"),
            dhmf_config_path=str(
                dhmf.get("config_path") or "example/a/config_open.yaml"
            ),
            query_mode=_normalize_query_mode(dhmf.get("query_mode") or "agent"),
            csv_path=str(ds.get("csv_path") or default_csv),
            n_limit=_opt_int(ds.get("n_limit")),
            id_list=_opt_id_list(ds.get("id_list")),
            output_dir=str(paths.get("output_dir") or "benchmark3_results"),
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
            enable_llm_only=_as_bool(ev.get("enable_llm_only", False), default=False),
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

    def resolve_paths(self) -> "Benchmark3Config":
        self.dhmf_config_path = str(resolve_path(self.dhmf_config_path))
        self.csv_path = str(resolve_path(self.csv_path))
        self.output_dir = str(resolve_path(self.output_dir))
        for attr in (
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
            "csv_path": self.csv_path,
            "n_limit": self.n_limit,
            "id_list": self.id_list,
            "eval_num_thread": self.eval_num_thread,
            "enable_llm_only": self.enable_llm_only,
            "judge_model": (self.judge_model_args or {}).get("model"),
            "llm_only_timeout": self.llm_only_timeout,
            "llm_only_model": (self.llm_only_model_args or {}).get("model"),
            "llm_only_enable_thinking": (self.llm_only_model_args or {}).get(
                "enable_thinking"
            ),
        }
