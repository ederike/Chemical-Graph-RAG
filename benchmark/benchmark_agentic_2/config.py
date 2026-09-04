"""
benchmark_agentic_2 配置加载。

优先级：yaml 显式字段 > 内置默认。
本包自包含，不读取其他评测流的 config。
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .utils import resolve_path

DEFAULT_CONFIG_PATH = "benchmark/benchmark_agentic_2/config.yaml"

_RUN_MODE_ALIASES = {
    "stats": "stats",
    "stat": "stats",
    "statistics": "stats",
    "build_stats": "stats",
    "evaluate": "evaluate",
    "eval": "evaluate",
    "e": "evaluate",
    "rejudge": "rejudge",
    "judge": "rejudge",
    "re-judge": "rejudge",
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
    s = str(mode or "agentic").strip().lower().replace("-", "_")
    if s in ("agentic", "agentic_query", "tool_loop", "react", "loop"):
        return "agentic"
    raise ValueError(
        f"benchmark_agentic_2 only supports query_mode=agentic, got {mode!r}"
    )


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

    base = resolve_path(output_dir or "benchmark/benchmark_agentic_2/results")
    name = (filename or "out.json").strip() or "out.json"
    if "{timestamp}" in name:
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        name = name.replace("{timestamp}", ts)
    return base / name


def _normalize_run_mode(mode: Any) -> str:
    s = str(mode or "all").strip().lower()
    key = _RUN_MODE_ALIASES.get(s)
    if key is None:
        raise ValueError(
            f"Unknown run.mode={mode!r}. "
            "Supported: 'stats' | 'evaluate' | 'rejudge' | 'report' | 'excel' | 'all'"
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
class DatasetSpec:
    id: str
    excel_path: str
    questions_sheet: Optional[str] = None
    stats_sheet: Optional[str] = None
    design_sheet: Optional[str] = None
    enabled: bool = True


@dataclass
class BenchmarkAgentic2Config:
    run_mode: str = "all"

    dhmf_config_path: str = "example/a/config_open.yaml"
    query_mode: str = "agentic"

    datasets: List[DatasetSpec] = field(default_factory=list)
    n_limit: Optional[int] = None
    id_list: Optional[List[str]] = None

    output_dir: str = "benchmark/benchmark_agentic_2/results"

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
    enable_doc_recall: bool = False
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

    # active dataset context (set by workflow when iterating)
    active_dataset_id: Optional[str] = None
    active_excel_path: Optional[str] = None
    active_questions_sheet: Optional[str] = None
    active_stats_sheet: Optional[str] = None
    active_design_sheet: Optional[str] = None

    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path, None] = None,
        *,
        overrides: Optional[dict] = None,
    ) -> "BenchmarkAgentic2Config":
        raw: Dict[str, Any] = {}
        cfg_path: Optional[Path] = None

        if path is not None:
            cfg_path = resolve_path(path)
            if not cfg_path.exists():
                raise FileNotFoundError(f"benchmark_agentic_2 配置不存在: {cfg_path}")
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

        datasets: List[DatasetSpec] = []
        for i, item in enumerate(raw.get("datasets") or []):
            if not isinstance(item, dict):
                continue
            did = _opt_str(item.get("id")) or f"set{i+1}"
            excel = _opt_str(item.get("excel_path"))
            if not excel:
                continue
            datasets.append(
                DatasetSpec(
                    id=did,
                    excel_path=excel,
                    questions_sheet=_opt_str(item.get("questions_sheet")),
                    stats_sheet=_opt_str(item.get("stats_sheet")),
                    design_sheet=_opt_str(item.get("design_sheet")),
                    enabled=_as_bool(item.get("enabled", True), default=True),
                )
            )
        if not datasets:
            # fallback: single excel_path under dataset
            excel = _opt_str(ds.get("excel_path"))
            if excel:
                datasets.append(
                    DatasetSpec(
                        id="default",
                        excel_path=excel,
                        questions_sheet=_opt_str(ds.get("questions_sheet")),
                        stats_sheet=_opt_str(ds.get("stats_sheet")),
                        design_sheet=_opt_str(ds.get("design_sheet")),
                        enabled=True,
                    )
                )

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
            query_mode=_normalize_query_mode(dhmf.get("query_mode") or "agentic"),
            datasets=datasets,
            n_limit=_opt_int(ds.get("n_limit")),
            id_list=_opt_id_list(ds.get("id_list")),
            output_dir=str(
                paths.get("output_dir") or "benchmark/benchmark_agentic_2/results"
            ),
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
            enable_doc_recall=_as_bool(ev.get("enable_doc_recall", False), default=False),
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

    def enabled_datasets(self) -> List[DatasetSpec]:
        return [d for d in self.datasets if d.enabled]

    def bind_dataset(self, spec: DatasetSpec) -> "BenchmarkAgentic2Config":
        self.active_dataset_id = spec.id
        self.active_excel_path = str(resolve_path(spec.excel_path))
        self.active_questions_sheet = spec.questions_sheet
        self.active_stats_sheet = spec.stats_sheet
        self.active_design_sheet = spec.design_sheet
        return self

    def resolve_paths(self) -> "BenchmarkAgentic2Config":
        self.dhmf_config_path = str(resolve_path(self.dhmf_config_path))
        self.output_dir = str(resolve_path(self.output_dir))
        for i, d in enumerate(self.datasets):
            self.datasets[i] = DatasetSpec(
                id=d.id,
                excel_path=str(resolve_path(d.excel_path)),
                questions_sheet=d.questions_sheet,
                stats_sheet=d.stats_sheet,
                design_sheet=d.design_sheet,
                enabled=d.enabled,
            )
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

    def _prefixed_name(self, filename: str) -> str:
        did = self.active_dataset_id
        if not did:
            return filename
        p = Path(filename)
        return f"{did}_{p.name}"

    def stats_file(self) -> Path:
        if self.stats_path and not self.active_dataset_id:
            return resolve_file_path(self.output_dir, "", self.stats_path)
        return resolve_file_path(
            self.output_dir,
            self._prefixed_name(self.stats_filename or "stats.json"),
            None if self.active_dataset_id else self.stats_path,
        )

    def dataset_file(self) -> Path:
        if self.dataset_path and not self.active_dataset_id:
            return resolve_file_path(self.output_dir, "", self.dataset_path)
        return resolve_file_path(
            self.output_dir,
            self._prefixed_name(self.dataset_filename or "dataset.json"),
            None if self.active_dataset_id else self.dataset_path,
        )

    def eval_results_file(self) -> Path:
        if self.eval_results_path and not self.active_dataset_id:
            return resolve_file_path(self.output_dir, "", self.eval_results_path)
        return resolve_file_path(
            self.output_dir,
            self._prefixed_name(self.eval_results_filename or "evals.json"),
            None if self.active_dataset_id else self.eval_results_path,
        )

    def report_source_file(self) -> Path:
        if self.report_source_path and not self.active_dataset_id:
            return resolve_file_path(self.output_dir, "", self.report_source_path)
        fname = _opt_str(self.report_source_filename)
        if fname:
            return resolve_file_path(
                self.output_dir, self._prefixed_name(fname), None
            )
        return self.eval_results_file()

    def report_file(self) -> Path:
        if self.report_path and not self.active_dataset_id:
            return resolve_file_path(self.output_dir, "", self.report_path)
        return resolve_file_path(
            self.output_dir,
            self._prefixed_name(self.report_filename or "report.json"),
            None if self.active_dataset_id else self.report_path,
        )

    def report_excel_file(self) -> Path:
        if self.report_excel_path and not self.active_dataset_id:
            return resolve_file_path(self.output_dir, "", self.report_excel_path)
        fname = _opt_str(self.report_excel_filename)
        if not fname:
            json_name = self.report_filename or "report.json"
            fname = str(Path(json_name).with_suffix(".xlsx"))
        return resolve_file_path(
            self.output_dir,
            self._prefixed_name(fname),
            None if self.active_dataset_id else self.report_excel_path,
        )

    def combined_report_file(self) -> Path:
        return resolve_file_path(self.output_dir, "combined_report.json", None)

    def combined_excel_file(self) -> Path:
        return resolve_file_path(self.output_dir, "combined_report.xlsx", None)

    def to_meta_snapshot(self) -> dict:
        return {
            "config_file": self.config_file,
            "dhmf_config_path": self.dhmf_config_path,
            "query_mode": self.query_mode,
            "active_dataset_id": self.active_dataset_id,
            "excel_path": self.active_excel_path,
            "n_limit": self.n_limit,
            "eval_num_thread": self.eval_num_thread,
            "enable_llm_only": self.enable_llm_only,
            "enable_doc_recall": self.enable_doc_recall,
            "judge_model": (self.judge_model_args or {}).get("model"),
            "llm_only_timeout": self.llm_only_timeout,
            "llm_only_model": (self.llm_only_model_args or {}).get("model"),
            "llm_only_enable_thinking": (self.llm_only_model_args or {}).get(
                "enable_thinking"
            ),
        }
