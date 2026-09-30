"""
benchmark_agentic_5 配置加载。

优先级：config.yaml 内字段；本包默认路径为本目录 config.yaml。
不读取、不引用 useless/ 下任何配置。
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .utils import parse_doc_id_range, parse_hop_spec, resolve_path

DEFAULT_CONFIG_PATH = "benchmark/benchmark_agentic_5/config.yaml"

_QUESTION_GEN_PROMPT_ALIASES = {
    "default": "QUESTION_GEN_USER",
    "standard": "QUESTION_GEN_USER",
    "user": "QUESTION_GEN_USER",
    "fuzzy": "QUESTION_GEN_Fuzzy_Matching",
    "fuzzy_matching": "QUESTION_GEN_Fuzzy_Matching",
}


def _normalize_question_gen_prompt(name: Any) -> str:
    from .prompts import Benchmark_PROMPT

    raw = str(name or "QUESTION_GEN_USER").strip()
    if not raw:
        raw = "QUESTION_GEN_USER"

    def _user_prompt_keys():
        return sorted(
            k
            for k in Benchmark_PROMPT
            if k.startswith("QUESTION_GEN") and k != "QUESTION_GEN_SYSTEM"
        )

    if raw in Benchmark_PROMPT and raw != "QUESTION_GEN_SYSTEM" and raw.startswith(
        "QUESTION_GEN"
    ):
        return raw

    key = _QUESTION_GEN_PROMPT_ALIASES.get(raw.lower().replace("-", "_"))
    if key is not None:
        return key

    lower_map = {k.lower(): k for k in _user_prompt_keys()}
    hit = lower_map.get(raw.lower())
    if hit is not None:
        return hit

    raise ValueError(
        f"Unknown generate.question_gen_prompt={name!r}. "
        f"Supported keys: {_user_prompt_keys()}; "
        f"aliases: default/standard/user, fuzzy/fuzzy_matching"
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


def _looks_like_example_a(path: Any) -> bool:
    s = str(path or "").replace("\\", "/")
    return "/example/a/" in f"/{s.strip('/')}/" or s.endswith("/example/a") or "example/a/" in s


def assert_example_b_workspace(*paths: Any) -> None:
    """本包只允许 example/b，碰到 example/a 直接拒绝。"""
    for p in paths:
        if _looks_like_example_a(p):
            raise ValueError(
                f"benchmark_agentic_5 禁止引用 example/a（得到 {p!r}）。"
                "请把 dhmf.config_path / paths.db_path 指到 example/b。"
            )


def parse_weknora_settings(raw: Optional[dict]) -> dict:
    wk = raw if isinstance(raw, dict) else {}
    chat_mode = str(wk.get("chat_mode") or "agent").strip().lower()
    if chat_mode in ("agent-chat", "agent_chat", "quick-answer", "quick"):
        chat_mode = "agent"
    elif chat_mode in ("knowledge-chat", "knowledge_chat", "qa"):
        chat_mode = "knowledge"
    return {
        "weknora_base_url": _opt_str(wk.get("base_url"))
        or "http://192.168.110.219:8081",
        "weknora_email": _opt_str(wk.get("email")),
        "weknora_password": wk.get("password") if wk.get("password") is not None else None,
        "weknora_knowledge_base_id": _opt_str(wk.get("knowledge_base_id")),
        "weknora_knowledge_base_name": _opt_str(wk.get("knowledge_base_name")),
        "weknora_chat_mode": chat_mode or "agent",
        "weknora_agent_id": _opt_str(wk.get("agent_id")) or "builtin-quick-answer",
        "weknora_agent_enabled": _as_bool(wk.get("agent_enabled", True), default=True),
        "weknora_timeout": float(wk.get("timeout", 300) or 300),
        "weknora_max_retries": max(1, int(wk.get("max_retries", 2) or 2)),
        "weknora_new_session_per_question": _as_bool(
            wk.get("new_session_per_question", True), default=True
        ),
        "weknora_delete_session": _as_bool(wk.get("delete_session", True), default=True),
        "weknora_session_title_prefix": _opt_str(wk.get("session_title_prefix"))
        or "benchmark_agentic_5",
    }


def _normalize_run_mode(mode: Any) -> str:
    s = str(mode or "all").strip().lower()
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

    base = resolve_path(output_dir or "results/benchmark_agentic_5")
    name = (filename or "out.json").strip() or "out.json"
    if "{timestamp}" in name:
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        name = name.replace("{timestamp}", ts)
    return base / name


@dataclass
class BenchmarkConfig:
    run_mode: str = "all"

    dhmf_config_path: str = "example/b/config_open.yaml"
    query_mode: str = "agentic"

    db_path: str = "example/b/DB/main.db"
    output_dir: str = "results/benchmark_agentic_5"

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

    hop_counts: Dict[int, int] = field(default_factory=lambda: {1: 10, 2: 50})
    seed: int = 42
    max_chars_per_doc: int = -1
    doc_id_min: Optional[int] = None
    doc_id_max: Optional[int] = None
    original_pdf_only: bool = True
    gen_max_retries: int = 3
    gen_sleep_between: float = 0.0
    gen_num_thread: int = 4
    gen_use_cache: bool = False
    question_gen_prompt: str = "QUESTION_GEN_USER"
    gen_api_key: Optional[str] = None
    gen_base_url: Optional[str] = None
    gen_timeout: float = 180.0
    gen_model_args: Dict[str, Any] = field(default_factory=dict)

    enable_weknora: bool = True
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

    weknora_base_url: str = "http://192.168.110.219:8081"
    weknora_email: Optional[str] = None
    weknora_password: Optional[str] = None
    weknora_knowledge_base_id: Optional[str] = None
    weknora_knowledge_base_name: Optional[str] = None
    weknora_chat_mode: str = "agent"
    weknora_agent_id: str = "builtin-quick-answer"
    weknora_agent_enabled: bool = True
    weknora_timeout: float = 300.0
    weknora_max_retries: int = 2
    weknora_new_session_per_question: bool = True
    weknora_delete_session: bool = True
    weknora_session_title_prefix: str = "benchmark_agentic_5"

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
                raise FileNotFoundError(f"benchmark_agentic_5 配置不存在: {cfg_path}")
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
            for k in (
                "run", "dhmf", "paths", "generate", "evaluate", "weknora", "logging"
            )
        ):
            raw = _deep_merge(raw, overrides)

        run = raw.get("run") or {}
        dhmf = raw.get("dhmf") or {}
        paths = raw.get("paths") or {}
        gen = raw.get("generate") or {}
        ev = raw.get("evaluate") or {}
        wk_raw = raw.get("weknora") or {}
        log = raw.get("logging") or {}

        hop_counts = parse_hop_spec(gen.get("hop_counts") or {1: 10, 2: 50})

        range_lo, range_hi = parse_doc_id_range(gen.get("doc_id_range"))
        doc_id_min = range_lo if range_lo is not None else _opt_int(gen.get("doc_id_min"))
        doc_id_max = range_hi if range_hi is not None else _opt_int(gen.get("doc_id_max"))

        q_name = _opt_str(paths.get("questions_filename")) or "questions.json"
        eval_results_name = (
            _opt_str(paths.get("eval_results_filename")) or "evals.json"
        )
        report_out_name = _opt_str(paths.get("report_filename")) or "report.json"

        eval_use_cache = bool(ev.get("use_cache", False))
        judge_timeout = float(ev.get("timeout", 180))
        weknora = parse_weknora_settings(wk_raw)

        query_mode = str(dhmf.get("query_mode") or "agentic").strip().lower()
        if query_mode not in ("agentic", "agentic_query"):
            raise ValueError(
                f"benchmark_agentic_5 只支持 agentic_query，"
                f"得到 dhmf.query_mode={dhmf.get('query_mode')!r}"
            )

        dhmf_config_path = str(dhmf.get("config_path") or "example/b/config_open.yaml")
        db_path = str(paths.get("db_path") or "example/b/DB/main.db")
        assert_example_b_workspace(dhmf_config_path, db_path)

        cfg = cls(
            run_mode=_normalize_run_mode(run.get("mode") or "generate"),
            dhmf_config_path=dhmf_config_path,
            query_mode="agentic",
            db_path=db_path,
            output_dir=str(paths.get("output_dir") or "results/benchmark_agentic_5"),
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
            hop_counts=hop_counts,
            seed=int(gen.get("seed", 42)),
            max_chars_per_doc=int(gen.get("max_chars_per_doc", -1)),
            doc_id_min=doc_id_min,
            doc_id_max=doc_id_max,
            original_pdf_only=_as_bool(gen.get("original_pdf_only", True), default=True),
            gen_max_retries=int(gen.get("max_retries", 3)),
            gen_sleep_between=float(gen.get("sleep_between", 0.0)),
            gen_num_thread=max(1, int(gen.get("num_thread", 4) or 4)),
            gen_use_cache=bool(gen.get("use_cache", False)),
            question_gen_prompt=_normalize_question_gen_prompt(
                gen.get("question_gen_prompt") or "QUESTION_GEN_USER"
            ),
            gen_api_key=_opt_str(gen.get("api_key")),
            gen_base_url=_opt_str(gen.get("base_url")),
            gen_timeout=float(gen.get("timeout", 180)),
            gen_model_args=_clean_model_args(gen.get("model_args")),
            enable_weknora=_as_bool(ev.get("enable_weknora", True), default=True),
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
            weknora_base_url=weknora["weknora_base_url"],
            weknora_email=weknora["weknora_email"],
            weknora_password=(
                str(weknora["weknora_password"])
                if weknora["weknora_password"] is not None
                else None
            ),
            weknora_knowledge_base_id=weknora["weknora_knowledge_base_id"],
            weknora_knowledge_base_name=weknora["weknora_knowledge_base_name"],
            weknora_chat_mode=weknora["weknora_chat_mode"],
            weknora_agent_id=weknora["weknora_agent_id"],
            weknora_agent_enabled=weknora["weknora_agent_enabled"],
            weknora_timeout=weknora["weknora_timeout"],
            weknora_max_retries=weknora["weknora_max_retries"],
            weknora_new_session_per_question=weknora[
                "weknora_new_session_per_question"
            ],
            weknora_delete_session=weknora["weknora_delete_session"],
            weknora_session_title_prefix=weknora["weknora_session_title_prefix"],
            log_level=_log_level(log.get("level", "INFO")),
            config_file=str(cfg_path) if cfg_path else None,
            raw=raw,
        )
        return cfg

    def resolve_paths(self) -> "BenchmarkConfig":
        self.dhmf_config_path = str(resolve_path(self.dhmf_config_path))
        self.db_path = str(resolve_path(self.db_path))
        self.output_dir = str(resolve_path(self.output_dir))
        assert_example_b_workspace(self.dhmf_config_path, self.db_path)
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
            "hop_counts": {str(k): v for k, v in self.hop_counts.items()},
            "seed": self.seed,
            "doc_id_min": self.doc_id_min,
            "doc_id_max": self.doc_id_max,
            "question_gen_prompt": self.question_gen_prompt,
            "original_pdf_only": self.original_pdf_only,
            "enable_doc_recall": self.enable_doc_recall,
            "enable_weknora": self.enable_weknora,
            "eval_num_thread": self.eval_num_thread,
            "gen_model": (self.gen_model_args or {}).get("model"),
            "judge_model": (self.judge_model_args or {}).get("model"),
            "weknora_base_url": self.weknora_base_url,
            "weknora_knowledge_base_id": self.weknora_knowledge_base_id,
            "weknora_knowledge_base_name": self.weknora_knowledge_base_name,
            "weknora_chat_mode": self.weknora_chat_mode,
            "weknora_agent_id": self.weknora_agent_id,
        }
