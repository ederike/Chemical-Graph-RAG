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
    if s in ("all", "both", "full"):
        return "all"
    raise ValueError(
        f"Unknown run.mode={mode!r}. Supported: 'generate' | 'evaluate' | 'all'"
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
    run_mode: str = "all"  # generate | evaluate | all

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

    report_filename: str = "eval_report.json"
    report_path: Optional[str] = None  # 覆盖报告写出路径

    # generate
    hop_counts: Dict[int, int] = field(default_factory=lambda: {1: 5, 2: 3, 3: 2})
    seed: int = 42
    max_chars_per_doc: int = 4000
    gen_max_retries: int = 3
    gen_sleep_between: float = 0.0
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
    save_summary_txt: bool = True
    eval_use_cache: bool = False
    dhmf_retrieve_use_cache: Optional[bool] = None
    judge_api_key: Optional[str] = None
    judge_base_url: Optional[str] = None
    judge_timeout: float = 180.0
    judge_model_args: Dict[str, Any] = field(default_factory=dict)

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

        q_name = _opt_str(paths.get("questions_filename")) or "questions.json"
        r_name = _opt_str(paths.get("report_filename")) or "eval_report.json"

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
            report_filename=r_name,
            report_path=_opt_str(paths.get("report_path")),
            hop_counts=hop_counts,
            seed=int(gen.get("seed", 42)),
            max_chars_per_doc=int(gen.get("max_chars_per_doc", 4000)),
            gen_max_retries=int(gen.get("max_retries", 3)),
            gen_sleep_between=float(gen.get("sleep_between", 0.0)),
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
            save_summary_txt=bool(ev.get("save_summary_txt", True)),
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

    def resolve_paths(self) -> "BenchmarkConfig":
        """解析固定路径字段为绝对路径；filename 保持原样。"""
        self.dhmf_config_path = str(resolve_path(self.dhmf_config_path))
        self.db_path = str(resolve_path(self.db_path))
        self.output_dir = str(resolve_path(self.output_dir))
        # custom path 字段存的是用户配置字符串；真正拼装在 questions_file() 等里做
        # 这里只规范化非空 custom path 的绝对形式，便于日志
        if self.questions_path:
            p = Path(self.questions_path)
            if not p.is_absolute():
                self.questions_path = str(resolve_path(p))
        if self.eval_questions_path:
            p = Path(self.eval_questions_path)
            if not p.is_absolute():
                self.eval_questions_path = str(resolve_path(p))
        if self.report_path:
            p = Path(self.report_path)
            if not p.is_absolute():
                self.report_path = str(resolve_path(p))
        return self

    # ------------------------------------------------------------------
    # 三类文件路径
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

    def report_file(self, *, timestamp: Optional[str] = None) -> Path:
        """评测报告写出路径。"""
        return resolve_file_path(
            self.output_dir,
            self.report_filename or "eval_report.json",
            self.report_path,
            timestamp=timestamp,
        )

    def summary_file(self, report: Optional[Path] = None) -> Path:
        """summary 与 report 同目录：{stem}.summary.txt。"""
        rp = report or self.report_file()
        return rp.with_name(rp.stem + ".summary.txt")

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
