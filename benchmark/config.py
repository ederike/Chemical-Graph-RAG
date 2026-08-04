"""
benchmark 配置加载与合并。

优先级（高 → 低）：
  CLI 显式参数 > BenchmarkConfig 实例字段覆盖 > config.yaml > 内置默认
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .utils import parse_hop_spec, project_root, resolve_path

# 默认配置文件位置（相对项目根）
DEFAULT_CONFIG_PATH = "benchmark/config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并 dict；override 中值为 None 的键不覆盖 base。"""
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
    """去掉 model=null / 空字符串，便于后续继承。"""
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


@dataclass
class BenchmarkConfig:
    """
    工作流全部可配置项的扁平视图，便于 workflow / CLI 使用。
    """

    # dhmf
    dhmf_config_path: str = "example/a/config_open.yaml"
    query_mode: str = "dual_path"

    # paths
    db_path: str = "example/a/DB/main.db"
    output_dir: str = "benchmark/outputs"
    questions_path: Optional[str] = None
    report_path: Optional[str] = None
    summary_path: Optional[str] = None
    dataset_path: Optional[str] = None

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
    max_source_chars: int = -1  # -1 = 评判时送入完整文档
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

    # 记录加载来源
    config_file: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(
        cls,
        path: Union[str, Path, None] = None,
        *,
        overrides: Optional[dict] = None,
    ) -> "BenchmarkConfig":
        """
        读取 yaml；path 为 None 时用 DEFAULT_CONFIG_PATH（不存在则用内置默认）。
        overrides: 嵌套或扁平字段，用于 CLI 覆盖。
        """
        root = project_root()
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

        if overrides:
            # 支持嵌套 dict 覆盖
            if any(k in overrides for k in ("dhmf", "paths", "generate", "evaluate", "logging")):
                raw = _deep_merge(raw, overrides)
            else:
                # 扁平 overrides 稍后直接写字段
                pass

        dhmf = raw.get("dhmf") or {}
        paths = raw.get("paths") or {}
        gen = raw.get("generate") or {}
        ev = raw.get("evaluate") or {}
        log = raw.get("logging") or {}

        hop_raw = gen.get("hop_counts") or {1: 5, 2: 3, 3: 2}
        hop_counts = parse_hop_spec(hop_raw)

        cfg = cls(
            dhmf_config_path=str(dhmf.get("config_path") or "example/a/config_open.yaml"),
            query_mode=str(dhmf.get("query_mode") or "dual_path"),
            db_path=str(paths.get("db_path") or "example/a/DB/main.db"),
            output_dir=str(paths.get("output_dir") or "benchmark/outputs"),
            questions_path=_opt_str(paths.get("questions_path")),
            report_path=_opt_str(paths.get("report_path")),
            summary_path=_opt_str(paths.get("summary_path")),
            dataset_path=_opt_str(paths.get("dataset_path")),
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
            max_source_chars=int(
                ev.get(
                    "max_source_chars",
                    gen.get("max_chars_per_doc", -1),
                )
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

        # 扁平 overrides（CLI）：只覆盖非 None 字段
        if overrides and not any(
            k in overrides for k in ("dhmf", "paths", "generate", "evaluate", "logging")
        ):
            cfg.apply_flat_overrides(overrides)

        return cfg

    def apply_flat_overrides(self, overrides: Dict[str, Any]) -> "BenchmarkConfig":
        """用扁平 dict 覆盖已有字段（值 None 跳过）。"""
        if not overrides:
            return self
        valid = {f.name for f in fields(self)}
        for k, v in overrides.items():
            if v is None or k not in valid:
                continue
            if k == "hop_counts":
                setattr(self, k, parse_hop_spec(v))
            elif k == "log_level":
                setattr(self, k, _log_level(v))
            elif k in ("gen_model_args", "judge_model_args"):
                setattr(self, k, _clean_model_args(v))
            else:
                setattr(self, k, v)
        return self

    def resolve_paths(self) -> "BenchmarkConfig":
        """把相对路径解析为绝对路径（基于项目根）。"""
        self.dhmf_config_path = str(resolve_path(self.dhmf_config_path))
        self.db_path = str(resolve_path(self.db_path))
        self.output_dir = str(resolve_path(self.output_dir))
        if self.questions_path:
            self.questions_path = str(resolve_path(self.questions_path))
        if self.report_path:
            self.report_path = str(resolve_path(self.report_path))
        if self.summary_path:
            self.summary_path = str(resolve_path(self.summary_path))
        if self.dataset_path:
            self.dataset_path = str(resolve_path(self.dataset_path))
        return self

    def to_dict(self) -> dict:
        """便于写入报告 meta。"""
        return {
            "config_file": self.config_file,
            "dhmf_config_path": self.dhmf_config_path,
            "query_mode": self.query_mode,
            "db_path": self.db_path,
            "output_dir": self.output_dir,
            "questions_path": self.questions_path,
            "report_path": self.report_path,
            "summary_path": self.summary_path,
            "dataset_path": self.dataset_path,
            "hop_counts": {str(k): v for k, v in self.hop_counts.items()},
            "seed": self.seed,
            "max_chars_per_doc": self.max_chars_per_doc,
            "gen_model_args": self.gen_model_args,
            "judge_model_args": self.judge_model_args,
            "gen_api_key_set": bool(self.gen_api_key),
            "gen_base_url": self.gen_base_url,
            "judge_api_key_set": bool(self.judge_api_key),
            "judge_base_url": self.judge_base_url,
            "query_mode": self.query_mode,
        }


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
