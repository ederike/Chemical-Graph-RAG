"""
多跳测试集生成 + DHMF RAG 评测主工作流。

全部参数写在 benchmark/config.yaml，脚本入口::

    python benchmark.py
    # 或
    python -m benchmark.run

run.mode: generate | evaluate | all
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .config import BenchmarkConfig, DEFAULT_CONFIG_PATH
from .evaluator import QueryEvaluator
from .question_gen import QuestionGenerator
from .utils import project_root, resolve_path

logger = logging.getLogger("benchmark.workflow")


class TestQueryWorkflow:
    """
    主调用类：问题生成与 RAG 评测。

    配置全部来自 yaml；入口调用 run() 按 run.mode 分发。
    """

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg.resolve_paths()
        self.root = project_root()

        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dhmf_config = None  # src.utils.config.Config
        self.dhmf = None
        self.llm = None          # 出题
        self.judge_llm = None    # 评判
        self._dataset: Optional[Dict[str, Any]] = None
        self._report: Optional[Dict[str, Any]] = None

        self._setup_logging(self.cfg.log_level)

    # ------------------------------------------------------------------
    # factory
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path, None] = DEFAULT_CONFIG_PATH,
    ) -> "TestQueryWorkflow":
        """从 yaml 加载配置（默认 benchmark/config.yaml）。"""
        cfg = BenchmarkConfig.from_yaml(config_path)
        return cls(cfg)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _setup_logging(self, level: int):
        """
        benchmark 默认安静：只让 ERROR 以上走 console。
        若 config logging.level=DEBUG 才放开中间日志。
        """
        root_logger = logging.getLogger("benchmark")
        if not root_logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            root_logger.addHandler(handler)
        # 用户设 DEBUG 时保留细节；否则抬到 ERROR，进度条用 fail_print
        if level <= logging.DEBUG:
            root_logger.setLevel(logging.DEBUG)
        else:
            root_logger.setLevel(logging.ERROR)

    def _ensure_sys_path(self):
        root_s = str(self.root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)

    def _load_dhmf_config(self):
        if self.dhmf_config is not None:
            return self.dhmf_config
        self._ensure_sys_path()
        from src.utils.config import Config

        self.dhmf_config = Config.from_yaml(self.cfg.dhmf_config_path)
        # 可选：覆盖 retrieve / agent 的 use_cache（评测时常关缓存）
        if self.cfg.dhmf_retrieve_use_cache is not None:
            flag = bool(self.cfg.dhmf_retrieve_use_cache)
            try:
                self.dhmf_config.retrieve.use_cache = flag
            except Exception:
                pass
            try:
                self.dhmf_config.agent.use_cache = flag
            except Exception:
                pass
        return self.dhmf_config

    def _resolve_llm_endpoint(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> tuple:
        """显式配置优先，否则 dhmf retrieve → settings。"""
        from src.utils.config import resolve_credentials

        dcfg = self._load_dhmf_config()
        def_key, def_url = resolve_credentials(dcfg, dcfg.retrieve)
        key = (api_key or "").strip() or def_key
        url = (base_url or "").strip() or def_url
        return key, url

    def _merge_model_args(self, user_ma: dict) -> dict:
        """用户 model_args 覆盖 dhmf retrieve.model_args；用户未写 model 则继承。"""
        dcfg = self._load_dhmf_config()
        base = dict(getattr(dcfg.retrieve, "model_args", None) or {})
        over = dict(user_ma or {})
        # 空 model 不覆盖
        if not over.get("model"):
            over.pop("model", None)
        merged = {**base, **over}
        return merged

    def setup_llm(self, force: bool = False):
        """初始化出题 LLM。"""
        if self.llm is not None and not force:
            return self.llm
        self._ensure_sys_path()
        from src.utils.OpenAIAPI import LLM

        api_key, base_url = self._resolve_llm_endpoint(
            self.cfg.gen_api_key, self.cfg.gen_base_url
        )
        self.llm = LLM(api_key, base_url, timeout=self.cfg.gen_timeout)
        self.cfg.gen_model_args = self._merge_model_args(self.cfg.gen_model_args)
        return self.llm

    def setup_judge_llm(self, force: bool = False):
        """初始化评判 LLM（可与出题不同）。"""
        if self.judge_llm is not None and not force:
            return self.judge_llm
        self._ensure_sys_path()
        from src.utils.OpenAIAPI import LLM

        # 若评判端点与出题完全一致且已建 llm，可复用
        same_endpoint = (
            (self.cfg.judge_api_key or self.cfg.gen_api_key)
            == (self.cfg.gen_api_key or self.cfg.judge_api_key)
            and (self.cfg.judge_base_url or self.cfg.gen_base_url)
            == (self.cfg.gen_base_url or self.cfg.judge_base_url)
        )
        api_key, base_url = self._resolve_llm_endpoint(
            self.cfg.judge_api_key, self.cfg.judge_base_url
        )
        if (
            same_endpoint
            and self.llm is not None
            and not force
            and abs(float(self.cfg.judge_timeout) - float(self.cfg.gen_timeout)) < 1e-6
        ):
            self.judge_llm = self.llm
        else:
            self.judge_llm = LLM(api_key, base_url, timeout=self.cfg.judge_timeout)

        self.cfg.judge_model_args = self._merge_model_args(self.cfg.judge_model_args)
        return self.judge_llm

    def setup_dhmf(self, force: bool = False):
        """加载 Config + DHMF（评测需要检索索引）。"""
        if self.dhmf is not None and not force:
            return self.dhmf
        self._ensure_sys_path()
        from src.DHMF import DHMF

        dcfg = self._load_dhmf_config()
        # 初始化时也尽量安静
        prev = logging.getLogger("DHMF").level
        logging.getLogger("DHMF").setLevel(logging.ERROR)
        try:
            self.dhmf = DHMF(dcfg)
            # 实例 logger 名是 DHMF.<id>
            if getattr(self.dhmf, "logger", None) is not None:
                self.dhmf.logger.setLevel(logging.ERROR)
                for h in self.dhmf.logger.handlers:
                    h.setLevel(logging.ERROR)
        finally:
            logging.getLogger("DHMF").setLevel(prev)
        return self.dhmf

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        return p if p.is_absolute() else resolve_path(p)

    def save_json(self, data: dict, path: Union[str, Path]) -> Path:
        path = self._resolve_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[saved] {path}", file=sys.stderr)
        return path

    def load_dataset(self, path: Union[str, Path]) -> Dict[str, Any]:
        path = self._resolve_path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "questions" not in data:
            raise ValueError(f"非法数据集（缺少 questions）: {path}")
        self._dataset = data
        print(f"[loaded] questions <- {path}", file=sys.stderr)
        return data

    # ------------------------------------------------------------------
    # steps
    # ------------------------------------------------------------------
    def generate_questions(self, save: bool = True) -> Dict[str, Any]:
        """
        步骤 1：生成多跳问题集。
        写出路径完全由 config.paths 决定。
        """
        self.setup_llm()
        gen = QuestionGenerator(
            self.llm,
            model_args=self.cfg.gen_model_args,
            db_path=self.cfg.db_path,
            hop_counts=self.cfg.hop_counts,
            seed=self.cfg.seed,
            max_chars_per_doc=self.cfg.max_chars_per_doc,
            use_cache=self.cfg.gen_use_cache,
            max_retries=self.cfg.gen_max_retries,
            sleep_between=self.cfg.gen_sleep_between,
        )
        dataset = gen.generate_all()
        dataset.setdefault("meta", {})["benchmark_config"] = self.cfg.to_dict()
        self._dataset = dataset

        if save:
            out = self.cfg.questions_file()
            self.save_json(dataset, out)
            dataset.setdefault("meta", {})["saved_path"] = str(out)
            # 同进程 all 模式：evaluate 直接用内存中的 dataset

        return dataset

    def evaluate(
        self,
        dataset: Optional[Dict[str, Any]] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        步骤 2：读问题集 → DHMF.query / agent_query → 评判。
        问题集与报告路径完全由 config.paths 决定。
        """
        if dataset is None:
            if self._dataset is not None:
                dataset = self._dataset
            else:
                path = self.cfg.eval_questions_file()
                if not path.is_file():
                    raise FileNotFoundError(
                        "未找到问题集 JSON，无法 evaluate。\n"
                        f"  期望路径: {path}\n"
                        "  请先将 run.mode 设为 generate 生成问题集，或在 "
                        "paths.eval_questions_filename / eval_questions_path 指定已有文件。"
                    )
                dataset = self.load_dataset(path)

        self.setup_dhmf()
        self.setup_judge_llm()

        evaluator = QueryEvaluator(
            self.dhmf,
            judge_llm=self.judge_llm,
            judge_model_args=self.cfg.judge_model_args,
            query_mode=self.cfg.query_mode,
            use_cache=self.cfg.eval_use_cache,
            max_judge_retries=self.cfg.eval_max_retries,
            max_source_chars=self.cfg.max_source_chars,
            sleep_between=self.cfg.eval_sleep_between,
            enable_doc_recall=self.cfg.enable_doc_recall,
        )
        report = evaluator.evaluate_all(dataset)
        report.setdefault("meta", {})["benchmark_config"] = self.cfg.to_dict()
        self._report = report

        table = report.get("summary_table") or ""
        if table:
            print(table)

        if save:
            out = self.cfg.report_file()
            self.save_json(report, out)
            report.setdefault("meta", {})["saved_path"] = str(out)
            if self.cfg.save_summary_txt and table:
                txt_path = self.cfg.summary_file(out)
                txt_path.parent.mkdir(parents=True, exist_ok=True)
                txt_path.write_text(table, encoding="utf-8")
                print(f"[saved] {txt_path}", file=sys.stderr)
                report.setdefault("meta", {})["summary_path"] = str(txt_path)

        return report

    def run_all(self) -> Dict[str, Any]:
        """生成 + 评测；路径全部走 config.paths。"""
        dataset = self.generate_questions(save=True)
        return self.evaluate(dataset=dataset, save=True)

    def run(self) -> Any:
        """
        按 config.run.mode 执行：
          generate → 只生成问题集
          evaluate → 只评测
          all      → 生成 + 评测
        """
        mode = (self.cfg.run_mode or "all").strip().lower()
        print(
            f"[benchmark] mode={mode} config={self.cfg.config_file}",
            file=sys.stderr,
        )
        if mode == "generate":
            return self.generate_questions(save=True)
        if mode == "evaluate":
            return self.evaluate(save=True)
        if mode == "all":
            return self.run_all()
        raise ValueError(
            f"Unknown run.mode={self.cfg.run_mode!r}. "
            f"Set run.mode in benchmark/config.yaml: generate | evaluate | all"
        )
