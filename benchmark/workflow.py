"""
多跳测试集生成 + DHMF RAG 评测 + 汇总报告。

全部参数写在 benchmark/config.yaml，脚本入口::

    python benchmark.py
    # 或
    python -m benchmark.run

run.mode: generate | evaluate | report | all
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
    主调用类：问题生成 / RAG 评测 / 汇总报告。

    三步均可独立运行；配置全部来自 yaml。
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
        self._eval_data: Optional[Dict[str, Any]] = None
        self._report: Optional[Dict[str, Any]] = None

        self._setup_logging(self.cfg.log_level)

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path, None] = DEFAULT_CONFIG_PATH,
    ) -> "TestQueryWorkflow":
        """从 yaml 加载配置（默认 benchmark/config.yaml）。"""
        cfg = BenchmarkConfig.from_yaml(config_path)
        return cls(cfg)

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
        if not over.get("model"):
            over.pop("model", None)
        return {**base, **over}

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
        prev = logging.getLogger("DHMF").level
        logging.getLogger("DHMF").setLevel(logging.ERROR)
        try:
            self.dhmf = DHMF(dcfg)
            if getattr(self.dhmf, "logger", None) is not None:
                self.dhmf.logger.setLevel(logging.ERROR)
                for h in self.dhmf.logger.handlers:
                    h.setLevel(logging.ERROR)
        finally:
            logging.getLogger("DHMF").setLevel(prev)
        return self.dhmf

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        return p if p.is_absolute() else resolve_path(p)

    def save_json(
        self,
        data: dict,
        path: Union[str, Path],
        *,
        quiet: bool = False,
        atomic: bool = True,
    ) -> Path:
        """
        写 JSON。默认原子写入（先 .tmp 再 replace），避免读到半截文件。
        quiet=True 时不打 [saved]（用于逐题增量保存）。
        """
        path = self._resolve_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        if atomic:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(path)
        else:
            path.write_text(payload, encoding="utf-8")
        if not quiet:
            print(f"[saved] {path}", file=sys.stderr)
        return path

    def load_json(self, path: Union[str, Path]) -> Dict[str, Any]:
        path = self._resolve_path(path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_dataset(self, path: Union[str, Path]) -> Dict[str, Any]:
        path = self._resolve_path(path)
        data = self.load_json(path)
        if "questions" not in data:
            raise ValueError(f"非法数据集（缺少 questions）: {path}")
        self._dataset = data
        # print(f"[loaded] questions <- {path}", file=sys.stderr)
        return data

    def load_eval_results(self, path: Union[str, Path]) -> Dict[str, Any]:
        path = self._resolve_path(path)
        data = self.load_json(path)
        if "results" not in data:
            raise ValueError(
                f"非法评测结果（缺少 results）: {path}\n"
                "  需要 evaluate 写出的 JSON（含 results 列表）。"
            )
        self._eval_data = data
        print(f"[loaded] eval_results <- {path}", file=sys.stderr)
        return data

    def generate_questions(self, save: bool = True) -> Dict[str, Any]:
        """
        步骤 1：生成多跳问题集。
        写出路径：paths.questions_filename / questions_path
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
            num_thread=self.cfg.gen_num_thread,
            question_gen_prompt=self.cfg.question_gen_prompt,
        )
        dataset = gen.generate_all()
        dataset.setdefault("meta", {})["config"] = self.cfg.to_meta_snapshot()
        self._dataset = dataset

        if save:
            out = self.cfg.questions_file()
            self.save_json(dataset, out)

        return dataset

    def evaluate(
        self,
        dataset: Optional[Dict[str, Any]] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        步骤 2：读问题集 → DHMF.query / agent_query → 评判。

        写出 eval_results（逐题明细，每题增量落盘）；
        不写汇总 report（由 report 步骤独立完成）。
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
                        "  请先 run.mode=generate，或配置 "
                        "paths.eval_questions_filename / eval_questions_path。"
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
            num_thread=self.cfg.eval_num_thread,
        )

        out = self.cfg.eval_results_file() if save else None
        # if out is not None:
        #     print(f"[eval] 实时结果 → {out}", file=sys.stderr)

        def _on_progress(mid_report: dict, index: int, total: int) -> None:
            if out is None:
                return
            mid_report.setdefault("meta", {})["config"] = self.cfg.to_meta_snapshot()
            mid_report.setdefault("meta", {})["progress"] = {
                "done": index,
                "total": total,
                "pct": round(100.0 * index / total, 1) if total else None,
            }
            self.save_json(mid_report, out, quiet=True)

        eval_data = evaluator.evaluate_all(
            dataset,
            on_progress=_on_progress if save else None,
        )
        eval_data.setdefault("meta", {})["config"] = self.cfg.to_meta_snapshot()
        self._eval_data = eval_data

        if save and out is not None:
            eval_data.setdefault("meta", {})["done"] = True
            # 完成后 progress 无意义，去掉避免冗余
            eval_data.get("meta", {}).pop("progress", None)
            self.save_json(eval_data, out)

        return eval_data

    def report(
        self,
        eval_data: Optional[Dict[str, Any]] = None,
        *,
        source_path: Optional[Union[str, Path]] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        步骤 3：对指定评测结果文件做统计汇总，写出 report JSON。

        输入优先级：
          1) 显式 eval_data
          2) 同进程 evaluate 的内存结果
          3) paths.report_source_* / 默认 eval_results 文件
        """
        src_path: Optional[Path] = None
        if eval_data is None:
            if self._eval_data is not None and source_path is None:
                eval_data = self._eval_data
                src_path = self.cfg.eval_results_file()
            else:
                src_path = (
                    self._resolve_path(source_path)
                    if source_path is not None
                    else self.cfg.report_source_file()
                )
                if not src_path.is_file():
                    raise FileNotFoundError(
                        "未找到评测结果 JSON，无法 report。\n"
                        f"  期望路径: {src_path}\n"
                        "  请先 run.mode=evaluate，或配置 "
                        "paths.report_source_filename / report_source_path / "
                        "eval_results_filename。"
                    )
                eval_data = self.load_eval_results(src_path)
        elif source_path is not None:
            src_path = self._resolve_path(source_path)

        report_doc = QueryEvaluator.build_report_document(
            eval_data,
            source_path=str(src_path) if src_path is not None else None,
            enable_doc_recall=self.cfg.enable_doc_recall,
        )
        # 用当前配置补全缺失字段
        meta = report_doc.setdefault("meta", {})
        snap = self.cfg.to_meta_snapshot()
        for k in (
            "db_path",
            "hop_counts",
            "seed",
            "query_mode",
            "judge_model",
            "config_file",
            "dhmf_config_path",
        ):
            if meta.get(k) is None and snap.get(k) is not None:
                meta[k] = snap[k]
        self._report = report_doc

        if save:
            out = self.cfg.report_file()
            self.save_json(report_doc, out)

        return report_doc

    def run_all(self) -> Dict[str, Any]:
        """generate → evaluate → report。"""
        dataset = self.generate_questions(save=True)
        eval_data = self.evaluate(dataset=dataset, save=True)
        return self.report(eval_data=eval_data, save=True)

    def run(self) -> Any:
        """
        按 config.run.mode 执行：
          generate → 只生成问题集
          evaluate → 只评测（写 eval_results）
          report   → 只汇总（读评测结果，写 report）
          all      → 生成 + 评测 + 汇总
        """
        mode = (self.cfg.run_mode or "all").strip().lower()
        print(
            # f"[benchmark] mode={mode} config={self.cfg.config_file}",
            file=sys.stderr,
        )
        if mode == "generate":
            return self.generate_questions(save=True)
        if mode == "evaluate":
            return self.evaluate(save=True)
        if mode == "report":
            return self.report(save=True)
        if mode == "all":
            return self.run_all()
        raise ValueError(
            f"Unknown run.mode={self.cfg.run_mode!r}. "
            f"Set run.mode in benchmark/config.yaml: "
            f"generate | evaluate | report | all"
        )
