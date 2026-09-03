"""
工作流：问题生成 → agentic_query vs 纯 LLM 评测 → 汇总 Excel。

全部参数写在 benchmark/benchmark_agentic_1/config.yaml::

    python -m benchmark.benchmark_agentic_1

run.mode: generate | evaluate | report | excel | all
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .config import (
    DEFAULT_CONFIG_PATH,
    BenchmarkConfig,
    merge_llm_only_model_args,
)
from .evaluator import QueryEvaluator
from .question_gen import QuestionGenerator
from .report import build_report_document
from .utils import (
    atomic_write_json,
    load_json,
    pin_retrieve_for_eval,
    project_root,
    resolve_path,
    unpin_retrieve_for_eval,
)

logger = logging.getLogger("benchmark_agentic_1.workflow")


class AgenticBenchmarkWorkflow:
    """三步均可独立跑；配置全部来自本包 config.yaml，不读 useless/。"""

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg.resolve_paths()
        self.root = project_root()
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dhmf_config = None
        self.dhmf = None
        self.llm = None
        self.judge_llm = None
        self.answer_llm = None
        self._dataset: Optional[Dict[str, Any]] = None
        self._eval_data: Optional[Dict[str, Any]] = None
        self._report: Optional[Dict[str, Any]] = None
        self._save_lock = threading.Lock()
        self._setup_logging(self.cfg.log_level)

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path, None] = DEFAULT_CONFIG_PATH,
    ) -> "AgenticBenchmarkWorkflow":
        cfg = BenchmarkConfig.from_yaml(config_path)
        return cls(cfg)

    def _setup_logging(self, level: int):
        root_logger = logging.getLogger("benchmark_agentic_1")
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
                self.dhmf_config.agentic.use_cache = flag
            except Exception:
                pass
        return self.dhmf_config

    def _resolve_llm_endpoint(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> tuple:
        from src.utils.config import resolve_credentials

        dcfg = self._load_dhmf_config()
        def_key, def_url = resolve_credentials(dcfg, dcfg.retrieve)
        key = (api_key or "").strip() or def_key
        url = (base_url or "").strip() or def_url
        return key, url

    def _merge_model_args(self, user_ma: dict) -> dict:
        dcfg = self._load_dhmf_config()
        base = dict(getattr(dcfg.retrieve, "model_args", None) or {})
        over = dict(user_ma or {})
        if not over.get("model"):
            over.pop("model", None)
        return {**base, **over}

    def setup_llm(self, force: bool = False):
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

    def _merge_llm_only_model_args(self, user_ma: dict) -> dict:
        dcfg = self._load_dhmf_config()
        base = dict(getattr(dcfg.retrieve, "model_args", None) or {})
        return merge_llm_only_model_args(base, user_ma)

    def setup_answer_llm(self, force: bool = False):
        if self.answer_llm is not None and not force:
            return self.answer_llm
        if not self.cfg.enable_llm_only:
            return None
        self._ensure_sys_path()
        from src.utils.OpenAIAPI import LLM

        api_key, base_url = self._resolve_llm_endpoint(
            self.cfg.llm_only_api_key, self.cfg.llm_only_base_url
        )
        self.answer_llm = LLM(api_key, base_url, timeout=self.cfg.llm_only_timeout)
        self.cfg.llm_only_model_args = self._merge_llm_only_model_args(
            self.cfg.llm_only_model_args
        )
        return self.answer_llm

    def setup_dhmf(self, force: bool = False):
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
    ) -> Path:
        path = self._resolve_path(path)
        with self._save_lock:
            atomic_write_json(data, path)
        if not quiet:
            print(f"[saved] {path}", file=sys.stderr)
        return path

    def load_json(self, path: Union[str, Path]) -> Dict[str, Any]:
        return load_json(self._resolve_path(path))

    def load_dataset(self, path: Union[str, Path]) -> Dict[str, Any]:
        path = self._resolve_path(path)
        data = self.load_json(path)
        if "questions" not in data:
            raise ValueError(f"非法数据集（缺少 questions）: {path}")
        self._dataset = data
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
        """步骤 1：从 doc 表指定范围抽样，按 hop_counts 出题；每完成一题实时落盘。"""
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
            doc_id_min=self.cfg.doc_id_min,
            doc_id_max=self.cfg.doc_id_max,
        )
        out = self.cfg.questions_file() if save else None
        existing = []
        if save and out is not None and out.is_file():
            try:
                prev = self.load_json(out)
                rows = prev.get("questions") if isinstance(prev, dict) else None
                if isinstance(rows, list):
                    existing = rows
                    print(f"[resume] 已有 {len(rows)} 条问题 <- {out}", file=sys.stderr)
            except Exception as e:
                print(f"[resume] 无法读取已有问题集 {out}: {e}", file=sys.stderr)

        def _on_progress(mid: dict, index: int, total: int) -> None:
            if out is None:
                return
            mid.setdefault("meta", {})["config"] = self.cfg.to_meta_snapshot()
            mid.setdefault("meta", {})["progress"] = {
                "done": index,
                "total": total,
                "pct": round(100.0 * index / total, 1) if total else None,
            }
            self.save_json(mid, out, quiet=True)

        dataset = gen.generate_all(
            on_progress=_on_progress if save else None,
            existing_questions=existing,
        )
        dataset.setdefault("meta", {})["config"] = self.cfg.to_meta_snapshot()
        dataset.get("meta", {}).pop("progress", None)
        self._dataset = dataset
        if save and out is not None:
            self.save_json(dataset, out)
        return dataset

    def evaluate(
        self,
        dataset: Optional[Dict[str, Any]] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """步骤 2：每题 agentic_query 一次 + 纯 LLM 一次 → Judge；逐题增量落盘。"""
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
        answer_llm = self.setup_answer_llm() if self.cfg.enable_llm_only else None
        pinned = pin_retrieve_for_eval(self.dhmf)

        evaluator = QueryEvaluator(
            self.dhmf,
            judge_llm=self.judge_llm,
            judge_model_args=self.cfg.judge_model_args,
            answer_llm=answer_llm,
            answer_model_args=self.cfg.llm_only_model_args,
            use_cache=self.cfg.eval_use_cache,
            max_judge_retries=self.cfg.eval_max_retries,
            max_llm_only_retries=self.cfg.llm_only_max_retries,
            llm_only_use_cache=self.cfg.llm_only_use_cache,
            max_source_chars=self.cfg.max_source_chars,
            sleep_between=self.cfg.eval_sleep_between,
            enable_doc_recall=self.cfg.enable_doc_recall,
            enable_llm_only=self.cfg.enable_llm_only,
            num_thread=self.cfg.eval_num_thread,
        )

        out = self.cfg.eval_results_file() if save else None
        existing = []
        if save and out is not None and out.is_file():
            try:
                prev = self.load_json(out)
                rows = prev.get("results") if isinstance(prev, dict) else None
                if isinstance(rows, list):
                    existing = rows
                    print(f"[resume] 已有 {len(rows)} 条 <- {out}", file=sys.stderr)
            except Exception as e:
                print(f"[resume] 无法读取已有结果 {out}: {e}", file=sys.stderr)

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

        try:
            eval_data = evaluator.evaluate_all(
                dataset,
                existing_results=existing,
                on_progress=_on_progress if save else None,
                extra_meta={"config": self.cfg.to_meta_snapshot()},
            )
        finally:
            unpin_retrieve_for_eval(self.dhmf, pinned)

        eval_data.setdefault("meta", {})["config"] = self.cfg.to_meta_snapshot()
        self._eval_data = eval_data
        if save and out is not None:
            eval_data.setdefault("meta", {})["done"] = True
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
        """步骤 3：读 evals，写 report JSON + Excel。"""
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
                        "  请先 run.mode=evaluate。"
                    )
                eval_data = self.load_eval_results(src_path)
        elif source_path is not None:
            src_path = self._resolve_path(source_path)

        report_doc = build_report_document(
            eval_data,
            source_path=str(src_path) if src_path is not None else None,
            enable_doc_recall=self.cfg.enable_doc_recall,
        )
        meta = report_doc.setdefault("meta", {})
        snap = self.cfg.to_meta_snapshot()
        for k in (
            "db_path", "hop_counts", "seed", "query_mode", "judge_model",
            "config_file", "dhmf_config_path", "enable_llm_only",
            "doc_id_min", "doc_id_max",
        ):
            if meta.get(k) is None and snap.get(k) is not None:
                meta[k] = snap[k]
        self._report = report_doc

        if save:
            out = self.cfg.report_file()
            self.save_json(report_doc, out)
            self.export_excel(report_doc=report_doc, eval_data=eval_data, save=True)
        return report_doc

    def export_excel(
        self,
        report_doc: Optional[Dict[str, Any]] = None,
        eval_data: Optional[Dict[str, Any]] = None,
        *,
        save: bool = True,
    ) -> Path:
        if report_doc is None:
            if self._report is not None:
                report_doc = self._report
            else:
                rp = self.cfg.report_file()
                if rp.is_file():
                    report_doc = self.load_json(rp)
                    self._report = report_doc
                    print(f"[loaded] report <- {rp}", file=sys.stderr)
                elif eval_data is None:
                    eval_data = self._eval_data
                    if eval_data is None:
                        ev = self.cfg.eval_results_file()
                        if ev.is_file():
                            eval_data = self.load_eval_results(ev)
                    if eval_data is not None:
                        report_doc = build_report_document(
                            eval_data,
                            source_path=str(self.cfg.eval_results_file()),
                            enable_doc_recall=self.cfg.enable_doc_recall,
                        )
                        self._report = report_doc
        if report_doc is None:
            raise FileNotFoundError(
                "未找到 report / evals JSON，无法导出 Excel。\n"
                f"  report: {self.cfg.report_file()}\n"
                f"  evals:  {self.cfg.eval_results_file()}\n"
            )

        if eval_data is None:
            eval_data = self._eval_data
        if eval_data is None:
            ev = self.cfg.eval_results_file()
            if ev.is_file():
                try:
                    eval_data = self.load_json(ev)
                except Exception as e:
                    print(f"[excel] 无法读取 evals {ev}: {e}", file=sys.stderr)
                    eval_data = None

        out = self.cfg.report_excel_file()
        if save:
            from .export_excel import write_report_workbook

            write_report_workbook(report_doc, out, eval_data=eval_data)
            print(f"[saved] {out}", file=sys.stderr)
        return out

    def run_all(self) -> Dict[str, Any]:
        dataset = self.generate_questions(save=True)
        eval_data = self.evaluate(dataset=dataset, save=True)
        return self.report(eval_data=eval_data, save=True)

    def run(self) -> Any:
        mode = (self.cfg.run_mode or "all").strip().lower()
        print(
            f"[benchmark_agentic_1] mode={mode} config={self.cfg.config_file}",
            file=sys.stderr,
        )
        if mode == "generate":
            return self.generate_questions(save=True)
        if mode == "evaluate":
            return self.evaluate(save=True)
        if mode == "report":
            return self.report(save=True)
        if mode == "excel":
            return self.export_excel(save=True)
        if mode == "all":
            return self.run_all()
        raise ValueError(
            f"Unknown run.mode={self.cfg.run_mode!r}. "
            f"Set run.mode in benchmark/benchmark_agentic_1/config.yaml: "
            f"generate | evaluate | report | excel | all"
        )
