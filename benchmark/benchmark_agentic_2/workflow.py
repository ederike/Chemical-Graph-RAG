"""
Excel 统计 + agentic/纯LLM 评测 + 分类汇总。

配置：benchmark/benchmark_agentic_2/config.yaml
运行：python -m benchmark.benchmark_agentic_2

run.mode: stats | evaluate | rejudge | report | excel | all
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import (
    DEFAULT_CONFIG_PATH,
    BenchmarkAgentic2Config,
    DatasetSpec,
    merge_llm_only_model_args,
)
from .dataset import build_stats_document, load_excel_dataset
from .evaluator import ExcelAgenticEvaluator
from .export_excel import write_report_workbook
from .report import build_report_document, format_summary_text
from .utils import pin_retrieve_for_eval, project_root, resolve_path, unpin_retrieve_for_eval

logger = logging.getLogger("benchmark_agentic_2.workflow")


class BenchmarkAgentic2Workflow:
    def __init__(self, cfg: BenchmarkAgentic2Config):
        self.cfg = cfg.resolve_paths()
        self.root = project_root()
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dhmf_config = None
        self.dhmf = None
        self.judge_llm = None
        self.answer_llm = None
        self._setup_logging(self.cfg.log_level)

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path, None] = DEFAULT_CONFIG_PATH,
    ) -> "BenchmarkAgentic2Workflow":
        cfg = BenchmarkAgentic2Config.from_yaml(config_path)
        return cls(cfg)

    def _setup_logging(self, level: int):
        root_logger = logging.getLogger("benchmark_agentic_2")
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

    def _resolve_llm_endpoint(self, api_key: Optional[str], base_url: Optional[str]):
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

    def setup_judge_llm(self, force: bool = False):
        if self.judge_llm is not None and not force:
            return self.judge_llm
        self._ensure_sys_path()
        from src.utils.OpenAIAPI import LLM

        api_key, base_url = self._resolve_llm_endpoint(
            self.cfg.judge_api_key, self.cfg.judge_base_url
        )
        self.judge_llm = LLM(api_key, base_url, timeout=self.cfg.judge_timeout)
        self.cfg.judge_model_args = self._merge_model_args(self.cfg.judge_model_args)
        return self.judge_llm

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
        dcfg = self._load_dhmf_config()
        base = dict(getattr(dcfg.retrieve, "model_args", None) or {})
        self.cfg.llm_only_model_args = merge_llm_only_model_args(
            base, self.cfg.llm_only_model_args
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

    def save_json(self, data: dict, path: Union[str, Path], *, quiet: bool = False) -> Path:
        path = Path(path) if Path(path).is_absolute() else resolve_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        if not quiet:
            print(f"[saved] {path}", file=sys.stderr)
        return path

    def load_json(self, path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(path) if Path(path).is_absolute() else resolve_path(path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _make_evaluator(self) -> ExcelAgenticEvaluator:
        self.setup_dhmf()
        self.setup_judge_llm()
        self.setup_answer_llm()
        return ExcelAgenticEvaluator(
            self.dhmf,
            judge_llm=self.judge_llm,
            judge_model_args=self.cfg.judge_model_args,
            answer_llm=self.answer_llm,
            answer_model_args=self.cfg.llm_only_model_args,
            use_cache=self.cfg.eval_use_cache,
            max_judge_retries=self.cfg.eval_max_retries,
            max_llm_only_retries=self.cfg.llm_only_max_retries,
            llm_only_use_cache=self.cfg.llm_only_use_cache,
            sleep_between=self.cfg.eval_sleep_between,
            enable_doc_recall=self.cfg.enable_doc_recall,
            enable_llm_only=self.cfg.enable_llm_only,
            num_thread=self.cfg.eval_num_thread,
        )

    def build_stats_for(self, spec: DatasetSpec, save: bool = True) -> Dict[str, Any]:
        self.cfg.bind_dataset(spec)
        dataset = load_excel_dataset(
            excel_path=spec.excel_path,
            questions_sheet=spec.questions_sheet,
            stats_sheet=spec.stats_sheet,
            design_sheet=spec.design_sheet,
            n_limit=self.cfg.n_limit,
            id_list=self.cfg.id_list,
        )
        # stamp dataset id onto questions
        for q in dataset.get("questions") or []:
            q["dataset_id"] = spec.id
        meta = dataset.setdefault("meta", {})
        meta["dataset_id"] = spec.id

        stats = dataset.get("stats") or build_stats_document(
            excel_path=spec.excel_path,
            questions_sheet=spec.questions_sheet,
            stats_sheet=spec.stats_sheet,
            design_sheet=spec.design_sheet,
        )
        stats.setdefault("meta", {})["dataset_id"] = spec.id

        if save:
            self.save_json(stats, self.cfg.stats_file())
            self.save_json(dataset, self.cfg.dataset_file())
        return {"stats": stats, "dataset": dataset}

    def evaluate_for(self, spec: DatasetSpec, save: bool = True) -> Dict[str, Any]:
        self.cfg.bind_dataset(spec)
        ds_path = self.cfg.dataset_file()
        if ds_path.is_file():
            dataset = self.load_json(ds_path)
        else:
            packed = self.build_stats_for(spec, save=True)
            dataset = packed["dataset"]

        questions = list(dataset.get("questions") or [])
        for q in questions:
            q.setdefault("dataset_id", spec.id)

        existing = []
        out_path = self.cfg.eval_results_file()
        if self.cfg.eval_resume and out_path.is_file():
            try:
                existing = list((self.load_json(out_path).get("results") or []))
            except Exception as e:
                print(f"[resume] skip broken evals: {e}", file=sys.stderr)

        evaluator = self._make_evaluator()
        pinned = pin_retrieve_for_eval(self.dhmf)
        try:
            def _on_progress(r, done, total):
                # periodic checkpoint: rewrite current full-ish list is handled by resume files
                if done == total or done % 10 == 0:
                    print(f"[evaluate:{spec.id}] {done}/{total}", file=sys.stderr, flush=True)

            eval_data = evaluator.evaluate(
                questions,
                existing_results=existing,
                resume=self.cfg.eval_resume,
                on_progress=_on_progress,
                dataset_meta=dataset.get("meta"),
                config_meta=self.cfg.to_meta_snapshot(),
            )
        finally:
            unpin_retrieve_for_eval(self.dhmf, pinned)

        if save:
            self.save_json(eval_data, out_path)
        return eval_data

    def rejudge_for(self, spec: DatasetSpec, save: bool = True) -> Dict[str, Any]:
        self.cfg.bind_dataset(spec)
        path = self.cfg.eval_results_file()
        if not path.is_file():
            raise FileNotFoundError(f"evals not found for rejudge: {path}")
        data = self.load_json(path)
        evaluator = self._make_evaluator()
        eval_data = evaluator.rejudge(
            data.get("results") or [],
            dataset_meta=(data.get("meta") or {}).get("dataset"),
            config_meta=self.cfg.to_meta_snapshot(),
        )
        if save:
            self.save_json(eval_data, path)
        return eval_data

    def report_for(self, spec: DatasetSpec, save: bool = True) -> Dict[str, Any]:
        self.cfg.bind_dataset(spec)
        eval_path = self.cfg.report_source_file()
        if not eval_path.is_file():
            raise FileNotFoundError(f"eval results missing: {eval_path}")
        eval_data = self.load_json(eval_path)
        stats = None
        stats_path = self.cfg.stats_file()
        if stats_path.is_file():
            stats = self.load_json(stats_path)
        report = build_report_document(eval_data, stats=stats)
        print(format_summary_text(report.get("summary") or {}), file=sys.stderr)
        if save:
            self.save_json(report, self.cfg.report_file())
        return report

    def excel_for(self, spec: DatasetSpec) -> Path:
        self.cfg.bind_dataset(spec)
        report_path = self.cfg.report_file()
        if not report_path.is_file():
            self.report_for(spec, save=True)
            report_path = self.cfg.report_file()
        report = self.load_json(report_path)
        eval_data = None
        ev_path = self.cfg.eval_results_file()
        if ev_path.is_file():
            eval_data = self.load_json(ev_path)
        out = self.cfg.report_excel_file()
        write_report_workbook(report, out, eval_data=eval_data)
        print(f"[saved] {out}", file=sys.stderr)
        return out

    def _run_one_mode(self, mode: str, spec: DatasetSpec):
        print(f"\n===== [{spec.id}] mode={mode} =====", file=sys.stderr)
        if mode == "stats":
            self.build_stats_for(spec)
        elif mode == "evaluate":
            self.evaluate_for(spec)
        elif mode == "rejudge":
            self.rejudge_for(spec)
        elif mode == "report":
            self.report_for(spec)
        elif mode == "excel":
            self.excel_for(spec)
        else:
            raise ValueError(f"unknown mode {mode}")

    def _write_combined(self, specs: List[DatasetSpec]):
        reports = []
        evals_all = []
        for spec in specs:
            self.cfg.bind_dataset(spec)
            rp = self.cfg.report_file()
            ep = self.cfg.eval_results_file()
            if rp.is_file():
                reports.append({"dataset_id": spec.id, "report": self.load_json(rp)})
            if ep.is_file():
                ed = self.load_json(ep)
                for r in ed.get("results") or []:
                    r = dict(r)
                    r.setdefault("dataset_id", spec.id)
                    evals_all.append(r)
        if not reports and not evals_all:
            return
        combined_eval = {
            "schema_version": 1,
            "meta": {
                "datasets": [s.id for s in specs],
                "enable_llm_only": self.cfg.enable_llm_only,
                "enable_doc_recall": self.cfg.enable_doc_recall,
                "n_results": len(evals_all),
            },
            "results": evals_all,
        }
        combined_report = build_report_document(combined_eval)
        combined_report["per_dataset"] = [
            {"dataset_id": x["dataset_id"], "summary": (x["report"].get("summary") or {})}
            for x in reports
        ]
        self.save_json(combined_report, self.cfg.combined_report_file())
        write_report_workbook(
            combined_report,
            self.cfg.combined_excel_file(),
            eval_data=combined_eval,
        )
        print(f"[saved] {self.cfg.combined_excel_file()}", file=sys.stderr)

    def run(self, mode: Optional[str] = None):
        mode = mode or self.cfg.run_mode
        specs = self.cfg.enabled_datasets()
        if not specs:
            raise ValueError("config.datasets 为空或全部 disabled")

        sequence = {
            "stats": ["stats"],
            "evaluate": ["stats", "evaluate"],
            "rejudge": ["rejudge"],
            "report": ["report"],
            "excel": ["report", "excel"],
            "all": ["stats", "evaluate", "report", "excel"],
        }.get(mode)
        if sequence is None:
            raise ValueError(f"Unknown mode={mode!r}")

        need_dhmf = any(m in ("evaluate", "rejudge") for m in sequence)
        if need_dhmf:
            self.setup_dhmf()
            self.setup_judge_llm()
            self.setup_answer_llm()

        for spec in specs:
            for step in sequence:
                self._run_one_mode(step, spec)

        if len(specs) > 1 and any(m in ("report", "excel", "all") for m in (mode,)):
            if mode in ("report", "excel", "all"):
                self._write_combined(specs)
