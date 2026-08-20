"""
Excel 统计 JSON + RAG 评测 + 分类汇总。

全部参数写在 benchmark2/config.yaml，脚本入口::

    python benchmark2.py
    python -m benchmark2.run

run.mode: stats | evaluate | report | excel | all
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from benchmark.utils import project_root, resolve_path

from .config import DEFAULT_CONFIG_PATH, Benchmark2Config
from .dataset import build_stats_document, load_excel_dataset
from .evaluator import ExcelQueryEvaluator
from .report import build_report_document, format_summary_text

logger = logging.getLogger("benchmark2.workflow")


class Benchmark2Workflow:
    def __init__(self, cfg: Benchmark2Config):
        self.cfg = cfg.resolve_paths()
        self.root = project_root()

        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dhmf_config = None
        self.dhmf = None
        self.judge_llm = None
        self._dataset: Optional[Dict[str, Any]] = None
        self._stats: Optional[Dict[str, Any]] = None
        self._eval_data: Optional[Dict[str, Any]] = None
        self._report: Optional[Dict[str, Any]] = None

        self._setup_logging(self.cfg.log_level)

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path, None] = DEFAULT_CONFIG_PATH,
    ) -> "Benchmark2Workflow":
        cfg = Benchmark2Config.from_yaml(config_path)
        return cls(cfg)

    def _setup_logging(self, level: int):
        root_logger = logging.getLogger("benchmark2")
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
        atomic: bool = True,
    ) -> Path:
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

    def _load_existing_eval(self, path: Path) -> list:
        if not path.is_file():
            return []
        try:
            data = self.load_json(path)
        except Exception as e:
            print(f"[resume] 无法读取已有结果 {path}: {e}", file=sys.stderr)
            return []
        rows = data.get("results") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return []
        print(f"[resume] 已有 {len(rows)} 条 <- {path}", file=sys.stderr)
        return rows

    def build_stats(self, save: bool = True) -> Dict[str, Any]:
        """步骤 1：Excel 统计页 + 题目分类 → 结构化 stats.json，并写出 dataset.json。"""
        dataset = load_excel_dataset(
            excel_path=self.cfg.excel_path,
            questions_sheet=self.cfg.questions_sheet,
            stats_sheet=self.cfg.stats_sheet,
            design_sheet=self.cfg.design_sheet,
            n_limit=self.cfg.n_limit,
            id_list=self.cfg.id_list,
        )
        stats = dataset.get("stats") or build_stats_document(
            excel_path=self.cfg.excel_path,
            questions_sheet=self.cfg.questions_sheet,
            stats_sheet=self.cfg.stats_sheet,
            design_sheet=self.cfg.design_sheet,
        )
        self._dataset = dataset
        self._stats = stats

        if save:
            self.save_json(stats, self.cfg.stats_file())
            self.save_json(dataset, self.cfg.dataset_file())
            meta = stats.get("meta") or {}
            print(
                f"[stats] schema={meta.get('schema')}  "
                f"n={meta.get('n_questions')}  "
                f"primary={((stats.get('category_fields') or {}).get('primary'))}",
                file=sys.stderr,
            )

        return stats

    def _ensure_dataset(self) -> Dict[str, Any]:
        if self._dataset is not None:
            return self._dataset
        ds_path = self.cfg.dataset_file()
        if ds_path.is_file():
            data = self.load_json(ds_path)
            if "questions" in data:
                self._dataset = data
                if isinstance(data.get("stats"), dict):
                    self._stats = data["stats"]
                print(f"[loaded] dataset <- {ds_path}", file=sys.stderr)
                return data
        dataset = load_excel_dataset(
            excel_path=self.cfg.excel_path,
            questions_sheet=self.cfg.questions_sheet,
            stats_sheet=self.cfg.stats_sheet,
            design_sheet=self.cfg.design_sheet,
            n_limit=self.cfg.n_limit,
            id_list=self.cfg.id_list,
        )
        self._dataset = dataset
        self._stats = dataset.get("stats")
        return dataset

    def _ensure_stats(self) -> Optional[Dict[str, Any]]:
        if self._stats is not None:
            return self._stats
        path = self.cfg.stats_file()
        if path.is_file():
            self._stats = self.load_json(path)
            return self._stats
        if self._dataset and isinstance(self._dataset.get("stats"), dict):
            self._stats = self._dataset["stats"]
            return self._stats
        return None

    def evaluate(
        self,
        dataset: Optional[Dict[str, Any]] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """步骤 2：逐题超图 / 纯LLM 问答 + 评判。"""
        if dataset is None:
            dataset = self._ensure_dataset()

        self.setup_dhmf()
        self.setup_judge_llm()

        evaluator = ExcelQueryEvaluator(
            self.dhmf,
            judge_llm=self.judge_llm,
            judge_model_args=self.cfg.judge_model_args,
            query_mode=self.cfg.query_mode,
            use_cache=self.cfg.eval_use_cache,
            max_judge_retries=self.cfg.eval_max_retries,
            sleep_between=self.cfg.eval_sleep_between,
            num_thread=self.cfg.eval_num_thread,
            enable_llm_only=self.cfg.enable_llm_only,
        )

        out = self.cfg.eval_results_file() if save else None
        existing = []
        if save and self.cfg.eval_resume and out is not None:
            existing = self._load_existing_eval(out)

        def _on_progress(mid_report: dict, index: int, total: int) -> None:
            if out is None:
                return
            mid_report.setdefault("meta", {})["config"] = self.cfg.to_meta_snapshot()
            mid_report.setdefault("meta", {})["progress"] = {
                "done": index,
                "total": total,
                "pct": round(100.0 * index / total, 1) if total else None,
            }
            if dataset.get("stats"):
                mid_report.setdefault("stats_meta", dataset["stats"].get("meta"))
            self.save_json(mid_report, out, quiet=True)

        eval_data = evaluator.evaluate_all(
            dataset,
            existing_results=existing,
            on_progress=_on_progress if save else None,
        )
        eval_data.setdefault("meta", {})["config"] = self.cfg.to_meta_snapshot()
        if dataset.get("stats"):
            eval_data["stats"] = dataset["stats"]
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
        """步骤 3：两路各自汇总 + 按统计分类切分。"""
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

        ev_meta = eval_data.get("meta") if isinstance(eval_data.get("meta"), dict) else {}
        n_res = len(eval_data.get("results") or [])
        planned = ev_meta.get("n_total_planned") or ev_meta.get("n_questions")
        if ev_meta.get("done") is False:
            print(
                f"[report] 评测结果未完成：results={n_res} "
                f"planned={planned}  "
                f"（evals 是中断进度，report 只会汇总已写出的题；"
                f"请先把 run.mode 改回 evaluate 跑完）",
                file=sys.stderr,
            )

        stats = None
        if isinstance(eval_data.get("stats"), dict):
            stats = eval_data["stats"]
        else:
            stats = self._ensure_stats()

        report_doc = build_report_document(
            eval_data,
            stats=stats,
            source_path=str(src_path) if src_path is not None else None,
        )
        meta = report_doc.setdefault("meta", {})
        snap = self.cfg.to_meta_snapshot()
        for k in (
            "query_mode",
            "judge_model",
            "config_file",
            "dhmf_config_path",
            "excel_path",
            "enable_llm_only",
        ):
            if meta.get(k) is None and snap.get(k) is not None:
                meta[k] = snap[k]
        self._report = report_doc

        if save:
            out = self.cfg.report_file()
            self.save_json(report_doc, out)
            print(format_summary_text(report_doc.get("summary") or {}), file=sys.stderr)
            self.export_excel(
                report_doc=report_doc,
                eval_data=eval_data,
                save=True,
            )

        return report_doc

    def export_excel(
        self,
        report_doc: Optional[Dict[str, Any]] = None,
        eval_data: Optional[Dict[str, Any]] = None,
        *,
        save: bool = True,
    ) -> Path:
        """把 report / evals 摊成多 sheet 的 xlsx。"""
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
                        stats = None
                        if isinstance(eval_data.get("stats"), dict):
                            stats = eval_data["stats"]
                        else:
                            stats = self._ensure_stats()
                        report_doc = build_report_document(eval_data, stats=stats)
                        self._report = report_doc
        if report_doc is None:
            raise FileNotFoundError(
                "未找到 report / evals JSON，无法导出 Excel。\n"
                f"  report: {self.cfg.report_file()}\n"
                f"  evals:  {self.cfg.eval_results_file()}\n"
                "  请先 run.mode=report，或 python -m benchmark2.export_excel。"
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
        self.build_stats(save=True)
        eval_data = self.evaluate(dataset=self._dataset, save=True)
        return self.report(eval_data=eval_data, save=True)

    def run(self) -> Any:
        mode = (self.cfg.run_mode or "all").strip().lower()
        print(f"[benchmark2] mode={mode}", file=sys.stderr)
        if mode == "stats":
            return self.build_stats(save=True)
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
            "Set run.mode in benchmark2/config.yaml: "
            "stats | evaluate | report | excel | all"
        )
