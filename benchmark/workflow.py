"""
多跳测试集生成 + DHMF RAG 评测主工作流。

典型用法::

    from benchmark.workflow import TestQueryWorkflow

    # 推荐：全部设置走 config.yaml
    wf = TestQueryWorkflow.from_config("benchmark/config.yaml")
    report = wf.run_all()

    # 或代码里覆盖 hop / 路径
    wf = TestQueryWorkflow.from_config(
        "benchmark/config.yaml",
        hop_counts={1: 10, 2: 5},
    )
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .config import BenchmarkConfig, DEFAULT_CONFIG_PATH
from .evaluator import QueryEvaluator
from .question_gen import QuestionGenerator
from .utils import parse_hop_spec, project_root, resolve_path

logger = logging.getLogger("benchmark.workflow")


class TestQueryWorkflow:
    """
    主调用类：把「问题生成」与「RAG 评测」组织成工作流。

    方法：
      - setup_llm()           初始化出题 LLM
      - setup_judge_llm()     初始化评判 LLM（可与出题不同端点）
      - setup_dhmf()          加载 DHMF（评测需要）
      - generate_questions()  抽样 + LLM 生成多跳问题 JSON
      - evaluate()            query + 评判 + 汇总
      - run_all()             生成 → 评测一条龙
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
        **flat_overrides,
    ) -> "TestQueryWorkflow":
        """
        从 yaml 加载；flat_overrides 可覆盖 BenchmarkConfig 扁平字段，例如::

            hop_counts={1:10}, questions_path='...', seed=1
        """
        # 过滤 None，避免把未传 CLI 项冲掉
        clean = {k: v for k, v in flat_overrides.items() if v is not None}
        cfg = BenchmarkConfig.from_yaml(config_path, overrides=clean if clean else None)
        if clean:
            cfg.apply_flat_overrides(clean)
        return cls(cfg)

    @classmethod
    def from_dict(cls, data: dict) -> "TestQueryWorkflow":
        """从嵌套 dict（与 yaml 同结构）构造。"""
        cfg = BenchmarkConfig.from_yaml(None, overrides=data)
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
        # 可选：覆盖 retrieve.use_cache
        if self.cfg.dhmf_retrieve_use_cache is not None:
            try:
                self.dhmf_config.retrieve.use_cache = bool(
                    self.cfg.dhmf_retrieve_use_cache
                )
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
    def _stable_questions_path(self) -> Path:
        """
        问题集稳定路径（generate / evaluate 独立运行时共用）。
        优先 config.paths.questions_path；否则 {output_dir}/questions.json
        """
        if self.cfg.questions_path:
            p = Path(self.cfg.questions_path)
            return p if p.is_absolute() else resolve_path(p)
        return self.output_dir / "questions.json"

    def _default_dataset_path(self) -> Path:
        """generate 默认写出路径 = 稳定问题集路径。"""
        return self._stable_questions_path()

    def _default_report_path(self) -> Path:
        if self.cfg.report_path:
            p = Path(self.cfg.report_path)
            return p if p.is_absolute() else resolve_path(p)
        # 稳定默认，便于只跑 evaluate 时也能找到最近一次报告之外的新结果
        # 仍带时间戳避免覆盖历史报告；问题集用稳定名
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"eval_report_{ts}.json"

    def _summary_path_for(self, report_path: Path) -> Path:
        if self.cfg.summary_path:
            p = Path(self.cfg.summary_path)
            return p if p.is_absolute() else resolve_path(p)
        return report_path.with_suffix(".summary.txt")

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        return p if p.is_absolute() else resolve_path(p)

    def resolve_questions_path(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """
        解析评测用问题集文件路径（存在才返回）。

        查找顺序：
          1) 显式 dataset_path 参数
          2) config.paths.dataset_path
          3) config.paths.questions_path
          4) {output_dir}/questions.json（稳定默认）
          5) {output_dir}/questions_*.json 中最新一份（兼容旧时间戳命名）
        """
        candidates: list[Path] = []
        for raw in (
            dataset_path,
            self.cfg.dataset_path,
            self.cfg.questions_path,
        ):
            if raw:
                candidates.append(self._resolve_path(raw))

        stable = self._stable_questions_path()
        if stable not in candidates:
            candidates.append(stable)

        for p in candidates:
            if p.is_file():
                return p

        # 兼容历史：questions_YYYYMMDD_HHMMSS.json
        pattern_hits = sorted(
            self.output_dir.glob("questions_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if pattern_hits:
            return pattern_hits[0]
        return None

    def save_json(self, data: dict, path: Union[str, Path]) -> Path:
        path = self._resolve_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # 结束时一行提示保存位置（非中间过程）
        print(f"[saved] {path}", file=sys.stderr)
        return path

    def load_dataset(self, path: Union[str, Path]) -> Dict[str, Any]:
        path = self._resolve_path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "questions" not in data:
            raise ValueError(f"非法数据集（缺少 questions）: {path}")
        self._dataset = data
        # 记住路径，便于同进程再次 evaluate / 配置回写
        self.cfg.questions_path = str(path)
        print(f"[loaded] questions <- {path}", file=sys.stderr)
        return data

    # ------------------------------------------------------------------
    # steps
    # ------------------------------------------------------------------
    def generate_questions(
        self,
        hop_counts: Optional[Dict[Any, int]] = None,
        output_path: Optional[Union[str, Path]] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        步骤 1：从 doc 表按跳数抽样，LLM 生成多跳场景问题。

        默认写入稳定路径 {output_dir}/questions.json（或 config.paths.questions_path），
        之后可单独调用 evaluate() 读取，无需同进程先 generate。
        """
        self.setup_llm()
        counts = (
            parse_hop_spec(hop_counts) if hop_counts is not None else self.cfg.hop_counts
        )
        gen = QuestionGenerator(
            self.llm,
            model_args=self.cfg.gen_model_args,
            db_path=self.cfg.db_path,
            hop_counts=counts,
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
            out = (
                self._resolve_path(output_path)
                if output_path
                else self._default_dataset_path()
            )
            self.save_json(dataset, out)
            dataset.setdefault("meta", {})["saved_path"] = str(out)
            # 回写配置路径，evaluate() 同进程/跨进程都能找到
            self.cfg.questions_path = str(out)
            self.cfg.dataset_path = str(out)

        return dataset

    def evaluate(
        self,
        dataset: Optional[Dict[str, Any]] = None,
        dataset_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        save: bool = True,
        save_summary_txt: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        步骤 2：DHMF.query → LLM 评判 →（可选）文档召回与 token/时延统计。

        可独立运行：自动从 questions_path / output_dir/questions.json /
        最新 questions_*.json 加载问题集。
        """
        if dataset is None:
            if self._dataset is not None and dataset_path is None:
                dataset = self._dataset
            else:
                path = self.resolve_questions_path(dataset_path)
                if path is not None:
                    dataset = self.load_dataset(path)
                else:
                    tried = [
                        dataset_path,
                        self.cfg.dataset_path,
                        self.cfg.questions_path,
                        str(self._stable_questions_path()),
                        str(self.output_dir / "questions_*.json"),
                    ]
                    tried_s = ", ".join(str(x) for x in tried if x)
                    raise FileNotFoundError(
                        "未找到问题集 JSON，无法 evaluate。\n"
                        f"  已尝试: {tried_s}\n"
                        "  请先运行 wf.generate_questions()，或在 config.paths 设置 "
                        "questions_path / dataset_path，或传入 dataset= / dataset_path=。"
                    )

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

        do_summary = (
            self.cfg.save_summary_txt
            if save_summary_txt is None
            else bool(save_summary_txt)
        )

        if save:
            out = Path(output_path) if output_path else self._default_report_path()
            if not out.is_absolute():
                out = resolve_path(out)
            self.save_json(report, out)
            report.setdefault("meta", {})["saved_path"] = str(out)
            if do_summary and table:
                txt_path = self._summary_path_for(out)
                txt_path.parent.mkdir(parents=True, exist_ok=True)
                txt_path.write_text(table, encoding="utf-8")
                print(f"[saved] {txt_path}", file=sys.stderr)
                report.setdefault("meta", {})["summary_path"] = str(txt_path)

        return report

    def run_all(
        self,
        hop_counts: Optional[Dict[Any, int]] = None,
        questions_path: Optional[Union[str, Path]] = None,
        report_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """生成问题 + 评测一条龙。"""
        q_out = questions_path or self.cfg.questions_path
        r_out = report_path or self.cfg.report_path
        dataset = self.generate_questions(
            hop_counts=hop_counts,
            output_path=q_out,
            save=True,
        )
        report = self.evaluate(
            dataset=dataset,
            output_path=r_out,
            save=True,
        )
        return report
