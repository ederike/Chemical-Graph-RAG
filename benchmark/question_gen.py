"""多跳问题生成：从 doc 表抽样 + 本地 LLM 出题。"""

from __future__ import annotations

import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .prompts import QUESTION_GEN_SYSTEM, QUESTION_GEN_USER
from .utils import (
    call_llm,
    extract_json_object,
    fail_print,
    format_docs_block,
    load_docs_from_db,
    parse_hop_spec,
    progress_iter,
    resolve_path,
)


class QuestionGenerator:
    """
    按跳数配置从 doc.content 抽样，调用 LLM 生成场景问题。

    hop_counts 示例: {1: 50, 2: 30, 3: 20}
    - 1 跳：每次抽 1 条 content
    - 2 跳：每次抽 2 条 content
    - n 跳：每次抽 n 条 content
    跨题可重复抽样；同一题内尽量不重复文档。
    """

    def __init__(
        self,
        llm,
        *,
        model_args: Optional[dict] = None,
        db_path: str = "example/a/DB/main.db",
        hop_counts: Optional[Dict[Any, int]] = None,
        seed: int = 42,
        max_chars_per_doc: int = 4000,
        use_cache: bool = False,
        max_retries: int = 3,
        sleep_between: float = 0.0,
    ):
        self.llm = llm
        self.model_args = dict(model_args or {})
        # 出题需要一定创造性，若未指定 temperature 给个温和值
        self.model_args.setdefault("temperature", 0.4)
        self.model_args.setdefault("enable_thinking", False)
        # 强制 JSON 输出（本地 Qwen 兼容时有效）
        self.model_args.setdefault(
            "response_format", {"type": "json_object"}
        )

        self.db_path = str(resolve_path(db_path))
        self.hop_counts = parse_hop_spec(hop_counts or {1: 5, 2: 3, 3: 2})
        self.seed = int(seed)
        self.max_chars_per_doc = int(max_chars_per_doc)
        self.use_cache = bool(use_cache)
        self.max_retries = max(1, int(max_retries))
        self.sleep_between = float(sleep_between)

        self._rng = random.Random(self.seed)
        self.docs: List[Dict[str, Any]] = []

    def load_docs(self, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
        path = db_path or self.db_path
        self.docs = load_docs_from_db(path)
        self.db_path = str(resolve_path(path))
        return self.docs

    def sample_docs(self, hop: int) -> List[Dict[str, Any]]:
        """同一题内尽量无重复；文档数不足时允许重复填充。"""
        if not self.docs:
            self.load_docs()
        n = max(1, int(hop))
        pool = list(self.docs)
        if len(pool) >= n:
            return self._rng.sample(pool, n)
        # 不足 n 篇：先全取再有放回补齐
        chosen = list(pool)
        while len(chosen) < n:
            chosen.append(self._rng.choice(pool))
        self._rng.shuffle(chosen)
        return chosen[:n]

    def _build_user_prompt(self, hop: int, docs: Sequence[Dict[str, Any]]) -> str:
        return QUESTION_GEN_USER.format(
            hop=hop,
            n_docs=len(docs),
            docs_block=format_docs_block(docs, self.max_chars_per_doc),
        )

    def generate_one(self, hop: int, q_index: int) -> Dict[str, Any]:
        """生成单条问题；失败时带 error 字段仍返回结构。"""
        docs = self.sample_docs(hop)
        source_docs = [
            {
                "doc_id": d["doc_id"],
                "name": d["name"],
                "content": d["content"],
            }
            for d in docs
        ]
        base = {
            "id": f"q_{hop}hop_{q_index:04d}",
            "hop": hop,
            "question": "",
            "ground_truth_answer": "",
            "explanation": "",
            "source_docs": source_docs,
            "source_names": [d["name"] for d in docs],
            "gen_status": 0,
            "gen_error": None,
            "gen_latency_s": None,
            "gen_usage": {},
        }

        user = self._build_user_prompt(hop, docs)
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            resp = call_llm(
                self.llm,
                system=QUESTION_GEN_SYSTEM,
                user=user,
                model_args=self.model_args,
                use_cache=self.use_cache,
            )
            base["gen_latency_s"] = resp.get("latency_s")
            base["gen_usage"] = {
                "prompt_tokens": resp.get("usage_prompt_tokens"),
                "completion_tokens": resp.get("usage_completion_tokens"),
                "total_tokens": resp.get("usage_total_tokens"),
            }
            if resp.get("status") != 1:
                last_err = (
                    f"llm status={resp.get('status')} "
                    f"answer={str(resp.get('answer'))[:200]}"
                )
                continue

            obj = extract_json_object(resp.get("answer") or "")
            if not obj:
                last_err = f"json parse failed: {str(resp.get('answer'))[:200]}"
                continue

            q = (obj.get("question") or "").strip()
            ans = (obj.get("ground_truth_answer") or obj.get("answer") or "").strip()
            exp = (obj.get("explanation") or "").strip()
            if not q or not ans:
                last_err = f"missing question/answer fields: keys={list(obj.keys())}"
                continue

            base["question"] = q
            base["ground_truth_answer"] = ans
            base["explanation"] = exp
            base["gen_status"] = 1
            base["gen_error"] = None
            return base

        base["gen_error"] = last_err or "unknown failure"
        return base

    def generate_all(self) -> Dict[str, Any]:
        """按 hop_counts 批量生成，返回完整数据集 dict。"""
        if not self.docs:
            self.load_docs()

        # 展开任务列表，统一进度条
        tasks: List[tuple] = []
        for hop, count in self.hop_counts.items():
            for i in range(1, count + 1):
                tasks.append((hop, i))

        questions: List[Dict[str, Any]] = []
        fail_n = 0
        t0 = time.perf_counter()
        pbar = progress_iter(tasks, total=len(tasks), desc="生成问题", unit="题")
        for hop, i in pbar:
            item = self.generate_one(hop, i)
            questions.append(item)
            if item.get("gen_status") != 1:
                fail_n += 1
                err = item.get("gen_error") or "unknown"
                fail_print(
                    f"生成 {item.get('id')} hop={hop} | {err} | "
                    f"sources={item.get('source_names')}"
                )
            # 进度条 postfix 只显示计数，不刷内容
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(ok=len(questions) - fail_n, fail=fail_n, refresh=False)
            if self.sleep_between > 0:
                time.sleep(self.sleep_between)

        ok_n = sum(1 for q in questions if q.get("gen_status") == 1)
        dataset = {
            "meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "db_path": self.db_path,
                "hop_counts": {str(k): v for k, v in self.hop_counts.items()},
                "seed": self.seed,
                "model": (self.model_args or {}).get("model"),
                "total": len(questions),
                "success": ok_n,
                "failed": len(questions) - ok_n,
                "elapsed_s": round(time.perf_counter() - t0, 3),
            },
            "questions": questions,
        }
        return dataset
