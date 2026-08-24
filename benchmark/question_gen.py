"""多跳问题生成：从 doc 表抽样 + 本地 LLM 出题（可多线程）。"""

from __future__ import annotations

import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .prompts import Benchmark_PROMPT
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

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None

class QuestionGenerator:
    """
    按跳数配置从 doc.content 抽样，调用 LLM 生成场景问题。

    hop_counts 示例: {1: 50, 2: 30, 3: 20}
    - 1 跳：每次抽 1 条 content
    - 2 跳：每次抽 2 条 content
    - n 跳：每次抽 n 条 content
    跨题可重复抽样；同一题内尽量不重复文档。

    多线程：主线程按 seed 预抽样（可复现），线程池只并行 LLM 出题。
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
        num_thread: int = 1,
        question_gen_prompt: str = "QUESTION_GEN_USER",
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
        try:
            nt = int(num_thread)
        except (TypeError, ValueError):
            nt = 1
        self.num_thread = max(1, nt)
        self.question_gen_prompt = self._resolve_prompt_key(question_gen_prompt)

        self._rng = random.Random(self.seed)
        self.docs: List[Dict[str, Any]] = []

    @staticmethod
    def _resolve_prompt_key(name: str) -> str:
        """校验并返回 Benchmark_PROMPT 中的出题 user prompt 键。"""
        key = str(name or "QUESTION_GEN_USER").strip() or "QUESTION_GEN_USER"
        if key not in Benchmark_PROMPT:
            allowed = sorted(
                k
                for k in Benchmark_PROMPT
                if k.startswith("QUESTION_GEN") and k != "QUESTION_GEN_SYSTEM"
            )
            raise ValueError(
                f"Unknown question_gen_prompt={name!r}. "
                f"Available: {allowed}"
            )
        if key == "QUESTION_GEN_SYSTEM":
            raise ValueError(
                "question_gen_prompt 不能使用 QUESTION_GEN_SYSTEM（那是 system 提示）"
            )
        return key

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
        template = Benchmark_PROMPT[self.question_gen_prompt]
        return template.format(
            hop=hop,
            n_docs=len(docs),
            docs_block=format_docs_block(
                docs, self.max_chars_per_doc, numbered=False
            ),
        )

    def generate_one(
        self,
        hop: int,
        q_index: int,
        docs: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """生成单条问题；失败时带 error 字段仍返回结构。docs 可预传入（多线程时主线程抽样）。"""
        docs = list(docs) if docs is not None else self.sample_docs(hop)
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
                system=Benchmark_PROMPT.get('QUESTION_GEN_SYSTEM', ''),
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
            if self.sleep_between > 0:
                time.sleep(self.sleep_between)
            return base

        base["gen_error"] = last_err or "unknown failure"
        if self.sleep_between > 0:
            time.sleep(self.sleep_between)
        return base

    def _plan_tasks(self) -> List[Tuple[int, int, List[Dict[str, Any]]]]:
        """
        主线程按 seed 顺序预抽样，保证多线程下文档抽样可复现。
        返回 [(hop, q_index, docs), ...]
        """
        if not self.docs:
            self.load_docs()
        planned: List[Tuple[int, int, List[Dict[str, Any]]]] = []
        for hop, count in self.hop_counts.items():
            for i in range(1, int(count) + 1):
                planned.append((int(hop), i, self.sample_docs(int(hop))))
        return planned

    def generate_all(self) -> Dict[str, Any]:
        """按 hop_counts 批量生成（num_thread>1 时并行 LLM），返回完整数据集 dict。"""
        planned = self._plan_tasks()
        n = len(planned)
        questions: List[Optional[Dict[str, Any]]] = [None] * n
        fail_n = 0
        done_n = 0
        t0 = time.perf_counter()
        workers = min(self.num_thread, max(1, n))

        def _on_item(item: Dict[str, Any]) -> None:
            nonlocal fail_n, done_n
            done_n += 1
            if item.get("gen_status") != 1:
                fail_n += 1
                err = item.get("gen_error") or "unknown"
                fail_print(
                    f"生成 {item.get('id')} hop={item.get('hop')} | {err} | "
                    f"sources={item.get('source_names')}"
                )

        if workers <= 1:
            pbar = progress_iter(range(n), total=n, desc="生成问题", unit="题")
            for idx in pbar:
                hop, i, docs = planned[idx]
                item = self.generate_one(hop, i, docs=docs)
                questions[idx] = item
                _on_item(item)
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(
                        ok=done_n - fail_n, fail=fail_n, thr=1, refresh=False
                    )
        else:
            if _tqdm is not None:
                pbar = _tqdm(
                    total=n,
                    desc=f"生成问题×{workers}",
                    unit="题",
                    file=sys.stderr,
                    dynamic_ncols=True,
                    leave=True,
                    mininterval=0.2,
                )
            else:
                pbar = None

            with ThreadPoolExecutor(max_workers=workers) as pool:
                fut_to_idx = {
                    pool.submit(self.generate_one, hop, i, docs): idx
                    for idx, (hop, i, docs) in enumerate(planned)
                }
                for fut in as_completed(fut_to_idx):
                    idx = fut_to_idx[fut]
                    try:
                        item = fut.result()
                    except Exception as e:
                        hop, i, docs = planned[idx]
                        item = {
                            "id": f"q_{hop}hop_{i:04d}",
                            "hop": hop,
                            "question": "",
                            "ground_truth_answer": "",
                            "explanation": "",
                            "source_docs": [
                                {
                                    "doc_id": d.get("doc_id"),
                                    "name": d.get("name"),
                                    "content": d.get("content"),
                                }
                                for d in docs
                            ],
                            "source_names": [d.get("name") for d in docs],
                            "gen_status": 0,
                            "gen_error": f"worker exception: {e}",
                            "gen_latency_s": None,
                            "gen_usage": {},
                        }
                    questions[idx] = item
                    _on_item(item)
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(
                            ok=done_n - fail_n, fail=fail_n, thr=workers, refresh=False
                        )
                    else:
                        print(
                            f"\r生成问题×{workers}: {done_n}/{n} "
                            f"ok={done_n - fail_n} fail={fail_n}",
                            end="",
                            file=sys.stderr,
                            flush=True,
                        )
            if pbar is not None:
                pbar.close()
            elif n:
                print(file=sys.stderr)

        out_questions: List[Dict[str, Any]] = [
            q for q in questions if q is not None
        ]
        ok_n = sum(1 for q in out_questions if q.get("gen_status") == 1)
        dataset = {
            "meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "db_path": self.db_path,
                "hop_counts": {str(k): v for k, v in self.hop_counts.items()},
                "seed": self.seed,
                "question_gen_prompt": self.question_gen_prompt,
                "num_thread": workers,
                "model": (self.model_args or {}).get("model"),
                "total": len(out_questions),
                "success": ok_n,
                "failed": len(out_questions) - ok_n,
                "elapsed_s": round(time.perf_counter() - t0, 3),
            },
            "questions": out_questions,
        }
        return dataset
