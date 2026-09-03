"""多跳问题生成：从 doc 表指定 id 范围抽样，按跳数成组交给 LLM 出题。"""

from __future__ import annotations

import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .prompts import Benchmark_PROMPT
from .utils import (
    call_llm,
    extract_json_object,
    fail_print,
    format_docs_block,
    load_doc_index_from_db,
    load_docs_by_ids,
    parse_hop_spec,
    progress_iter,
    resolve_path,
)

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None

OnProgress = Optional[Callable[[Dict[str, Any], int, int], None]]


class QuestionGenerator:
    """
    hop_counts 示例: {1: 10, 2: 50}
    - 1 跳 10 题：抽 10 篇文档，每篇单独出题
    - 2 跳 50 题：抽 50*2 篇，每两篇一组交给 LLM
    同一跳内尽量不重复文档；文档不足时才跨题复用。
    """

    def __init__(
        self,
        llm,
        *,
        model_args: Optional[dict] = None,
        db_path: str = "example/a/DB/main.db",
        hop_counts: Optional[Dict[Any, int]] = None,
        seed: int = 42,
        max_chars_per_doc: int = -1,
        use_cache: bool = False,
        max_retries: int = 3,
        sleep_between: float = 0.0,
        num_thread: int = 1,
        question_gen_prompt: str = "QUESTION_GEN_USER",
        doc_id_min: Optional[int] = None,
        doc_id_max: Optional[int] = None,
    ):
        self.llm = llm
        self.model_args = dict(model_args or {})
        self.model_args.setdefault("temperature", 0.4)
        self.model_args.setdefault("enable_thinking", False)
        self.model_args.setdefault("response_format", {"type": "json_object"})

        self.db_path = str(resolve_path(db_path))
        self.hop_counts = parse_hop_spec(hop_counts or {1: 5, 2: 3})
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
        self.doc_id_min = doc_id_min
        self.doc_id_max = doc_id_max

        self._rng = random.Random(self.seed)
        self.docs: List[Dict[str, Any]] = []

    @staticmethod
    def _resolve_prompt_key(name: str) -> str:
        key = str(name or "QUESTION_GEN_USER").strip() or "QUESTION_GEN_USER"
        if key not in Benchmark_PROMPT:
            allowed = sorted(
                k
                for k in Benchmark_PROMPT
                if k.startswith("QUESTION_GEN") and k != "QUESTION_GEN_SYSTEM"
            )
            raise ValueError(
                f"Unknown question_gen_prompt={name!r}. Available: {allowed}"
            )
        if key == "QUESTION_GEN_SYSTEM":
            raise ValueError("question_gen_prompt 不能使用 QUESTION_GEN_SYSTEM")
        return key

    def load_docs(self, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
        path = db_path or self.db_path
        self.docs = load_doc_index_from_db(
            path, doc_id_min=self.doc_id_min, doc_id_max=self.doc_id_max
        )
        self.db_path = str(resolve_path(path))
        return self.docs

    def _fill_content(self, groups: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        ids = [d["doc_id"] for g in groups for d in g]
        full = {d["doc_id"]: d for d in load_docs_by_ids(self.db_path, ids)}
        out: List[List[Dict[str, Any]]] = []
        for g in groups:
            filled = []
            for d in g:
                item = full.get(d["doc_id"])
                if item is None:
                    raise RuntimeError(f"抽样到的 doc_id={d['doc_id']} 读不到 content")
                filled.append(item)
            out.append(filled)
        return out

    def _sample_groups(self, hop: int, count: int) -> List[List[Dict[str, Any]]]:
        """抽 hop*count 篇文档，按 hop 切成 count 组。文档够则组间不重复。"""
        if not self.docs:
            self.load_docs()
        n = max(1, int(hop))
        k = max(1, int(count))
        pool = list(self.docs)
        need = n * k
        if len(pool) >= need:
            picked = self._rng.sample(pool, need)
            return [picked[i * n:(i + 1) * n] for i in range(k)]
        groups: List[List[Dict[str, Any]]] = []
        for _ in range(k):
            if len(pool) >= n:
                groups.append(self._rng.sample(pool, n))
            else:
                chosen = list(pool)
                while len(chosen) < n:
                    chosen.append(self._rng.choice(pool))
                self._rng.shuffle(chosen)
                groups.append(chosen[:n])
        return groups

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
        if docs is None:
            docs = self._fill_content(self._sample_groups(hop, 1))[0]
        docs = list(docs)
        if any(not (d.get("content") or "").strip() for d in docs):
            docs = self._fill_content([docs])[0]
        source_docs = [
            {"doc_id": d["doc_id"], "name": d["name"], "content": d["content"]}
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
            "source_doc_ids": [d["doc_id"] for d in docs],
            "gen_status": 0,
            "gen_error": None,
            "gen_latency_s": None,
            "gen_usage": {},
        }

        user = self._build_user_prompt(hop, docs)
        last_err = None
        for _attempt in range(1, self.max_retries + 1):
            resp = call_llm(
                self.llm,
                system=Benchmark_PROMPT.get("QUESTION_GEN_SYSTEM", ""),
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
        if not self.docs:
            self.load_docs()
        planned: List[Tuple[int, int, List[Dict[str, Any]]]] = []
        for hop, count in self.hop_counts.items():
            groups = self._fill_content(self._sample_groups(int(hop), int(count)))
            for i, docs in enumerate(groups, start=1):
                planned.append((int(hop), i, docs))
        return planned

    def _dataset_shell(
        self,
        questions: List[Optional[Dict[str, Any]]],
        *,
        t0: float,
        workers: int,
        done: bool,
    ) -> Dict[str, Any]:
        out_q = [q for q in questions if q is not None]
        ok_n = sum(1 for q in out_q if q.get("gen_status") == 1)
        return {
            "meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "db_path": self.db_path,
                "hop_counts": {str(k): v for k, v in self.hop_counts.items()},
                "seed": self.seed,
                "question_gen_prompt": self.question_gen_prompt,
                "doc_id_min": self.doc_id_min,
                "doc_id_max": self.doc_id_max,
                "n_source_docs": len(self.docs),
                "num_thread": workers,
                "model": (self.model_args or {}).get("model"),
                "total": len(out_q),
                "success": ok_n,
                "failed": len(out_q) - ok_n,
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "done": bool(done),
            },
            "questions": out_q,
        }

    def generate_all(
        self,
        *,
        on_progress: OnProgress = None,
        existing_questions: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        planned = self._plan_tasks()
        n = len(planned)
        questions: List[Optional[Dict[str, Any]]] = [None] * n
        by_id = {}
        for q in existing_questions or []:
            if isinstance(q, dict) and q.get("id") and q.get("gen_status") == 1:
                by_id[str(q["id"])] = q
        n_resumed = 0
        todo_idx: List[int] = []
        for idx, (hop, i, _docs) in enumerate(planned):
            qid = f"q_{hop}hop_{i:04d}"
            if qid in by_id:
                questions[idx] = by_id[qid]
                n_resumed += 1
            else:
                todo_idx.append(idx)
        if n_resumed:
            print(f"[resume] 出题已完成 {n_resumed}/{n}，待生成 {len(todo_idx)}", file=sys.stderr)
        fail_n = 0
        done_n = n_resumed
        t0 = time.perf_counter()
        workers = min(self.num_thread, max(1, len(todo_idx))) if todo_idx else 1

        n_needed = sum(h * c for h, c in self.hop_counts.items())
        print(
            f"[generate] docs={len(self.docs)} range=[{self.doc_id_min}, {self.doc_id_max}] "
            f"hop_counts={self.hop_counts} need_docs={n_needed} questions={n}",
            file=sys.stderr,
        )
        if len(self.docs) < n_needed:
            print(
                f"[generate] 范围内文档 {len(self.docs)} < 需求 {n_needed}，"
                "同一跳内题与题之间可能复用文档",
                file=sys.stderr,
            )

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

        def _emit() -> None:
            if on_progress is None:
                return
            mid = self._dataset_shell(questions, t0=t0, workers=workers, done=False)
            try:
                on_progress(mid, done_n, n)
            except Exception as e:
                fail_print(f"on_progress 保存失败: {e}")

        if not todo_idx:
            return self._dataset_shell(questions, t0=t0, workers=workers, done=True)

        if workers <= 1:
            pbar = progress_iter(todo_idx, total=len(todo_idx), desc="生成问题", unit="题")
            for idx in pbar:
                hop, i, docs = planned[idx]
                item = self.generate_one(hop, i, docs=docs)
                questions[idx] = item
                _on_item(item)
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(
                        ok=done_n - fail_n, fail=fail_n, thr=1, refresh=False
                    )
                _emit()
        else:
            if _tqdm is not None:
                pbar = _tqdm(
                    total=len(todo_idx),
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
                    pool.submit(self.generate_one, *planned[idx]): idx
                    for idx in todo_idx
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
                            "source_doc_ids": [d.get("doc_id") for d in docs],
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
                    _emit()
            if pbar is not None:
                pbar.close()
            elif n:
                print(file=sys.stderr)

        dataset = self._dataset_shell(questions, t0=t0, workers=workers, done=True)
        return dataset
