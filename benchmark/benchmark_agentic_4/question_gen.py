"""多类型配方问题生成：从专利 doc 抽样，按题型分别用 LLM 出题。"""

from __future__ import annotations

import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .prompts import QUESTION_GEN_PROMPT_KEYS, Benchmark_PROMPT
from .utils import (
    call_llm,
    extract_json_object,
    fail_print,
    format_docs_block,
    llm_attempt_should_retry,
    load_doc_index_from_db,
    load_docs_by_ids,
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
    type_counts 示例: {product_to_formula: 10, application_to_formula: 10}
    - 共抽 sum(counts) 篇专利（尽量不重复）
    - 一篇文档生成一道对应类型的题
    """

    def __init__(
        self,
        llm,
        *,
        model_args: Optional[dict] = None,
        db_path: str = "example/a/DB/main.db",
        type_counts: Optional[Dict[str, int]] = None,
        seed: int = 42,
        max_chars_per_doc: int = 12000,
        use_cache: bool = False,
        max_retries: int = 3,
        sleep_between: float = 0.0,
        num_thread: int = 1,
        doc_id_min: Optional[int] = None,
        doc_id_max: Optional[int] = None,
        doc_id_range: Optional[object] = None,
        n_limit: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        self.llm = llm
        self.model_args = dict(model_args or {})
        self.model_args.setdefault("temperature", 0.2)
        self.model_args.setdefault("enable_thinking", False)
        self.model_args.setdefault("response_format", {"type": "json_object"})

        self.db_path = str(resolve_path(db_path))
        self.type_counts = {
            str(k): int(v) for k, v in (type_counts or {}).items() if int(v) > 0
        }
        if not self.type_counts:
            raise ValueError("type_counts 为空")
        for qt in self.type_counts:
            if qt not in QUESTION_GEN_PROMPT_KEYS:
                raise ValueError(f"未知 question_type={qt}")

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
        self.doc_id_min = doc_id_min
        self.doc_id_max = doc_id_max
        if doc_id_range is not None:
            self.doc_id_range = doc_id_range
        elif doc_id_min is None and doc_id_max is None:
            self.doc_id_range = None
        elif doc_id_min == 0 and doc_id_max is not None:
            self.doc_id_range = doc_id_max
        else:
            self.doc_id_range = [doc_id_min, doc_id_max]
        self.n_limit = int(n_limit) if n_limit and int(n_limit) > 0 else None
        self.timeout = float(timeout) if timeout is not None else None

        self._rng = random.Random(self.seed)
        self.docs: List[Dict[str, Any]] = []

    def load_docs(self, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
        path = db_path or self.db_path
        self.docs = load_doc_index_from_db(
            path,
            doc_id_min=self.doc_id_min,
            doc_id_max=self.doc_id_max,
        )
        self.db_path = str(resolve_path(path))
        return self.docs

    def _fill_content(self, docs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ids = [d["doc_id"] for d in docs]
        full = {d["doc_id"]: d for d in load_docs_by_ids(self.db_path, ids)}
        out = []
        for d in docs:
            item = full.get(d["doc_id"])
            if item is None:
                raise RuntimeError(f"抽样的 doc_id={d['doc_id']} 读不到 content")
            out.append(item)
        return out

    def _sample_assignments(self) -> List[Tuple[str, int, Dict[str, Any]]]:
        """返回 [(question_type, local_index, doc_meta), ...]。尽量不重复文档。"""
        if not self.docs:
            self.load_docs()
        need = sum(self.type_counts.values())
        pool = list(self.docs)
        if len(pool) >= need:
            picked = self._rng.sample(pool, need)
        else:
            picked = list(pool)
            while len(picked) < need:
                picked.append(self._rng.choice(pool))
            self._rng.shuffle(picked)

        assignments: List[Tuple[str, int, Dict[str, Any]]] = []
        cursor = 0
        for qt, count in self.type_counts.items():
            for i in range(1, count + 1):
                assignments.append((qt, i, picked[cursor]))
                cursor += 1

        if self.n_limit is not None:
            assignments = assignments[: self.n_limit]
        return assignments

    def _build_user_prompt(self, question_type: str, docs: Sequence[Dict[str, Any]]) -> str:
        key = QUESTION_GEN_PROMPT_KEYS[question_type]
        template = Benchmark_PROMPT[key]
        return template.format(
            docs_block=format_docs_block(
                docs, self.max_chars_per_doc, numbered=False
            ),
        )

    def generate_one(
        self,
        question_type: str,
        q_index: int,
        docs: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if docs is None:
            raise ValueError("docs 不能为空")
        docs = list(docs)
        if any(not (d.get("content") or "").strip() for d in docs):
            docs = self._fill_content(docs)
        source_docs = [
            {"doc_id": d["doc_id"], "name": d["name"], "content": d["content"]}
            for d in docs
        ]
        qid = f"q_{question_type}_{q_index:04d}"
        base: Dict[str, Any] = {
            "id": qid,
            "question_type": question_type,
            "question": "",
            "gold_answer": "",
            "ground_truth_answer": "",
            "explanation": "",
            "entities": {},
            "source_docs": source_docs,
            "source_names": [d["name"] for d in docs],
            "source_doc_ids": [d["doc_id"] for d in docs],
            "gen_status": 0,
            "gen_error": None,
            "gen_latency_s": None,
            "gen_usage": {},
        }

        user = self._build_user_prompt(question_type, docs)
        last_err = None
        for attempt in range(1, self.max_retries + 1):
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
                if not llm_attempt_should_retry(
                    resp, attempt=attempt, max_retries=self.max_retries, timeout_s=self.timeout
                ):
                    break
                continue

            obj = extract_json_object(resp.get("answer") or "")
            if not isinstance(obj, dict):
                last_err = "JSON 解析失败"
                if attempt < self.max_retries:
                    continue
                break

            question = str(obj.get("question") or "").strip()
            gold = str(
                obj.get("gold_answer")
                or obj.get("ground_truth_answer")
                or ""
            ).strip()
            explanation = str(obj.get("explanation") or "").strip()
            entities = obj.get("entities") if isinstance(obj.get("entities"), dict) else {}
            qt = str(obj.get("question_type") or question_type).strip() or question_type
            if not question or not gold:
                last_err = "缺少 question 或 gold_answer"
                if attempt < self.max_retries:
                    continue
                break

            base.update(
                {
                    "question": question,
                    "gold_answer": gold,
                    "ground_truth_answer": gold,
                    "explanation": explanation,
                    "entities": entities,
                    "question_type": qt,
                    "gen_status": 1,
                    "gen_error": None,
                }
            )
            if self.sleep_between > 0:
                time.sleep(self.sleep_between)
            return base

        base["gen_error"] = last_err or "unknown"
        return base

    def _plan_tasks(self) -> List[Tuple[str, int, List[Dict[str, Any]]]]:
        assignments = self._sample_assignments()
        # 批量填 content
        metas = [a[2] for a in assignments]
        filled = self._fill_content(metas)
        by_id = {d["doc_id"]: d for d in filled}
        planned: List[Tuple[str, int, List[Dict[str, Any]]]] = []
        for qt, idx, meta in assignments:
            doc = by_id.get(meta["doc_id"]) or meta
            planned.append((qt, idx, [doc]))
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
        by_type: Dict[str, int] = {}
        for q in out_q:
            if q.get("gen_status") == 1:
                qt = str(q.get("question_type") or "")
                by_type[qt] = by_type.get(qt, 0) + 1
        return {
            "meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "db_path": self.db_path,
                "type_counts": dict(self.type_counts),
                "seed": self.seed,
                "n_limit": self.n_limit,
                "doc_id_range": self.doc_id_range,
                "n_source_docs": len(self.docs),
                "num_thread": workers,
                "model": (self.model_args or {}).get("model"),
                "total": len(out_q),
                "success": ok_n,
                "failed": len(out_q) - ok_n,
                "by_type_success": by_type,
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
        for idx, (qt, i, _docs) in enumerate(planned):
            qid = f"q_{qt}_{i:04d}"
            if qid in by_id:
                questions[idx] = by_id[qid]
                n_resumed += 1
            else:
                todo_idx.append(idx)
        if n_resumed:
            print(
                f"[resume] 出题已完成 {n_resumed}/{n}，待生成 {len(todo_idx)}",
                file=sys.stderr,
            )
        fail_n = 0
        done_n = n_resumed
        t0 = time.perf_counter()
        workers = min(self.num_thread, max(1, len(todo_idx))) if todo_idx else 0

        print(
            f"[generate] docs={len(self.docs)} doc_id_range={self.doc_id_range!r} "
            f"type_counts={self.type_counts} questions={n}",
            file=sys.stderr,
        )

        def _on_item(item: Dict[str, Any]) -> None:
            nonlocal fail_n, done_n
            done_n += 1
            if item.get("gen_status") != 1:
                fail_n += 1
                err = item.get("gen_error") or "unknown"
                fail_print(
                    f"失败 {item.get('id')} type={item.get('question_type')} | {err} | "
                    f"sources={item.get('source_names')}"
                )

        def _emit() -> None:
            if on_progress is None:
                return
            mid = self._dataset_shell(questions, t0=t0, workers=workers, done=False)
            try:
                on_progress(mid, done_n, n)
            except Exception as e:
                fail_print(f"on_progress 回调异常: {e}")

        if not todo_idx:
            return self._dataset_shell(questions, t0=t0, workers=workers, done=True)

        if workers <= 1:
            pbar = progress_iter(todo_idx, total=len(todo_idx), desc="生成问题", unit="题")
            for idx in pbar:
                qt, i, docs = planned[idx]
                item = self.generate_one(qt, i, docs=docs)
                questions[idx] = item
                _on_item(item)
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(ok=done_n - fail_n, fail=fail_n, thr=1, refresh=False)
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
                        qt, i, docs = planned[idx]
                        item = {
                            "id": f"q_{qt}_{i:04d}",
                            "question_type": qt,
                            "question": "",
                            "gold_answer": "",
                            "ground_truth_answer": "",
                            "explanation": "",
                            "entities": {},
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

        return self._dataset_shell(questions, t0=t0, workers=workers, done=True)
