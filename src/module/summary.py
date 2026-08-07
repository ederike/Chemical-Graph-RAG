"""
文档总结步骤（insert 之后、chunk 之前，可在 main 中独立运行）：

  doc.content（全文识别）→ LLM 总结 → hyperedge.content

每条 doc 对应一条 hyperedge；分块时用 hyperedge 总结作为 head，
doc 识别全文按 token 切成 body_n。
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import copy
import json
import threading
import time

from ..utils.OpenAIAPI import LLM
from ..utils.prompt import PROMPT
from ..utils.utils import Retry, TQDM_BAR_FORMAT
from ..utils.config import resolve_credentials, Config
from ..utils.database import BaseDB
from typing import Dict
import logging
import tiktoken

class Summary:
    def __init__(
        self,
        db: Dict[str, BaseDB],
        logger: logging.Logger,
        config: Config,
    ):
        self.config = config
        self.logger = logger
        self.doc_db = db['doc']
        self.hyperedge_db = db['hyperedge']
        self._flushed_count = 0
        self._buffer_lock = threading.Lock()
        self.metrics = None  # set by DHMF

        stage = getattr(config, 'summary', None)
        api_key, base_url = resolve_credentials(config, stage)
        llm_timeout = 120.0
        try:
            r = getattr(stage, 'retry', None) if stage is not None else None
            if r is not None:
                llm_timeout = float(getattr(r, 'timeout', llm_timeout) or llm_timeout)
        except Exception:
            pass
        self.llmmodel = LLM(
            api_key,
            base_url,
            timeout=max(30.0, llm_timeout),
            max_retries=0,
        )
        self.tokener = tiktoken.get_encoding('cl100k_base')
        self.usage_prompt_tokens = 0
        self.usage_completion_tokens = 0

    def _flush_every(self) -> int:
        stage = getattr(self.config, 'summary', None)
        try:
            n = int(getattr(stage, 'flush_every', 500) or 500)
        except (TypeError, ValueError):
            n = 500
        return max(1, n)

    def _use_cache(self) -> bool:
        stage = getattr(self.config, 'summary', None)
        return bool(getattr(stage, 'use_cache', True)) if stage is not None else True

    def _num_thread(self) -> int:
        stage = getattr(self.config, 'summary', None)
        try:
            n = int(getattr(stage, 'num_thread', 1) or 1)
        except (TypeError, ValueError):
            n = 1
        return max(1, n)

    def _model_args(self) -> dict:
        stage = getattr(self.config, 'summary', None)
        return dict(getattr(stage, 'model_args', None) or {}) if stage is not None else {}

    def _prompt_name(self) -> str:
        stage = getattr(self.config, 'summary', None)
        name = getattr(stage, 'prompt', 'doc_summary') if stage is not None else 'doc_summary'
        if name not in PROMPT:
            return 'doc_summary'
        return name

    def _existing_hyperedge_by_doc(self, doc_id) -> dict:
        rows = self.hyperedge_db.search('doc_id', doc_id) or []
        return rows[0] if rows else None

    @Retry(max_attempt=3, wait=0.1, timeout=120, config_attr='summary.retry')
    def processing_single_task(self, doc, **kwargs):
        attempt = int(kwargs.get('attempt', 1) or 1)
        max_attempt = int(kwargs.get('max_attempt', 3) or 3)
        doc_id = doc.get('id')
        name = doc.get('name') or f'doc_{doc_id}'
        content = (doc.get('content') or '').strip()
        t0 = time.perf_counter()

        if not content:
            raise ValueError(f"Empty doc content for summary: {name}")

        if attempt > 1:
            self.logger.debug(
                f"Doc {name} summary attempt {attempt}/{max_attempt}"
            )

        prompt_name = self._prompt_name()
        template = PROMPT.get(prompt_name, PROMPT['doc_summary'])
        # 用 replace，避免正文中的 {} 触发 str.format 报错
        if '{content}' in template:
            user_prompt = template.replace('{content}', content)
        else:
            user_prompt = f"{template}\n\n{content}"
        model_args = copy.deepcopy(self._model_args())
        model_args.setdefault('enable_thinking', False)
        model_args.pop('response_format', None)
        if attempt > 1:
            model_args['temperature'] = max(
                float(model_args.get('temperature') or 0.0), 1.0
            )

        prompt = {'system': PROMPT.get('doc_summary_system', ''), 'user': user_prompt}
        response = self.llmmodel.generate(
            prompt=prompt,
            model_args=model_args,
            attempt=attempt,
            use_cache=self._use_cache(),
        )

        if response.get('status') != 1:
            err = str(response.get('answer') or '')[:400]
            raise RuntimeError(
                f"Summary API failed for {name} "
                f"(attempt {attempt}/{max_attempt}): {err}"
            )

        summary_text = (response.get('answer') or '').strip()
        # strip accidental code fences
        if summary_text.startswith('```'):
            lines = summary_text.splitlines()
            if lines and lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            summary_text = '\n'.join(lines).strip()

        if not summary_text:
            # 毒缓存清理
            try:
                self.llmmodel.drop_generate_cache(prompt, model_args)
            except Exception:
                pass
            raise ValueError(
                f"Summary empty for {name} (attempt {attempt}/{max_attempt})"
            )

        cache_hit = bool(response.get('_cache_hit'))
        prompt_tok = response.get('usage_prompt_tokens') or 0
        completion_tok = response.get('usage_completion_tokens') or 0
        total_tok = response.get('usage_total_tokens')
        if total_tok is None:
            total_tok = (prompt_tok or 0) + (completion_tok or 0)

        if not cache_hit:
            self.usage_prompt_tokens += prompt_tok or 0
            self.usage_completion_tokens += completion_tok or 0

        extra = {
            'source': 'summary',
            'role': 'head',
            'is_head': True,
            'doc_name': name,
            'cost': {
                'usage_prompt_tokens': prompt_tok,
                'usage_completion_tokens': completion_tok,
                'usage_total_tokens': total_tok,
                'cache_hit': cache_hit,
            },
        }
        he_row = {
            'doc_id': doc_id,
            'name': 'head',
            'content': summary_text,
            'status': 'summary',
            'extra': json.dumps(extra, ensure_ascii=False),
        }
        # tokens optional
        try:
            he_row['tokens'] = len(self.tokener.encode(summary_text))
        except Exception:
            pass

        existing = self._existing_hyperedge_by_doc(doc_id)
        with self._buffer_lock:
            if existing:
                he_row['id'] = existing['id']
                self.hyperedge_db.buffer.append(he_row)
            else:
                # new insert (no id)
                self.hyperedge_db.buffer.append(he_row)

            self.doc_db.buffer.append({
                'id': doc_id,
                'status': 'summary',
            })

            if self.metrics is not None:
                self.metrics.record(
                    'summary',
                    time.perf_counter() - t0,
                    cache_hit=cache_hit,
                    prompt_tokens=prompt_tok if not cache_hit else 0,
                    completion_tokens=completion_tok if not cache_hit else 0,
                    total_tokens=total_tok if not cache_hit else 0,
                    name=name,
                    extra=f'attempt={attempt}',
                    log=False,
                    accumulate_time=False,
                )
            self._maybe_flush()
        return True

    def _maybe_flush(self, force: bool = False):
        # called under lock from single-task; processing also flushes at end
        n = len(self.hyperedge_db.buffer) + len(self.doc_db.buffer)
        if n <= 0:
            return
        if not force and len(self.hyperedge_db.buffer) < self._flush_every():
            return
        self.save()

    def prepare(self):
        if self.config.settings.debug:
            self.tasks = self.doc_db.search_all() or []
        else:
            self.tasks = self.doc_db.search('status', 'new') or []
        self.doc_db.buffer_clear()
        self.hyperedge_db.buffer_clear()
        self._flushed_count = 0
        self.usage_prompt_tokens = 0
        self.usage_completion_tokens = 0
        self.logger.debug(
            f"Number of documents to summarize: {len(self.tasks)}"
        )

    def processing(self):
        n = len(self.tasks)
        if n == 0:
            return

        t_wall = time.perf_counter()
        num_thread = self._num_thread()

        def _postfix():
            pf = {'total': n}
            if self.metrics is not None:
                s = self.metrics.stage_snapshot('summary')
                pf['real'] = s['real']
            return pf

        def _run_one(task):
            """@Retry 耗尽返回 None；成功返回 True。"""
            try:
                out = self.processing_single_task(task)
            except Exception as e:
                self.logger.error(
                    f"Summary failed for "
                    f"{task.get('name') or task.get('id')}: {e}"
                )
                return None
            if out is None:
                self.logger.error(
                    f"Summary failed for "
                    f"{task.get('name') or task.get('id')} after all retries"
                )
            return out

        if num_thread <= 1:
            bar = tqdm(
                self.tasks,
                desc='summary',
                unit='doc',
                bar_format=TQDM_BAR_FORMAT,
            )
            for task in bar:
                _run_one(task)
                bar.set_postfix(**_postfix())
        else:
            with ThreadPoolExecutor(max_workers=num_thread) as executor:
                futures = {
                    executor.submit(_run_one, task): task
                    for task in self.tasks
                }
                bar = tqdm(
                    as_completed(futures),
                    total=n,
                    desc='summary',
                    unit='doc',
                    bar_format=TQDM_BAR_FORMAT,
                )
                for future in bar:
                    try:
                        future.result()
                    except Exception as e:
                        task = futures[future]
                        self.logger.error(
                            f"Summary failed for "
                            f"{task.get('name') or task.get('id')}: {e}"
                        )
                    bar.set_postfix(**_postfix())

        # final flush
        with self._buffer_lock:
            self._maybe_flush(force=True)

        if self.metrics is not None:
            self.metrics.finalize_stage_wall_time(
                'summary', time.perf_counter() - t_wall
            )

    def save(self):
        """
        Flush buffers: hyperedge rows with id → update; without id → add.
        Doc status updates always go through update.
        """
        he_buf = list(self.hyperedge_db.buffer or [])
        doc_buf = list(self.doc_db.buffer or [])
        if not he_buf and not doc_buf:
            return

        he_add = [r for r in he_buf if r.get('id') is None]
        he_upd = [r for r in he_buf if r.get('id') is not None]

        if he_add:
            self.hyperedge_db.add(he_add)
        if he_upd:
            self.hyperedge_db.update(he_upd)
        if doc_buf:
            self.doc_db.update(doc_buf)

        self._flushed_count += len(he_buf)
        self.hyperedge_db.buffer_clear()
        self.doc_db.buffer_clear()
        self.logger.info(
            f"[summary] flush hyperedge_add={len(he_add)} "
            f"hyperedge_upd={len(he_upd)} docs={len(doc_buf)} "
            f"total_he_flushed={self._flushed_count}"
        )

    def clear(self):
        """
        Clear summary outputs: delete all hyperedges.
        Reset doc status summary/chunk → new（chunk 正文需另行 chunk_clear）。
        Does not touch chunk/node 表内容.
        """
        self.hyperedge_db.clear()
        docs = self.doc_db.search_all() or []
        updates = [
            {'id': d['id'], 'status': 'new'}
            for d in docs
            if d.get('status') in ('summary', 'chunk')
        ]
        if updates:
            self.doc_db.update(updates)
        self.logger.info(
            f"[summary] clear: hyperedges wiped, "
            f"docs reset to new: {len(updates)}"
        )
