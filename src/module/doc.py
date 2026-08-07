from typing import Dict
import logging
from ..utils.database import BaseDB
from ..utils.config import Config
import tiktoken

from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from tqdm import tqdm
import base64
import hashlib
import json
import threading
import time

from ..utils.OpenAIAPI import LLM
from ..utils.prompt import PROMPT
from ..utils.utils import hash_str, CacheDB, Retry, NonRetryableError, TQDM_BAR_FORMAT
from ..utils.config import resolve_credentials

class Doc:
    def _flush_every(self) -> int:
        try:
            n = int(getattr(self.config.doc, 'flush_every', 1000) or 1000)
        except (TypeError, ValueError):
            n = 1000
        return max(1, n)

    def _maybe_flush(self, force: bool = False):
        n = len(self.doc_db.buffer)
        if n <= 0:
            return
        if not force and n < self._flush_every():
            return
        self.save()

    def processing(self):
        n = len(self.tasks)
        bar = tqdm(
            self.tasks,
            desc='insert',
            unit='doc',
            bar_format=TQDM_BAR_FORMAT,
        )
        for task in bar:
            self.processing_single_task(task)
            bar.set_postfix(total=n)

    def save(self):
        buf = self.doc_db.buffer
        if not buf:
            return
        self.doc_db.add(buf)
        self._flushed_count += len(buf)
        self.logger.info(
            f"[doc] flush n={len(buf)} total_flushed={self._flushed_count}"
        )
        self.doc_db.buffer_clear()

    def clear(self):
        self.doc_db.clear()

    def __init__(self, db: Dict[str, BaseDB], logger: logging.Logger, config: Config):
        self.config = config
        self.doc_db = db['doc']
        self.logger = logger
        self._flushed_count = 0
        if self.config.doc.count_token:
            self.tokener = tiktoken.get_encoding('cl100k_base')

        recog = getattr(self.config.doc, 'recognition', None)
        api_key, base_url = resolve_credentials(self.config, recog)
        llm_timeout = 300.0
        llm_retries = 0
        try:
            r = getattr(recog, 'retry', None) if recog is not None else None
            if r is not None:
                llm_timeout = float(getattr(r, 'timeout', llm_timeout) or llm_timeout)
        except Exception:
            pass
        self.llmmodel = LLM(
            api_key,
            base_url,
            timeout=max(60.0, llm_timeout),
            max_retries=llm_retries,
        )
        self._recog_cache = CacheDB('cache/OpenAI', 'pdf_recognize_cache')
        self.metrics = None  # set by DHMF

    def resolve_doc_dir(self) -> Path:
        """Document directory: working_path/doc (e.g. example/a/doc)."""
        doc_subdir = getattr(self.config.doc, 'doc_dir', 'doc') or 'doc'
        return Path(self.config.settings.working_path) / doc_subdir

    def list_pdf_files(self):
        """Collect PDF paths under working_path/doc."""
        doc_dir = self.resolve_doc_dir()
        if not doc_dir.exists():
            return []
        return sorted(doc_dir.glob('*.pdf'))

    @staticmethod
    def classify_pdf_magic(path: Path) -> str:
        """
        快速校验文件头是否像 PDF。
        返回 '' 表示通过；否则返回可读原因（假 PDF / 空文件等）。
        """
        path = Path(path)
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                return 'missing or empty'
            magic = path.read_bytes()[:8]
        except OSError as e:
            return f'unreadable: {e}'
        if magic.startswith(b'%PDF'):
            return ''
        kind = 'unknown'
        if magic.startswith(b'\x89PNG'):
            kind = 'PNG'
        elif magic[:2] == b'\xff\xd8':
            kind = 'JPEG'
        elif magic.startswith(b'PK'):
            kind = 'ZIP/OOXML'
        elif magic.startswith(b'\xd0\xcf\x11\xe0'):
            kind = 'OLE/CFB (doc/xls)'
        elif magic.lstrip().startswith(b'<!') or magic.lstrip().startswith(b'<'):
            kind = 'HTML'
        elif magic.startswith(b'Can not') or magic.startswith(b'Cannot '):
            kind = 'error-text'
        return f'Not a real PDF (magic={magic!r}, kind={kind})'

    def get_existing_doc_names(self) -> set:
        """Set of file names already present in the doc table."""
        rows = self.doc_db.search_all() or []
        return {r['name'] for r in rows if r.get('name')}

    def get_docs_by_name(self, name: str):
        return self.doc_db.search('name', name) or []

    def is_already_inserted(self, name: str = None, content_hash: str = None) -> bool:
        """True if a doc with the same name or content hash already exists."""
        if name and self.get_docs_by_name(name):
            return True
        if content_hash:
            rows = self.doc_db.search('hash', content_hash) or []
            if rows:
                return True
        return False

    def pdf_to_images_b64(self, pdf_path: Path):
        """Render each page of a PDF to base64 data-URL at configured DPI."""
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError(
                "PyMuPDF (pymupdf) is required for PDF recognition. "
                "Install with: pip install pymupdf"
            ) from e

        pdf_path = Path(pdf_path)
        if not pdf_path.is_file() or pdf_path.stat().st_size <= 0:
            raise NonRetryableError(f"PDF missing or empty: {pdf_path}")

        magic = pdf_path.read_bytes()[:8]
        if not magic.startswith(b'%PDF'):
            kind = 'unknown'
            if magic.startswith(b'\x89PNG'):
                kind = 'PNG'
            elif magic[:2] == b'\xff\xd8':
                kind = 'JPEG'
            elif magic.startswith(b'PK'):
                kind = 'ZIP/OOXML'
            elif magic.lstrip().startswith(b'<!') or magic.lstrip().startswith(b'<'):
                kind = 'HTML'
            raise NonRetryableError(
                f"Not a real PDF (magic={magic!r}, kind={kind}): {pdf_path.name}"
            )

        dpi = self.config.doc.recognition.dpi
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        fmt = str(
            getattr(self.config.doc.recognition, 'image_format', 'jpeg') or 'jpeg'
        ).lower()
        if fmt in ('jpg', 'jpeg'):
            fitz_fmt = 'jpeg'
            mime = 'jpeg'
        else:
            fitz_fmt = 'png'
            mime = 'png'

        images = []
        doc = fitz.open(str(pdf_path))
        try:
            if doc.page_count <= 0:
                raise ValueError(f"PDF has 0 pages: {pdf_path.name}")
            for page in doc:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img_bytes = pix.tobytes(fitz_fmt)
                b64 = base64.b64encode(img_bytes).decode('ascii')
                images.append(f'data:image/{mime};base64,{b64}')
        finally:
            doc.close()

        return images

    def _prompt_hash(self) -> str:
        recog = self.config.doc.recognition
        prompt_key = getattr(recog, 'prompt', 'pdf_recognize')
        prompt_body = PROMPT.get(prompt_key, PROMPT.get('pdf_recognize', ''))
        system_body = PROMPT.get('pdf_recognize_system', '')
        return hashlib.md5(
            f'{system_body}\n---\n{prompt_body}'.encode('utf-8')
        ).hexdigest()[:16]

    def _recognition_cache_key(self, pdf_path: Path, file_hash: str) -> str:
        """Whole-document recognition result cache key."""
        recog = self.config.doc.recognition
        payload = {
            'scope': 'file',
            'file_hash': file_hash,
            'name': pdf_path.name,
            'model_args': recog.model_args,
            'dpi': recog.dpi,
            'prompt': getattr(recog, 'prompt', 'pdf_recognize'),
            'prompt_hash': self._prompt_hash(),
            'pipeline': 'whole_file_plain_text_v1',
            'image_format': getattr(recog, 'image_format', 'jpeg'),
        }
        return hashlib.md5(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')
        ).hexdigest()

    def clear_recognition_cache(self, name: str) -> int:
        """Remove PDF recognition cache entries for a given file name."""
        deleted = self._recog_cache.delete_by_name(name)
        self.logger.info(f"Recognition cache cleared for {name!r}: {deleted} row(s)")
        return deleted

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        cleaned = (text or '').strip()
        if cleaned.startswith('```'):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned = '\n'.join(lines).strip()
        return cleaned

    @classmethod
    def normalize_recognition_text(cls, text: str) -> str:
        """Normalize VLM plain-text whole-document recognition output."""
        return cls._strip_code_fence(text)

    @staticmethod
    def doc_extra_payload(
        *,
        page_count: int = 0,
        source_pdf: str = '',
        file_hash: str = '',
        recognition_cost=None,
    ) -> dict:
        """doc.extra metadata (full recognition text lives in doc.content)."""
        return {
            'page_count': page_count,
            'source_pdf': source_pdf,
            'file_hash': file_hash,
            'recognition_cost': recognition_cost or {},
            'pipeline': 'whole_file_plain_text_v1',
        }

    @Retry(
        max_attempt=4,
        wait=[10, 30, 60],
        timeout=300,
        config_attr='doc.recognition.retry',
    )
    def recognize_pdf(self, pdf_path: Path, **kwargs):
        """
        PDF → 全部页面图像 → 一次 VLM 多图识别 → 全文纯文本。

        文件级并行由 prepare_from_pdfs 控制。

        返回 dict 供 insert：
          content : 全文纯文本
          extra   : 元数据 JSON
          hash    : content 哈希
        """
        attempt = int(kwargs.get('attempt', 1) or 1)
        max_attempt = int(kwargs.get('max_attempt', 1) or 1)

        pdf_path = Path(pdf_path)
        raw_bytes = pdf_path.read_bytes()
        file_hash = hashlib.md5(raw_bytes).hexdigest()
        recog = self.config.doc.recognition
        cache_key = self._recognition_cache_key(pdf_path, file_hash)

        if recog.use_cache:
            cached = self._recog_cache.search_cache(cache_key)
            if cached is not None:
                out = json.loads(cached)
                if self.metrics is not None:
                    self.metrics.record(
                        'recognize',
                        0.0,
                        cache_hit=True,
                        name=pdf_path.name,
                        log=False,
                        accumulate_time=False,
                    )
                return out

        if attempt > 1:
            self.logger.debug(
                f"PDF {pdf_path.name} recognition attempt {attempt}/{max_attempt}"
            )

        images_b64 = self.pdf_to_images_b64(pdf_path)
        if not images_b64:
            raise NonRetryableError(f"No pages rendered from PDF: {pdf_path.name}")

        page_count = len(images_b64)
        user_prompt = PROMPT.get(
            getattr(recog, 'prompt', 'pdf_recognize'),
            PROMPT['pdf_recognize'],
        )
        system_prompt = PROMPT.get('pdf_recognize_system', '')
        user_prompt = (
            f"{user_prompt}\n\n"
            f"本文件共 {page_count} 页，以下按页序给出全部页面图片。"
            f"请用中文直接输出整份文档的识别正文（纯文本，不要 JSON）；"
            f"数值范围、限值与临界条件须完整保留上下端点与测定条件。"
        )

        model_args = dict(recog.model_args or {})
        model_args.setdefault('enable_thinking', False)
        model_args.pop('response_format', None)
        if attempt > 1:
            model_args['temperature'] = max(
                float(model_args.get('temperature') or 0.0), 1.0
            )

        response = self.llmmodel.generate_vision(
            prompt={'system': system_prompt, 'user': user_prompt},
            images=images_b64,
            model_args=model_args,
        )

        if response.get('status') != 1:
            err = str(response.get('answer') or '')[:500]
            raise RuntimeError(
                f"PDF recognition API failed for {pdf_path.name} "
                f"(attempt {attempt}/{max_attempt}, pages={page_count}, "
                f"model={model_args.get('model')!r}): {err}"
            )

        raw_answer = (response.get('answer') or '').strip()
        content = self.normalize_recognition_text(raw_answer)
        if not content:
            raise RuntimeError(
                f"PDF recognition empty answer for {pdf_path.name} "
                f"(attempt {attempt}/{max_attempt}, pages={page_count})"
            )

        cost = {
            'usage_prompt_tokens': response.get('usage_prompt_tokens'),
            'usage_completion_tokens': response.get('usage_completion_tokens'),
            'usage_total_tokens': response.get('usage_total_tokens'),
        }
        extra_obj = self.doc_extra_payload(
            page_count=page_count,
            source_pdf=str(pdf_path),
            file_hash=file_hash,
            recognition_cost=cost,
        )
        result = {
            'name': pdf_path.name,
            'content': content,
            'extra': json.dumps(extra_obj, ensure_ascii=False),
            'hash': hash_str(content),
            'file_hash': file_hash,
            'source_pdf': str(pdf_path),
            'page_count': page_count,
            'recognition_cost': cost,
        }

        if recog.use_cache:
            self._recog_cache.update_cache(
                json.dumps(
                    {'name': pdf_path.name, 'file_hash': file_hash},
                    ensure_ascii=False,
                ),
                json.dumps(result, ensure_ascii=False),
                cache_key,
            )

        if self.metrics is not None:
            self.metrics.record(
                'recognize',
                0.0,
                cache_hit=False,
                prompt_tokens=response.get('usage_prompt_tokens') or 0,
                completion_tokens=response.get('usage_completion_tokens') or 0,
                total_tokens=response.get('usage_total_tokens'),
                name=pdf_path.name,
                extra=f'pages={page_count} attempt={attempt}',
                log=False,
                accumulate_time=False,
            )
        return result

    def prepare_from_pdfs(
        self,
        pdf_paths=None,
        skip_existing: bool = True,
        progress_total: int = None,
    ):
        """
        Recognize PDFs into doc_list tasks (before insert).
        Each PDF becomes one independent document; multi-thread by file.
        """
        if pdf_paths is None:
            pdf_paths = self.list_pdf_files()
        else:
            pdf_paths = [Path(p) for p in pdf_paths]

        if not pdf_paths:
            self.logger.warning(
                f"No PDF files found under {self.resolve_doc_dir()} "
            )

        existing_names = self.get_existing_doc_names() if skip_existing else set()
        to_process = []
        skipped = []
        invalid = []  # (name, reason) 假 PDF / 空文件，进池前剔除
        for p in pdf_paths:
            if skip_existing and p.name in existing_names:
                skipped.append(p.name)
                continue
            bad = self.classify_pdf_magic(p)
            if bad:
                invalid.append((p.name, bad))
                continue
            to_process.append(p)

        if skipped:
            self.logger.info(
                f"Skip {len(skipped)} PDF(s) already in doc table: "
                f"{skipped[:10]}{'...' if len(skipped) > 10 else ''}"
            )
            if self.metrics is not None:
                for name in skipped:
                    self.metrics.record(
                        'recognize',
                        0.0,
                        skipped=True,
                        name=name,
                        log=False,
                        accumulate_time=False,
                    )

        if invalid:
            self.logger.warning(
                f"Skip {len(invalid)} non-PDF file(s) "
                f"(wrong magic / empty; not sent to VLM): "
                f"{invalid[:5]}{'...' if len(invalid) > 5 else ''}"
            )
            if self.metrics is not None:
                for name, _reason in invalid:
                    self.metrics.record(
                        'recognize',
                        0.0,
                        skipped=True,
                        name=name,
                        log=False,
                        accumulate_time=False,
                    )

        if not to_process:
            self.logger.info("No new PDFs to recognize.")
            return []

        if progress_total is None:
            progress_total = len(to_process)
        else:
            try:
                progress_total = max(int(progress_total), len(to_process))
            except (TypeError, ValueError):
                progress_total = len(to_process)

        recog = self.config.doc.recognition
        num_thread = max(1, int(getattr(recog, 'num_thread', 1) or 1))
        # 与 LLM 客户端一致，用于心跳日志说明「最长可能等多久」
        try:
            http_timeout = float(
                getattr(getattr(recog, 'retry', None), 'timeout', 300) or 300
            )
        except (TypeError, ValueError):
            http_timeout = 300.0
        results = []
        results_lock = threading.Lock()

        def _record_skip(name: str):
            if self.metrics is not None:
                self.metrics.record(
                    'recognize',
                    0.0,
                    skipped=True,
                    name=name,
                    log=False,
                    accumulate_time=False,
                )

        def _one(path):
            """单文件识别；任何错误只跳过当前文件，不中断整批。"""
            try:
                out = self.recognize_pdf(path)
            except NonRetryableError as e:
                # 假 PDF / 空页等：不可重试，勿写成「重试耗尽」
                self.logger.warning(
                    f"Skip PDF (invalid/unreadable, no retry): {path.name}: {e}"
                )
                _record_skip(path.name)
                return None
            except Exception as e:
                self.logger.warning(
                    f"Skip PDF after error: {path.name}: {e}"
                )
                _record_skip(path.name)
                return None
            if out is None:
                self.logger.warning(
                    f"Skip PDF (API retries exhausted or empty result): "
                    f"{path.name}"
                )
                _record_skip(path.name)
            return out

        def _postfix(inflight: int = 0):
            pf = {'total': progress_total}
            if inflight:
                pf['run'] = inflight
            if self.metrics is not None:
                s = self.metrics.stage_snapshot('recognize')
                pf['real'] = s['real']
            return pf

        self.logger.info(
            f"PDF recognition start: {len(to_process)} file(s), "
            f"num_thread={num_thread}, http_timeout={http_timeout:.0f}s "
            f"(whole-file multi-image per PDF)"
        )
        t0 = time.perf_counter()

        if num_thread <= 1:
            bar = tqdm(
                to_process,
                desc='recognize',
                unit='pdf',
                bar_format=TQDM_BAR_FORMAT,
            )
            for p in bar:
                result = _one(p)
                if result is not None:
                    results.append(result)
                bar.set_postfix(**_postfix())
        else:
            # wait + 心跳：避免 as_completed 在末尾长请求上「假死」无日志
            with ThreadPoolExecutor(max_workers=num_thread) as executor:
                pending = {
                    executor.submit(_one, p): (p, time.perf_counter())
                    for p in to_process
                }
                bar = tqdm(
                    total=len(pending),
                    desc='recognize',
                    unit='pdf',
                    bar_format=TQDM_BAR_FORMAT,
                )
                heartbeat_s = 30.0
                while pending:
                    done, _ = wait(
                        set(pending.keys()),
                        timeout=heartbeat_s,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        now = time.perf_counter()
                        aged = sorted(
                            (
                                (now - started, path.name)
                                for path, started in pending.values()
                            ),
                            reverse=True,
                        )
                        top = ', '.join(
                            f'{name}({elapsed:.0f}s)'
                            for elapsed, name in aged[:8]
                        )
                        more = f' +{len(aged) - 8} more' if len(aged) > 8 else ''
                        self.logger.info(
                            f"[recognize] still in-flight={len(pending)} "
                            f"(each call may take up to ~{http_timeout:.0f}s "
                            f"HTTP timeout + retries): {top}{more}"
                        )
                        bar.set_postfix(**_postfix(len(pending)))
                        continue

                    for future in done:
                        path, started = pending.pop(future)
                        try:
                            result = future.result()
                            if result is not None:
                                with results_lock:
                                    results.append(result)
                        except Exception as e:
                            self.logger.error(
                                f"Failed to recognize {path.name}: {e}"
                            )
                            _record_skip(path.name)
                        bar.update(1)
                        bar.set_postfix(**_postfix(len(pending)))
                bar.close()

        wall = time.perf_counter() - t0
        results.sort(key=lambda x: x.get('name', ''))

        if self.metrics is not None:
            self.metrics.finalize_stage_wall_time('recognize', wall)
            self.metrics.log_stage('recognize')

        self.logger.info(
            f"Recognized {len(results)}/{len(to_process)} valid PDF documents "
            f"(skipped existing={len(skipped)}, invalid_non_pdf={len(invalid)}, "
            f"threads={num_thread}, wall_time={wall:.3f}s)."
        )
        return results

    def processing_single_task(self, task):
        """
        Insert one recognized document (full plain-text content).
        Skip if the same file name or content hash already exists.
        """
        text = task['content']
        file_name = task['name']
        content_hash = task.get('hash') or hash_str(text)

        existing_by_name = self.get_docs_by_name(file_name)
        if existing_by_name:
            self.logger.info(
                f"Skip insert (name already in DB): {file_name} "
                f"-> existing id(s)={[r['id'] for r in existing_by_name]}"
            )
            return

        existing_by_hash = self.doc_db.search('hash', content_hash) or []
        if existing_by_hash:
            self.logger.info(
                f"Skip insert (content hash already in DB): {file_name} "
                f"-> existing name(s)={[r['name'] for r in existing_by_hash]}"
            )
            return

        add_doc = {
            'name': file_name,
            'content': text,
            'hash': content_hash,
            'status': 'new',
        }

        if task.get('extra') is not None:
            add_doc['extra'] = (
                task['extra']
                if isinstance(task['extra'], str)
                else json.dumps(task['extra'], ensure_ascii=False)
            )
        else:
            add_doc['extra'] = json.dumps(
                self.doc_extra_payload(
                    page_count=int(task.get('page_count') or 0),
                    source_pdf=str(task.get('source_pdf') or ''),
                    file_hash=str(task.get('file_hash') or ''),
                    recognition_cost=task.get('recognition_cost') or {},
                ),
                ensure_ascii=False,
            )

        if self.config.doc.count_token:
            add_doc['tokens'] = len(self.tokener.encode(text))

        self.doc_db.buffer.append(add_doc)
        self._maybe_flush()

    def prepare(self, doc_list):
        """Filter out already-inserted docs before buffering."""
        existing_names = self.get_existing_doc_names()
        filtered = []
        for item in doc_list:
            name = item.get('name')
            if name and name in existing_names:
                self.logger.info(f"Skip prepare (already in DB): {name}")
                continue
            filtered.append(item)
        self.tasks = filtered
        self.doc_db.buffer_clear()
        self._flushed_count = 0
        self.logger.debug(
            f"Number of documents to insert: {len(self.tasks)} "
            f"(filtered from {len(doc_list)})"
        )
