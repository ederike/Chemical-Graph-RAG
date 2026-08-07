from ..module.doc import BaseDoc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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


class Doc(BaseDoc):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    # ------------------------------------------------------------------
    # PDF discovery
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # PDF → images
    # ------------------------------------------------------------------
    def pdf_to_images_b64(self, pdf_path: Path):
        """Render each page of a PDF to base64-encoded PNG at configured DPI."""
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
        images = []

        doc = fitz.open(str(pdf_path))
        try:
            if doc.page_count <= 0:
                raise ValueError(f"PDF has 0 pages: {pdf_path.name}")
            for page in doc:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img_bytes = pix.tobytes('png')
                b64 = base64.b64encode(img_bytes).decode('ascii')
                images.append(b64)
        finally:
            doc.close()

        return images

    # ------------------------------------------------------------------
    # Vision recognition cache
    # ------------------------------------------------------------------
    def _prompt_hash(self) -> str:
        recog = self.config.doc.recognition
        prompt_key = getattr(recog, 'prompt', 'pdf_recognize')
        prompt_body = PROMPT.get(prompt_key, PROMPT.get('pdf_recognize', ''))
        system_body = PROMPT.get('pdf_recognize_system', '')
        return hashlib.md5(
            f'{system_body}\n---\n{prompt_body}'.encode('utf-8')
        ).hexdigest()[:16]

    def _file_recognition_cache_key(self, pdf_path: Path, file_hash: str) -> str:
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
            'pipeline': 'page_plain_text_v1',
        }
        return hashlib.md5(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')
        ).hexdigest()

    def _page_recognition_cache_key(
        self,
        pdf_path: Path,
        file_hash: str,
        page_idx: int,
        prev_text: str,
    ) -> str:
        """Per-page cache key (includes previous-page text hash for context)."""
        recog = self.config.doc.recognition
        prev_hash = hashlib.md5((prev_text or '').encode('utf-8')).hexdigest()[:16]
        payload = {
            'scope': 'page',
            'file_hash': file_hash,
            'name': pdf_path.name,
            'page_idx': int(page_idx),
            'prev_hash': prev_hash,
            'model_args': recog.model_args,
            'dpi': recog.dpi,
            'prompt': getattr(recog, 'prompt', 'pdf_recognize'),
            'prompt_hash': self._prompt_hash(),
            'pipeline': 'page_plain_text_v1',
        }
        return hashlib.md5(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')
        ).hexdigest()

    def clear_recognition_cache(self, name: str) -> int:
        """Remove PDF recognition cache entries for a given file name."""
        deleted = self._recog_cache.delete_by_name(name)
        self.logger.info(f"Recognition cache cleared for {name!r}: {deleted} row(s)")
        return deleted

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------
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
    def normalize_page_text(cls, text: str) -> str:
        """Normalize VLM plain-text page recognition output."""
        return cls._strip_code_fence(text)

    @staticmethod
    def join_page_texts(page_texts: list) -> str:
        """Concatenate page recognitions in order with blank-line separators."""
        parts = [(t or '').strip() for t in (page_texts or [])]
        parts = [p for p in parts if p]
        return '\n\n'.join(parts)

    @staticmethod
    def doc_extra_payload(
        *,
        page_count: int = 0,
        source_pdf: str = '',
        file_hash: str = '',
        recognition_cost=None,
        page_texts=None,
    ) -> dict:
        """doc.extra metadata (full recognition text lives in doc.content)."""
        extra = {
            'page_count': page_count,
            'source_pdf': source_pdf,
            'file_hash': file_hash,
            'recognition_cost': recognition_cost or {},
            'pipeline': 'page_plain_text_v1',
        }
        if page_texts is not None:
            extra['page_count'] = len(page_texts)
        return extra

    # ------------------------------------------------------------------
    # Single-page vision recognition
    # ------------------------------------------------------------------
    @Retry(max_attempt=3, wait=0.1, timeout=300, config_attr='doc.recognition.retry')
    def recognize_page(
        self,
        image_b64: str,
        *,
        pdf_path: Path,
        file_hash: str,
        page_idx: int,
        page_count: int,
        prev_text: str = '',
        **kwargs,
    ):
        """
        Recognize one page image with VLM.
        For page_idx > 0, multi-turn history carries only the previous page text.
        """
        attempt = int(kwargs.get('attempt', 1) or 1)
        max_attempt = int(kwargs.get('max_attempt', 1) or 1)
        pdf_path = Path(pdf_path)
        recog = self.config.doc.recognition
        prev_text = (prev_text or '').strip()

        cache_key = self._page_recognition_cache_key(
            pdf_path, file_hash, page_idx, prev_text
        )
        if recog.use_cache:
            cached = self._recog_cache.search_cache(cache_key)
            if cached is not None:
                out = json.loads(cached)
                if self.metrics is not None:
                    self.metrics.record(
                        'recognize',
                        0.0,
                        cache_hit=True,
                        name=f'{pdf_path.name}#p{page_idx + 1}',
                        log=False,
                        accumulate_time=False,
                    )
                return out

        if attempt > 1:
            self.logger.debug(
                f"PDF {pdf_path.name} page {page_idx + 1}/{page_count} "
                f"recognition attempt {attempt}/{max_attempt}"
            )

        prompt_key = getattr(recog, 'prompt', 'pdf_recognize')
        user_prompt = PROMPT.get(prompt_key, PROMPT['pdf_recognize'])
        user_prompt = (
            f"{user_prompt}\n\n"
            f"当前为第 {page_idx + 1}/{page_count} 页，"
            f"请直接输出本页的识别正文（纯文本，不要 JSON）。"
        )
        system_prompt = PROMPT.get('pdf_recognize_system', '')

        history = None
        if page_idx > 0 and prev_text:
            # 仅携带上一页识别文本，不跨文件、不回传上一页图片
            history = [
                {
                    'role': 'user',
                    'content': '上一页图片的识别结果如下，供当前页衔接参考：\n' + prev_text,
                },
                {
                    'role': 'assistant',
                    'content': prev_text,
                },
            ]

        model_args = dict(recog.model_args or {})
        model_args.setdefault('enable_thinking', False)
        # 明文输出，禁止强制 JSON
        model_args.pop('response_format', None)
        if attempt > 1:
            model_args['temperature'] = max(
                float(model_args.get('temperature') or 0.0), 1.0
            )

        response = self.llmmodel.generate_vision(
            prompt={'system': system_prompt, 'user': user_prompt},
            images=[image_b64],
            model_args=model_args,
            history=history,
        )

        if response.get('status') != 1:
            err = str(response.get('answer') or '')[:500]
            raise RuntimeError(
                f"PDF page recognition API failed for {pdf_path.name} "
                f"page {page_idx + 1}/{page_count} "
                f"(attempt {attempt}/{max_attempt}): {err}"
            )

        raw_answer = (response.get('answer') or '').strip()
        page_text = self.normalize_page_text(raw_answer)
        if not page_text:
            raise RuntimeError(
                f"PDF page recognition empty for {pdf_path.name} "
                f"page {page_idx + 1}/{page_count} "
                f"(attempt {attempt}/{max_attempt})"
            )

        cost = {
            'usage_prompt_tokens': response.get('usage_prompt_tokens'),
            'usage_completion_tokens': response.get('usage_completion_tokens'),
            'usage_total_tokens': response.get('usage_total_tokens'),
        }
        result = {
            'page_idx': page_idx,
            'text': page_text,
            'recognition_cost': cost,
        }

        if recog.use_cache:
            self._recog_cache.update_cache(
                json.dumps(
                    {
                        'name': pdf_path.name,
                        'file_hash': file_hash,
                        'page_idx': page_idx,
                    },
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
                name=f'{pdf_path.name}#p{page_idx + 1}',
                extra=f'attempt={attempt}',
                log=False,
                accumulate_time=False,
            )
        return result

    # ------------------------------------------------------------------
    # Whole-PDF recognition (sequential pages, one VLM call per page)
    # ------------------------------------------------------------------
    def recognize_pdf(self, pdf_path: Path, **kwargs):
        """
        PDF → 逐页图像 → 每页单独 VLM 识别 → 按序拼接为文档正文。

        页间：仅中间页在对话历史中携带上一页识别文本（不跨文件、不回传上页图）。
        多线程在文件级（prepare_from_pdfs），单文件内页序串行。

        返回 dict 供 insert：
          content : 全文纯文本（各页识别结果按序拼接）
          extra   : 元数据 JSON
          hash    : content 哈希
        """
        pdf_path = Path(pdf_path)
        raw_bytes = pdf_path.read_bytes()
        file_hash = hashlib.md5(raw_bytes).hexdigest()
        recog = self.config.doc.recognition
        file_cache_key = self._file_recognition_cache_key(pdf_path, file_hash)

        if recog.use_cache:
            cached = self._recog_cache.search_cache(file_cache_key)
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

        images_b64 = self.pdf_to_images_b64(pdf_path)
        if not images_b64:
            raise ValueError(f"No pages rendered from PDF: {pdf_path}")

        page_count = len(images_b64)
        page_texts = []
        total_cost = {
            'usage_prompt_tokens': 0,
            'usage_completion_tokens': 0,
            'usage_total_tokens': 0,
        }
        prev_text = ''

        for page_idx, img_b64 in enumerate(images_b64):
            page_out = self.recognize_page(
                img_b64,
                pdf_path=pdf_path,
                file_hash=file_hash,
                page_idx=page_idx,
                page_count=page_count,
                prev_text=prev_text,
            )
            if page_out is None:
                raise RuntimeError(
                    f"Failed to recognize {pdf_path.name} page {page_idx + 1}/{page_count} "
                    f"after all retries"
                )
            text = (page_out.get('text') or '').strip()
            if not text:
                raise RuntimeError(
                    f"Empty page text for {pdf_path.name} page {page_idx + 1}/{page_count}"
                )
            page_texts.append(text)
            prev_text = text
            cost = page_out.get('recognition_cost') or {}
            for k in total_cost:
                try:
                    total_cost[k] += int(cost.get(k) or 0)
                except (TypeError, ValueError):
                    pass

        content = self.join_page_texts(page_texts)
        if not content.strip():
            raise RuntimeError(f"PDF recognition empty document for {pdf_path.name}")

        extra_obj = self.doc_extra_payload(
            page_count=page_count,
            source_pdf=str(pdf_path),
            file_hash=file_hash,
            recognition_cost=total_cost,
            page_texts=page_texts,
        )
        result = {
            'name': pdf_path.name,
            'content': content,
            'extra': json.dumps(extra_obj, ensure_ascii=False),
            'hash': hash_str(content),
            'file_hash': file_hash,
            'source_pdf': str(pdf_path),
            'page_count': page_count,
            'recognition_cost': total_cost,
        }

        if recog.use_cache:
            self._recog_cache.update_cache(
                json.dumps(
                    {'name': pdf_path.name, 'file_hash': file_hash},
                    ensure_ascii=False,
                ),
                json.dumps(result, ensure_ascii=False),
                file_cache_key,
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
        for p in pdf_paths:
            if skip_existing and p.name in existing_names:
                skipped.append(p.name)
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
        results = []
        results_lock = threading.Lock()

        def _one(path):
            try:
                out = self.recognize_pdf(path)
            except Exception as e:
                self.logger.error(f"Failed to recognize {path}: {e}")
                return None
            if out is None:
                self.logger.error(
                    f"Failed to recognize {path} after all retries"
                )
            return out

        def _postfix():
            pf = {'total': progress_total}
            if self.metrics is not None:
                s = self.metrics.stage_snapshot('recognize')
                pf['real'] = s['real']
            return pf

        self.logger.info(
            f"PDF recognition start: {len(to_process)} file(s), "
            f"num_thread={num_thread} (file-level; pages sequential per file)"
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
            with ThreadPoolExecutor(max_workers=num_thread) as executor:
                futures = {executor.submit(_one, p): p for p in to_process}
                bar = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc='recognize',
                    unit='pdf',
                    bar_format=TQDM_BAR_FORMAT,
                )
                for future in bar:
                    p = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            with results_lock:
                                results.append(result)
                    except Exception as e:
                        self.logger.error(f"Failed to recognize {p}: {e}")
                    bar.set_postfix(**_postfix())

        wall = time.perf_counter() - t0
        results.sort(key=lambda x: x.get('name', ''))

        if self.metrics is not None:
            self.metrics.finalize_stage_wall_time('recognize', wall)
            self.metrics.log_stage('recognize')

        self.logger.info(
            f"Recognized {len(results)}/{len(to_process)} new PDF documents "
            f"(skipped existing: {len(skipped)}, threads={num_thread}, "
            f"wall_time={wall:.3f}s)."
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
