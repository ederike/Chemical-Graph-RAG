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
import re
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
        # self.logger.info(
        #     f"[doc] flush n={len(buf)} total_flushed={self._flushed_count}"
        # )
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

    def _open_pdf_validated(self, pdf_path: Path):
        """Open PDF after basic existence / magic checks. Caller must close."""
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

        return fitz.open(str(pdf_path))

    def pdf_page_count(self, pdf_path: Path) -> int:
        """Return page count without rendering (for slice planning)."""
        doc = self._open_pdf_validated(pdf_path)
        try:
            n = int(doc.page_count or 0)
            if n <= 0:
                raise NonRetryableError(f"PDF has 0 pages: {Path(pdf_path).name}")
            return n
        finally:
            doc.close()

    @staticmethod
    def slice_doc_name(base_name: str, slice_index: int) -> str:
        """
        切片输出文档名：首段 (index=0) 保持原名；
        多出来的段为「原名_{n}」，n 从 1 起（index=1 → _1）。
        """
        if slice_index <= 0:
            return base_name
        return f"{base_name}_{int(slice_index)}"

    @staticmethod
    def slice_family_base(name: str) -> str:
        """
        foo.pdf_1 → foo.pdf；非切片名原样返回。
        只剥「扩展名后的 _数字」，避免把 report_2024.pdf 误当成切片。
        """
        raw = Path(str(name or '')).name.strip()
        if not raw:
            return ''
        m = re.match(r'^(.+\.[A-Za-z0-9]+)_(\d+)$', raw)
        return m.group(1) if m else raw

    @staticmethod
    def _parse_doc_extra(raw):
        if not raw:
            return {}
        if isinstance(raw, dict):
            return raw
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _source_keys_of_row(self, row) -> set:
        keys = set()
        extra = self._parse_doc_extra(row.get('extra'))
        for field in ('source_name', 'source_pdf'):
            v = extra.get(field)
            if v:
                keys.add(Path(str(v)).name)
        return {k for k in keys if k}

    def collect_docs_for_delete(self, name: str) -> list:
        """
        Resolve a file name to the full slice family.

        A long PDF becomes foo.pdf, foo.pdf_1, foo.pdf_2. Deleting any
        member (or the source file name) collects all of them via
        extra.source_name / extra.source_pdf and the _N name pattern.
        """
        requested = Path(str(name or '')).name.strip()
        if not requested:
            return []

        identity_rows = self.doc_db.db.execute(
            f"SELECT id, name, extra FROM {self.doc_db.table}"
        ) or []

        bases = {requested, self.slice_family_base(requested)}
        bases.discard('')

        changed = True
        while changed:
            changed = False
            for row in identity_rows:
                n = (row.get('name') or '').strip()
                src_keys = self._source_keys_of_row(row)
                family_n = self.slice_family_base(n)
                if n in bases or family_n in bases or (src_keys & bases):
                    new_keys = set()
                    if n:
                        new_keys.add(n)
                        new_keys.add(family_n)
                    new_keys |= src_keys
                    extra_keys = new_keys - bases
                    extra_keys.discard('')
                    if extra_keys:
                        bases |= extra_keys
                        changed = True

        out = []
        seen = set()
        for row in identity_rows:
            n = (row.get('name') or '').strip()
            src_keys = self._source_keys_of_row(row)
            family_n = self.slice_family_base(n)
            if n in bases or family_n in bases or (src_keys & bases):
                did = row.get('id')
                if did is None or did in seen:
                    continue
                seen.add(did)
                out.append(row)
        out.sort(key=lambda r: (r.get('name') or '', r.get('id') or 0))
        return out

    def max_pages_per_doc(self) -> int:
        try:
            n = int(
                getattr(self.config.doc.recognition, 'max_pages_per_doc', 12) or 12
            )
        except (TypeError, ValueError):
            n = 12
        return max(1, n)

    def plan_pdf_slices(self, pdf_path: Path) -> list:
        """
        按 max_pages_per_doc 规划识别单元。
        每项: path, doc_name, page_start(0-based), page_end(exclusive),
              slice_index, total_pages, n_slices。
        页数 ≤ 上限时仅一段，doc_name = 原文件名。
        """
        pdf_path = Path(pdf_path)
        total_pages = self.pdf_page_count(pdf_path)
        max_pages = self.max_pages_per_doc()
        units = []
        slice_index = 0
        for page_start in range(0, total_pages, max_pages):
            page_end = min(page_start + max_pages, total_pages)
            units.append({
                'path': pdf_path,
                'doc_name': self.slice_doc_name(pdf_path.name, slice_index),
                'page_start': page_start,
                'page_end': page_end,
                'slice_index': slice_index,
                'total_pages': total_pages,
            })
            slice_index += 1
        for u in units:
            u['n_slices'] = len(units)
        return units

    def pdf_to_images_b64(
        self,
        pdf_path: Path,
        page_start: int = 0,
        page_end: int = None,
    ):
        """
        Render PDF pages to base64 data-URLs at configured DPI.
        page_start / page_end: 0-based half-open range; None end = all remaining.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError(
                "PyMuPDF (pymupdf) is required for PDF recognition. "
                "Install with: pip install pymupdf"
            ) from e

        pdf_path = Path(pdf_path)
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
        doc = self._open_pdf_validated(pdf_path)
        try:
            n = int(doc.page_count or 0)
            if n <= 0:
                raise NonRetryableError(f"PDF has 0 pages: {pdf_path.name}")
            start = max(0, int(page_start or 0))
            end = n if page_end is None else min(int(page_end), n)
            if start >= end:
                raise NonRetryableError(
                    f"Empty page range [{start}, {end}) for {pdf_path.name} "
                    f"(total_pages={n})"
                )
            for i in range(start, end):
                page = doc.load_page(i)
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

    def _recognition_cache_key(
        self,
        pdf_path: Path,
        file_hash: str,
        *,
        doc_name: str = None,
        page_start: int = 0,
        page_end: int = None,
        slice_index: int = 0,
    ) -> str:
        """Per-slice recognition result cache key."""
        recog = self.config.doc.recognition
        payload = {
            'scope': 'page_slice',
            'file_hash': file_hash,
            'name': pdf_path.name,
            'doc_name': doc_name or pdf_path.name,
            'page_start': int(page_start or 0),
            'page_end': page_end,
            'slice_index': int(slice_index or 0),
            'max_pages_per_doc': self.max_pages_per_doc(),
            'model_args': recog.model_args,
            'dpi': recog.dpi,
            'prompt': getattr(recog, 'prompt', 'pdf_recognize'),
            'prompt_hash': self._prompt_hash(),
            'pipeline': 'page_slice_plain_text_v1',
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
        source_name: str = '',
        file_hash: str = '',
        recognition_cost=None,
        slice_index: int = 0,
        n_slices: int = 1,
        page_start: int = 0,
        page_end: int = None,
        total_pages: int = 0,
    ) -> dict:
        """doc.extra metadata (full recognition text lives in doc.content)."""
        return {
            'page_count': page_count,
            'source_pdf': source_pdf,
            'source_name': source_name or '',
            'file_hash': file_hash,
            'recognition_cost': recognition_cost or {},
            'pipeline': 'page_slice_plain_text_v1',
            'slice_index': int(slice_index or 0),
            'n_slices': int(n_slices or 1),
            'page_start': int(page_start or 0),
            'page_end': page_end,
            'total_pages': int(total_pages or page_count or 0),
        }

    @Retry(
        max_attempt=4,
        wait=[10, 30, 60],
        timeout=300,
        config_attr='doc.recognition.retry',
    )
    def recognize_pdf(self, pdf_path: Path, **kwargs):
        """
        PDF 页切片 → 页面图像 → 一次 VLM 多图识别 → 纯文本。

        可通过 kwargs 指定单段切片（prepare_from_pdfs 已规划好）：
          doc_name, page_start, page_end, slice_index, n_slices, total_pages

        未传切片参数时按 max_pages_per_doc 自动规划并识别首段
        （兼容旧调用；多段请用 prepare_from_pdfs）。

        返回 dict 供 insert：
          content : 本段纯文本
          name    : 文档名（首段原名，后续 原名_{n}）
          extra   : 元数据 JSON
          hash    : content 哈希
        """
        attempt = int(kwargs.get('attempt', 1) or 1)
        max_attempt = int(kwargs.get('max_attempt', 1) or 1)

        pdf_path = Path(pdf_path)
        page_start = int(kwargs.get('page_start', 0) or 0)
        page_end = kwargs.get('page_end', None)
        if page_end is not None:
            page_end = int(page_end)
        slice_index = int(kwargs.get('slice_index', 0) or 0)
        n_slices = int(kwargs.get('n_slices', 1) or 1)
        total_pages = kwargs.get('total_pages', None)
        doc_name = kwargs.get('doc_name') or self.slice_doc_name(
            pdf_path.name, slice_index
        )

        # 无切片参数的旧调用：按规划取第 0 段（≤上限即全文）
        if (
            'page_end' not in kwargs
            and 'page_start' not in kwargs
            and 'slice_index' not in kwargs
        ):
            units = self.plan_pdf_slices(pdf_path)
            unit = units[0]
            page_start = unit['page_start']
            page_end = unit['page_end']
            slice_index = unit['slice_index']
            n_slices = unit['n_slices']
            total_pages = unit['total_pages']
            doc_name = unit['doc_name']

        raw_bytes = pdf_path.read_bytes()
        file_hash = hashlib.md5(raw_bytes).hexdigest()
        recog = self.config.doc.recognition
        cache_key = self._recognition_cache_key(
            pdf_path,
            file_hash,
            doc_name=doc_name,
            page_start=page_start,
            page_end=page_end,
            slice_index=slice_index,
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
                        name=doc_name,
                        log=False,
                        accumulate_time=False,
                    )
                return out

        if attempt > 1:
            self.logger.debug(
                f"PDF {doc_name} recognition attempt {attempt}/{max_attempt} "
                f"(source={pdf_path.name}, pages={page_start}:{page_end})"
            )

        images_b64 = self.pdf_to_images_b64(
            pdf_path, page_start=page_start, page_end=page_end
        )
        if not images_b64:
            raise NonRetryableError(
                f"No pages rendered from PDF: {pdf_path.name} "
                f"[{page_start}:{page_end}]"
            )

        page_count = len(images_b64)
        if total_pages is None:
            total_pages = page_count if n_slices <= 1 else page_end
        total_pages = int(total_pages or page_count)

        user_prompt = PROMPT.get(
            getattr(recog, 'prompt', 'pdf_recognize'),
            PROMPT['pdf_recognize'],
        )
        system_prompt = PROMPT.get('pdf_recognize_system', '')
        # 对模型按「本段文档」描述；页码用源文件 1-based 区间便于对齐
        human_from = page_start + 1
        human_to = page_start + page_count
        if n_slices > 1:
            span_note = (
                f"本段为源文件第 {human_from}-{human_to} 页"
                f"（共 {total_pages} 页，第 {slice_index + 1}/{n_slices} 段），"
                f"以下按页序给出本段全部页面图片。"
            )
        else:
            span_note = (
                f"本文件共 {page_count} 页，以下按页序给出全部页面图片。"
            )
        user_prompt = (
            f"{user_prompt}\n\n"
            f"{span_note}"
            f"请用中文直接输出本段文档的识别正文（纯文本，不要 JSON）；"
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
                f"PDF recognition API failed for {doc_name} "
                f"(source={pdf_path.name}, attempt {attempt}/{max_attempt}, "
                f"pages={page_count} [{human_from}-{human_to}/{total_pages}], "
                f"model={model_args.get('model')!r}): {err}"
            )

        raw_answer = (response.get('answer') or '').strip()
        content = self.normalize_recognition_text(raw_answer)
        if not content:
            raise RuntimeError(
                f"PDF recognition empty answer for {doc_name} "
                f"(source={pdf_path.name}, attempt {attempt}/{max_attempt}, "
                f"pages={page_count} [{human_from}-{human_to}/{total_pages}])"
            )

        cost = {
            'usage_prompt_tokens': response.get('usage_prompt_tokens'),
            'usage_completion_tokens': response.get('usage_completion_tokens'),
            'usage_total_tokens': response.get('usage_total_tokens'),
        }
        extra_obj = self.doc_extra_payload(
            page_count=page_count,
            source_pdf=str(pdf_path),
            source_name=pdf_path.name,
            file_hash=file_hash,
            recognition_cost=cost,
            slice_index=slice_index,
            n_slices=n_slices,
            page_start=page_start,
            page_end=page_start + page_count,
            total_pages=total_pages,
        )
        result = {
            'name': doc_name,
            'content': content,
            'extra': json.dumps(extra_obj, ensure_ascii=False),
            'hash': hash_str(content),
            'file_hash': file_hash,
            'source_pdf': str(pdf_path),
            'source_name': pdf_path.name,
            'page_count': page_count,
            'slice_index': slice_index,
            'n_slices': n_slices,
            'recognition_cost': cost,
        }

        if recog.use_cache:
            self._recog_cache.update_cache(
                json.dumps(
                    {
                        'name': doc_name,
                        'source_name': pdf_path.name,
                        'file_hash': file_hash,
                        'page_start': page_start,
                        'page_end': page_start + page_count,
                        'slice_index': slice_index,
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
                name=doc_name,
                extra=(
                    f'pages={page_count} slice={slice_index + 1}/{n_slices} '
                    f'attempt={attempt}'
                ),
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

        渲染页数超过 max_pages_per_doc 时切成多段，每段作为独立文档：
        首段名=原文件名，后续=原文件名_{n}。切片级多线程并行。
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
        to_process = []  # list of slice units
        skipped = []  # doc_name already in DB
        invalid = []  # (name, reason) 假 PDF / 空文件 / 规划失败
        max_pages = self.max_pages_per_doc()
        n_source_files = 0
        n_sliced_files = 0

        for p in pdf_paths:
            bad = self.classify_pdf_magic(p)
            if bad:
                invalid.append((p.name, bad))
                continue
            try:
                units = self.plan_pdf_slices(p)
            except Exception as e:
                invalid.append((p.name, str(e)))
                continue
            n_source_files += 1
            if len(units) > 1:
                n_sliced_files += 1
            for unit in units:
                name = unit['doc_name']
                if skip_existing and name in existing_names:
                    skipped.append(name)
                    continue
                to_process.append(unit)

        if skipped:
            self.logger.info(
                f"Skip {len(skipped)} slice doc(s) already in doc table: "
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
                f"Skip {len(invalid)} non-PDF/unreadable file(s) "
                f"(wrong magic / empty / plan failed; not sent to VLM): "
                # f"{invalid[:5]}{'...' if len(invalid) > 5 else ''}"
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
            self.logger.info("No new PDF slices to recognize.")
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

        def _unit_label(unit) -> str:
            return unit.get('doc_name') or Path(unit['path']).name

        def _one(unit):
            """单切片识别；任何错误只跳过当前切片，不中断整批。"""
            label = _unit_label(unit)
            try:
                out = self.recognize_pdf(
                    unit['path'],
                    doc_name=unit['doc_name'],
                    page_start=unit['page_start'],
                    page_end=unit['page_end'],
                    slice_index=unit['slice_index'],
                    n_slices=unit['n_slices'],
                    total_pages=unit['total_pages'],
                )
            except NonRetryableError as e:
                self.logger.warning(
                    f"Skip PDF slice (invalid/unreadable, no retry): {label}: {e}"
                )
                _record_skip(label)
                return None
            except Exception as e:
                self.logger.warning(
                    f"Skip PDF slice after error: {label}: {e}"
                )
                _record_skip(label)
                return None
            if out is None:
                self.logger.warning(
                    f"Skip PDF slice (API retries exhausted or empty result): "
                    f"{label}"
                )
                _record_skip(label)
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
            f"PDF recognition start: sources={n_source_files}, "
            f"slices={len(to_process)}, sliced_files={n_sliced_files}, "
            f"max_pages_per_doc={max_pages}, num_thread={num_thread}, "
            f"http_timeout={http_timeout:.0f}s "
            f"(page-slice multi-image per unit)"
        )
        t0 = time.perf_counter()

        if num_thread <= 1:
            bar = tqdm(
                to_process,
                desc='recognize',
                unit='slice',
                bar_format=TQDM_BAR_FORMAT,
            )
            for unit in bar:
                result = _one(unit)
                if result is not None:
                    results.append(result)
                bar.set_postfix(**_postfix())
        else:
            # wait + 心跳：避免 as_completed 在末尾长请求上「假死」无日志
            with ThreadPoolExecutor(max_workers=num_thread) as executor:
                pending = {
                    executor.submit(_one, unit): (unit, time.perf_counter())
                    for unit in to_process
                }
                bar = tqdm(
                    total=len(pending),
                    desc='recognize',
                    unit='slice',
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
                                (now - started, _unit_label(unit))
                                for unit, started in pending.values()
                            ),
                            reverse=True,
                        )
                        top = ', '.join(
                            f'{name}({elapsed:.0f}s)'
                            for elapsed, name in aged[:8]
                        )
                        more = f' +{len(aged) - 8} more' if len(aged) > 8 else ''
                        # self.logger.info(
                        #     f"[recognize] still in-flight={len(pending)} "
                        #     f"(each call may take up to ~{http_timeout:.0f}s "
                        #     f"HTTP timeout + retries): {top}{more}"
                        # )
                        bar.set_postfix(**_postfix(len(pending)))
                        continue

                    for future in done:
                        unit, started = pending.pop(future)
                        label = _unit_label(unit)
                        try:
                            result = future.result()
                            if result is not None:
                                with results_lock:
                                    results.append(result)
                        except Exception as e:
                            self.logger.error(
                                f"Failed to recognize {label}: {e}"
                            )
                            _record_skip(label)
                        bar.update(1)
                        bar.set_postfix(**_postfix(len(pending)))
                bar.close()

        wall = time.perf_counter() - t0
        results.sort(key=lambda x: x.get('name', ''))

        if self.metrics is not None:
            self.metrics.finalize_stage_wall_time('recognize', wall)
            self.metrics.log_stage('recognize')

        self.logger.info(
            f"Recognized {len(results)}/{len(to_process)} valid PDF slice docs "
            f"(sources={n_source_files}, sliced_files={n_sliced_files}, "
            f"skipped existing={len(skipped)}, invalid={len(invalid)}, "
            f"max_pages_per_doc={max_pages}, threads={num_thread}, "
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
