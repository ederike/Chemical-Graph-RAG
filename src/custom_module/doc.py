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
from ..utils.utils import hash_str, CacheDB, Retry
from ..utils.config import resolve_credentials


class Doc(BaseDoc):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        recog = getattr(self.config.doc, 'recognition', None)
        api_key, base_url = resolve_credentials(self.config, recog)
        self.llmmodel = LLM(api_key, base_url)
        self._recog_cache = CacheDB('cache/OpenAI', 'pdf_recognize_cache')
        self.metrics = None  # set by DHMF to PipelineMetrics

    # ------------------------------------------------------------------
    # PDF discovery — same directory as original txt: working_path/doc
    # ------------------------------------------------------------------
    def resolve_doc_dir(self) -> Path:
        """Document directory: working_path/doc (e.g. example/b/doc)."""
        doc_subdir = getattr(self.config.doc, 'doc_dir', 'doc') or 'doc'
        return Path(self.config.settings.working_path) / doc_subdir

    def list_pdf_files(self):
        """Collect PDF paths under working_path/doc (same place as .txt)."""
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
        if name:
            if self.get_docs_by_name(name):
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

        dpi = self.config.doc.recognition.dpi
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        images = []

        doc = fitz.open(str(pdf_path))
        try:
            for page in doc:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img_bytes = pix.tobytes('png')
                b64 = base64.b64encode(img_bytes).decode('ascii')
                images.append(b64)
        finally:
            doc.close()

        return images

    # ------------------------------------------------------------------
    # Vision recognition cache helpers
    # ------------------------------------------------------------------
    def _recognition_cache_key(self, pdf_path: Path, file_hash: str) -> str:
        recog = self.config.doc.recognition
        # 包含提示词正文哈希，避免只改 PROMPT 文本时仍命中旧缓存
        prompt_key = getattr(recog, 'prompt', 'pdf_recognize')
        prompt_body = PROMPT.get(prompt_key, PROMPT.get('pdf_recognize', ''))
        system_body = PROMPT.get('pdf_recognize_system', '')
        prompt_hash = hashlib.md5(
            f'{system_body}\n---\n{prompt_body}'.encode('utf-8')
        ).hexdigest()[:16]
        payload = {
            'file_hash': file_hash,
            'name': pdf_path.name,
            'model_args': recog.model_args,
            'dpi': recog.dpi,
            'prompt': prompt_key,
            'prompt_hash': prompt_hash,
        }
        return hashlib.md5(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')
        ).hexdigest()

    def clear_recognition_cache(self, name: str) -> int:
        """
        Remove PDF recognition cache entries for a given file name.
        Returns number of cache rows deleted.
        """
        deleted = self._recog_cache.delete_by_name(name)
        self.logger.info(f"Recognition cache cleared for {name!r}: {deleted} row(s)")
        return deleted

    # ------------------------------------------------------------------
    # Vision recognition response parsing  →  {"head": str, "body": str}
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

    @staticmethod
    def _as_str(value) -> str:
        if value is None:
            return ''
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value).strip()

    @classmethod
    def parse_recognition_json(cls, text: str):
        """
        解析 PDF 识别 JSON，统一为 {"head": str, "body": str}。
        仅接受 head/body（及极少数兼容别名）；解析失败返回 None。
        """
        cleaned = cls._strip_code_fence(text)
        if not cleaned:
            return None
        try:
            obj = json.loads(cleaned)
        except Exception:
            start_i, end_i = cleaned.find('{'), cleaned.rfind('}')
            if start_i < 0 or end_i <= start_i:
                return None
            try:
                obj = json.loads(cleaned[start_i:end_i + 1])
            except Exception:
                return None

        if not isinstance(obj, dict):
            return None

        # 标准键；兼容旧缓存中的中文键 / 别名（仅作读取回退）
        if 'head' in obj:
            head = cls._as_str(obj.get('head'))
        else:
            head = cls._as_str(
                obj.get('关键信息')
                or obj.get('key_info')
                or obj.get('key_information')
                or obj.get('summary')
            )
        if 'body' in obj:
            body = cls._as_str(obj.get('body'))
        else:
            body = cls._as_str(
                obj.get('正文')
                or obj.get('content')
                or obj.get('full_text')
            )
        if not head and not body:
            return None
        return {'head': head, 'body': body}

    # 兼容旧调用名
    _parse_recognition_json = parse_recognition_json

    @classmethod
    def normalize_recognition(cls, text: str, *, require_json: bool = True) -> dict:
        """
        将模型输出规范为 {"head": str, "body": str}。
        require_json=True（默认）时解析失败抛 ValueError。
        """
        parsed = cls.parse_recognition_json(text)
        if parsed is not None:
            return parsed
        cleaned = cls._strip_code_fence(text)
        if require_json:
            raise ValueError(
                'PDF recognition expected JSON {"head": "...", "body": "..."}, '
                f'got: {cleaned[:240]!r}'
            )
        return {'head': '', 'body': cleaned}

    @staticmethod
    def _wants_json_object(model_args: dict) -> bool:
        rf = (model_args or {}).get('response_format')
        if not isinstance(rf, dict):
            return False
        return str(rf.get('type', '')).lower() == 'json_object'

    @staticmethod
    def doc_extra_payload(
        head: str,
        body: str,
        *,
        page_count: int = 0,
        source_pdf: str = '',
        file_hash: str = '',
        recognition_cost=None,
    ) -> dict:
        """doc.extra 标准结构：供分块 / 超边使用。"""
        return {
            'head': head or '',
            'body': body or '',
            'page_count': page_count,
            'source_pdf': source_pdf,
            'file_hash': file_hash,
            'recognition_cost': recognition_cost or {},
        }

    # ------------------------------------------------------------------
    # Vision recognition
    # ------------------------------------------------------------------
    @Retry(max_attempt=3, wait=0.1, timeout=60)
    def recognize_pdf(self, pdf_path: Path, **kwargs):
        """
        PDF → 图像 → 多模态 LLM → JSON {head, body}。

        适配 @Retry：
          - 接受 attempt / max_attempt（由装饰器注入）
          - 失败 raise，触发重试；全部失败时装饰器返回 None
          - 缓存命中直接成功返回，不进入重试逻辑

        返回 dict 供 insert：
          content : 规范 JSON 字符串 {"head":"...","body":"..."}
          extra   : 结构化 JSON（含 head/body，供 chunk 直接读取）
          hash    : content 哈希

        Metrics: cache hit 只计数；真实识别记 token；墙钟在 prepare_from_pdfs 汇总。
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
            raise ValueError(f"No pages rendered from PDF: {pdf_path}")

        prompt_key = recog.prompt
        user_prompt = PROMPT.get(prompt_key, PROMPT['pdf_recognize'])
        user_prompt = (
            f"{user_prompt}\n\n"
            f"本文件共 {len(images_b64)} 页，以下按页序给出全部页面图片，"
            f'请输出 JSON：{{"head":"...","body":"..."}}。'
        )
        system_prompt = PROMPT.get('pdf_recognize_system', '')

        model_args = dict(recog.model_args or {})
        model_args.setdefault('enable_thinking', False)
        model_args.setdefault('response_format', {'type': 'json_object'})
        if attempt > 1:
            model_args['temperature'] = max(float(model_args.get('temperature') or 0.0), 1.0)
        require_json = self._wants_json_object(model_args)

        response = self.llmmodel.generate_vision(
            prompt={'system': system_prompt, 'user': user_prompt},
            images=images_b64,
            model_args=model_args,
        )

        if response.get('status') != 1:
            raise RuntimeError(
                f"PDF recognition failed for {pdf_path.name} "
                f"(attempt {attempt}/{max_attempt}): {response.get('answer')}"
            )

        raw_answer = (response.get('answer') or '').strip()
        try:
            parsed = self.normalize_recognition(raw_answer, require_json=require_json)
        except ValueError as e:
            raise RuntimeError(
                f"PDF recognition JSON parse failed for {pdf_path.name} "
                f"(attempt {attempt}/{max_attempt}): {e}"
            ) from e

        head = parsed.get('head') or ''
        body = parsed.get('body') or ''
        if not head.strip() and not body.strip():
            raise RuntimeError(
                f"PDF recognition empty head/body for {pdf_path.name} "
                f"(attempt {attempt}/{max_attempt})"
            )

        content_obj = {'head': head, 'body': body}
        content_json = json.dumps(content_obj, ensure_ascii=False)
        cost = {
            'usage_prompt_tokens': response.get('usage_prompt_tokens'),
            'usage_completion_tokens': response.get('usage_completion_tokens'),
            'usage_total_tokens': response.get('usage_total_tokens'),
        }
        extra_obj = self.doc_extra_payload(
            head,
            body,
            page_count=len(images_b64),
            source_pdf=str(pdf_path),
            file_hash=file_hash,
            recognition_cost=cost,
        )

        result = {
            'name': pdf_path.name,
            'content': content_json,
            'extra': json.dumps(extra_obj, ensure_ascii=False),
            'hash': hash_str(content_json),
            'file_hash': file_hash,
            'source_pdf': str(pdf_path),
            'page_count': len(images_b64),
            'head': head,
            'body': body,
            'recognition_cost': cost,
        }

        if recog.use_cache:
            self._recog_cache.update_cache(
                json.dumps({'name': pdf_path.name, 'file_hash': file_hash}, ensure_ascii=False),
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
                extra=f"pages={len(images_b64)} attempt={attempt}",
                log=False,
                accumulate_time=False,
            )

        return result

    def prepare_from_pdfs(self, pdf_paths=None, skip_existing: bool = True):
        """
        Recognize PDFs into doc_list tasks (before insert).
        Each PDF becomes one independent document; texts are not mixed.

        Multi-threaded via config.doc.recognition.num_thread.
        Logs only the real wall-clock total for the whole batch (not per-file times).

        skip_existing: if True (default), skip PDFs whose name is already in doc table
                       — avoids re-recognize and re-insert of already processed files.
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

        recog = self.config.doc.recognition
        num_thread = max(1, int(getattr(recog, 'num_thread', 1) or 1))
        results = []
        results_lock = threading.Lock()

        def _one(path):
            """
            调用带 @Retry 的 recognize_pdf。
            装饰器耗尽后返回 None（不抛错），此处统一当成失败处理。
            """
            out = self.recognize_pdf(path)
            if out is None:
                self.logger.error(
                    f"Failed to recognize {path} after all retries "
                    f"(@Retry exhausted, returned None)"
                )
            return out

        def _postfix():
            if self.metrics is None:
                return {}
            s = self.metrics.stage_snapshot('recognize')
            return {
                'real': s['real'],
                'cache': s['cache'],
                'tok': s['tokens'],
            }

        self.logger.info(
            f"PDF recognition start: {len(to_process)} file(s), "
            f"num_thread={num_thread}"
        )
        t0 = time.perf_counter()

        if num_thread <= 1:
            bar = tqdm(to_process, desc='PDF recognize', unit='pdf')
            for p in bar:
                try:
                    result = _one(p)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to recognize {p}: {e}")
                bar.set_postfix(**_postfix())
        else:
            with ThreadPoolExecutor(max_workers=num_thread) as executor:
                futures = {executor.submit(_one, p): p for p in to_process}
                bar = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc='PDF recognize',
                    unit='pdf',
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
            # Replace any per-task time with true batch wall-clock only
            self.metrics.finalize_stage_wall_time('recognize', wall)
            self.metrics.log_stage('recognize')

        # Single summary line: real total elapsed only (no per-file times)
        self.logger.info(
            f"Recognized {len(results)}/{len(to_process)} new PDF documents "
            f"(skipped existing: {len(skipped)}, threads={num_thread}, "
            f"wall_time={wall:.3f}s)."
        )
        return results

    def processing_single_task(self, task):
        """
        Insert one recognized document.
        Skip if the same file name or content hash already exists in doc table
        (prevents duplicate rows / id drift on re-run).
        """
        text = task['content']
        file_name = task['name']
        content_hash = task.get('hash') or hash_str(text)

        # Dedup by name (primary): already processed file must not re-insert
        existing_by_name = self.get_docs_by_name(file_name)
        if existing_by_name:
            self.logger.info(
                f"Skip insert (name already in DB): {file_name} "
                f"-> existing id(s)={[r['id'] for r in existing_by_name]}"
            )
            return

        # Dedup by content hash (secondary): same text under another name rare but safe
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
        }

        if task.get('extra') is not None:
            add_doc['extra'] = task['extra'] if isinstance(task['extra'], str) else json.dumps(
                task['extra'], ensure_ascii=False
            )
        else:
            # 兼容：从 content JSON 或顶层 head/body 构造 extra
            head = task.get('head') or ''
            body = task.get('body') or ''
            if not head and not body and isinstance(text, str):
                parsed = self.parse_recognition_json(text)
                if parsed:
                    head, body = parsed.get('head') or '', parsed.get('body') or ''
            add_doc['extra'] = json.dumps(
                self.doc_extra_payload(head, body),
                ensure_ascii=False,
            )

        if self.config.doc.count_token:
            add_doc['tokens'] = len(self.tokener.encode(text))

        self.doc_db.buffer.append(add_doc)

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
        self.logger.debug(
            f"Number of documents to insert: {len(self.tasks)} "
            f"(filtered from {len(doc_list)})"
        )
