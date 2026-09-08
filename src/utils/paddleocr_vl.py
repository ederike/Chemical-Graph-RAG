"""PaddleOCR-VL vLLM 客户端（文档识别加速路径）。

契约与踩坑见 useless/ocr_api_client.py。对接入口：
  patch_genai_semaphore() 必须在构造 PaddleOCRVL 之前调用；
  make_pipeline() + parse_document()。
"""
import logging
import shutil
import tempfile
import time
from pathlib import Path

logger = logging.getLogger("paddleocr_vl")

DEFAULT_MODEL = "PaddleOCR-VL-1.6"
TABLE_OPEN, TABLE_CLOSE = "<table", "</table>"


class EmptyOCRError(ValueError):
    """空白页或无识别正文：不应重试。"""


def patch_genai_semaphore() -> dict:
    """Py3.9 + paddlex 3.7: 在运行中的 loop 内懒创建 semaphore。"""
    import asyncio
    import threading
    from paddlex.inference.models.common import genai as genai_mod

    lock = threading.Lock()
    semaphores = {}
    stats = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _create_chat_completion(self, messages, *, return_future=False, **kwargs):
        async def _with_sem():
            loop = asyncio.get_running_loop()
            with lock:
                sem = semaphores.setdefault(
                    (id(self), id(loop)),
                    asyncio.Semaphore(getattr(self, "_max_concurrency", 8) or 8),
                )
            async with sem:
                r = await self._client.chat.completions.create(
                    model=self._model_name, messages=messages, **kwargs
                )
                stats["requests"] += 1
                u = getattr(r, "usage", None)
                if u is not None:
                    stats["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
                    stats["completion_tokens"] += int(
                        getattr(u, "completion_tokens", 0) or 0
                    )
                return r

        return genai_mod.run_async(_with_sem(), return_future=return_future)

    genai_mod.GenAIClient.create_chat_completion = _create_chat_completion
    logger.info("Patched GenAIClient semaphore (cross-loop) + token stats")
    return stats


def _jsonify(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return obj


def _encode_image_jpeg(img, quality: int = 95) -> str:
    import base64
    import io
    from PIL import Image
    import numpy as np

    arr = np.asarray(img)
    if arr.ndim == 2:
        im = Image.fromarray(arr, mode="L")
    else:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        im = Image.fromarray(arr.astype("uint8"), mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class RemoteLayoutDetector:
    """把 PP-DocLayoutV3 调用转到远程 /layout，返回与本地模型相同的 boxes 结构。"""

    def __init__(self, url: str, local_model=None, timeout: float = 120.0):
        self.url = (url or "").rstrip("/")
        self._local = local_model
        self.timeout = float(timeout or 120.0)
        if self._local is not None and hasattr(self._local, "batch_sampler"):
            self.batch_sampler = self._local.batch_sampler
        else:
            class _BS:
                batch_size = 8
            self.batch_sampler = _BS()

    def __call__(self, images, **kwargs):
        import json
        import urllib.error
        import urllib.request

        if images is None:
            return []
        if not isinstance(images, (list, tuple)):
            images = [images]
        payload = {
            "images": [_encode_image_jpeg(im) for im in images],
            "threshold": kwargs.get("threshold"),
            "layout_nms": kwargs.get("layout_nms"),
            "layout_unclip_ratio": kwargs.get("layout_unclip_ratio"),
            "layout_merge_bboxes_mode": kwargs.get("layout_merge_bboxes_mode"),
            "layout_shape_mode": kwargs.get("layout_shape_mode"),
            "filter_overlap_boxes": kwargs.get("filter_overlap_boxes"),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url + "/layout",
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "paddleocr-layout-client",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(
                f"远程版面切框失败 HTTP {e.code} {self.url}/layout: {err}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"连不上版面切框服务 {self.url}/layout "
                f"（请在 CPU/GPU 强机启动 scripts/paddleocr_layout_server.py，"
                f"并把 doc.recognition.paddleocr_layout_url 指过去）: {e}"
            ) from e
        results = body.get("results")
        if not isinstance(results, list) or len(results) != len(images):
            raise RuntimeError(
                f"版面服务返回异常: n_images={len(images)} n_results="
                f"{0 if not isinstance(results, list) else len(results)}"
            )
        return results

    def __getattr__(self, name):
        if self._local is not None:
            return getattr(self._local, name)
        raise AttributeError(name)


def ping_layout_server(url: str, timeout: float = 5.0) -> None:
    import urllib.error
    import urllib.request

    health = (url or "").rstrip("/") + "/health"
    try:
        req = urllib.request.Request(health, headers={"User-Agent": "paddleocr-layout-client"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read(256)
    except urllib.error.HTTPError as e:
        if e.code >= 500:
            raise RuntimeError(f"版面切框服务异常 HTTP {e.code}: {health}") from e
    except Exception as e:
        raise RuntimeError(
            f"连不上版面切框服务 {health} "
            f"（空着 paddleocr_layout_url 则回退本机 CPU）: {e}"
        ) from e


def ping_server(url: str, timeout: float = 5.0) -> None:
    """GET {url}/models；连不上时给出明确地址，避免 paddlex 只报 Connection error。"""
    import urllib.error
    import urllib.request

    base = (url or "").rstrip("/")
    if not base.endswith("/v1"):
        models = base + "/v1/models"
    else:
        models = base + "/models"
    try:
        req = urllib.request.Request(models, headers={"User-Agent": "paddleocr-vl-client"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read(256)
    except urllib.error.HTTPError as e:
        if e.code >= 500:
            raise RuntimeError(
                f"PaddleOCR-VL 服务异常 HTTP {e.code}: {models}"
            ) from e
    except Exception as e:
        raise RuntimeError(
            f"连不上 PaddleOCR-VL 服务 {models} "
            f"（请检查 doc.recognition.paddleocr_url，本机 8001 未监听时应填 GPU 机地址）: {e}"
        ) from e


def make_pipeline(
    url: str,
    model_name: str = DEFAULT_MODEL,
    concurrency: int = 16,
    layout_url: str = "",
):
    from paddleocr import PaddleOCRVL
    ping_server(url)
    layout_url = (layout_url or "").strip()
    if layout_url:
        ping_layout_server(layout_url)

    pipe = PaddleOCRVL(
        pipeline_version="v1.6",
        vl_rec_backend="vllm-server",
        vl_rec_server_url=url,
        vl_rec_api_model_name=model_name or DEFAULT_MODEL,
        vl_rec_max_concurrency=max(1, int(concurrency or 16)),
    )
    if layout_url:
        inner = getattr(pipe, "paddlex_pipeline", pipe)
        local = getattr(inner, "layout_det_model", None)
        inner.layout_det_model = RemoteLayoutDetector(layout_url, local_model=local)
        logger.info("Layout detection -> remote %s/layout", layout_url)
    return pipe


def render_pdf_pages(
    pdf_path: Path,
    page_dir: Path,
    dpi: int,
    max_pages: int = 0,
    page_start: int = 0,
    page_end: int = None,
) -> list:
    import fitz

    page_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    try:
        n_all = doc.page_count
        start = max(0, int(page_start or 0))
        end = n_all if page_end is None else min(int(page_end), n_all)
        if max_pages and max_pages > 0:
            end = min(end, start + int(max_pages))
        paths = []
        for i in range(start, end):
            p = page_dir / f"page_{i + 1:04d}.png"
            if not p.exists():
                doc[i].get_pixmap(dpi=dpi).save(str(p))
            paths.append(str(p))
        return paths
    finally:
        doc.close()


def inputs_visually_blank(paths: list, white_min: int = 250) -> bool:
    """全部输入图几乎纯白（常见于 PDF 末尾空白页）。"""
    from PIL import Image
    import numpy as np

    if not paths:
        return True
    for p in paths:
        im = np.asarray(Image.open(p).convert("L"))
        if im.size == 0:
            continue
        if float((im < white_min).mean()) > 0.002:
            return False
    return True


def join_markdown(results: list) -> str:
    parts = []
    for res in results:
        md = res.markdown
        t = md["markdown_texts"]
        if isinstance(t, str):
            parts.append(t)
        else:
            parts.extend(t)
    return "\n\n".join(parts)


def check_truncation(md_text: str, name: str) -> int:
    opened = md_text.count(TABLE_OPEN)
    closed = md_text.count(TABLE_CLOSE)
    if opened != closed:
        logger.warning(
            "%s: %d <table> vs %d </table> -> possible output truncation",
            name, opened, closed,
        )
    return opened - closed


def parse_document(
    pipeline,
    src_path: Path,
    stats: dict,
    *,
    dpi: int = 200,
    page_start: int = 0,
    page_end: int = None,
    is_image: bool = False,
    work_dir: Path = None,
    keep_pages: bool = False,
) -> str:
    """对给定页范围（或单图）predict + restructure，返回 markdown 文本。"""
    t0 = time.time()
    src_path = Path(src_path)
    own_work = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="paddleocr_pages_"))
        own_work = True
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        if is_image:
            inputs = [str(src_path)]
        else:
            page_dir = work_dir / (src_path.stem + "_pages")
            inputs = render_pdf_pages(
                src_path, page_dir, dpi, 0, page_start, page_end,
            )
        if not inputs:
            raise ValueError(f"{src_path.name}: no pages/images to OCR")
        if inputs_visually_blank(inputs):
            raise EmptyOCRError(
                f"{src_path.name}: visually blank pages "
                f"[{page_start}:{page_end}], skip OCR"
            )

        results = list(pipeline.predict(inputs, markdown_ignore_labels=[]))
        if not results:
            raise ValueError(f"{src_path.name}: predict returned no result")

        restructured = pipeline.restructure_pages(
            results,
            merge_tables=True,
            relevel_titles=True,
            concatenate_pages=False,
        )
        md_text = join_markdown(restructured)
        if len(md_text.strip()) < 20:
            raise EmptyOCRError(
                f"{src_path.name}: markdown nearly empty ({len(md_text)} chars) "
                f"pages=[{page_start}:{page_end}]"
            )
        check_truncation(md_text, src_path.name)
        logger.info(
            "%s: paddleocr done in %.1fs, %d input(s)",
            src_path.name, time.time() - t0, len(inputs),
        )
        return md_text
    finally:
        if own_work and not keep_pages:
            shutil.rmtree(work_dir, ignore_errors=True)
