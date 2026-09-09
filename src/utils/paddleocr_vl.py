"""PaddleOCR-VL vLLM 客户端（文档识别加速路径）。

契约与踩坑见 useless/ocr_api_client.py。对接入口：
  patch_genai_semaphore() 必须在构造 PaddleOCRVL 之前调用；
  make_pipeline() + parse_document()。

paddlex 版面推理不是线程安全的：同一进程里并发 predict 会把不同 batch
的 DETR 框叠到一起（[B,300,4] broadcast）。有 paddleocr_layout_url 时
本机不再加载 PP-DocLayout，切框纯 HTTP；切片并发用 pipeline 池，每份
pipeline 同一时刻只服务一个切片。VL 请求跨 pipeline 共用一个全局槽位。
"""
import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path

logger = logging.getLogger("paddleocr_vl")

DEFAULT_MODEL = "PaddleOCR-VL-1.6"
TABLE_OPEN, TABLE_CLOSE = "<table", "</table>"

_CREATE_MODEL_LOCK = threading.Lock()
_VL_GATE = None
_VL_GATE_LOCK = threading.Lock()


class EmptyOCRError(ValueError):
    """空白页或无识别正文：不应重试。"""


def patch_genai_semaphore(concurrency: int = 16) -> dict:
    """Py3.9 + paddlex 3.7: VL HTTP 用线程闸门，跨 pipeline 共享上限。不记 token。"""
    import asyncio
    from paddlex.inference.models.common import genai as genai_mod

    global _VL_GATE
    n = max(1, int(concurrency or 16))
    with _VL_GATE_LOCK:
        if _VL_GATE is None:
            _VL_GATE = threading.BoundedSemaphore(n)
        gate = _VL_GATE
    stats = {"requests": 0}
    lock = threading.Lock()

    def _create_chat_completion(self, messages, *, return_future=False, **kwargs):
        async def _with_sem():
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, gate.acquire)
            try:
                r = await self._client.chat.completions.create(
                    model=self._model_name, messages=messages, **kwargs
                )
                with lock:
                    stats["requests"] += 1
                return r
            finally:
                gate.release()

        return genai_mod.run_async(_with_sem(), return_future=return_future)

    genai_mod.GenAIClient.create_chat_completion = _create_chat_completion
    logger.info("Patched GenAIClient VL gate concurrency=%d", n)
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


class _BatchSize:
    def __init__(self, batch_size: int = 8):
        self.batch_size = max(1, int(batch_size or 8))


class RemoteLayoutDetector:
    """把 PP-DocLayoutV3 调用转到远程 /layout。不持有、不转发本机 paddle 模型。

    一次 HTTP 对应 cv worker 的一批页图（通常即一个切片）。
    切片级并发由 Doc 的 pipeline 池对齐 /health 的 workers。
    """

    def __init__(
        self,
        url: str,
        timeout: float = 120.0,
        workers: int = 1,
        batch_size: int = 8,
    ):
        self.url = (url or "").rstrip("/")
        self.timeout = float(timeout or 120.0)
        self.workers = max(1, int(workers or 1))
        self.batch_sampler = _BatchSize(batch_size)

    def close(self):
        return None

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


def ping_layout_server(url: str, timeout: float = 5.0) -> int:
    """探活并返回远端 worker 进程数（失败则抛错；解析不到则 1）。"""
    import json
    import urllib.error
    import urllib.request

    health = (url or "").rstrip("/") + "/health"
    try:
        req = urllib.request.Request(health, headers={"User-Agent": "paddleocr-layout-client"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(4096)
    except urllib.error.HTTPError as e:
        if e.code >= 500:
            raise RuntimeError(f"版面切框服务异常 HTTP {e.code}: {health}") from e
        return 1
    except Exception as e:
        raise RuntimeError(
            f"连不上版面切框服务 {health} "
            f"（空着 paddleocr_layout_url 则回退本机 CPU）: {e}"
        ) from e
    workers = 1
    try:
        body = json.loads(raw.decode("utf-8"))
        workers = max(1, int(body.get("workers") or 1))
    except Exception:
        workers = 1
    return workers


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


def _is_layout_config(config) -> bool:
    if not isinstance(config, dict):
        return False
    module = str(config.get("module_name") or "")
    name = str(config.get("model_name") or "")
    return module == "layout_detection" or name.startswith("PP-DocLayout")


def _close_local_layout(model) -> None:
    if model is None or isinstance(model, RemoteLayoutDetector):
        return
    closer = getattr(model, "close", None)
    if not callable(closer):
        return
    try:
        closer()
    except Exception:
        logger.debug("close local layout model failed", exc_info=True)


def _attach_remote_layout(pipe, layout_url: str, workers: int, batch_size: int = 8):
    inner = getattr(pipe, "paddlex_pipeline", pipe)
    current = getattr(inner, "layout_det_model", None)
    if isinstance(current, RemoteLayoutDetector):
        current.workers = max(1, int(workers or 1))
        return current
    det = RemoteLayoutDetector(
        layout_url, workers=workers, batch_size=batch_size,
    )
    inner.layout_det_model = det
    _close_local_layout(current)
    return det


def make_pipeline(
    url: str,
    model_name: str = DEFAULT_MODEL,
    concurrency: int = 16,
    layout_url: str = "",
):
    from paddleocr import PaddleOCRVL
    ping_server(url)
    layout_url = (layout_url or "").strip()
    layout_workers = ping_layout_server(layout_url) if layout_url else 1
    conc = max(1, int(concurrency or 16))
    kwargs = dict(
        pipeline_version="v1.6",
        vl_rec_backend="vllm-server",
        vl_rec_server_url=url,
        vl_rec_api_model_name=model_name or DEFAULT_MODEL,
        vl_rec_max_concurrency=conc,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )
    if not layout_url:
        return PaddleOCRVL(**kwargs)

    from paddlex.inference.pipelines.base import BasePipeline

    orig = BasePipeline.create_model

    def _create_model(self, config, **kw):
        if _is_layout_config(config):
            bs = int((config or {}).get("batch_size") or 8)
            logger.debug(
                "Skip local %s; layout -> %s/layout",
                (config or {}).get("model_name") or "PP-DocLayout",
                layout_url,
            )
            return RemoteLayoutDetector(
                layout_url, workers=layout_workers, batch_size=bs,
            )
        return orig(self, config, **kw)

    with _CREATE_MODEL_LOCK:
        BasePipeline.create_model = _create_model
        try:
            pipe = PaddleOCRVL(**kwargs)
        finally:
            BasePipeline.create_model = orig

    det = _attach_remote_layout(pipe, layout_url, layout_workers)
    logger.info(
        "Layout detection -> remote %s/layout workers=%d (pipeline pool)",
        layout_url, det.workers,
    )
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
    stats: dict = None,
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
