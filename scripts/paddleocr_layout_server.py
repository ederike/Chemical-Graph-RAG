#!/usr/bin/env python3
"""远程 PP-DocLayoutV3 切框服务：多进程，每进程一份模型。

paddlex 版面推理不是线程安全的，并行靠进程。默认不占满整机核，自行改下面常量。

    python scripts/paddleocr_layout_server.py

客户端:
    paddleocr_layout_url: 'http://HOST:8002'
"""

# ---- 启动默认值（改这里；命令行 --workers 等会覆盖）----
HOST = "0.0.0.0"
PORT = 8002
MODEL = "PP-DocLayoutV3"
WORKERS = 8          # 进程数，每进程一份模型
OMP_THREADS = 8      # 每进程内部 OpenMP/MKL 线程；占用核约 WORKERS * OMP_THREADS
# -------------------------------------------------------

import argparse
import base64
import io
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import get_context
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("layout_server")

_MODEL = None
_POOL = None  # type: Optional[ProcessPoolExecutor]
_META = {"workers": 1, "omp_threads": 1, "model": "PP-DocLayoutV3", "cpu": 1}


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
    return str(obj)


def _boxes_from_result(res) -> dict:
    if isinstance(res, dict):
        boxes = res.get("boxes") or []
    else:
        try:
            boxes = res["boxes"]
        except Exception:
            boxes = getattr(res, "json", {}).get("res", {}).get("boxes") or []
            if not boxes and hasattr(res, "get"):
                boxes = res.get("boxes") or []
    out = []
    for b in boxes:
        if not isinstance(b, dict):
            continue
        item = {
            "cls_id": _jsonify(b.get("cls_id", 0)),
            "label": b.get("label", "text"),
            "score": float(b.get("score") or 0.0),
            "coordinate": _jsonify(b.get("coordinate") or []),
        }
        for k in ("poly", "points", "shape"):
            if k in b:
                item[k] = _jsonify(b[k])
        out.append(item)
    return {"boxes": out}


def _init_worker(model_name: str, omp_threads: int):
    """每个进程独立加载模型；必须在 import paddle 之前设线程数。"""
    global _MODEL
    n = max(1, int(omp_threads or 1))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    os.environ["CPU_NUM"] = str(n)
    from paddlex import create_model

    _MODEL = create_model(model_name)


def _worker_predict(images_b64: list) -> dict:
    from PIL import Image
    import numpy as np

    arrays = []
    for b64 in images_b64:
        raw = base64.b64decode(b64)
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        arrays.append(np.asarray(im))
    raw = list(_MODEL.predict(arrays))
    results = [_boxes_from_result(r) for r in raw]
    return {
        "pid": os.getpid(),
        "n_img": len(arrays),
        "n_boxes": sum(len(r["boxes"]) for r in results),
        "results": results,
    }


def _worker_ready():
    return {"pid": os.getpid(), "ok": _MODEL is not None}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info("%s - " + fmt, self.address_string(), *args)

    def _send(self, code: int, obj: dict):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            self._send(200, {"ok": True, **_META})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/layout":
            self._send(404, {"error": "not found"})
            return
        if _POOL is None:
            self._send(503, {"error": "worker pool not ready"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            self._send(400, {"error": f"invalid json: {e}"})
            return
        images_b64 = body.get("images") or []
        if not images_b64:
            self._send(400, {"error": "images empty"})
            return
        try:
            out = _POOL.submit(_worker_predict, images_b64).result()
        except Exception as e:
            logger.exception("layout predict failed")
            self._send(500, {"error": str(e)})
            return
        logger.info(
            "layout pid=%s n_img=%d n_boxes=%d",
            out.get("pid"), out.get("n_img"), out.get("n_boxes"),
        )
        self._send(200, {"results": out["results"]})


def main() -> None:
    global _POOL, _META
    cpu = os.cpu_count() or 0
    ap = argparse.ArgumentParser(
        description="Multi-process PP-DocLayoutV3 HTTP server",
    )
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help=f"进程数（每进程一份模型）。默认 {WORKERS}",
    )
    ap.add_argument(
        "--omp-threads",
        type=int,
        default=OMP_THREADS,
        help=f"每进程 OpenMP/MKL 线程。默认 {OMP_THREADS}",
    )
    args = ap.parse_args()
    workers = max(1, int(args.workers))
    omp = max(1, int(args.omp_threads))
    _META = {
        "workers": workers,
        "omp_threads": omp,
        "model": args.model,
        "cpu": cpu,
        "approx_threads": workers * omp,
    }
    logger.info(
        "start workers=%d omp_threads=%d cpu=%s approx_used=%d model=%s "
        "(不自动占满核，改脚本顶部 WORKERS / OMP_THREADS)",
        workers, omp, cpu or "?", workers * omp, args.model,
    )
    ctx = get_context("spawn")
    _POOL = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(args.model, omp),
    )
    logger.info("loading %d copies of %s ...", workers, args.model)
    futs = [_POOL.submit(_worker_ready) for _ in range(workers)]
    pids = []
    for f in as_completed(futs):
        r = f.result()
        pids.append(r["pid"])
    logger.info("all workers ready pids=%s", pids)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("Layout server http://%s:%d/layout", args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("stop")
        httpd.server_close()
        _POOL.shutdown(wait=False, cancel_futures=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
