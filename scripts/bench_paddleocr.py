#!/usr/bin/env python3
"""测 PaddleOCR-VL 吞吐：按生产切片（max_pages_per_doc）串行 predict。

  python scripts/bench_paddleocr.py
  python scripts/bench_paddleocr.py --n-slices 8 --max-pages 4 --dpi 150
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.paddleocr_vl import (  # noqa: E402
    EmptyOCRError,
    make_pipeline,
    parse_document,
    patch_genai_semaphore,
)


def _metrics_url(ocr_url: str) -> str:
    u = ocr_url.rstrip("/")
    if u.endswith("/v1"):
        u = u[:-3]
    return u.rstrip("/") + "/metrics"


def _fetch_metrics(ocr_url: str, timeout: float = 5.0) -> dict:
    out = {}
    try:
        raw = urllib.request.urlopen(_metrics_url(ocr_url), timeout=timeout).read().decode()
    except Exception as e:
        return {"_error": str(e)}
    keys = (
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:gpu_cache_usage_perc",
        "vllm:avg_generation_throughput_toks_per_s",
        "vllm:avg_prompt_throughput_toks_per_s",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:request_success_total",
    )
    for line in raw.splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        k, v = line.rsplit(" ", 1)
        base = k.split("{", 1)[0]
        if any(base == x or k.startswith(x) for x in keys):
            try:
                out[k] = float(v)
            except ValueError:
                continue
    return out


def _pick_pdfs(doc_dir: Path, n_slices: int, max_pages: int) -> list[Path]:
    import fitz

    pdfs = sorted(p for p in doc_dir.iterdir() if p.suffix.lower() == ".pdf")
    picked = []
    for p in pdfs:
        try:
            d = fitz.open(str(p))
            n = int(d.page_count or 0)
            d.close()
        except Exception:
            continue
        if n >= 1:
            picked.append(p)
        if len(picked) >= n_slices:
            break
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://113.108.154.212:8001/v1")
    ap.add_argument("--model", default="PaddleOCR-VL-1.6")
    ap.add_argument("--doc-dir", default="example/a/doc")
    ap.add_argument("--n-slices", type=int, default=8)
    ap.add_argument("--max-pages", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    doc_dir = Path(args.doc_dir)
    pdfs = _pick_pdfs(doc_dir, args.n_slices + args.warmup, args.max_pages)
    if len(pdfs) < args.warmup + 1:
        raise SystemExit(f"not enough PDFs under {doc_dir}")

    print(
        f"OCR bench url={args.url} model={args.model} conc={args.conc} "
        f"dpi={args.dpi} max_pages={args.max_pages} n_slices={args.n_slices} "
        f"warmup={args.warmup}",
        flush=True,
    )
    print("vLLM metrics before:", json.dumps(_fetch_metrics(args.url), ensure_ascii=False), flush=True)

    t_init = time.perf_counter()
    stats = patch_genai_semaphore()
    pipe = make_pipeline(args.url, args.model, args.conc)
    init_s = time.perf_counter() - t_init
    print(f"pipeline_init_s={init_s:.2f}", flush=True)

    work = Path(tempfile.mkdtemp(prefix="ocr_bench_"))
    rows = []

    def run_one(pdf: Path, tag: str) -> dict | None:
        snap = dict(stats)
        t0 = time.perf_counter()
        try:
            md = parse_document(
                pipe,
                pdf,
                stats,
                dpi=args.dpi,
                page_start=0,
                page_end=args.max_pages,
                work_dir=work / pdf.stem,
                keep_pages=True,
            )
        except EmptyOCRError as e:
            dt = time.perf_counter() - t0
            print(f"  SKIP {tag} {pdf.name}: {e} ({dt:.2f}s)", flush=True)
            return None
        dt = time.perf_counter() - t0
        reqs = stats["requests"] - snap["requests"]
        ptok = stats["prompt_tokens"] - snap["prompt_tokens"]
        ctok = stats["completion_tokens"] - snap["completion_tokens"]
        nchar = len(md)
        # 实际渲染页数
        page_dir = work / pdf.stem / (pdf.stem + "_pages")
        n_pages = len(list(page_dir.glob("*.png"))) if page_dir.exists() else args.max_pages
        rec = {
            "name": pdf.name,
            "tag": tag,
            "seconds": round(dt, 3),
            "pages": n_pages,
            "chars": nchar,
            "requests": reqs,
            "prompt_tokens": ptok,
            "completion_tokens": ctok,
            "pages_per_s": round(n_pages / dt, 3) if dt else 0,
            "gen_tok_per_s": round(ctok / dt, 1) if dt else 0,
        }
        print(
            f"  {tag} {pdf.name}: {dt:.2f}s pages={n_pages} chars={nchar} "
            f"reqs={reqs} prompt={ptok} completion={ctok} "
            f"{rec['pages_per_s']} page/s gen={rec['gen_tok_per_s']} tok/s",
            flush=True,
        )
        return rec

    warmup_pdfs = pdfs[: args.warmup]
    measure_pdfs = pdfs[args.warmup : args.warmup + args.n_slices]

    for p in warmup_pdfs:
        run_one(p, "warmup")

    t_batch = time.perf_counter()
    snap_batch = dict(stats)
    for p in measure_pdfs:
        rec = run_one(p, "run")
        if rec is not None:
            rows.append(rec)
    batch_s = time.perf_counter() - t_batch

    if not rows:
        raise SystemExit("all slices empty/skipped")

    pages = sum(r["pages"] for r in rows)
    secs = [r["seconds"] for r in rows]
    ptok = stats["prompt_tokens"] - snap_batch["prompt_tokens"]
    ctok = stats["completion_tokens"] - snap_batch["completion_tokens"]
    reqs = stats["requests"] - snap_batch["requests"]
    summary = {
        "url": args.url,
        "model": args.model,
        "conc": args.conc,
        "dpi": args.dpi,
        "max_pages_per_slice": args.max_pages,
        "pipeline_init_s": round(init_s, 3),
        "n_slices_ok": len(rows),
        "n_pages": pages,
        "wall_s": round(batch_s, 3),
        "slice_s_mean": round(statistics.mean(secs), 3),
        "slice_s_p50": round(statistics.median(secs), 3),
        "slice_s_min": round(min(secs), 3),
        "slice_s_max": round(max(secs), 3),
        "pages_per_s": round(pages / batch_s, 3) if batch_s else 0,
        "slices_per_s": round(len(rows) / batch_s, 3) if batch_s else 0,
        "requests": reqs,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "gen_tok_per_s": round(ctok / batch_s, 1) if batch_s else 0,
        "total_tok_per_s": round((ptok + ctok) / batch_s, 1) if batch_s else 0,
        "vllm_metrics_after": _fetch_metrics(args.url),
        "slices": rows,
    }
    print("SUMMARY", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    out = _ROOT / "scripts" / "outputs" / "ocr_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
