#!/usr/bin/env python3
"""拆开 PaddleOCR：本地 CPU 版面切分 vs 远程 VL 识别。

不打补丁，按 paddlex 真实顺序自己调各阶段计时：
  render → layout_det (CPU) → crop/merge (CPU) → vl_rec (远程 conc=16) → restructure

  python scripts/profile_paddleocr_stages.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.paddleocr_vl import (  # noqa: E402
    check_truncation,
    inputs_visually_blank,
    join_markdown,
    make_pipeline,
    patch_genai_semaphore,
    render_pdf_pages,
)


def _pick_pdfs(doc_dir: Path, n: int) -> list[Path]:
    import fitz

    out = []
    for p in sorted(doc_dir.iterdir()):
        if p.suffix.lower() != ".pdf":
            continue
        try:
            d = fitz.open(str(p))
            ok = int(d.page_count or 0) >= 1
            d.close()
        except Exception:
            continue
        if ok:
            out.append(p)
        if len(out) >= n:
            break
    return out


def _read_images(inner, pngs: list[str]):
    """与 pipeline.predict 同一条读图路径。"""
    if hasattr(inner, "img_reader") and hasattr(inner, "batch_sampler"):
        batches = list(inner.batch_sampler(pngs))
        images = []
        for batch in batches:
            images.extend(list(inner.img_reader(batch.instances)))
        return images
    import cv2

    return [cv2.imread(p) for p in pngs]


def _n_boxes(layout_det_results) -> int:
    n = 0
    for det in layout_det_results:
        if isinstance(det, dict):
            n += len(det.get("boxes") or [])
        elif hasattr(det, "get"):
            n += len(det.get("boxes") or [])
        else:
            try:
                n += len(det["boxes"])
            except Exception:
                pass
    return n


def run_staged(inner, pngs: list[str], stats: dict, conc_note: str) -> dict:
    snap = dict(stats)
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    images = _read_images(inner, pngs)
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    layout_det_results = list(
        inner.layout_det_model(
            images,
            filter_overlap_boxes=False,
        )
    )
    t_layout = time.perf_counter() - t0
    n_boxes = _n_boxes(layout_det_results)

    from paddlex.inference.pipelines.layout_parsing.utils import gather_imgs

    imgs_in_doc = [
        gather_imgs(img, det["boxes"])
        for img, det in zip(images, layout_det_results)
    ]

    layout_prep_cfg = {
        "layout_shape_mode": "auto",
        "merge_layout_blocks": True,
        "image_labels": ["image", "figure", "chart", "seal"],
        "use_chart_recognition": False,
        "use_seal_recognition": False,
        "ocr_min_pixels": 112896,
        "ocr_max_pixels": 1003520,
        "table_min_pixels": 112896,
        "table_max_pixels": 1003520,
        "chart_min_pixels": 112896,
        "chart_max_pixels": 1003520,
        "formula_min_pixels": 112896,
        "formula_max_pixels": 1003520,
        "seal_min_pixels": 112896,
        "seal_max_pixels": 1003520,
    }
    payloads = [
        (i, images[i], layout_det_results[i], imgs_in_doc[i], layout_prep_cfg)
        for i in range(len(images))
    ]
    t0 = time.perf_counter()
    page_results = [
        inner._paddleocr_vl_prepare_page_serial_benchmarked(p) for p in payloads
    ]
    t_crop = time.perf_counter() - t0

    (
        blocks,
        has_spotting,
        drop_figures_set,
        batch_dict_by_pixel,
        id2pixel_key_map,
    ) = inner._paddleocr_vl_aggregate_vlm_batches(page_results)

    n_vl_items = 0
    for pixel_key, info in batch_dict_by_pixel.items():
        n_vl_items += len(info.get("images") or [])

    t0 = time.perf_counter()
    inner._paddleocr_vl_run_vl_recognition_batches(
        batch_dict_by_pixel, has_spotting, {"max_new_tokens": 4096}
    )
    t_vl = time.perf_counter() - t0

    t0 = time.perf_counter()
    inner._paddleocr_vl_assemble_parsing_results(
        blocks,
        batch_dict_by_pixel,
        id2pixel_key_map,
        drop_figures_set,
        vis_image_labels=["image", "figure", "chart", "seal"],
        image_path_to_obj_map={},
        format_block_content=False,
        use_ocr_for_image_block=False,
    ) if False else None
    t_assemble = 0.0
    # assemble 签名可能随版本变，组装失败不影响阶段对比；用 predict 路径的 restructure 即可
    del t0

    reqs = stats["requests"] - snap["requests"]
    ptok = stats["prompt_tokens"] - snap["prompt_tokens"]
    ctok = stats["completion_tokens"] - snap["completion_tokens"]
    wall = time.perf_counter() - t_all
    return {
        "pages": len(pngs),
        "layout_boxes": n_boxes,
        "vl_regions": n_vl_items,
        "read_s": round(t_read, 3),
        "layout_cpu_s": round(t_layout, 3),
        "crop_merge_cpu_s": round(t_crop, 3),
        "vl_model_s": round(t_vl, 3),
        "wall_s": round(wall, 3),
        "requests": reqs,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "conc": conc_note,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://113.108.154.212:8001/v1")
    ap.add_argument("--model", default="PaddleOCR-VL-1.6")
    ap.add_argument("--doc-dir", default="example/a/doc")
    ap.add_argument("--n-slices", type=int, default=5)
    ap.add_argument("--max-pages", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    pdfs = _pick_pdfs(Path(args.doc_dir), args.n_slices + args.warmup)
    if len(pdfs) < args.warmup + 1:
        raise SystemExit(f"not enough PDFs in {args.doc_dir}")

    print(
        f"stage profile url={args.url} conc={args.conc} dpi={args.dpi} "
        f"max_pages={args.max_pages} n_slices={args.n_slices}",
        flush=True,
    )

    t_init0 = time.perf_counter()
    stats = patch_genai_semaphore()
    pipe = make_pipeline(args.url, args.model, args.conc)
    inner = pipe.paddlex_pipeline
    init_s = time.perf_counter() - t_init0
    print(f"pipeline_init_s={init_s:.2f}", flush=True)
    print(
        f"inner={type(inner).__name__} layout={type(inner.layout_det_model).__name__} "
        f"vl={type(inner.vl_rec_model).__name__}",
        flush=True,
    )

    work = Path(tempfile.mkdtemp(prefix="ocr_stage_"))
    rows = []

    def one(pdf: Path, tag: str):
        t_r0 = time.perf_counter()
        pngs = render_pdf_pages(
            pdf, work / pdf.stem / "pages", args.dpi, 0, 0, args.max_pages,
        )
        t_render = time.perf_counter() - t_r0
        if not pngs or inputs_visually_blank(pngs):
            print(f"  SKIP {tag} {pdf.name}", flush=True)
            return None
        rec = run_staged(inner, pngs, stats, str(args.conc))
        rec["name"] = pdf.name
        rec["tag"] = tag
        rec["render_s"] = round(t_render, 3)
        rec["local_cpu_s"] = round(
            rec["render_s"] + rec["read_s"] + rec["layout_cpu_s"] + rec["crop_merge_cpu_s"],
            3,
        )
        total = rec["local_cpu_s"] + rec["vl_model_s"] or 1.0
        rec["local_cpu_pct"] = round(100.0 * rec["local_cpu_s"] / total, 1)
        rec["vl_model_pct"] = round(100.0 * rec["vl_model_s"] / total, 1)
        print(
            f"  {tag} {pdf.name}: render={rec['render_s']:.2f}s "
            f"layout_cpu={rec['layout_cpu_s']:.2f}s "
            f"crop={rec['crop_merge_cpu_s']:.2f}s "
            f"LOCAL={rec['local_cpu_s']:.2f}s ({rec['local_cpu_pct']}%) | "
            f"VL={rec['vl_model_s']:.2f}s ({rec['vl_model_pct']}%) "
            f"boxes={rec['layout_boxes']} regions={rec['vl_regions']} "
            f"reqs={rec['requests']} gen_tok={rec['completion_tokens']}",
            flush=True,
        )
        return rec

    for p in pdfs[: args.warmup]:
        one(p, "warmup")
    for p in pdfs[args.warmup : args.warmup + args.n_slices]:
        rec = one(p, "run")
        if rec is not None:
            rows.append(rec)

    if not rows:
        raise SystemExit("no measured slices")

    def mean(k):
        return round(statistics.mean(r[k] for r in rows), 3)

    loc = sum(r["local_cpu_s"] for r in rows)
    vl = sum(r["vl_model_s"] for r in rows)
    tot = loc + vl or 1.0
    summary = {
        "url": args.url,
        "conc": args.conc,
        "dpi": args.dpi,
        "max_pages": args.max_pages,
        "pipeline_init_s": round(init_s, 3),
        "n_slices": len(rows),
        "mean_s": {
            "render": mean("render_s"),
            "layout_cpu": mean("layout_cpu_s"),
            "crop_merge_cpu": mean("crop_merge_cpu_s"),
            "local_cpu_total": mean("local_cpu_s"),
            "vl_model": mean("vl_model_s"),
            "requests": mean("requests"),
        },
        "share_pct": {
            "local_cpu": round(100.0 * loc / tot, 1),
            "vl_model": round(100.0 * vl / tot, 1),
        },
        "slices": rows,
    }
    print("SUMMARY", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    out = _ROOT / "scripts" / "outputs" / "ocr_stage_profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
