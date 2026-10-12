#!/usr/bin/env python3
"""Run benchmark_agentic_1 with MSDS OOM guards. Resume-safe via evals.json."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")


def main() -> int:
    os.chdir(ROOT)
    import importlib.util

    path = ROOT / "scripts" / "oom_safe_retrieve.py"
    spec = importlib.util.spec_from_file_location("oom_safe_retrieve", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.apply_oom_guards()
    from benchmark.benchmark_agentic_1.run import main as bench_main

    return int(bench_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
