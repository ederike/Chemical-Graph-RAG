#!/usr/bin/env python3
"""
benchmark_agentic_1 入口（无 CLI 传参）。

全部选项写在 benchmark/benchmark_agentic_1/config.yaml::

    run:
      mode: all          # generate | evaluate | report | excel | all

运行::

    python -m benchmark.benchmark_agentic_1
    python -m benchmark.benchmark_agentic_1.run
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from benchmark.benchmark_agentic_1.config import DEFAULT_CONFIG_PATH
    from benchmark.benchmark_agentic_1.workflow import AgenticBenchmarkWorkflow

    try:
        wf = AgenticBenchmarkWorkflow.from_config(DEFAULT_CONFIG_PATH)
        wf.run()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[benchmark_agentic_1] failed: {e}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
