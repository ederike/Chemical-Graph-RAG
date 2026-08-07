#!/usr/bin/env python3
"""
benchmark 脚本入口（无 CLI 传参）。

全部选项写在 benchmark/config.yaml，例如::

    run:
      mode: all          # generate | evaluate | report | all

运行::

    python -m benchmark.run
    # 或项目根目录
    python benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

def main() -> int:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from benchmark.config import DEFAULT_CONFIG_PATH
    from benchmark.workflow import TestQueryWorkflow

    try:
        wf = TestQueryWorkflow.from_config(DEFAULT_CONFIG_PATH)
        wf.run()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[benchmark] failed: {e}", file=sys.stderr)
        raise
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
