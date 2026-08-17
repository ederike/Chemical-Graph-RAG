#!/usr/bin/env python3
"""
benchmark2 入口（无 CLI 传参）。

全部选项写在 benchmark2/config.yaml::

    run:
      mode: all          # stats | evaluate | report | all

运行::

    python -m benchmark2.run
    python benchmark2.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from benchmark2.config import DEFAULT_CONFIG_PATH
    from benchmark2.workflow import Benchmark2Workflow

    try:
        wf = Benchmark2Workflow.from_config(DEFAULT_CONFIG_PATH)
        wf.run()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[benchmark2] failed: {e}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
