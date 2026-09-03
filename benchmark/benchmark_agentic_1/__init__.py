"""
独立评测模块：多跳问题生成 + agentic_query vs 纯 LLM + Judge + Excel 汇总。

与 useless/ 下旧 benchmark 无任何依赖。运行::

    python -m benchmark.benchmark_agentic_1
"""

from .config import DEFAULT_CONFIG_PATH, BenchmarkConfig
from .workflow import AgenticBenchmarkWorkflow

__all__ = [
    "AgenticBenchmarkWorkflow",
    "BenchmarkConfig",
    "DEFAULT_CONFIG_PATH",
]
