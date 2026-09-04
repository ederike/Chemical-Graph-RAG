"""
建筑涂料 Excel 测试集独立评测：agentic + 纯 LLM 对照。

配置：benchmark/benchmark_agentic_2/config.yaml
运行：python -m benchmark.benchmark_agentic_2

run.mode: stats | evaluate | rejudge | report | excel | all
与 benchmark2 / benchmark_agentic_1 无 import 关系。
"""

from .config import DEFAULT_CONFIG_PATH, BenchmarkAgentic2Config
from .workflow import BenchmarkAgentic2Workflow

__all__ = [
    "BenchmarkAgentic2Workflow",
    "BenchmarkAgentic2Config",
    "DEFAULT_CONFIG_PATH",
]
