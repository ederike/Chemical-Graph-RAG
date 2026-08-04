"""
多跳问答测试集生成 + DHMF RAG 评测。

配置：benchmark/config.yaml
运行：python benchmark.py  或  python -m benchmark.run
"""

from .config import BenchmarkConfig, DEFAULT_CONFIG_PATH
from .workflow import TestQueryWorkflow

__all__ = ["TestQueryWorkflow", "BenchmarkConfig", "DEFAULT_CONFIG_PATH"]
