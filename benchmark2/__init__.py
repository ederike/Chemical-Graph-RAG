"""
Excel 测试集评测：统计 JSON + RAG 问答评判 + 分类汇总。

配置：benchmark2/config.yaml
运行：python benchmark2.py  或  python -m benchmark2.run

run.mode: stats | evaluate | rejudge | report | excel | all
"""

from .config import Benchmark2Config, DEFAULT_CONFIG_PATH
from .workflow import Benchmark2Workflow

__all__ = ["Benchmark2Workflow", "Benchmark2Config", "DEFAULT_CONFIG_PATH"]
