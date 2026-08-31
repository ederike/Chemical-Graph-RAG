"""
建筑涂料专利集 CSV 评测：超图问答 + 按得分维度评判 + Excel 汇总。

配置：benchmark3/config.yaml
运行：python benchmark3.py  或  python -m benchmark3.run

run.mode: evaluate | rejudge | report | excel | all
"""

from .config import DEFAULT_CONFIG_PATH, Benchmark3Config
from .workflow import Benchmark3Workflow

__all__ = ["Benchmark3Workflow", "Benchmark3Config", "DEFAULT_CONFIG_PATH"]
