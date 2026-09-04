# benchmark_agentic_2

独立评测模块：对「建筑涂料测试问题集1/2」跑 **agentic** 与 **纯 LLM** 两路，再成对评判、汇总、导出 Excel。

与 `benchmark2/`、`benchmark/benchmark_agentic_1/` **无代码 import 关系**；提示词与数据集解析逻辑为本包自有副本。

## 环境

仓库根目录，使用项目 conda 环境（示例）：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate test
cd /root/projects/Chemical-Graph-RAG
```

## 运行

```bash
# 按 config.yaml 的 run.mode 执行（默认 stats，避免误跑全量评测）
python -m benchmark.benchmark_agentic_2

# 等价
python -m benchmark.benchmark_agentic_2.run

# 仅构建统计
python -m benchmark.benchmark_agentic_2.build_stats
```

修改 `benchmark/benchmark_agentic_2/config.yaml`：

- `run.mode`: `stats` | `evaluate` | `rejudge` | `report` | `excel` | `all`
- `datasets[].enabled`: 控制跑 set1 / set2 / both
- `dataset.n_limit` / `id_list`: 抽样调试

## 输出

默认写到 `benchmark/benchmark_agentic_2/results/`，按数据集分文件：

- `set1_stats.json` / `set2_stats.json`
- `set1_dataset.json` / `set2_dataset.json`
- `set1_evals.json` / `set2_evals.json`
- `set1_report.json` / `set2_report.json`
- `set1_report.xlsx` / `set2_report.xlsx`
- `combined_report.json` / `combined_report.xlsx`（两集合并汇总，若两者都跑过）

## 数据

测试集 Excel 已放在本包 `data/` 下，config 只指向本包路径。
