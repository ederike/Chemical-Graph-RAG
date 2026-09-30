# benchmark_agentic_5

在 `example/b` 的 200 篇原始 PDF 上，对本项目 `agentic_query` 与云端 WeKnora 问答做对照评测。

本包不读取、不引用 `example/a`，也不 import `benchmark_agentic_1/2/4`。配置、SQLite、FAISS 一律指向 `example/b`。

## 流程

1. **generate**：从 `example/b/DB/main.db` 的 `doc` 表抽样原始 PDF（排除 `name.pdf_N` 切片），按 `hop_counts` 出题。出题逻辑与 `benchmark_agentic_1` 相同。默认 1 跳 100 / 2 跳 50 / 3 跳 50；200 篇原始 PDF 不够 2/3 跳无重复时会跨题复用。
2. **evaluate**：每题两路作答——本项目 `DHMF.agentic_query`，以及 WeKnora `agent-chat` + 内置「快速问答」；然后用同一把尺子做 pairwise LLM 评判，并统计文档召回、延迟。
3. **report / excel**：写出 `report.json` 和可读性较好的 `report.xlsx`（目录、总体指标、系统对比、按跳数、文档召回、延迟、逐题明细、差异题、失败列表）。

## 运行

全部选项写在本目录 `config.yaml`：

```bash
python -m benchmark.benchmark_agentic_5
```

`run.mode`：

| 值 | 作用 |
|---|---|
| `generate` | 只出题（默认，避免误跑全量评测） |
| `evaluate` | 读问题集，两路作答 + 评判 |
| `report` | 读 evals，写 report JSON + Excel |
| `excel` | 只重出 Excel |
| `all` | generate → evaluate → report |

产物默认落在 `results/benchmark_agentic_5/`。中断后同一文件可续跑。

## 路径约定

- DHMF 配置：`example/b/config_open.yaml`（`working_path: example/b`）
- 数据库：`example/b/DB/main.db`
- 原始 PDF：`example/b/doc/`
- 检索窗口：`search_range` 必须为 `0`（全库）。生产 id 窗口会漏掉 b 库里 1 起步的 chunk/node。

加载配置时若路径里出现 `example/a`，会直接报错。评测加载 DHMF 时也会把 `search_range` 强制改成全库。

## WeKnora

默认对接：

- 地址 `http://192.168.110.219:8081`
- 知识库 `test`（`677e7f95-5530-4872-8412-55abbe459b7f`）
- `POST /api/v1/agent-chat/{session}`，`agent_id: builtin-quick-answer`

每题新建会话，答完删除。
