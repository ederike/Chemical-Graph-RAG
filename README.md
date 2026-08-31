# Chemical-Graph-RAG

化工超图 RAG 知识库（仓库名：`ky_knowledge`）。面向涂料、助剂等化工产品的 TDS / MSDS / 专利等 PDF，把文档识别、摘要、实体抽取做成可检索的超图，再用向量 + 关键词 + Agent 回答产品参数、牌号、配方和跨文档问题。

典型问题例如：某牌号的 NFPA 评级、相对密度、自燃温度；某公司除该产品外还有哪些抗氧化剂。系统先从文档中召回证据，再由大模型作答，并可与「不检索的纯 LLM」对照。

---

## 运行条件

运行该项目必须具备下列环境和依赖。构建阶段会大量调用视觉/语言模型，检索阶段还需要 Embedding 与 Reranker。

### 条件一：Python 环境与 Python 包

- Python 3.10+（开发环境为 3.13）
- 建议独立虚拟环境
- 安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖：

| 用途                    | 包                                                                             |
| ----------------------- | ------------------------------------------------------------------------------ |
| 配置与数据              | `pydantic`、`pyyaml`、`python-dotenv`、`numpy`、`tqdm`、`openpyxl` |
| LLM 调用（OpenAI 兼容） | `openai`、`tiktoken`、`httpx`                                            |
| 多跳 Agent              | `langgraph`                                                                  |
| 向量检索                | `faiss-cpu`                                                                  |
| PDF 渲染                | `pymupdf`                                                                    |
| 可选：对象存储与业务库  | `oss2`、`pymysql`、`cryptography`                                        |
| 可选：超图可视化        | `matplotlib`、`networkx`                                                   |

不需要 GPU。磁盘需能放下工作目录：SQLite 主库、FAISS 分片、LLM/嵌入缓存、原始 PDF。万级文档规模下库体可达数 GB。

### 条件二：模型服务

配置文件里的接口需为 **OpenAI 兼容** 的 `/v1` 服务（`api_key` 可填 `EMPTY`，视网关而定）。至少要能访问：

1. **LLM / 视觉模型**：PDF 按页转写、文档摘要、实体抽取、问答与 Agent 规划。当前示例使用 `qwen3.6-27b`。
2. **Embedding**：chunk / node 向量化与查询向量。当前示例使用 `Qwen3-Embedding-8B`（768 维）。
3. **Reranker**：检索终排。当前示例使用 `Qwen3-Reranker-0.6B`。

上述地址写在 YAML 的 `settings.base_url`、`vectorization.base_url`、`retrieve.rerank_base_url` 等字段中，按实际部署修改。

### 条件三：工作目录、配置文件与数据源

- 项目根目录下需要一份流水线配置 YAML。入口 `main.py` 默认读取 `example/a/config_open.yaml`（`example/` 不入库，需本地创建）。
- 待入库文件放在 `{working_path}/{doc.doc_dir}`，默认 `example/a/doc/`，支持 PDF 以及 jpg/png；也可设 `doc.source_type: txt`。
- 各阶段以 SQLite 行级 `status` 断点续跑，产物在 `{working_path}/DB/`（`main.db` + `vdb/` 分片）。
- **可选**：从业务 MySQL + 阿里云 OSS 拉产品附件时，配置 `dm_data_mysql`、`ali_oss`、`oss_download`。只做本地 PDF 时不需要。

---

## 运行说明

整体顺序：**配配置 → 放文档（或 OSS 下载）→ 构建索引 → 查询**。构建流水线可重复执行，已完成的阶段会跳过。

### 操作一：准备配置与文档

1. 克隆仓库，安装依赖（见条件一）。
2. 创建工作目录和配置，例如：

```bash
mkdir -p example/a/doc
```

3. 编写 `example/a/config_open.yaml`（密钥、内网地址请按环境填写，不要提交真实口令）：

```yaml
settings:
  working_path: 'example/a'
  debug: false
  api_key: 'EMPTY'
  base_url: 'http://<LLM网关>/v1'

doc:
  source_type: 'pdf'   # pdf | txt
  doc_dir: 'doc'
  recognition:
    model_args:
      model: 'qwen3.6-27b'
      temperature: 0.2
      enable_thinking: false
    dpi: 100
    max_pages_per_doc: 4
    num_thread: 8

summary:
  model_args: { model: 'qwen3.6-27b', temperature: 0.2, enable_thinking: false }
  num_thread: 16

chunk:
  chunk_size_max: 512
  chunk_overlap: 0.1

extract:
  extract_prompt: 'extract_product'
  model_args:
    model: 'qwen3.6-27b'
    temperature: 0.2
    enable_thinking: false
    response_format: { type: 'json_object' }
  num_thread: 16

build:
  target: ['hyperedge', 'node']

vectorization:
  base_url: 'http://<Embedding网关>/v1'
  model_args: { model: 'Qwen3-Embedding-8B', dimension: 768 }
  default_target: ['chunk', 'node']
  shard_max_vectors: 100000
  index_type: hnsw
  index_quant: fp16

retrieve:
  chunk_candidate_k: 30
  node_candidate_k: 30
  enable_keyword_exact: true
  enable_keyword_minority: true   # 少数词（牌号/CAS/货号），默认开
  enable_keyword_majority: false  # 多数词（字段名），默认关
  enable_rerank: true
  rerank_top_k: 10
  embedding_base_url: 'http://<Embedding网关>/v1'
  rerank_base_url: 'http://<Rerank网关>/v1'
  rerank_model_args: { model: 'Qwen3-Reranker-0.6B' }

agent:
  max_steps: 12
  enable_direct_retrieve: false  # 多跳时用原问题再检索作答一次给 synthesize 参考；单跳跳过。默认关
```

4. 把 PDF 放到 `example/a/doc/`。若走 OSS，在 YAML 中配置 `oss_download` / `dm_data_mysql` / `ali_oss`，并在 `main.py` 中打开 `graph.download_from_oss()`（以及按需的 `graph.dedupe_downloaded_docs()`）。

### 操作二：构建知识库

`main.py` 按固定顺序跑完整条构建链：

```text
insert_default → summary → chunk → extract → build → vectorization
```

```bash
python main.py
```

各步含义：

| 步骤               | 作用                                                                                      |
| ------------------ | ----------------------------------------------------------------------------------------- |
| `insert_default` | PDF 按页渲染，VLM 转写为文本，写入`doc`；超长 PDF 按 `max_pages_per_doc` 切成多条文档 |
| `summary`        | 对每篇文档生成摘要，写入该文档唯一超边`hyperedge`                                       |
| `chunk`          | 摘要作为 head 块，正文按 token 切 body 块（默认 512，重叠 10%）                           |
| `extract`        | 从块中抽取「实体名 → 描述」                                                              |
| `build`          | 实体物化为`node`（当前不建 pairwise `edge`）                                          |
| `vectorization`  | 对 chunk、node 建分片 HNSW 索引                                                           |

中断后直接再跑 `python main.py` 即可续跑。日志在 `{working_path}/log/`。构建耗时、token 累计写入 `DB/build_metrics.json`。

也可在 Python 中分步调用：

```python
from src.DHMF import DHMF
from src.utils.config import Config

graph = DHMF(Config.from_yaml('example/a/config_open.yaml'))
graph.insert_default()
graph.summary()
graph.chunk()
graph.extract()
graph.build()
graph.vectorization()
```

向量分片过大时，可用 `python scripts/repack_vdb.py --config example/a/config_open.yaml --shard-max 500000` 在不重新嵌入的情况下合并分片。

### 操作三：问答查询

库构建完成后，用同一配置加载 `DHMF` 查询。

**单跳检索问答**（三路召回 + 可选 rerank，一次生成）：

```python
print(graph.query('该产品的相对密度是多少？', mode='dual_path', pretty=True))
```

**多跳 Agent**（规划检索步，单跳可一次检索作答，多跳按中间结论继续检索，最多 `agent.max_steps` 步；`agent.enable_direct_retrieve: true` 时多跳会再用原问题检索并作答一次给最终汇总参考，单跳自动跳过）：

```python
print(graph.agent_query(
    '满足 NFPA 评级为健康1、相对密度约1.10、自燃温度高于400°C 的产品，其生产公司还有什么抗氧化剂产品吗？',
    pretty=True,
))
```

检索默认行为：chunk 向量与 node 向量各召回 30 条；LLM 抽取牌号/CAS 等少数值做 SQLite FTS5 精确匹配（多数词默认关闭）；三路按 chunk 并集后用 Reranker 保留 top-10 文档。可通过 `retrieve.enable_full_body_context`、`enable_keyword_exact`、`enable_keyword_minority`、`enable_keyword_majority`、`enable_rerank` 等开关调整。

超图可视化（可选）：

```bash
python scripts/draw_hypergraph.py --db example/a/DB/main.db --max-hyperedges 12
```

---

## 测试说明

仓库提供两套评测，都依赖已构建好的 DHMF 库，并用 LLM 做裁判；每题同时跑「超图回答」和「纯 LLM（不检索）」，一次对比打分。

### 1. 库内自生成评测（`benchmark`）

从库中抽样生成 1-hop / 2-hop / 3-hop 问题，统计正确率、gold 文档召回，并导出 Excel。

```bash
# 编辑 benchmark/config.yaml：run.mode、dhmf.config_path、paths.output_dir 等
python benchmark.py
```

`run.mode` 可选：`generate` | `evaluate` | `report` | `excel` | `all`。
`dhmf.query_mode` 可选：`agent`（默认）或 `dual_path`。

当前示例配置会生成 200 题（1/2/3-hop = 100/50/50），输出目录由 `paths.output_dir` 指定。

### 2. 外部 Excel 测试集（`benchmark2`）

读取建筑涂料外部题集，按身份、难度、能力维度、跳数等字段出分（0–3 分：正确 / 部分正确 / 错误）。

```bash
# 编辑 benchmark2/config.yaml：dataset.excel_path、paths.output_dir、run.mode
python benchmark2.py
```

仓库已带两份题集：

- `benchmark2/建筑涂料测试问题集1.xlsx`
- `benchmark2/建筑涂料测试问题集2.xlsx`

`run.mode` 可选：`stats` | `evaluate` | `rejudge` | `report` | `excel` | `all`。可用 `n_limit`、`id_list` 先跑子集；`evaluate.resume: true` 支持断点续跑。

评测会真实打到 LLM / Embedding / Reranker，耗时长、费用高，请先确认服务可用，并单独指定 `output_dir`，避免覆盖已有结果。

---

## 技术架构

核心类是 `src/DHMF.py` 中的 `DHMF`：组装文档、摘要、分块、抽取、构图、向量化、检索模块，以及 LangGraph Agent。

```text
                    ┌──────────────────────────────────────────┐
                    │              数据接入                       │
                    │  本地 PDF/图片/txt  或  MySQL + OSS 下载    │
                    └──────────────────┬───────────────────────┘
                                       ▼
 PDF 页渲染 (PyMuPDF) ── VLM 转写 ──► doc（SQLite）
                                       │
                                       ▼
                              summary ──► hyperedge（一篇一超边）
                                       │
                                       ▼
                         chunk ──► head + body_n（token 切分）
                                       │
                                       ▼
                         extract ──► 实体名/描述（JSON）
                                       │
                                       ▼
                         build ──► node（实体挂到文档/块）
                                       │
                                       ▼
              vectorization ──► FAISS 分片 HNSW（chunk / node，可选 fp16）
                              SQLite FTS5（chunk 小写正文，牌号/CAS）
                                       │
           ┌───────────────────────────┴───────────────────────────┐
           │                     在线检索                           │
           │  查询向量 + instruct 前缀                               │
           │  ① chunk ANN  ② node ANN→所属 chunk  ③ FTS 关键词路     │
           │  三路并集 → 全文扩展 → Qwen3-Reranker → 证据文档          │
           └───────────────────────────┬───────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
             query(dual_path)                      agent_query
             一次检索 + 一次作答            LangGraph: plan → execute ⟲ → synthesize
                                                    单跳一次完成，多跳按依赖展开
```

**存储**

| 层       | 实现                     | 内容                                                                        |
| -------- | ------------------------ | --------------------------------------------------------------------------- |
| 结构化库 | SQLite`DB/main.db`     | `doc` / `chunk` / `hyperedge` / `node` / `edge`，行级 status 断点 |
| 向量库   | FAISS`DB/vdb/*.shards` | 默认 HNSW + fp16，按`shard_max_vectors` 分片；删除打墓碑                  |
| 全文     | SQLite FTS5 trigram      | 关键词精确匹配（少数词 / 多数词可独立开关；少数词默认开，多数词默认关）     |
| 缓存     | `cache/OpenAI/*.db`    | LLM、Embedding、PDF 识别结果                                                |

**检索与问答**

- **dual_path**：chunk 向量 + node 向量（+ 可选关键词）合并后生成答案。
- **agent**：`src/agent/`，LangGraph 编排规划、检索 skill、纯 LLM 步与最终综合；单跳不强制多轮。
- 关键词路可拆成少数词 / 多数词两路召回：`enable_keyword_minority`（默认开）只用少数词专用提示词抽取牌号、CAS、货号并只检索这些词；`enable_keyword_majority`（默认关）抽取字段名。总开关 `enable_keyword_exact` 为 false 时两路都关。匹配走预计算小写正文 FTS5 `MATCH`，避免全表 Python 扫描。

**目录结构（源码）**

```text
main.py                 构建入口
src/DHMF.py             流水线门面
src/module/             insert / summary / chunk / extract / build / vectorization / retrieve
src/agent/              多跳 Agent（graph / nodes / skill / runner）
src/utils/              配置、SQLite/FAISS、LLM 客户端、OSS 下载、prompt
benchmark/              库内自生成评测
benchmark2/             外部 Excel 评测
scripts/                超图可视化、向量分片重打包
```
