---
description: "知识库构建与问答的 Python 包：DHMF 是唯一对外入口，构建阶段在 module/，两套问答在 agent/ 与 agentic/，模型、存储和配置在 utils/。"
kind: "package-group"
---

# src

中文

## Summary

`src` 是化工知识超图的库代码。调用方只应构造 `DHMF`，再走构建步骤或三条问答入口；不要绕过它直接拼模块。构建按固定顺序写 SQLite 和 FAISS。问答有三条互不替代的路径：单次检索生成、多跳规划、工具循环。`agent` 与 `agentic` 各自读自己的配置段，不回退到 `retrieve`。

## Table of Contents

- [Packages](#packages)
- [Pipeline](#pipeline)
- [Related documentation](#related-documentation)
- [Dev Note](#dev-note)

-----

<a id="packages"></a>

## Packages

| 路径 | 职责 |
| --- | --- |
| [`DHMF.py`](DHMF.py) | 门面。组装数据库、向量库和各阶段模块，暴露构建步骤与 `query` / `agent_query` / `agentic_query` |
| [`module/`](module/README.md) | 入库流水线：识别、总结、分块、抽取、构图、向量化、检索 |
| [`agent/`](agent/README.md) | 多跳规划问答。图为 plan → execute → synthesize |
| [`agentic/`](agentic/README.md) | 工具循环问答。模型自行决定检索、读文档和作答 |
| [`utils/`](utils/README.md) | 配置、SQLite、FAISS、LLM / 嵌入 / 重排客户端、提示词、指标 |

-----

<a id="pipeline"></a>

## Pipeline

构建顺序由 `main.py` 选择调用哪些步骤，顺序本身是：

`download_from_oss`（可选）→ `insert` / `insert_default` → `summary` → `chunk` → `extract` → `build` → `vectorization`

每步都有对应的 `*_clear`。`vectorization()` 默认只处理 `vectorization.default_target`（当前为 `chunk` 与 `node`）。检索与向量化不要同时把全部分片 pin 进内存。

问答入口：

| 方法 | 配置段 | 行为 |
| --- | --- | --- |
| `query` | `retrieve` | 三路召回后一次生成。默认 `mode='dual_path'` |
| `agent_query` | `agent` | 先规划子问题，再检索或直答，最后汇总 |
| `agentic_query` | `agentic` | 在同一条消息上循环调用工具，直到给出答案 |

-----

<a id="related-documentation"></a>

## Related documentation

- [`docs/API入门.md`](../docs/API入门.md) — HTTP 怎么调这三套问答。
- [`docs/agentic原理与实现详解.md`](../docs/agentic原理与实现详解.md) — 工具循环的轮次、停机和证据槽。
- [`example/a/config_open.yaml`](../example/a/config_open.yaml) — 开发机当前配置。字段名是 `model`，不是 `model-uid`。

<a id="dev-note"></a>

## Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

- 新增构建阶段时，在 `DHMF` 上加同名方法和 `*_clear`，并在 [`module/README.md`](module/README.md) 的顺序表里补一行。不要让阶段互相 import 问答包。
- `agent` 与 `agentic` 的 LLM、轮次和召回宽度只读各自配置。改一处不会自动改另一处。
- 向量维数来自 `vectorization.model_args` 的 `dimension`（加载后规范成 `dimensions`），并写入 FAISS。已有索引是 768 维；换嵌入模型前先确认服务端按请求降维且 L2 范数为 1。

</details>
