---
description: "构建和问答共用的底层：YAML 配置、SQLite 与 FAISS、OpenAI 兼容的对话/嵌入/重排、提示词、重试、指标和进度事件。"
kind: "package-reference"
---

# src/utils

中文

## Summary

这里没有业务流程。`module`、`agent`、`agentic` 和 `api` 都从这里拿配置、存储和模型客户端。改存储表结构或客户端请求形状会影响每一条流水线，所以先看调用方 README 再改这里。配置加载会把嵌入的 `dimension` 规范成请求字段 `dimensions`，并同步 FAISS 的维数。

## Table of Contents

- [Source map](#source-map)
- [Further Exploration](#further-exploration)
- [Dev Note](#dev-note)

-----

<a id="source-map"></a>

## Source map

| 文件 | 职责 |
| --- | --- |
| [`config.py`](config.py) | YAML → pydantic。各阶段的地址、模型、重试、检索开关。`resolve_credentials` 在阶段未写地址时回退到 `settings` |
| [`storage.py`](storage.py) | `doc` / `chunk` / `hyperedge` / `node` / `edge` 五张表，以及块正文的 FTS5 trigram 索引 |
| [`database.py`](database.py) | SQLite 与 FAISS 的基类。分片、HNSW、fp16、pin、按 id 区间搜索 |
| [`OpenAIAPI.py`](OpenAIAPI.py) | `LLM`、`Embedding`、`Reranker`。本地模型的思考开关、嵌入批量与缓存、重排的 `/v1/rerank` |
| [`prompt.py`](prompt.py) | 识别、总结、抽取用的长提示词。问答提示词在 `agent` 和 `agentic` 自己的包里 |
| [`utils.py`](utils.py) | 重试、磁盘缓存、HTML/Markdown 转纯文本、进度条格式 |
| [`metrics.py`](metrics.py) | 本次构建与累计的耗时、token。累计写在 `{working_path}/DB/build_metrics.json` |
| [`progress.py`](progress.py) | 线程局部的进度回调。HTTP 流式接口绑定它；命令行没绑定时调用是空操作 |
| [`chat_history.py`](chat_history.py) | 把追问补成独立检索句。只影响查询文本，不改库 |
| [`oss_download.py`](oss_download.py) | 从 MySQL 查产品文件并下载 OSS。不识别 PDF |
| [`paddleocr_vl.py`](paddleocr_vl.py) | PaddleOCR-VL 客户端。仅当识别配置打开 OCR 时使用 |

块的 FTS 用 trigram。短于 3 个字符的词 trigram 盖不住，关键词检索改用 `instr`。

-----

<a id="further-exploration"></a>

## Further Exploration

- [`src/README.md`](../README.md) — 谁在什么顺序里用这些工具。
- [`src/module/README.md`](../module/README.md) — 表和向量是哪一步写入的。
- [`api/README.md`](../../api/README.md) — HTTP 进程如何加载同一份配置。

<a id="dev-note"></a>

## Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

- 本地对话会丢掉顶层 `enable_thinking`，只对模型名含 `qwen3` 或网关 id `llm-medium` 注入 `chat_template_kwargs.enable_thinking`。换一个不含这两个特征的名字时，YAML 里的开关不会生效。
- 嵌入客户端不截断、不再做归一化。768 维必须由服务端按请求的 `dimensions` 降维并保证 L2 范数为 1，否则和现有 FAISS 对不上。
- 重排请求发到 `{base}/rerank`（base 已含 `/v1`）。模型 id 用网关上的名字，不使用权重目录名。
- `Cache` 的键含模型参数。改 `model` 或维数后，旧缓存自然不会命中。

</details>
