---
description: "构建和问答共用的底层：YAML 如何变成配置对象，五张业务表和 FAISS 分片如何读写，以及对话、嵌入、重排请求如何发给 OpenAI 兼容服务。"
kind: "package-reference"
---

# src/utils

中文

## Summary

这里没有“先识别再分块”这种业务流程。`module`、`agent`、`agentic` 和 `api` 通过这些文件读配置、落库、调模型。改一个函数的请求形状，会同时影响识别、抽取、检索和两条问答路径。业务规则放在调用方，这里只保证：配置字段有确定的类型，向量和状态要么一起落盘、要么都不算完成，模型请求在本地网关上不会因为多余字段直接 400。

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Source map](#source-map)
- [Known Limitations](#known-limitations)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>

## Use this package

加载配置：

```text
from src.utils.config import Config
config = Config.from_yaml("example/a/config_open.yaml")
```

阶段自己的 `base_url` 优先。没写时 `resolve_credentials(config, stage)` 回退到 `settings.base_url`，`api_key` 空则用 `"EMPTY"`。超时用 `resolve_llm_timeout(stage)`，读该段的 `timeout`，没有则 300 秒。

对话、嵌入、重排不要各自新写 HTTP。用这三个类，它们处理超时、缓存和本地网关的字段差异：

```text
from src.utils.OpenAIAPI import LLM, Embedding, Reranker

llm = LLM(api_key, base_url, timeout=300)
llm.generate(prompt={"system": "...", "user": "..."}, model_args=model_args)

emb = Embedding(api_key, base_url, timeout=120, max_retries=3)
emb.generate("乙醇", {"model": "embedding-medium", "dimension": 768})
emb.generate_batch(["丙酮", "硫酸"], {"model": "embedding-medium", "dimension": 768})

rr = Reranker(api_key, base_url, timeout=120)
rr.rerank("丙酮的沸点", ["……", "……"], model_args={"model": "reranker-xsmall"})
```

`dimension`、`dim`、`dims` 都会在进缓存和进请求之前改成 `dimensions`。调用方继续写 YAML 里的 `dimension` 即可。

进度：检索和两条问答在工作线程里调用 `progress.emit(stage, title, preview)`。HTTP 层用 `progress.bind` 接住这些事件并写成 SSE。没绑定时 `emit` 直接返回，命令行和评测不受影响。

-----

<a id="understand-the-implementation"></a>

## Understand the implementation

### 配置 `config.py`

YAML 的每一段对应一个 pydantic 模型：`settings`、`app_config`、`doc`、`summary`、`chunk`、`extract`、`build`、`vectorization`、`retrieve`、`agent`、`agentic`、`search_range`。布尔值接受 `true/false/yes/no/on/off/1/0`。空键变成空字符串，避免后面拼接 URL 时出现 `"None"`。

`vectorization` 在校验末尾做两件事：把嵌入参数里的维数别名收成 `dimensions`；若写了维数，把它抄到 `vectorization.dim`，FAISS 建索引时用这个整数。`shard_max_vectors` 大于 0 时，`flush_every`、`index_save_every`、`task_page_size` 被强制设成同一个数，这样“从数据库取一页”“写入一片”“封存一片”是同一批向量。

`search_range` 的 `chunk_max_vectors` 和 `node_max_vectors` 是检索和 pin 共用的 id 范围。不要在 `retrieve` 或 `agentic` 里再复制一份范围，否则页面问答和命令行会看到不同的库。

### 表和全文 `storage.py`、`database.py`

`storage.py` 定义 `DocDB`、`ChunkDB`、`HyperedgeDB`、`NodeDB`、`EdgeDB` 以及对应的 VDB。块插入时把正文 casefold 后写入 `chunk_fts`，分词器是 trigram。trigram 不索引短于 3 个字符的词，关键词检索对这类词改用 `instr`。

`database.py` 里 SQLite 和 FAISS 各用一把按文件路径分的锁，两者互不阻塞。FAISS 有两种形态：单个 `.vdb`，或 `*.shards/` 目录。分片模式下一片是一个 `IndexIDMap2`。HNSW 且 `index_quant=fp16` 时，内层是 `IndexHNSWSQ`。id 使用业务表的整数主键，检索返回的 id 能直接回到 SQLite。

`pin_shards(min_vectors, max_vectors)` 只把与这个 id 区间重叠的分片留在进程里。重复 pin 同一区间不会再次读盘。`add` 写入当前未封存的那一片；HNSW 不支持按 id 物理删除，删除用墓碑集合，重新插入同一 id 时清掉墓碑。

保存索引先写 `*.vdb.tmp`，再 `os.replace`。进程在写入中途退出时，旧文件还在。

### 模型客户端 `OpenAIAPI.py`

`LLM` 在内网地址或占位密钥（`EMPTY`、`none`、空字符串）上进入 `local_mode`。本地调用会去掉顶层的 `enable_thinking`、`thinking_budget` 等云厂商字段，否则 vLLM 会 400。思考开关改走 `chat_template_kwargs.enable_thinking`，并且只对两类模型注入：名字里含 `qwen3`，或网关 id 恰好是 `llm-medium`。没写这个开关时视为关闭。回答正文优先取 `message.content`；空的时候才用 `reasoning_content`。内容若以 `<think>...` 开头，会剥掉这段再返回。

`Embedding.generate` 走 `POST /v1/embeddings`。批量接口一次请求多条文本，缓存仍按单条文本加模型参数来读写，所以半批命中缓存时只请求未命中的那几条。返回顺序按响应里的 `index` 排回输入顺序。超时、连接错误和 502/503/504/429 会按 `max_retries` 重试。

`Reranker` 发 `POST {base}/rerank`（`base` 已含 `/v1`）。解析 `results[].relevance_score`，也接受 `score` 或 `relevance`。结果按分数降序。404 时再试不带 `/v1` 的路径。

磁盘缓存实现在 `utils.py` 的 `Cache`，库文件在 `cache/OpenAI/`。键是 prompt 与 `model_args` 的 JSON 的 MD5。`LLM.drop_generate_cache` 供抽取在 JSON 校验失败时删掉这一次的坏缓存。

### 其余文件

`prompt.py` 的 `PROMPT` 只放识别、总结、抽取这些构建提示词。问答提示词在 `agent/prompts.py` 和 `agentic/prompts.py`，避免改检索文案时碰到问答。

`metrics.py` 把本次构建的真实调用时间和 token 记在内存，结束时并入 `{working_path}/DB/build_metrics.json`。缓存命中不计入 token，也不计入“真实调用时间”。

`chat_history.py` 识别“它、该产品、继续”这类追问，把上一轮问题和答案缩进当前问句。它不读数据库。

`oss_download.py` 从配置的 MySQL 表查出对象键，再从 OSS 下载到 `working_path` 下的文档目录。它不调用视觉模型。

`paddleocr_vl.py` 仅在 `doc.recognition.use_paddleocr` 为真时使用。版面模型和 VL 请求的并发约束写在文件顶部的说明里：同一条 pipeline 不能并发 `predict`。

-----

<a id="source-map"></a>

## Source map

| 文件 | 对外符号 |
| --- | --- |
| [`config.py`](config.py) | `Config`、`resolve_credentials`、`resolve_llm_timeout` |
| [`storage.py`](storage.py) | `DocDB`、`ChunkDB`、`HyperedgeDB`、`NodeDB`、`EdgeDB` 及对应 VDB |
| [`database.py`](database.py) | `BaseDB`、`BaseVDB`、分片与 pin |
| [`OpenAIAPI.py`](OpenAIAPI.py) | `LLM`、`Embedding`、`Reranker`、`normalize_embedding_model_args` |
| [`prompt.py`](prompt.py) | `PROMPT` |
| [`utils.py`](utils.py) | `Retry`、`Cache`、`html_markdown_to_plain` |
| [`metrics.py`](metrics.py) | `PipelineMetrics` |
| [`progress.py`](progress.py) | `emit`、`bind` |
| [`chat_history.py`](chat_history.py) | `normalize_history`、`expand_retrieve_query` |
| [`oss_download.py`](oss_download.py) | `fetch_spider_products`、`download_rows_from_oss` |
| [`paddleocr_vl.py`](paddleocr_vl.py) | `make_pipeline`、`parse_document` |

-----

<a id="known-limitations"></a>

## Known Limitations

- 嵌入客户端相信服务端返回的维数。服务端忽略 `dimensions` 时，FAISS 的 `add_with_ids` 会在写入时失败，而不是悄悄截成 768。
- `llm-medium` 这个例外是按字符串写死的。网关若改了模型 id，思考开关会再次失效，回答可能只出现在 `reasoning_content`。
- FTS 的 trigram 对中文是按字符三元组，不是按化工词典。牌号和 CAS 靠精确匹配和 `instr`，不靠分词质量。
- `Cache` 没有容量上限。`cache/OpenAI/` 会随构建变大，清理是手工删库文件。

<a id="dev-note"></a>

## Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

- 给本地模型加新的非标准字段时，先放进 `_EXTRA_BODY_KEYS`。直接放在 `chat.completions.create` 的顶层，OpenAI SDK 或 vLLM 会拒掉整次请求。
- 改 FAISS 的 id 类型或分片文件名，要同时看 `vectorization.py` 的封存逻辑和 `retrieve.py` 的按区间打开。三处对“一片里最小 id、最大 id”的假设必须一致。
- 密码、OSS 密钥和数据库口令留在 YAML 或环境里，不要写进这个目录的默认值。

</details>
