---
description: "把资料写成可检索的库，并执行一次查询的七个阶段。每个文件只做一步，阶段之间用 SQLite 的状态字段交接。"
kind: "package-reference"
---

# src/module

中文

## Summary

这个目录是流水线本体。`DHMF` 按顺序调用这里的类，自己不实现识别或检索。每一步都遵守同一件事：只挑选“还没做完”的行，做完就改状态并分批落库，避免把整库载入内存。检索是唯一不写业务表的一步，它读已经建好的块、节点、FTS 和 FAISS。

阶段顺序不能颠倒：

`insert`（识别）→ `summary` → `chunk` → `extract` → `build` → `vectorization`

-----

<a id="use-this-package"></a>

## Use this package

不要直接 new 这些类来跑全库。用 `DHMF` 上的同名方法，它会先打开构建日志和指标。

```text
graph.insert_default()
graph.summary()
graph.chunk()
graph.extract()
graph.build()
graph.vectorization()
```

某一步要重做时，先清再跑。清掉的是这一层的产物，不会自动重跑上游。

| 要重做 | 先调用 | 实际清掉什么 |
| --- | --- | --- |
| 识别 | `insert_clear()` | `doc`，以及 summary 写下的 `hyperedge`，还有 doc 向量 |
| 总结 | `summary_clear()` | 超边正文；`doc` 的总结状态回到可再总结。不删块和节点 |
| 分块 | `chunk_clear()` | 块。不自动重做总结 |
| 抽取 | `extract_clear()` | 块上的抽取结果 |
| 构图 | `build_clear()` | 节点，以及 node/edge 向量。不删总结正文 |
| 向量 | `vectorization_clear("chunk")` 或 `"node"` | 该表状态回到 `undone`，并删掉对应 FAISS |

问答侧检索的入口是 `Retrieve.retrieve_items`。`DHMF.query` 和 Agent 的 `QuerySkill` 都走它。Agentic 的 `search` 也走它，但用调用参数盖住这一次的 top-k，不改 YAML。

```text
items = graph.retrieve_module.retrieve_items("丙酮 CAS 67-64-1 沸点")
```

返回的是块级命中列表。每个元素带内容、来源、分数和匹配类型。`get_last_timing()` 给出这一次各阶段秒数：`rewrite`、`embed`、`chunk`、`node`、`keyword`、`expand`、`rerank`、`total`。并行打开时 `total` 是墙钟，不等于各路相加。

-----

<a id="understand-the-implementation"></a>

## Understand the implementation

### 识别 `doc.py`

`insert_pdfs` 扫描 `working_path/doc`。PDF 页数超过 `recognition.max_pages_per_doc` 时切成多段独立文档：第一段保持原文件名，后面是 `原文件名_1`、`原文件名_2`。图片不切片。`skip_existing=True` 时按切片后的文档名跳过，所以长 PDF 只入库了第一段时，后面的段还会补。

每一段把页面渲成图片，连同 `prompt.py` 里的 `pdf_recognize` 提示词送给视觉模型。提示词要求中文叙述，CAS、化学式、型号、单位和数值范围保持原样。`enable_thinking` 在这一步被强制关掉，并且去掉 `response_format`，因为识别要的是纯文本不是 JSON。失败按 `recognition.retry` 重试。`use_paddleocr` 为真时改走 `utils/paddleocr_vl.py`，默认可关闭。

识别正文写入 `doc.content`。同一批按 `doc.flush_every` 个源文件入库一次，避免全文堆在内存里。

### 总结 `summary.py`

读还没总结的 `doc.content`，让 `summary` 段的模型写成一段中文摘要，写入 `hyperedge.content`。一份文档一条超边。长 PDF 被切成多段时，后段可以用 `enable_prior_context` 把同一源文件最近几段识别正文放进提示词，`prior_context_slices` 控制带几段；提示词要求结合前文，但重点写本段。关掉时每段只看自己，避免长文把上下文撑满。

### 分块 `chunk.py`

分块发生在总结之后。每个文档产生两类块：

- 一块名为 `head` 的头块，内容是超边总结，不再切分。
- 若干 `body_1`、`body_2`……，来自 `doc.content`。

切分用 `cl100k_base` 计数。先按句号、问号、叹号和换行切开，不在逗号或分号处切断。再把句子装进不超过 `chunk_size_max`（当前 512）的块。相邻块保留约 `chunk_overlap` 比例的尾部；比例大于 1 时当成百分数，上限 0.9。英文单词之间补空格，避免两句粘成一个词。块正文另外写入 FTS5，供关键词检索。

### 抽取 `extract.py`

对抽取状态未完成的块调用 `extract` 段的模型。`model_args` 里带 `response_format: json_object`，解析失败会丢掉这次缓存再试，避免坏 JSON 被永久命中。结果写回块的抽取字段。并发数是 `extract.num_thread`，满 `flush_every` 条就 `UPDATE` 一次。

### 构图 `build.py`

按文档把已经抽取完的块收在一起。找不到该文档的超边就记错误并跳过，所以构图之前必须先总结。

若 `build.target` 含 `hyperedge`，把头块上的实体合并进这条超边的 `extra`，并记下头块 id。若含 `node`，每个实体插一行节点：`name` 是实体名，`content` 是描述，`embedding_content` 是「名称、换行、描述」，供后面的向量化使用。节点记下 `doc_id`、`chunk_id`。这一步不插入 `edge`。处理过的块状态改成 `build`。

### 向量化 `vectorization.py`

默认目标是 `chunk` 和 `node`，由 `vectorization.default_target` 决定。`prepare` 只统计 `embedding_status='undone'` 的行数，不把全表读进内存。`processing` 按 `shard_max_vectors` 做 keyset 分页。每页再切成 `batch_size` 条一批去嵌，批与批由 `num_thread` 个线程并发。

单条文本用 `Embedding.generate`，多条用 `generate_batch`。缓存键包含模型名和 `dimensions`，所以换模型不会误用旧向量。请求里的维数来自 YAML 的 `dimension: 768`，加载配置时改名为 `dimensions`。客户端不截断、不再归一化，768 和单位范数必须由嵌入服务完成。

向量先进入内存中的 FAISS。SQLite 状态改成 `done` 发生在这一批已经写到磁盘之后。进程在这两步之间死掉时，这些行仍是 `undone`，下次会重试，缓存能命中。一片写满 `shard_max_vectors` 就封存并卸载，内存里只留正在写的一片。现有分片的量化如果和 YAML 不一致，新向量仍跟磁盘上的编码走，日志会警告；要换成另一种量化，必须 `vectorization_clear` 后重建。

### 检索 `retrieve.py`

`retrieve_items` 的顺序：

1. 可选。`enable_query_rewrite` 用 `rewrite_model_args` 把问句写得更具体。开发配置里这项是关的。
2. 可选。只给查询向量加 `query_instruct` 前缀，文档侧不加。当前是开的。
3. 查询向量对块向量取 `chunk_candidate_k` 条。0 表示跳过这一路。
4. 查询向量对节点向量取 `node_candidate_k` 条，再映回这些节点所属的块。
5. 可选关键词。模型从问句里抽出少数词（CAS、牌号、货号）和多数词（字段名）。只开少数词时用专用提示词。FTS5 用 trigram；短于 3 个字符的词改 `instr`。命中块数超过 `keyword_max_df` 的词当停用词丢掉。关键词路内部可以先做一次文档级重排，再截到 `keyword_top_k`。这一路只加候选，不删向量命中。
6. 三路按块取并集。同一块保留分数较高的那次。
7. 扩展正文。`enable_slice_family_expand` 优先：命中任一切片就把同源 PDF 的全部分段按原序补进来。它关掉时才看 `enable_full_body_context`：`false` 是头块加命中的正文块，`simple` 只在命中正文时补头块，`true` 写入该文档全部正文块。
8. 终轮重排，截到 `rerank_top_k`。0 表示不截断。

`enable_parallel_paths` 为真时，关键词路和向量路同时跑，块与节点在嵌入完成后也并行。为假时按块、节点、关键词串行，峰值内存更低。

`search_range.chunk_max_vectors` 和 `node_max_vectors` 限制参与检索的 id。`0` 是全库，一个整数 `N` 是 `0..N`，`[a, b]` 是闭区间。分片按 id 区间打开有重叠的那些片。关键词 FTS 使用同一段块 id。`retrieve_items` 的同名参数只覆盖这一次调用。

-----

<a id="source-map"></a>

## Source map

| 文件 | 类 | 配置段 |
| --- | --- | --- |
| [`doc.py`](doc.py) | `Doc` | `doc`、`doc.recognition` |
| [`summary.py`](summary.py) | `Summary` | `summary` |
| [`chunk.py`](chunk.py) | `Chunk` | `chunk` |
| [`extract.py`](extract.py) | `Extract` | `extract` |
| [`build.py`](build.py) | `Build` | `build` |
| [`vectorization.py`](vectorization.py) | `Vectorization` | `vectorization` |
| [`retrieve.py`](retrieve.py) | `Retrieve` | `retrieve`，范围在 `search_range` |

-----

<a id="known-limitations"></a>

## Known Limitations

- 识别按页渲染后整段送给视觉模型。`max_pages_per_doc` 设得太大时，单次请求的图片数会超过模型或网关能接受的长度。
- 抽取依赖模型稳定输出 JSON。坏结果会丢缓存重试，但重试耗尽后该块保持未完成。
- 关键词路的质量取决于抽词模型。抽不出 CAS 或牌号时，这一路就是空的，向量路仍然保留。
- 终轮重排看的是扩展之后的文档文本。`rerank_max_chars` 为 `-1` 时不按字符截断，长文档会拉高重排耗时。

<a id="dev-note"></a>

## Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

- 改块的切分规则后，已有块不会自动重切。需要 `chunk_clear`，然后重新抽取、构图、向量化，否则节点仍指向旧块。
- 向量化的 `save()` 会强制把缓冲区写入磁盘。不要在 `_append_embedding` 里提前把状态改成 `done`。
- 检索的临时覆盖通过 `_temporary_scope` 包住一次调用。新增的 top-k 参数也要走这条路径，避免并发请求改到共享配置。

</details>
