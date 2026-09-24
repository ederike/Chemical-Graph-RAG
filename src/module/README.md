---
description: "入库流水线的各个阶段：识别文档、总结、分块、抽实体、构图、向量化，以及问答用的三路检索。"
kind: "package-reference"
---

# src/module

中文

## Summary

每个文件是流水线里的一步，由 `DHMF` 按顺序调用。阶段之间用 SQLite 表交接，不在内存里传递整库。检索是唯一给问答用的阶段，它读已经建好的块、节点和 FAISS，不负责写入。

## Table of Contents

- [Use this package](#use-this-package)
- [Source map](#source-map)
- [Further Exploration](#further-exploration)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>

## Use this package

从项目根目录、用已加载的配置构造 `DHMF`，然后按需调用步骤。未完成的上游步骤不要跳过：分块依赖总结写入的超边，构图依赖块上的抽取结果，向量化依赖 `embedding_status='undone'` 的行。

```text
config = Config.from_yaml("example/a/config_open.yaml")
graph = DHMF(config)
graph.insert_default()
graph.summary()
graph.chunk()
graph.extract()
graph.build()
graph.vectorization()
```

问答侧只通过 `graph.query(...)` 进入检索。需要命中列表而不是答案时，HTTP 层调用 `graph.retrieve_module.retrieve_items`。

-----

<a id="source-map"></a>

## Source map

| 文件 | 阶段 | 读 | 写 |
| --- | --- | --- | --- |
| [`doc.py`](doc.py) | 识别 | `working_path` 下的 PDF / 图片 | `doc.content`。可选 PaddleOCR，默认可走视觉模型 |
| [`summary.py`](summary.py) | 总结 | `doc.content` | 每份文档一条 `hyperedge`，正文是 LLM 总结 |
| [`chunk.py`](chunk.py) | 分块 | 超边总结 + 识别全文 | 头块 `head`，正文按 token 切成 `body_n`，相邻块可重叠 |
| [`extract.py`](extract.py) | 抽取 | 块文本 | 块上的实体 JSON。要求模型返回 JSON 对象 |
| [`build.py`](build.py) | 构图 | 已有超边 + 块上的实体 | `node`。节点记下所属块和超边。不新建超边，也不写 `edge` |
| [`vectorization.py`](vectorization.py) | 向量化 | `embedding_status='undone'` 的行 | FAISS 分片，成功落盘后才把状态改成 `done` |
| [`retrieve.py`](retrieve.py) | 检索 | 查询文本、FAISS、FTS5 | 不写库。块向量、节点向量、关键词三路合并后再重排 |

向量化按页取未完成行，满 `shard_max_vectors` 封存一片并卸掉内存。检索范围由 `search_range` 的 id 区间决定，pin 也只加载与该区间重叠的分片。

-----

<a id="further-exploration"></a>

## Further Exploration

- [`src/README.md`](../README.md) — 门面和三条问答入口。
- [`src/utils/README.md`](../utils/README.md) — 这些阶段共用的配置、客户端和存储。
- [`docs/WeKnora与本项目知识库构建与检索对比.md`](../../docs/WeKnora与本项目知识库构建与检索对比.md) — 构建和检索与另一套系统的差别。

<a id="dev-note"></a>

## Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

- `build` 明确不写 `edge`。邻居关系在检索里按同一超边或同一文档共现，不是类型化的边。
- 改索引类型或量化（例如 L2 与 HNSW、fp16）后，必须先 `vectorization_clear` 再重跑。只改 YAML 不会重嵌已经标成 `done` 的行。
- 向量化进行中不要 pin。pin 把分片整份读进内存，和按片写入抢同一批文件。

</details>
