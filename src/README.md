---
description: "化工资料知识库的 Python 实现。DHMF 是唯一对外入口：按固定顺序把 PDF 写成 SQLite 和 FAISS，再用三条互不替代的路径回答问题。"
kind: "package-group"
---

# src

中文

## Summary

`src` 把一份化工技术资料（TDS、SDS/MSDS、说明书、专利）变成可检索的库，并在这套库上回答问题。调用方只构造 `DHMF`。构建步骤把原文、总结、文本块和实体分别写入 SQLite，把块和实体的向量写入 FAISS。问答不重新建库，只读已经写好的表和分片。

三条问答路径解决的问题不同，不能互相代替：

- `query`：一次检索，然后生成。适合问句已经指向某份资料里的一段事实。
- `agent_query`：先让模型把问题拆成若干子问题，再逐个检索，最后对照汇总。适合一句话里套了两三个必须分别查证的条件。
- `agentic_query`：模型自己决定下一轮是再搜、读原文、看同文档实体，还是作答。适合第一次检索不够、需要换词或核对数值的问题。

`agent` 和 `agentic` 各自读自己的配置段，不回退到 `retrieve` 的模型地址。向量索引和嵌入、重排服务是三套问答共用的。

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Packages](#packages)
- [Known Limitations](#known-limitations)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>

## Use this package

在仓库根目录、用一份 YAML 构造对象。开发机配置是 `example/a/config_open.yaml`。`settings.working_path` 决定 SQLite、FAISS、原文和构建日志放在哪，当前是 `example/a`。

```text
from src.DHMF import DHMF
from src.utils.config import Config

config = Config.from_yaml("example/a/config_open.yaml")
graph = DHMF(config)
```

构建按这个顺序调用。上游没跑完，下游会空转或跳过：总结读不到 `doc.content`，分块读不到超边，构图读不到超边会把整份文档跳过，向量化只处理 `embedding_status='undone'` 的行。

```text
graph.download_from_oss()   # 可选。只下载，不识别
graph.insert_default()      # 识别 working_path/doc 下的 PDF 和图片
graph.summary()             # 每份文档一条超边
graph.chunk()               # 头块 + 正文块
graph.extract()             # 每块一份实体 JSON
graph.build()               # 实体变成 node
graph.vectorization()       # 默认只嵌 chunk 和 node
```

`main.py` 里目前打开的是 `extract`、`build`、`vectorization`。前面几步保持注释，是因为这份库的识别、总结和分块已经跑过，重跑会按各步自己的“已完成则跳过”规则补缺，而不是清空重来。要整段重做，先调对应的 `*_clear`。

问答：

```text
graph.query("丙酮的沸点是多少？", mode="dual_path", pretty=False)
graph.agent_query("……", pretty=False, history=None)
graph.agentic_query("……", pretty=False, history=None)
```

`pretty=True` 返回可直接打印的字符串。`pretty=False` 返回字典，里面有 `answer`、`status`、来源和 token。`history` 是同一会话里已经问过的轮次，用来把“它的闪点呢”补成带产品名的句子，不改索引。

上线检索前可以 `graph.pin_retrieve_indexes()`，把 `search_range` 覆盖到的分片读进本进程。构建或 `vectorization()` 期间不要 pin，两边会抢同一批分片文件。用完 `unpin_retrieve_indexes()`。

-----

<a id="understand-the-implementation"></a>

## Understand the implementation

### 库里实际有什么

`DHMF.__init__` 打开 `{working_path}/DB/main.db` 和 `{working_path}/DB/vdb/`，并按 `vectorization` 里的维数、HNSW 参数创建五套存储。当前配置维数是 768，索引是 HNSW、fp16 量化，每片最多 10 万条向量。

| 表 | 谁写入 | 一行是什么 |
| --- | --- | --- |
| `doc` | `insert` | 一份资料（或长 PDF 的一段）的识别正文 |
| `hyperedge` | `summary`，`build` 只回写 extra | 这份资料的 LLM 总结。一文档一条 |
| `chunk` | `chunk` | 头块（总结）或正文块。正文同时进 FTS5 |
| `node` | `build` | 一个实体。向量文本是「名称 + 换行 + 描述」 |
| `edge` | 当前构图不写 | 表还在，检索不用它 |

向量分片在 `vdb/chunk.shards` 和 `vdb/node.shards`。分片元数据里的 `dim` 必须是 768。检索和 pin 看的不是“第几片”，而是向量 id 落在 `search_range` 的哪一段。

### 三条问答为什么分开

`query` 的检索参数、改写和生成模型都在 `retrieve`。`agent_query` 的规划和汇总模型在 `agent`，它内部的检索仍走已经建好的 `Retrieve`，但生成和可选改写用 `agent` 的模型，避免和检索配置绑死。`agentic_query` 的循环、工具宽度、上下文上限全在 `agentic`；工具里的 `search` 会临时覆盖召回个数，不改 `retrieve` 这份共享配置。

HTTP 服务和命令行用的是同一个 `DHMF`。差别只在 `api` 把调用放进线程池，并用 `progress.emit` 把步骤推到页面。

-----

<a id="packages"></a>

## Packages

| 路径 | 何时去读 |
| --- | --- |
| [`DHMF.py`](DHMF.py) | 要加一个对外步骤、清空某张表，或改问答返回字段 |
| [`module/`](module/README.md) | 要改识别、分块、抽取、构图、向量化或三路检索 |
| [`agent/`](agent/README.md) | 要改“先规划再执行”这条路径 |
| [`agentic/`](agentic/README.md) | 要改工具循环、停机条件或四个工具 |
| [`utils/`](utils/README.md) | 要改配置字段、表结构、FAISS 或模型请求 |

-----

<a id="known-limitations"></a>

## Known Limitations

- 构图不产生类型化的边。`graph_neighbors` 找的是同一超边或同一文档上的其他实体。
- 向量化默认不处理 `doc`、`hyperedge`、`edge`。检索用的是块向量和节点向量。
- 已标成 `embedding_status='done'` 的行，再跑 `vectorization()` 不会重嵌。换模型或换维数要先 `vectorization_clear`。
- 本进程把分片整份读进内存。开发机内存小于全库时，不要把 `search_range` 放成全库再 pin。

<a id="dev-note"></a>

## Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

- 新的构建阶段要同时提供方法和 `*_clear`。`clear` 必须把状态改回下一步能重新选中的值，而不是只删向量、留下 `done`。
- 不要在 `module` 里 import `agent` 或 `agentic`。问答依赖构建结果，构建不依赖问答。
- 生产机配置不随开发机 YAML 一起改。这里的说明只描述仓库里的代码和 `example/a` 这份开发配置。

</details>
