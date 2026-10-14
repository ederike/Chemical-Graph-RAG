# Chemical-Graph-RAG

## 文件在这台电脑的哪个位置

项目在 WSL2 里，发行版名 `Ubuntu-24.04`（Ubuntu 24.04），默认用户是 `root`。生产机是另一台，目录写在文末「生产机」。

从 Windows 进这个发行版：打开 PowerShell、命令提示符或 Windows 终端，执行

```bash
wsl
```

默认发行版就是 `Ubuntu-24.04`，也可以写明：

```bash
wsl -d Ubuntu-24.04
```

进来后提示符是 `root`，家目录 `/root`。再进入仓库：

```bash
cd /root/projects/Chemical-Graph-RAG
```

| 从哪里看                 | 路径                                                              |
| ------------------------ | ----------------------------------------------------------------- |
| WSL 里                   | `/root/projects/Chemical-Graph-RAG`                             |
| Windows 资源管理器地址栏 | `\\wsl.localhost\Ubuntu-24.04\root\projects\Chemical-Graph-RAG` |
| 发行版的虚拟磁盘文件     | `D:\WSL\Ubuntu-24.04\ext4.vhdx`                                 |

代码、知识库 `example/a/DB`、缓存 `cache/` 和 conda 都在这块 Linux 盘上。资源管理器要改文件时走上面的 `\\wsl.localhost\...`。`ext4.vhdx` 是整块盘的一个文件，用 `wsl` 进系统后再操作里面的目录。

后端流水线入口是仓库根目录的 `main.py`（开发主库 `example/a`）和 `main2.py`（小库 `example/b`），按文件里打开的那几行往下执行。原文、总结、文本块和实体进 SQLite，块和实体的向量进 FAISS。检索只读已经建好的库。

代码分两段：

- **构建**：`download_from_oss`（可选）→ `insert_default` → `summary` → `chunk` → `extract` → `build` → `vectorization`。每一步只处理还没做完的行，写完才改状态。
- **检索**：`pin_retrieve_indexes` → `query` / `agent_query` / `agentic_query` → `unpin_retrieve_indexes`。三条问答共用同一套索引，各读自己的配置段。

网站是后面单独一节的薄封装，浏览器不进入这条流水线。

## 本地运行

在仓库根目录执行。依赖见 `requirements.txt`。本机开发用 conda 环境 `test`：

```bash
source /root/miniconda3/bin/activate test
python main.py
```

`main.py` 固定读 `example/a/config_open.yaml`，`main2.py` 固定读 `example/b/config_open.yaml`。配置里的 `settings.working_path` 决定库放在哪，当前分别是 `example/a` 和 `example/b`。两个文件里的流水线调用目前都注释着：直接运行只会打开 `{working_path}/DB/main.db` 和 `{working_path}/DB/vdb/`，然后退出。

要跑某一步，在对应文件里取消那一行的注释，保存后再执行。一次只打开当前要补的那一步。`main.py` 里的调用就是下面这一串：

```python
config = Config.from_yaml('example/a/config_open.yaml')
graph = DHMF(config)

构建
graph.download_from_oss()
graph.dedupe_downloaded_docs()
graph.insert_default()
graph.summary()
graph.chunk()
graph.extract()
graph.build()
graph.vectorization()

检索
graph.pin_retrieve_indexes()
print(graph.query('...', mode='dual_path', pretty=True))
print(graph.agent_query('...', pretty=True))
res = graph.agentic_query('...', pretty=False)
print(res['answer'])
graph.unpin_retrieve_indexes()
```

构建开始后，日志写到 `{working_path}/log/build_日期_时间.log`。本次真实调用的耗时和 token 记在内存里，结束时并入 `{working_path}/DB/build_metrics.json`。磁盘缓存命中不计入 token。

`query` 检索一次再生成。`agent_query` 先把问题拆成子问题再逐个检索、最后汇总。`agentic_query` 由模型决定下一轮是再搜、读原文、看同文档实体，还是作答。`mode` 只接受 `dual_path`。

## 构建阶段

`settings` 是各步共用的：

```bash
python main.py
python main2.py
```

要跑某一步，在对应文件里取消那一行的注释，保存后再执行。一次只打开当前要补的那一步。已经完成的行会按各步自己的状态跳过，注释着是为了避免对已经建好的库误跑。日志在这一步开始时写到 `{working_path}/log/build_日期_时间.log`。本次真实调用的耗时和 token 记在内存里，结束时并入 `{working_path}/DB/build_metrics.json`；磁盘缓存命中不计入 token。

`settings` 是各步共用的：

| 字段                       | 作用                                                                                                |
| -------------------------- | --------------------------------------------------------------------------------------------------- |
| `working_path`           | 原文、SQLite、FAISS、构建日志的根目录。`example/a` 这份配置是 `example/a`                       |
| `debug`                  | `true` 时总结、分块、抽取、构图会放宽选行（已完成的文档或块也会再进任务）。日常补缺保持 `false` |
| `api_key` / `base_url` | 某一步自己的地址没写时回退到这里。空密钥按`EMPTY` 发给内网网关                                    |

下面按 `main.py` 里的调用顺序写。每一步的配置都在同一份 YAML 里，段名和代码里的配置对象一致。下文的「当前值」取自 `example/a/config_open.yaml`。

状态是断点。重跑只挑选还没做完的行：

| 步骤               | 挑选条件                            | 做完后                                                      |
| ------------------ | ----------------------------------- | ----------------------------------------------------------- |
| `insert_default` | 源文件还没按字节 MD5 或切片名入库   | `doc.status = new`                                        |
| `summary`        | `doc.status = new`                | `doc.status = summary`，写入一条超边                      |
| `chunk`          | `doc.status = summary`            | `doc.status = chunk`，写入 head 和 body 块                |
| `extract`        | `chunk.status = new`              | `chunk.status = extract`，`extra.extract` 里是实体 JSON |
| `build`          | `chunk.status = extract`          | `chunk.status = build`，实体写入 `node`                 |
| `vectorization`  | 目标表`embedding_status = undone` | 向量落盘之后才标`done`                                    |

上游没跑完时，下游任务列表是空的：没有 `doc.content` 就没有总结，没有超边就跳过该文档的分块和构图。

### 下载与本地去重（可选，默认注释）

`graph.download_from_oss()` 只把对象下载到 `{working_path}/{doc.doc_dir}`。它不识别、不入库。`graph.dedupe_downloaded_docs()` 是另一步：按文件字节 MD5 删掉目录里的重复 PDF，每种内容留一份。`insert` 不会调用它。

`oss_download`：

| 字段             | 作用                                                              | example/a 当前值      |
| ---------------- | ----------------------------------------------------------------- | --------------------- |
| `table`        | MySQL 里列出对象键的表                                            | `msds_information`  |
| `file_type`    | `1` TDS、`2` MSDS、`all` 两种都下。表没有 `type` 列时忽略 | `all`               |
| `limit`        | 最多下载多少行。`0` 或负数表示不限制                            | `-1`                |
| `download_dir` | 相对`working_path` 的落地目录。空则用 `doc.doc_dir`           | `doc`               |
| `bucket_key`   | `ali_oss` 里的桶名                                              | `ky-products-files` |
| `num_thread`   | 并行 GET 的线程数。每个线程自己持有一个桶客户端                   | `8`                 |
| `dedupe_local` | 下载结束后是否再跑一遍本地字节去重                                | `false`             |

连接信息在 `dm_data_mysql`（库、主机、账号）和 `ali_oss.<bucket_key>`（访问密钥）。只有这两步读取它们。下载按对象键和本地文件名跳过已有文件，不对文件内容做哈希。

### insert

`graph.insert_default()` 读取 `{working_path}/{doc.doc_dir}`。`source_type: pdf` 时识别 PDF 和图片（`.pdf`、`.jpg`、`.jpeg`、`.png`、`.webp`、`.gif`、`.bmp`、`.tif`、`.tiff`）；`txt` 时按文件名把纯文本直接插入，不调用视觉模型。

`example/a` 的 `doc_dir` 仍是 `doc`。现有三套原文在同级目录 `doc_tds_25367`、`doc_专利文件_8697`、`doc_msds_49688`。要识别其中一套，把 `doc_dir` 改成该目录名，或把文件放进 `example/a/doc`。

识别按源文件分批：一批渲染、调用模型、写入 `doc`，再处理下一批，避免把全部正文同时留在内存里。长 PDF 按 `max_pages_per_doc` 切成多条文档：第一段用原文件名，后面是 `原文件名_1`、`原文件名_2`。图片不切页。识别正文写入 `doc.content`，页范围、源文件名、字节 MD5 写在 `doc.extra`。`doc.hash` 是识别正文的哈希，列上有唯一约束。

去重分两处。送给模型之前：切片名已经在 `doc` 表里的段直接跳过（`skip_existing` 默认为真，中断后续跑只补缺段）；剩下的段再算源文件字节 MD5，这个 MD5 已经属于另一个源文件名时，整份不再识别。同一源文件的各页切片共用这个 MD5，尚未入库的段都会保留。算不出 MD5 时，识别完成、写入之前再按文档名和识别正文哈希跳过。

还有 `file_hash` 为空的旧行时，insert 按每批 2000 行把 `extra` 里的 MD5 回填到列上，直到没有空值。去重查找走 `file_hash` 和 `name` 上的索引，不读取 `doc.content`。

`doc` 与 `doc.recognition`：

| 字段                                                | 作用                                                          | example/a 当前值                       |
| --------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------- |
| `source_type`                                     | `pdf` 走识别；`txt` 直接读文本                            | `pdf`                                |
| `doc_dir`                                         | 相对`working_path` 的原文目录                               | `doc`                                |
| `count_token`                                     | 为真时用 cl100k 把正文字数写入`doc.tokens`                  | `true`                               |
| `flush_every`                                     | 每多少个源文件识别并入库一次                                  | `500`                                |
| `recognition.api_key` / `base_url`              | 视觉模型地址。空则回退`settings`                            | 与`settings` 相同的对话网关          |
| `recognition.model_args`                          | 模型名、温度、是否思考。温度会改变识别文本，从而改变正文哈希  | `llm-medium`，温度 `0.2`，思考关闭 |
| `recognition.use_cache`                           | 为真时按切片查`pdf_recognize_cache`                         | `true`                               |
| `recognition.prompt`                              | `src/utils/prompt.py` 里的提示词名                          | `pdf_recognize`                      |
| `recognition.dpi` / `image_format`              | 送给视觉模型的页图分辨率和格式                                | `100` / `jpeg`                     |
| `recognition.num_thread`                          | 视觉模型的切片并发。PaddleOCR 路径不使用它                    | `8`                                  |
| `recognition.max_pages_per_doc`                   | 一次请求最多带多少页。超出则切成多条`doc`                   | `4`                                  |
| `recognition.use_paddleocr`                       | 为真时改走 PaddleOCR-VL，切片和缓存键仍是这一套               | `false`                              |
| `recognition.paddleocr_url` / `paddleocr_model` | OCR 视觉服务                                                  | 配置里有地址，当前未启用               |
| `recognition.paddleocr_concurrency`               | OCR 的 HTTP 并发                                              | `16`                                 |
| `recognition.paddleocr_dpi`                       | OCR 渲染分辨率                                                | `200`                                |
| `recognition.paddleocr_layout_url`                | 版面切框服务。空则本机 CPU，并强制单线程                      | 配置里有地址，当前未启用               |
| `recognition.retry`                               | `max_attempt`、`wait`（秒，标量或列表）、单次 `timeout` | 3 次，等待 30/60/90 秒，超时 600 秒    |

`graph.insert_clear()` 删除 `doc` 表、总结写入的 `hyperedge`，并清掉 doc 向量。`chunk` 和 `node` 还在，重做整库时要继续调用后面的 clear。

### summary

`graph.summary()` 对每条 `status=new` 的文档调用对话模型，把总结写入 `hyperedge.content`。一篇文档一条超边。成功后 `doc.status` 改为 `summary`。

长 PDF 的后段切片可以在提示词里带上同一源文件更早切片的识别正文。`enable_prior_context` 为假时不带，全部走 `prompt` 指定的模板。为真且确实载入了前文时，后段改用 `doc_summary_continuation`。`prior_context_slices` 控制带最近几段；`0` 表示全部前文。

| 字段                       | 作用                                       | example/a 当前值                       |
| -------------------------- | ------------------------------------------ | -------------------------------------- |
| `api_key` / `base_url` | 总结用的对话模型                           | 与`settings` 相同                    |
| `model_args`             | 模型名、温度、思考开关。这些字段进入缓存键 | `llm-medium`，温度 `0.2`，思考关闭 |
| `prompt`                 | 提示词名                                   | `doc_summary`                        |
| `use_cache`              | 是否读写`llm_cache`                      | `true`                               |
| `num_thread`             | 并发文档数                                 | `8`                                  |
| `flush_every`            | 每多少条超边写一次库                       | `500`                                |
| `enable_prior_context`   | 后段切片是否注入前文                       | `false`                              |
| `prior_context_slices`   | 开启时注入最近几段前文                     | `1`                                  |
| `retry`                  | 失败重试                                   | 3 次，间隔 30 秒，超时 300 秒          |

`graph.summary_clear()` 删除全部超边，并把 `status` 为 `summary` 或 `chunk` 的文档改回 `new`，同时清掉 hyperedge 向量。它不删除 `chunk` 和 `node`。已经分过块时，先 `build_clear()`、`chunk_clear()`，再 `summary_clear()`，然后从 `summary()` 重跑。

### chunk

`graph.chunk()` 只处理 `doc.status=summary` 且已经有超边的文档。每篇文档产出：

- 一个头块，`name=head`，正文是超边总结，不分块。头块 id 回写到 `hyperedge.chunk_id`。
- 若干正文块，`name=body_1`、`body_2`……，正文来自 `doc.content`。按句号、问号、叹号和换行切句，再打包到 `chunk_size_max` 个 cl100k token。相邻块保留大约 `chunk_overlap × chunk_size_max` 个 token 的尾部重叠。

块的 `extra` 记录 `role`（`head` 或 `body`）、`chunk_index`、重叠比例和来源。插入时把正文 casefold 后写入 `chunk_fts`。完成后 `doc.status=chunk`，新块的 `status` 默认为 `new`，留给抽取。

这一步不调用模型。

| 字段               | 作用                                                                 | example/a 当前值 |
| ------------------ | -------------------------------------------------------------------- | ---------------- |
| `count_token`    | 为真时把块的 token 数写入`chunk.tokens`                            | `true`         |
| `chunk_size_max` | 正文块的 token 上限                                                  | `512`          |
| `chunk_overlap`  | 相邻块重叠比例。大于 1 时按百分数理解（`10` 即 `0.1`），上限 0.9 | `0.1`          |
| `flush_every`    | 每多少个块写一次库                                                   | `1000`         |

`graph.chunk_clear()` 清空 `chunk`（含全文索引）和 chunk 向量。有超边的文档状态改回 `summary`，没有超边的改回 `new`。

### extract

`graph.extract()` 读取 `chunk.status=new` 的块，用对话模型按 JSON 抽出「实体名 → 中文描述」。合法结果写进该块的 `extra.extract`，状态改为 `extract`。头块和正文块都抽。`extract_prompt` 指向 `src/utils/prompt.py` 里的 `extract_product`。配置里的名字缺失时，代码回退到 `extract`。

返回必须能解析成实体字典：`{"entities": {"名称": "描述"}}`、这种对象的列表，或「值全是字符串」的扁平对象。`reference`、`knowledge` 被丢掉。JSON 不合法时删掉这一次的 `llm_cache` 条目再重试，避免以后一直命中坏结果。第二次及以后的尝试会把温度改成 `1.0`，因此缓存键和第一次不同。

| 字段                       | 作用                                                                      | example/a 当前值                              |
| -------------------------- | ------------------------------------------------------------------------- | --------------------------------------------- |
| `api_key` / `base_url` | 抽取用的对话模型                                                          | 与`settings` 相同                           |
| `model_args`             | 模型、温度、思考、`response_format`。缺省会补上 JSON 对象格式和关闭思考 | `llm-medium`，温度 `0.2`，`json_object` |
| `extract_prompt`         | 提示词名，内容里的`{content}` 换成块正文                                | `extract_product`                           |
| `use_cache`              | 是否读写`llm_cache`                                                     | `true`                                      |
| `num_thread`             | 并发块数                                                                  | `16`                                        |
| `flush_every`            | 每多少个块更新一次                                                        | `2000`                                      |
| `retry`                  | 失败重试                                                                  | 3 次，间隔 30 秒，超时 180 秒                 |

`graph.extract_clear()` 把全部块的状态改回 `new`，并清空 `chunk.extra`。节点还在时先 `build_clear()`。

### build

`graph.build()` 按文档把 `chunk.status=extract` 的块收在一起。该文档没有超边则整篇跳过。

`build.target` 里的名字决定写什么：

- `hyperedge`：把头块上的实体聚合写进已有超边的 `extra`（`entities`、`source_chunk_id`、头块标记），并确认 `chunk_id` 指向头块。不新建超边，不改总结正文。
- `node`：每个块上的每个实体成为一行。`name` 是实体名，`content` 是描述，`embedding_content` 是「名称、换行、描述」。`doc_id` 和 `chunk_id` 指向来源。`hyperedge_id` 这一列留空；同一篇文档的实体靠 `doc_id` 聚在一起。检索拼材料时若这列为空，会按 `doc_id` 补上该文档的超边 id。

`edge` 表保留，这一步不写。完成后块状态改为 `build`。构图不调用嵌入模型，也不写向量分片。

| 字段            | 作用                                             | example/a 当前值        |
| --------------- | ------------------------------------------------ | ----------------------- |
| `target`      | 要更新的对象                                     | `hyperedge`、`node` |
| `flush_every` | 缓冲区里超边更新、节点、块状态合计达到该数就落库 | `500`                 |

`graph.build_clear()` 把块状态改回 `extract`，清空 `node` 和 `edge`，并清掉这两套向量。超边正文保留。

### vectorization

`graph.vectorization()` 不带参数时，按 `vectorization.default_target` 的顺序嵌表。当前是先 `chunk`，再 `node`。`graph.vectorization('node')` 只处理这一张表。

被嵌入的文本优先用 `embedding_content`，为空则用 `content`。因此块嵌的是块正文，节点嵌的是「名称 + 描述」。`doc`、`hyperedge`、`edge` 不在默认目标里，检索不读它们的向量。

只挑选 `embedding_status=undone` 的行，按主键分页读取，不把整表载入内存。请求按 `batch_size` 打包，`num_thread` 个批次并行。缓存按单条文本命中：一批里只有未命中的文本会发给嵌入服务。

`shard_max_vectors` 大于 0 时，`flush_every`、`index_save_every`、`task_page_size` 会被强制改成同一个数。满这一批就做一次 checkpoint：向量写入当前活跃分片，SQLite 才把这些行标成 `done`，然后活跃分片封存，再开一个空的活跃分片。进程峰值大约是一片索引，加上这一批的请求缓冲。checkpoint 之前崩溃的行仍是 `undone`，重跑会再嵌一次；嵌入缓存命中时不再向服务计费，向量写入当时的活跃片。跑完时不足一片的尾部也会封成只读片。已封分片不再追加。

构建或向量化期间不要调用 `pin_retrieve_indexes()`。pin 会把分片留在本进程里，和封片写盘抢同一批文件。

| 字段                       | 作用                                                                                                        | example/a 当前值                  |
| -------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `api_key` / `base_url` | 嵌入服务                                                                                                    | `http://182.92.180.215:9999/v1` |
| `model_args.model`       | 网关里的嵌入模型 id                                                                                         | `embedding-medium`              |
| `model_args.dimension`   | 请求的维数。加载配置时收成`dimensions`，并抄到 `vectorization.dim`。降维和 L2 归一化由服务端做          | `768`                           |
| `default_target`         | 无参数调用时嵌哪些表                                                                                        | `chunk`、`node`               |
| `use_cache`              | 是否读写`emb_cache`                                                                                       | `true`                          |
| `num_thread`             | 并行的嵌入请求数                                                                                            | `32`                            |
| `batch_size`             | 每个请求里的文本条数                                                                                        | `32`                            |
| `shard_max_vectors`      | 每片最多多少条向量，也是分页、落盘、封片的批大小。空或`0` 表示整库一个 `{表名}.vdb`，索引会整份留在内存 | `100000`                        |
| `index_type`             | `hnsw` 或 `flat_l2`                                                                                     | `hnsw`                          |
| `index_quant`            | `fp16` 或 `none`（float32）。和已有分片的编码必须一致                                                   | `fp16`                          |
| `hnsw_M`                 | 图的邻居数。越大召回越好，内存和建图越慢                                                                    | `32`                            |
| `hnsw_efConstruction`    | 建图时的搜索宽度                                                                                            | `256`                           |
| `hnsw_efSearch`          | 查询时的搜索宽度                                                                                            | `128`                           |
| `retry`                  | 单次嵌入失败的重试                                                                                          | 3 次，间隔 30 秒，超时 600 秒     |

换模型、换维数或换索引类型之前，先 `graph.vectorization_clear()`（或传入表名）。它把对应表的 `embedding_status` 改回 `undone`，删除该套 FAISS，并删掉同目录的墓碑文件。只删分片、留下 `done`，下次 `vectorization()` 会认为没有工作可做。按文件名删除和墓碑压缩见下一小节。

### 删除、墓碑与压缩

整段重做用各步末尾的 `*_clear`，调用顺序在「从中间重做」。按文件名去掉某一份资料用 `graph.delete`：业务行当场删掉，向量先记墓碑，检索跳过这些 id；分片里的 HNSW 要等 `compact_vectors` 才物理去掉。

| 调用                      | 作用                                                                                                           |
| ------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `insert_clear()`        | 清空`doc`、`hyperedge` 和 doc 向量。`chunk`、`node` 保留                                               |
| `summary_clear()`       | 清空超边和 hyperedge 向量。`status` 为 `summary` 或 `chunk` 的文档改回 `new`。`chunk`、`node` 保留 |
| `chunk_clear()`         | 清空`chunk`、全文索引和 chunk 向量。有超边的文档改回 `summary`，其余改回 `new`                           |
| `extract_clear()`       | 块状态改回`new`，清空 `chunk.extra`。节点保留                                                              |
| `build_clear()`         | 块状态改回`extract`，清空 `node`、`edge` 及这两套向量。超边正文保留                                      |
| `vectorization_clear()` | 指定表（省略则按`default_target`）的 `embedding_status` 改回 `undone`，删除该套 FAISS 和墓碑文件         |
| `delete(names)`         | 按文件名删掉一整族文档及其块、超边、节点、边，并给对应向量打墓碑                                               |

`*_clear` 不删 `cache/OpenAI/`。`vectorization_clear` 删掉一整套索引并留下空库，下次 `vectorization()` 再写入。`delete` 只处理点名的那一族，其余向量还在原分片里。

#### 按文件名删除

在 `main.py` 里打开库之后调用。`names` 是一个文件名，或一组文件名：

```python
graph.delete('某文件.pdf')
graph.delete(['a.pdf', 'b.pdf'], clear_cache=True)
```

传入文件名或其中一段切片名即可。代码按 `extra.source_name` 和 `名字_n` 把同一份长 PDF 的全部分段收齐，一次删掉。

顺序是先收集 id，再给向量打墓碑，最后删 SQLite。默认 `delete_vectors=True`，墓碑打在 `doc`、`chunk`、`hyperedge`、`node` 上。`delete_edge` 默认 `True`，连 `edge` 的行和向量一起处理；传 `False` 时留下 edge。SQL 按 `doc_id` 删除 node、hyperedge、chunk、edge，再删 doc。chunk 的删除会同步改 `chunk_fts`。

`clear_cache` 默认 `False`。设为 `True` 时，按这些文件名删除 `pdf_recognize_cache` 里的条目。库里找不到文档时，只要开了这个开关，仍按传入的名字清识别缓存。`llm_cache` 和 `emb_cache` 不在这个开关里。

返回 `{'_batch': ...}`，里面有删掉的文档名、各表行数、识别缓存条数，以及每套向量新记上的墓碑个数。检索那边已预载的字典会被标成需要重装。其余文档的 `status` 保持原样。压缩要另外调用下面的 `compact_vectors`。

#### 墓碑

当前索引类型是 HNSW，不能按 id 从已建成的图里摘掉节点。`remove()` 把 id 写进墓碑文件，检索时丢掉这些 id，分片文件本身留到压缩。

| 布局   | 文件                                                 |
| ------ | ---------------------------------------------------- |
| 分片   | `{working_path}/DB/vdb/{表名}.shards/deleted.json` |
| 单文件 | `{working_path}/DB/vdb/{表名}.deleted.json`        |

内容是 `{"ids": [按数字排序的 id]}`。有墓碑时，ANN 会在原来的 topk 之外再多取一些近邻，多取条数是墓碑个数和 `max(32, topk × 3)` 里较小的那个，滤掉墓碑 id 之后再截回 topk。分片的 `ntotal` 在压缩之前不变。

同一个 id 再次写入向量时，会从墓碑里划掉，检索又能命中它。`vectorization_clear` 删整套分片时连这份文件一起删。`index_type` 为 `flat_l2` 时，会先尝试按 id 从索引里物理删除，同时仍写入这份墓碑。

`delete` 和检索都不会顺手压缩。

查看当前墓碑：

```python
graph.vector_status()       # 五套都看
graph.vector_status('doc')  # 只看一套
```

返回每套的 `ntotal`、`deleted_count`、`deleted_ratio`，以及每一片里有多少 id 落在墓碑上。按片统计会打开该套的每一个分片文件。`node` 有两百多片，这一次查看会把这些文件逐个读一遍。

`example/a` 目前只有 `doc.shards/deleted.json`：id `25351`–`33882`，共 8532 个。这些 id 在 `doc` 表里已经没有行。该套 `sealed` 为空，活跃片 `0.vdb` 是空索引，墓碑是删除后留下的 id 清单。chunk、node、hyperedge、edge 没有墓碑文件。

#### 墓碑压缩

墓碑只让检索跳过 id。要把这些 id 从 HNSW 里拿掉，调用压缩。压缩不请求嵌入服务：向量从现有分片里还原，再按当前的量化方式和 HNSW 参数重建这一片，id 仍是业务表主键。每片条数和片号布局保持不动。要改每片上限，用后面的 `scripts/repack_vdb.py`；那一步同样丢掉表里已经没有的 id，成功后墓碑清空。

```python
graph.compact_vectors('doc')   # 必须指定一套：doc / chunk / hyperedge / node / edge
graph.compact_all_vectors()    # 五套依次做
```

压缩前先 `SELECT id` 读出该表全部主键。索引里有、表里没有的 id 从该片去掉。一整套都没有这种 id 时跳过重写，返回里 `skipped` 为真、`reason` 为 `all_ids_in_sql`，并把墓碑收成「主键还在表里」的那些 id。有分片被重写时，写完后删除该套的 `deleted.json`，主键仍在的向量重新参与检索。重写后一条不剩的分片文件会删掉。活跃片按同一规则就地重写。

读全表主键和重写一片都会占内存。`node` 约 2400 万行，主键集合本身就要一截内存；重写时峰值大约是正在处理的那一片（旧索引加上新建的索引）。压缩就地覆盖分片文件，和 `vectorization()`、`pin_retrieve_indexes()`、`python -m api` 错开再跑。

## 检索阶段

### pin、提问、unpin

这三步读已经写好的库，不改表状态。

`graph.pin_retrieve_indexes()` 按 `search_range` 把与 id 区间重叠的已封分片用 `faiss.read_index` 整份读进本进程并留下。重复 pin 同一区间不会再次读盘。没 pin 时，检索仍可用：用到一片就加载，搜完卸掉。pin 必须和后面的提问在同一个 `DHMF` 实例里。pin 还会把该 id 窗口内的块，以及这些块覆盖到的文档和超边，预载进进程里的字典，并确认 `chunk_fts` 与块行数一致。窗口很大时，预载本身就会占用大量内存。

`search_range` 同时约束检索和 pin。`0` 表示该表不限制；一个整数 `N` 表示 `0..N`；`[a, b]` 表示闭区间。不要在 `retrieve` 或 `agentic` 里再写一份范围。

`example/a` 配置文件注释里的 id 窗口是：文档 TDS `0–33883`、专利 `33884–57032`、MSDS `57033–257613`；节点 TDS `0–1551894`、专利 `1551895–4827583`、MSDS `4827584–24346941`；块 TDS `0–126911`、专利 `126912–377548`、MSDS `377542–1777008`。当前生效值是 MSDS 这一段：

```yaml
search_range:
  node_max_vectors: [4827585, 24346941]
  chunk_max_vectors: [377542, 1777008]
```

在这个窗口上 pin，会常驻 MSDS 重叠到的全部分片，并把这一段块载入内存。内存放不下时，把窗口改回要服务的那一段再 pin。传入 `chunk_max_vectors` / `node_max_vectors` 只覆盖这一次调用。

然后三选一：

```python
graph.pin_retrieve_indexes()
print(graph.query('丙酮的沸点是多少？', mode='dual_path', pretty=True))
print(graph.agent_query('……', pretty=True))
res = graph.agentic_query('……', pretty=False)
print(res['answer'])
graph.unpin_retrieve_indexes()
```

- `query` 只接受 `mode='dual_path'`：块向量、节点向量、可选关键词三路召回，扩展上下文后重排，再生成一次。参数全在 `retrieve`。
- `agent_query` 先规划子问题再检索汇总。规划和生成读 `agent`，检索仍用已经建好的索引。
- `agentic_query` 由模型决定下一轮调用哪个工具。循环、工具宽度、上下文上限全在 `agentic`。工具里的检索临时使用 `agentic` 的召回个数，不改 `retrieve` 里的共享数字。

`pretty=True` 返回可打印的字符串。`pretty=False` 返回字典，里面有 `answer`、`status`、来源和 token。`history` 用来把「它的闪点呢」补成带上一轮产品名的句子，不改索引。

`retrieve` 里和这次提问直接相关的字段（example/a）：

| 字段                                                             | 作用                                                                     | 当前值                        |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------- |
| `api_key` / `base_url` / `model_args`                      | 生成回答的对话模型                                                       | `llm-medium`，温度 `0.2`  |
| `timeout`                                                      | 这次生成的 HTTP 超时（秒）                                               | `10800`                     |
| `use_cache`                                                    | 生成与改写是否走`llm_cache`                                            | `false`                     |
| `embedding_base_url` / `embedding_model_args`                | 查询向量。维数要和库一致                                                 | `embedding-medium`，`768` |
| `enable_query_rewrite`                                         | 检索前是否让模型改写问句                                                 | `false`                     |
| `enable_query_instruct` / `query_instruct`                   | 只给查询向量加指令前缀，库内文档不加                                     | 开启                          |
| `chunk_candidate_k` / `node_candidate_k`                     | 两路向量各留多少条。`0` 关闭该路                                       | 各`40`                      |
| `enable_keyword_exact`                                         | 关键词总开关                                                             | `true`                      |
| `enable_keyword_minority` / `enable_keyword_majority`        | 少数词（CAS、牌号、货号）与多数词（字段名）                              | 只开少数词                    |
| `keyword_candidate_k` / `keyword_top_k` / `keyword_max_df` | 关键词候选、送去重排的条数、命中块数超过则当停用词                       | `20` / `10` / `2000`    |
| `enable_parallel_paths`                                        | 三路并行或串行。串行时峰值内存更低                                       | `false`                     |
| `enable_full_body_context`                                     | `true` 命中文档扩全部正文；`false` 头块加命中块；`simple` 只补头块 | `true`                      |
| `enable_slice_family_expand`                                   | 命中任一切片时扩到同源 PDF 的全部分段。开启后忽略上一行                  | `false`                     |
| `enable_rerank` / `rerank_base_url` / `rerank_model_args`  | 重排服务                                                                 | 开启，`reranker-xsmall`     |
| `rerank_top_k`                                                 | 重排后保留条数                                                           | `20`                        |
| `rerank_max_chars`                                             | 送进重排的正文上限。`-1` 不截断                                        | `4000`                      |

`agent`：`max_steps` 是规划步数上限（当前 `6`）；`use_cache` 管规划、改写和作答（当前 `false`）。规划之后代码总会再插一步不检索的回答，供汇总对照。`chunk_candidate_k` / `node_candidate_k` 为 `null` 时沿用检索模块当时的宽度。YAML 里的 `enable_pure_llm` 当前没有被读取。

`agentic`：`max_turns`（`32`）、`max_prompt_tokens`（`101072`）、`prompt_token_reserve`（`4096`）控制循环何时停。`use_cache` 当前为 `false`，工具循环本身也不进 `llm_cache`。`enable_search`、`enable_read_doc`、`enable_read_chunk`、`enable_graph_neighbors`、`enable_note_evidence` 分别开关五个工具。`graph_neighbors` 返回同一 `hyperedge_id` 或同一 `doc_id` 上的其他实体；构图没填 `hyperedge_id` 时，实际按同一文档展开。`neighbors_limit` 当前 `20`。这一段还有自己的 `chunk_candidate_k`、`node_candidate_k`、关键词开关、`rerank_top_k`（`10`）和正文扩展开关（当前关闭全文扩展）。

`graph.unpin_retrieve_indexes()` 卸掉 pin 留下的分片句柄，之后检索回到按片加载。可以只传 `'chunk'` 或 `'node'`。

### 从中间重做

| 想重做的起点 | 先调用                                                    | 再从这里跑                 |
| ------------ | --------------------------------------------------------- | -------------------------- |
| 总结         | `build_clear()`、`chunk_clear()`、`summary_clear()` | `summary()`              |
| 分块         | `build_clear()`、`chunk_clear()`                      | `chunk()`                |
| 抽取         | `build_clear()`、`extract_clear()`                    | `extract()`              |
| 构图         | `build_clear()`                                         | `build()`                |
| 全部向量     | `vectorization_clear()`                                 | `vectorization()`        |
| 某一套向量   | `vectorization_clear('chunk')` 或 `'node'`            | `vectorization('chunk')` |

只清状态、不删 `cache/OpenAI/` 时，相同提示词仍会命中磁盘缓存。要让模型真正再算一遍，还要关掉对应的 `use_cache`，或删掉下文「数据文件」一节里的缓存库。

## 用 scripts/repack_vdb.py 改分片大小

这个脚本把已经写在 FAISS 里的向量按新的每片条数重写成 HNSW，不调用嵌入服务，也不重跑 insert 到 build。向量从现有分片里解出（fp16 会先还原再按当前量化方式写入），id 保持业务表主键。SQLite 里已经不存在的 id 会被丢掉，重写结束后墓碑清空。

在仓库根目录：

```bash
# 把现有分片收成每片最多 50 万条
python scripts/repack_vdb.py --config example/a/config_open.yaml --shard-max 500000

# 只改 node
python scripts/repack_vdb.py --config example/a/config_open.yaml --shard-max 500000 --name node

# 一套合成一个 {名字}.vdb。进程内存要能放下这一整套索引
python scripts/repack_vdb.py --config example/a/config_open.yaml --mono --name chunk

# 当前布局已经符合目标时仍强制重写
python scripts/repack_vdb.py --config example/a/config_open.yaml --shard-max 500000 --name node --force
```

| 参数            | 作用                                                                         |
| --------------- | ---------------------------------------------------------------------------- |
| `--config`    | 流水线 YAML。用它的`working_path`、维数、HNSW 和量化参数打开库             |
| `--shard-max` | 新的每片条数上限，按条数切，不按字节。与`--mono` 二选一                    |
| `--mono`      | 每套合成一个文件                                                             |
| `--name`      | `doc`、`chunk`、`hyperedge`、`node`、`edge` 之一。省略则五套都处理 |
| `--force`     | 布局已经是目标大小时也重写                                                   |

脚本打印每套的 JSON 摘要。`skipped` 为真时 `reason` 可能是 `empty`（没有向量）、`already_mono`、`already_packed`（片数和每片上限已经符合）、`no_live_vectors`（相对 SQLite 没有可保留的 id）。

重写过程：

1. 先把活跃片落盘。
2. 在 `{working_path}/DB/vdb/{名字}.repack.tmp` 里按 id 顺序重建。目标片在封存前留在内存里，所以峰值大约是新的一片，`--mono` 则是整套。
3. 完成后把旧的 `{名字}.shards` 或 `{名字}.vdb` 移到 `{名字}.repack.bak`，再把新文件换上。成功后删除备份和临时目录。中途失败会尽量把备份移回。
4. 分片模式下会再留一个空的活跃片，下次向量化往这片追加。

磁盘上要再留出大约一整套向量的空闲空间，备份和临时目录同时存在的那一段最占地方。`node` 是五套里最大的，重打包之前先确认数据盘剩余。

改大每片条数或改成单文件之后，把 YAML 里的 `vectorization.shard_max_vectors` 改成同一个数；单文件则删掉该项或设为 `0`。否则下一次用这份 YAML 构造 `DHMF`（`python main.py`、`python -m api` 都算）会把 `meta.json` 里的 `shard_max_vectors` 写回旧值，新写入的片仍按旧的 10 万条封存。已经封好的大片不会被拆开，库里就会同时存在两种片长。

按 id 顺序装箱时，第一片装的是最小的那些 id。把每片上限放到超过 TDS 加专利的 id 跨度时，第一片会把后面的 MSDS 向量装进来。pin 按「分片 id 范围与 `search_range` 是否重叠」决定整片加载，这一片只要沾到要 pin 的窗口，片内其余向量也会进内存。需要只常驻前一段语料时，每片上限应落在该段 id 之内。

重打包期间不要启动 `python -m api`，也不要同时跑 `vectorization()` 或 `pin_retrieve_indexes()`。结束后用新进程打开库。

## 网站

网站是流水线外面的一层。浏览器提交问题，`api` 在同一个进程里调用已经打开的 `DHMF.query`、`agent_query` 或 `agentic_query`。建库用仓库根目录的 `main.py`。

本机直接起进程（conda 环境与建库相同）：

```bash
source /root/miniconda3/bin/activate test
export DHMF_CONFIG=example/a/config_open.yaml
python -m api
```

`DHMF_CONFIG` 缺省也是 `example/a/config_open.yaml`。监听地址在这份 YAML 的 `app_config`：`host` `0.0.0.0`，`port` `8000`。浏览器打开 `http://127.0.0.1:8000/`。`app_config.workers`（当前 `4`）是名为 `dhmf-api` 的线程池大小，同时能跑几路问答。uvicorn 的进程数在 `api/app.py` 的 `run()` 里写成 `1`，整份 FAISS 只加载一次。

启动时 `init_db()` 建好账号库，然后 `pin_retrieve_indexes()` 把 `search_range` 盖住的分片读进这个进程，退出时 `unpin`。`GET /health` 在库已 pin 时返回正常。当前 `example/a` 的 `search_range` 是 MSDS 窗口（node `4827585`–`24346941`，chunk `377542`–`1777008`）。用这份配置拉起网站会把这一大段索引 pin 进内存。本机只想打开登录和历史时：

```bash
CGR_WEB_ONLY=1 python -m api
```

这个开关下进程不构造 `DHMF`，问答接口会说明库未加载。要对 TDS 和专利做在线问答，先把 `search_range` 改成对应窗口再启动。网站进程开着的时候不要同时跑 `vectorization()` 或 `scripts/repack_vdb.py`。

页面静态文件在 `chemical-rag-web/web/`（`index.html`、`app.js`、`styles.css`）。`python -m api` 直接读磁盘上的这份目录。提问走 `POST /api/stream`，登录令牌放在 `Authorization: Bearer`。

### 账号

页面提供登录。开户、改密、停用在服务器上执行，命令写在 `api/accounts.py`，库文件默认是 `chemical-rag-web/data/app.db`。

```bash
python -m api.accounts list
python -m api.accounts add 用户名 密码
python -m api.accounts add 用户名 密码 --admin
python -m api.accounts passwd 用户名 新密码
python -m api.accounts disable 用户名
python -m api.accounts enable 用户名
```

空库第一次 `init_db()` 时，`api/store.py` 的 `_seed_admin` 写入一名管理员。仓库里若有 `chemical-rag-web/config.json`，用户名和初始口令取该文件的 `username`、`password`；没有这个文件时用户名为 `admin`，初始口令在 `_seed_admin` 里。部署后立刻执行 `passwd`。口令留在该文件和账号库中。`passwd` 和 `disable` 会删掉该用户的登录令牌，已打开的页面需要重新登录。

换一个账号库做试验时指定 `CGR_WEB_DB=/tmp/cgr-test.db`。服务正在写库时，整文件覆盖 `app.db` 会和 WAL 冲突；要换库先停进程。

### 用 Docker 拉起

```bash
cp docker/env.example .env
docker compose up -d --build
```

`docker-compose.yml` 起两个服务。`api` 用根目录 `Dockerfile`，环境变量 `DHMF_CONFIG=example/a/config_open.yaml`，把 `./example/a`、`./cache`、`./chemical-rag-web/data` 挂进容器，只在 Compose 网络里暴露 `8000`。`web` 用 `Dockerfile.web`，把页面和 `docker/nginx.conf` 烤进镜像，宿主机端口 `8060:8060`。浏览器打开 `http://127.0.0.1:8060/`。`web` 等 `api` 的健康检查通过后才启动。

`.env` 里的 `AUTH_USER`、`AUTH_PASSWORD` 传给 web 容器。`docker/99-htpasswd.sh` 在口令非空时写出 `/etc/nginx/chemical-rag.htpasswd`。当前 `docker/nginx.conf` 把 `/api/` 反代到 `http://api:8000/` 并转发 `Authorization`，配置里没有 `auth_basic`。网页登录以账号库为准，改 `.env` 之后仍用 `python -m api.accounts passwd` 改口令。

改 `src/` 或 `api/` 后重新构建 `api` 镜像。改 `chemical-rag-web/web/` 或 `docker/nginx.conf` 后重新构建 `web` 镜像。本机 `python -m api` 读的是磁盘上的代码和页面，不用构建镜像。

相关代码：`api/__main__.py` 启动进程，`api/app.py` 是 HTTP 和 pin，`api/accounts.py` 是账号命令，`api/store.py` 是账号库表结构和初始管理员。细节在 `api/README.md` 和 `chemical-rag-web/data/README.md`。

## 数据文件

知识库、向量、缓存、账号库都在仓库目录里，由运行时写出来。`.gitignore` 忽略了 `example/`、`cache/`、`*.db`、`.env`、`remote/`、`chemical-rag-web/config.json` 和 `chemical-rag-web/data/`。克隆之后这些要在本机另备。生产机上的对应路径，以及 2026-09-15 打到 OSS 的包，写在文末。

| 路径                                                                    | 内容                                                   |
| ----------------------------------------------------------------------- | ------------------------------------------------------ |
| `example/a/DB/main.db`                                                | 开发主库的业务表。`main.py` 使用                     |
| `example/a/DB/vdb/`                                                   | 五套 FAISS。块和实体是检索用的两套                     |
| `example/a/DB/build_metrics.json`                                     | 累计构建耗时和 token                                   |
| `example/a/log/`                                                      | `build_日期_时间.log`，以及打开轨迹时的 agentic 日志 |
| `example/a/doc_tds_25367`、`doc_专利文件_8697`、`doc_msds_49688`  | 三套原文。配置里的`doc.doc_dir` 仍是 `doc`         |
| `example/b/`                                                          | `main2.py` 的小库，结构与 `example/a` 相同         |
| `cache/OpenAI/`                                                       | 识别、对话、嵌入三份磁盘缓存。各`working_path` 共用  |
| `chemical-rag-web/data/app.db`                                        | 网站用户、令牌和会话                                   |
| `src/`                                                                | 建库和三条问答路径的代码                               |
| `Dockerfile`、`Dockerfile.web`、`docker-compose.yml`、`docker/` | 镜像和 Compose                                         |

### 知识库

业务库是一个 SQLite 文件：`{working_path}/DB/main.db`。打开时设置 `journal_mode=WAL`、`synchronous=NORMAL`。五张业务表在同一个文件里，向量不在这个文件里。

业务表一览：

| 表                 | 一行表示                                     | 何时写入                                                  |
| ------------------ | -------------------------------------------- | --------------------------------------------------------- |
| `doc`            | 一份资料，或长 PDF 的一个页段                | `insert`                                                |
| `hyperedge`      | 一篇文档的一条摘要                           | `summary` 创建，`chunk`、`build` 回写               |
| `chunk`          | 头块（摘要）或正文块                         | `chunk` 创建，`extract`、`build` 改状态和 `extra` |
| `node`           | 一个抽取出的实体                             | `build`                                                 |
| `edge`           | 预留的实体关系                               | 当前构图不插入行                                          |
| `chunk_fts`      | `chunk.content` 的全文索引（FTS5 trigram） | 随块插入、删除和`chunk_clear`                           |
| `chunk_fts_meta` | 全文索引已覆盖的行数和分词器                 | 随`chunk_fts`                                           |

`doc`：一份资料，或长 PDF 的一个页段。

| 列                    | 含义                                                                                      |
| --------------------- | ----------------------------------------------------------------------------------------- |
| `id`                | 主键，也是 doc 向量的 id                                                                  |
| `status`            | `new` → `summary` → `chunk`                                                       |
| `name`              | 切片文档名。首段为源文件名，后续为`源文件名_n`                                          |
| `content`           | 识别正文                                                                                  |
| `extra`             | JSON：页范围、`source_name`、`file_hash`、`slice_index`、`n_slices`、`pipeline` |
| `tokens`            | `count_token` 开启时的 cl100k 计数                                                      |
| `embedding_content` | 若填写，向量化优先用它。insert 不填                                                       |
| `embedding_status`  | 默认`undone`。默认向量化不处理 doc                                                      |
| `hash`              | 识别正文哈希，唯一                                                                        |
| `file_hash`         | 源文件字节 MD5。同一切片族相同                                                            |

辅助索引：`idx_doc_file_hash_name(file_hash, name)`、`idx_doc_name(name)`、`idx_doc_file_hash_pending`（只覆盖 `file_hash IS NULL` 的行，供回填）。

`hyperedge`：一篇文档一条。`summary` 写 `content`，`chunk` 回写头块的 `chunk_id`，`build` 回写 `extra`。

| 列                    | 含义                                               |
| --------------------- | -------------------------------------------------- |
| `id`                | 主键                                               |
| `status`            | 该行状态                                           |
| `doc_id`            | 所属文档                                           |
| `chunk_id`          | 头块 id。`chunk` 回写                            |
| `name`              | 文档名                                             |
| `content`           | 中文摘要。`summary` 写入                         |
| `extra`             | JSON。`build` 回写                               |
| `tokens`            | `count_token` 开启时的 cl100k 计数               |
| `embedding_content` | 若填写，向量化优先用它。默认向量化不处理 hyperedge |
| `embedding_status`  | 默认`undone`                                     |

`chunk`：头块或正文块。

| 列                                                        | 含义                                         |
| --------------------------------------------------------- | -------------------------------------------- |
| `id`                                                    | 主键，也是 chunk 向量的 id                   |
| `status`                                                | `new` → `extract` → `build`          |
| `doc_id`                                                | 所属文档                                     |
| `name`                                                  | `head` 或 `body_n`                       |
| `content`                                               | 头块是总结，正文块是识别正文的一段           |
| `extra`                                                 | 角色、序号、重叠；抽取之后还有`extract`    |
| `tokens` / `embedding_content` / `embedding_status` | 与 doc 相同含义。块的向量文本就是`content` |

全文在同一 SQLite 文件的虚表 `chunk_fts`（FTS5，`tokenize=trigram`），正文是 `content.casefold()`。`chunk_fts_meta` 记录已索引行数。三个字符及以上的关键词走 `MATCH`，更短的走 `instr`。块插入、删除、`chunk_clear` 会同步维护这张虚表。

`node`：一个实体。

| 列                                  | 含义                                        |
| ----------------------------------- | ------------------------------------------- |
| `id`                              | 主键，也是 node 向量的 id                   |
| `doc_id` / `chunk_id`           | 实体从哪篇文档的哪一块抽出                  |
| `hyperedge_id`                    | 列存在。当前构图不填，检索时按`doc_id` 补 |
| `name` / `content`              | 实体名和中文描述                            |
| `embedding_content`               | `名称\n描述`，向量化使用这一列            |
| `embedding_status`                | 向量落盘后为`done`                        |
| `status` / `extra` / `tokens` | 构图路径不依赖它们                          |

`edge`：当前构图不插入行，检索不遍历它。列与建表语句一致：

| 列                                                        | 含义                                                |
| --------------------------------------------------------- | --------------------------------------------------- |
| `id`                                                    | 主键                                                |
| `status`                                                | 预留                                                |
| `doc_id` / `chunk_id` / `hyperedge_id`              | 这条关系落在哪篇文档、哪一块、哪条超边              |
| `src_node_id` / `dst_node_id`                         | 起点实体和终点实体                                  |
| `name` / `content` / `extra`                        | 预留的名称、正文和 JSON                             |
| `tokens` / `embedding_content` / `embedding_status` | 与其他表相同的计数和向量字段。默认向量化不处理 edge |

向量在 `{working_path}/DB/vdb/`。五套存储各用一个名字：`doc`、`chunk`、`hyperedge`、`node`、`edge`。`shard_max_vectors` 大于 0 时，一套是目录 `{名字}.shards/`：

| 文件                     | 含义                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------ |
| `meta.json`            | 维数、`shard_max_vectors`、`sealed` 列表、`active` 文件名、HNSW 参数、量化方式 |
| `0.vdb`、`1.vdb`…… | 已封存的只读分片。向量 id 与业务表主键相同                                           |
| `active` 点名的文件    | 唯一可追加的一片。封存后`ntotal` 为 0，已写向量在 `sealed` 里                    |
| `deleted.json`         | 墓碑。`delete()` 把 id 记在这里，检索时过滤。物理去掉见构建阶段的「墓碑压缩」      |

未分片时，一整套是单个 `{名字}.vdb`。保存先写临时文件再替换，写入中途退出时旧文件还在。向量 id 与 SQLite 主键相同，检索命中后按 id 回表。

同目录的 `build_metrics.json` 记录累计构建成本。网站账号在下一小节的 `app.db`。

### 网站账号库

文件是 `chemical-rag-web/data/app.db`，由 `api/store.py` 的 `init_db()` 创建。环境变量 `CGR_WEB_DB` 可改路径。连接使用 WAL，旁边的 `app.db-wal`、`app.db-shm` 要和主文件放在一起。里面是网站用户、令牌和会话。删掉它只影响这些记录，`main.db` 和向量仍在 `{working_path}/DB/`。

| 表                | 一行表示               |
| ----------------- | ---------------------- |
| `users`         | 一个登录账号           |
| `auth_tokens`   | 一次登录发出的令牌     |
| `conversations` | 一个用户的一段会话     |
| `turns`         | 这一段会话里的一轮问答 |

`users`：

| 列                | 含义                                                     |
| ----------------- | -------------------------------------------------------- |
| `id`            | 主键                                                     |
| `username`      | 登录名。唯一约束`COLLATE NOCASE`，大小写视为同一个名字 |
| `password_hash` | 口令摘要。库里不存明文                                   |
| `is_admin`      | `1` 为管理员                                           |
| `disabled`      | `1` 为停用。停用后旧令牌也不能再通过校验               |
| `created_at`    | 创建时间                                                 |

`auth_tokens`：

| 列                                                 | 含义                                                 |
| -------------------------------------------------- | ---------------------------------------------------- |
| `token_hash`                                     | 主键，令牌的 SHA-256。原始令牌只在登录响应里出现一次 |
| `user_id`                                        | 所属用户。删用户时级联删除                           |
| `created_at` / `expires_at` / `last_used_at` | 签发、过期、最近使用                                 |

`conversations`：

| 列                              | 含义                                                 |
| ------------------------------- | ---------------------------------------------------- |
| `id`                          | 会话 id                                              |
| `user_id`                     | 所属用户。按 id 读取时都带这个条件                   |
| `title`                       | 会话标题                                             |
| `title_is_manual`             | `1` 表示标题被人工改过，之后不再用第一问自动覆盖   |
| `created_at` / `updated_at` | 创建和最近更新。列表按`(user_id, updated_at)` 排序 |

`turns`：

| 列                                                 | 含义                                           |
| -------------------------------------------------- | ---------------------------------------------- |
| `id`                                             | 主键                                           |
| `conversation_id` / `user_id` / `turn_index` | 属于哪段会话、哪个用户、会话内序号             |
| `query`                                          | 问句                                           |
| `mode`                                           | `retrieve`、`dual`、`agent`、`agentic` |
| `answer`                                         | 最终回答                                       |
| `status`                                         | 这一轮的完成状态                               |
| `latency_s`                                      | 耗时（秒）                                     |
| `sources_json`                                   | 引用的来源                                     |
| `result_json`                                    | 接口返回的完整结果                             |
| `stream_steps_json`                              | 过程栏里的步骤                                 |
| `created_at`                                     | 写入时间                                       |

删用户会级联删掉他的令牌、会话和回合。删会话会级联删掉回合。

### src 里的代码文件

入口和四个子目录。各目录的接口在自己的 README：`src/README.md`、`src/module/README.md`、`src/agent/README.md`、`src/agentic/README.md`、`src/utils/README.md`。网站进程在仓库的 `api/`，见上一节。

| 文件                            | 做什么                                                                                                                                                                                                                                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/DHMF.py`                 | 对外入口。打开`main.db` 和五套向量，提供下载、去重、`insert`、`summary`、`chunk`、`extract`、`build`、`vectorization`、`delete`、`vector_status`、`compact_vectors`、`repack_vectors`、`pin` / `unpin`、`query`、`agent_query`、`agentic_query`，以及各步的 `*_clear` |
| `src/module/doc.py`           | 识别 PDF 和图片，按切片名和字节 MD5 去重后写入`doc`                                                                                                                                                                                                                                                   |
| `src/module/summary.py`       | 每篇文档写一条超边摘要                                                                                                                                                                                                                                                                                  |
| `src/module/chunk.py`         | 切头块和正文块，并维护`chunk_fts`                                                                                                                                                                                                                                                                     |
| `src/module/extract.py`       | 从状态为`new` 的块抽出实体 JSON，写入 `chunk.extra`                                                                                                                                                                                                                                                 |
| `src/module/build.py`         | 把实体写成`node`。节点带 `doc_id`、`chunk_id`、名称和描述。这一步不插入 `edge`，也不填 `node.hyperedge_id`                                                                                                                                                                                    |
| `src/module/vectorization.py` | 把默认目标（chunk、node）里`embedding_status=undone` 的行写入 FAISS，checkpoint 后再标 `done`                                                                                                                                                                                                       |
| `src/module/retrieve.py`      | `pin`、三路召回和重排。`pin` 时把窗口内的 chunk、doc、hyperedge 读进进程                                                                                                                                                                                                                            |
| `src/agent/runner.py`         | `agent_query` 的入口                                                                                                                                                                                                                                                                                  |
| `src/agent/graph.py`          | 规划、执行、汇总这条状态图                                                                                                                                                                                                                                                                              |
| `src/agent/nodes.py`          | 规划、执行、汇总三个节点                                                                                                                                                                                                                                                                                |
| `src/agent/skill.py`          | 单步检索                                                                                                                                                                                                                                                                                                |
| `src/agent/state.py`          | 子问题、依赖和用量                                                                                                                                                                                                                                                                                      |
| `src/agent/prompts.py`        | 规划、改写、单步作答和汇总的提示词。只读配置段`agent`                                                                                                                                                                                                                                                 |
| `src/agentic/runner.py`       | `agentic_query` 的入口                                                                                                                                                                                                                                                                                |
| `src/agentic/loop.py`         | 多轮工具循环和停机                                                                                                                                                                                                                                                                                      |
| `src/agentic/tools.py`        | `search`、`read_doc`、`read_chunk`、`graph_neighbors`、`note_evidence`                                                                                                                                                                                                                        |
| `src/agentic/prompts.py`      | 工具循环的系统提示。只读配置段`agentic`                                                                                                                                                                                                                                                               |
| `src/utils/config.py`         | 把 YAML 读成配置对象                                                                                                                                                                                                                                                                                    |
| `src/utils/storage.py`        | `doc`、`hyperedge`、`chunk`、`node`、`edge` 和全文表的建表                                                                                                                                                                                                                                    |
| `src/utils/database.py`       | SQLite 连接、FAISS 分片的封存、pin、墓碑、压缩和`repack`                                                                                                                                                                                                                                              |
| `src/utils/OpenAIAPI.py`      | 对话、嵌入、重排三个客户端，以及磁盘缓存的读写                                                                                                                                                                                                                                                          |
| `src/utils/utils.py`          | `Cache` / `CacheDB`、重试                                                                                                                                                                                                                                                                           |
| `src/utils/prompt.py`         | 识别、总结、抽取等提示词文本，其中包括`extract_product`                                                                                                                                                                                                                                               |
| `src/utils/metrics.py`        | 构建耗时和 token，结束时写入`build_metrics.json`                                                                                                                                                                                                                                                      |
| `src/utils/progress.py`       | 步骤事件。网站过程栏订阅它                                                                                                                                                                                                                                                                              |
| `src/utils/chat_history.py`   | 把追问补成可单独检索的问句                                                                                                                                                                                                                                                                              |
| `src/utils/oss_download.py`   | 按配置从对象存储把原文拉到`doc_dir`                                                                                                                                                                                                                                                                   |
| `src/utils/paddleocr_vl.py`   | `use_paddleocr` 打开时的识别后端                                                                                                                                                                                                                                                                      |

### 缓存文件

模型调用有两层缓存，职责不同。

第一层在本机磁盘，命中后不再发 HTTP。实现是 `src/utils/utils.py` 的 `Cache` / `CacheDB`，库文件在仓库根目录的 `cache/OpenAI/`（与 `working_path` 无关，`example/a` 和 `example/b` 共用）：

| 文件                       | 谁写入                                                           | 键                                                              |
| -------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------- |
| `pdf_recognize_cache.db` | `doc.recognition.use_cache`                                    | 字节 MD5、切片页范围、模型参数、提示词哈希、dpi、后端、图片格式 |
| `llm_cache.db`           | 总结、抽取、检索改写和`query` 生成。各步有自己的 `use_cache` | `prompt` 与 `model_args` 的 JSON 的 MD5                     |
| `emb_cache.db`           | `vectorization.use_cache`，检索查询向量同样走嵌入客户端        | 单条文本加嵌入参数（含`dimensions`）的 MD5                    |

表结构都是 `hash`（唯一）、`input`、`output`。对话和嵌入成功后把去掉内部字段的响应包在 `{"output": ...}` 里。识别缓存直接存切片结果。`status` 不是 `1`，或调用方标了 `_cache_store=false` 的结果不写入。抽取在 JSON 校验失败时按同一次 `prompt` 和 `model_args` 删除那一条，避免坏 JSON 被永久复用。

嵌入的批量接口一次请求多条文本，查找仍按单条键。半批命中时只把未命中的文本放进同一次 API。单条 `generate` 和批量 `generate_batch` 共用 `emb_cache`，所以中断后续跑、或 `vectorization_clear` 之后再嵌，已算过的文本仍然命中。

这一层没有容量上限，也不会按时间淘汰。库会随构建变大。要清空某一类结果，在没有构建进程占用该文件时删除对应的 `.db`。

第二层在推理服务上。本地缓存未命中、请求真正发出之后，网关若复用了提示词前缀，会在补全用量里带上 `prompt_tokens_details.cached_tokens`。客户端记到响应的 `usage_cached_tokens`，并随成功结果一起进入第一层。它不代替第一层：有这个数字仍然算一次真实调用，token 仍计入本次构建。嵌入和 PDF 识别不读这个字段。`agentic` 的工具循环走多轮 `chat`，不写 `llm_cache`；服务端前缀缓存仍可能作用在重复的系统提示上。

两层和业务库里的 `status` / `embedding_status` 互相独立。状态决定这一步还要不要选中该行；磁盘缓存决定选中之后还要不要再付一次模型费用。`*_clear` 只改状态和库内产物，不删 `cache/OpenAI/`。删缓存文件也不把 `done` 改回 `undone`。

### Docker 文件

镜像里是代码和运行环境。知识库、缓存、账号库通过 volume 留在宿主机。

| 文件                                         | 作用                                                                                                                                                             |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Dockerfile`                               | API 镜像。基础是`python:3.13-slim-bookworm`，装 `requirements.txt`，复制 `api/`、`src/`、`main.py`。启动命令是 `python -m api`。知识库由 volume 挂入 |
| `Dockerfile.web`                           | 页面镜像。基础是`nginx:1.24-alpine`，复制 `docker/nginx.conf`、`docker/99-htpasswd.sh` 和 `chemical-rag-web/web/`，监听 `8060`                         |
| `docker-compose.yml`                       | 同时起`api` 和 `web`。挂载、端口和 `DHMF_CONFIG` 见上文「用 Docker 拉起」                                                                                  |
| `docker/nginx.conf`                        | 容器内站点配置。`/api/` 反代到 API，问答流式接口关闭代理缓冲                                                                                                   |
| `docker/env.example`                       | 复制成仓库根目录的`.env`。里面是 `AUTH_USER` 和 `AUTH_PASSWORD` 的占位                                                                                     |
| `docker/99-htpasswd.sh`                    | 容器启动时若`AUTH_PASSWORD` 非空，写出 `/etc/nginx/chemical-rag.htpasswd`                                                                                    |
| `chemical-rag-web/nginx-chemical-rag.conf` | 给宿主机 nginx 用的一份配置稿。Compose 使用的是镜像内的`docker/nginx.conf`                                                                                     |

## 生产机

项目在生产机端的 `/data/chemical-graph-rag`，是一份快照。公网入口 `http://182.92.85.174:8060/`。容器名 `chemical-graph-rag-api-1` 和 `chemical-graph-rag-web-1`，由该目录下的 Compose 拉起。登录方式和资产信息写在仓库的 `remote/remote_config.yaml`。口令和密钥内容留在那个文件里。

和生产 Compose 挂载对应的路径：

| 生产路径                                                  | 内容                                                                               |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `/data/chemical-graph-rag/example/a/config_open.yaml`   | 线上使用的流水线配置。改完后在该目录执行`docker compose restart api` 才进入进程  |
| `/data/chemical-graph-rag/example/a/DB/main.db`         | 知识库业务表                                                                       |
| `/data/chemical-graph-rag/example/a/DB/vdb/`            | FAISS 分片                                                                         |
| `/data/chemical-graph-rag/cache`                        | 模型磁盘缓存，挂到容器内`/app/cache`                                             |
| `/data/chemical-graph-rag/chemical-rag-web/data/app.db` | 线上账号和会话。这份库是在生产机上初始化的，和开发机的`app.db` 各用各的          |
| `/data/chemical-graph-rag/chemical-rag-web/web/`        | 页面源文件。线上页面烤在 web 镜像里，改磁盘上的静态文件后要重新构建 web 镜像再生效 |

线上检索窗口停在专利末尾：`node_max_vectors` 到 `4827583`，`chunk_max_vectors` 到 `377548`（TDS 加专利）。API 进程常驻大约 11 GiB。推理地址写在上面的 YAML 里。

目前体量：文档切片约 249077，正文块约 178 万，实体节点约 2435 万；SQLite 约 18.75 GiB；node 向量约 41.18 GiB，chunk 向量约 3.01 GiB；`vdb` 目录约 45 GB，`DB` 目录约 63 GB。上表的生产路径是这些文件在生产机上的对应位置。生产进程实际 pin 的是 TDS 加专利那一段，常驻内存按约 11 GiB 计。

### 曾经打包上传的 OSS

打的包在阿里云 OSS：

| 项       | 值                                     |
| -------- | -------------------------------------- |
| 桶       | `ky-knowledge`                       |
| Endpoint | `http://oss-cn-beijing.aliyuncs.com` |
| 前缀     | `chemical-graph-rag/`                |

该前缀下是两件对象：

| 对象                                           | 大小        |
| ---------------------------------------------- | ----------- |
| `chemical-graph-rag/cgr-db-20260915.tar.gz`  | 约 8736 MiB |
| `chemical-graph-rag/cgr-src-20260915.tar.gz` | 约 22.2 MiB |

原文 PDF 的下载是另一条配置：`example/a/config_open.yaml` 的 `oss_download.bucket_key` 为 `ky-products-files`，连接信息在同一文件的 `oss_download` 和旁边的数据库段。密钥留在配置文件里。
