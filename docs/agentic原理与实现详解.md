# Agentic RAG 原理与实现详解

## 目录

1. [系统架构总览](#1-系统架构总览)
2. [Agentic 的核心原理](#2-agentic-的核心原理)
3. [Function Call 的实现机制](#3-function-call-的实现机制)
4. [完整回答检索流程](#4-完整回答检索流程)
5. [代码逐层详解](#5-代码逐层详解)
6. [两种 Agent 模式对比](#6-两种-agent-模式对比)

---

## 1. 系统架构总览

Chemical-Graph-RAG 是一个面向化工/材料领域产品技术资料的 RAG（检索增强生成）系统。它构建了一套完整的数据流水线：

```
PDF 文档 → 识别(VLM) → 分块 → 实体抽取 → 超图构建 → 向量化 → 检索 → LLM 作答
```

系统提供了**三种问答模式**：

| 模式 | API 端点 | 特点 |
|------|---------|------|
| 单跳检索 | `POST /query` | 一次检索 + 一次生成，最简单 |
| 多跳 Agent | `POST /multihop-query` | LangGraph 编排，LLM 规划检索步骤再执行 |
| **Agentic 工具调用** | `POST /agentic-query` | LLM 边想边调工具，自主决策检索策略 |

本文重点讲解 **Agentic 工具调用模式** 的原理和实现。

---

## 2. Agentic 的核心原理

### 2.1 什么是 Agentic？

Agentic 的核心思想是：**让 LLM 像人一样，边思考边使用工具，自主决定检索策略**。

传统 RAG 的流程是固定的：改写查询 → 向量检索 → 拼接结果 → 生成答案。这种方式的问题是：
- 检索策略是预先写死的，不能根据中间结果动态调整
- 如果第一次检索不理想，无法自动换策略重试
- 无法主动阅读文档原文或跳转关联实体

Agentic 模式解决了这些问题：LLM 在同一个对话上下文中**循环执行 "思考 → 调用工具 → 观察结果 → 再思考"** 的流程，直到收集到足够信息后给出最终答案。

### 2.2 核心循环：Think → Tool → Observe

```
┌──────────────────────────────────────────────────┐
│                  Agentic Loop                     │
│                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │  THINK   │───▶│   TOOL   │───▶│ OBSERVE  │   │
│  │  思考下   │    │  调用    │    │  观察    │   │
│  │  一步    │◀───│  工具    │    │  结果    │   │
│  └──────────┘    └──────────┘    └──────────┘   │
│       │                                        │
│       │ 信息够了                                │
│       ▼                                        │
│  ┌──────────┐                                   │
│  │  ANSWER  │  输出最终答案                      │
│  └──────────┘                                   │
└──────────────────────────────────────────────────┘
```

这个循环在 [src/agentic/loop.py](file:///root/projects/Chemical-Graph-RAG/src/agentic/loop.py) 的 `run_agentic_loop` 函数中实现。

### 2.3 三大工具

Agentic 模式提供了三个工具，LLM 可以在任何一轮对话中按需调用：

| 工具 | 功能 | 使用场景 |
|------|------|---------|
| `search(query, mode)` | 多路检索知识库 | 初始搜索、换关键词重试、换检索模式 |
| `read_doc(doc_id)` | 阅读文档原文 | 核对具体数值/规格/配方 |
| `graph_neighbors(name/node_id/doc_id)` | 查看超图邻居 | 从产品找公司、从公司找其他产品 |

### 2.4 search 的四种检索模式

LLM 需要根据问题类型选择正确的 `search mode`：

| Mode | 检索方式 | 适用场景 |
|------|---------|---------|
| `keyword` | 抽取牌号/CAS/货号 → FTS 精确匹配 | 有具体标识号（R-902, 13463-67-7） |
| `node` | 实体节点向量检索 | 问某个主体（谁生产的、某公司有哪些产品） |
| `chunk` | 正文块语义向量检索 | 问用途/工艺/配方等描述性内容 |
| `hybrid` | 三路并行混合 | 拿不准或既有标识又有规格时 |

---

## 3. Function Call 的实现机制

### 3.1 两种协议：OpenAI Tools 与 JSON 降级

系统支持两种工具调用协议，在 [AgenticConfig](file:///root/projects/Chemical-Graph-RAG/src/utils/config.py#L798-L904) 中通过 `tool_protocol` 配置：

- **`openai`**（默认自动）：使用 OpenAI 兼容的 function calling API，将工具定义作为 `tools` 参数传给 LLM
- **`json`**：LLM 输出 JSON 格式的动作描述，由代码解析后执行

**协议自动降级**：当 `tool_protocol=auto` 时，优先使用 OpenAI tools 协议。如果 API 返回错误指示不支持 tools（如某些 vLLM 部署），会自动降级到 JSON 协议。降级逻辑在 [loop.py 的 `_switch_to_json`](file:///root/projects/Chemical-Graph-RAG/src/agentic/loop.py#L390-L396) 中：

```python
def _switch_to_json(reason: str) -> None:
    nonlocal use_openai, protocol, schemas
    if not use_openai:
        return
    use_openai = False
    protocol = "json"
    schemas = []
    messages[0] = {"role": "system", "content": _system_prompt("json")}
```

### 3.2 OpenAI Tools 模式

#### 3.2.1 工具 Schema 定义

工具的定义在 [src/agentic/tools.py](file:///root/projects/Chemical-Graph-RAG/src/agentic/tools.py) 的 `tool_schemas` 函数中。每个工具定义包含：

- `type: "function"`：标识这是一个可调用函数
- `function.name`：工具名称
- `function.description`：工具用途说明（LLM 据此决定何时调用）
- `function.parameters`：JSON Schema 格式的参数定义

以 `search` 工具为例：

```python
{
    "type": "function",
    "function": {
        "name": "search",
        "description": "在化工产品知识库中检索。返回短摘要和 doc_id / chunk_id。...",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "一句完整问句。正确：『CAS 号 13463-67-7 对应什么产品』..."
                },
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "keyword", "node", "chunk"],
                    "description": "hybrid（默认）=块+节点+抽词精确匹配..."
                }
            },
            "required": ["query"]
        }
    }
}
```

#### 3.2.2 发送给 LLM

在 [loop.py 的 `_chat` 函数](file:///root/projects/Chemical-Graph-RAG/src/agentic/loop.py#L383-L389) 中，tools 模式和普通模式用不同的参数调用 LLM：

```python
def _chat(*, with_tools: bool) -> dict:
    if with_tools and use_openai and schemas:
        return ctx.llm.chat(
            messages,
            dict(cfg.model_args or {}),
            tools=schemas,        # 传入工具定义
            tool_choice="auto",    # 让 LLM 自主决定是否调用工具
        )
    ma = dict(cfg.model_args or {})
    ma.pop("tools", None)
    ma.pop("tool_choice", None)
    return ctx.llm.chat(messages, ma)
```

#### 3.2.3 解析 LLM 的工具调用响应

LLM 返回的响应中包含 `tool_calls` 字段，在 [loop.py 的 `_openai_tool_calls_to_actions`](file:///root/projects/Chemical-Graph-RAG/src/agentic/loop.py#L295-L312) 中解析：

```python
def _openai_tool_calls_to_actions(resp: dict) -> List[dict]:
    actions = []
    thought = str(resp.get("answer") or "").strip()
    for tc in resp.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments") or "{}"
        if isinstance(args, str):
            try:
                args_obj = json.loads(args) if args else {}
            except Exception:
                args_obj = {"_raw": args}
        elif isinstance(args, dict):
            args_obj = args
        actions.append({
            "kind": "tool",
            "name": str(fn.get("name") or ""),
            "arguments": args_obj,
            "thought": thought,
        })
    return actions
```

### 3.3 JSON 降级模式

当 API 不支持 function calling 时，LLM 直接输出 JSON 格式的动作描述。解析逻辑在 [loop.py 的 `parse_json_action`](file:///root/projects/Chemical-Graph-RAG/src/agentic/loop.py#L82-L144) 中：

**工具调用 JSON 格式**：
```json
{"thought": "本轮判断", "tool": "search", "arguments": {"query": "...", "mode": "hybrid"}}
```

**作答 JSON 格式**：
```json
{"thought": "依据哪些证据", "answer": "最终回答全文"}
```

解析函数 `parse_json_action` 的处理逻辑：
1. 先去除 Markdown 代码块标记（` ```json ... ``` `）
2. 尝试直接 `json.loads` 解析
3. 失败则用正则提取 `{...}` 再解析
4. 判断是 `answer` 还是 `tool` 类型的动作
5. 支持 `tool_calls` 数组格式的兼容解析

### 3.4 工具执行与结果返回

工具的实际执行在 [src/agentic/tools.py](file:///root/projects/Chemical-Graph-RAG/src/agentic/tools.py) 的 `ToolContext.execute` 方法中：

```python
def execute(self, name: str, arguments) -> str:
    args = _parse_args(arguments)
    fn = {
        "search": self.search,
        "read_doc": self.read_doc,
        "graph_neighbors": self.graph_neighbors,
    }.get(name)
    if fn is None:
        return json.dumps({"error": f"未知工具 {name!r}", ...})
    try:
        payload = fn(args)
    except Exception as e:
        payload = {"error": f"{name} 失败: {e}"}
    if not isinstance(payload, str):
        payload = json.dumps(payload, ensure_ascii=False, indent=2)
    return payload
```

执行结果以 JSON 字符串返回，追加到 messages 中。在 OpenAI tools 模式下，以 `role: "tool"` 消息追加，并关联 `tool_call_id`；在 JSON 模式下，以 `role: "user"` 消息追加。

---

## 4. 完整回答检索流程

### 4.1 整体流程（以 Agentic 模式为例）

```
用户问题: "CAS 号 13463-67-7 对应什么产品，它的密度是多少？"
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Turn 1: LLM 思考 → 调用 search("CAS 号 13463-67-7 对应什么产品", mode="keyword")  │
│           → 检索模块执行 keyword 模式检索                               │
│           → 返回命中列表（含 doc_id、预览）                              │
├─────────────────────────────────────────────────────────────┤
│  Turn 2: LLM 观察结果 → 调用 read_doc(doc_id=42)                   │
│           → 读取文档全文，核对密度数值                                  │
│           → 返回文档正文                                               │
├─────────────────────────────────────────────────────────────┤
│  Turn 3: LLM 收集足够信息 → 输出最终答案                              │
│           "CAS 13463-67-7 对应金红石型钛白粉 R-902，密度约 4.0 g/cm³..."  │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 检索模块内部流程（retrieve_items）

无论哪种模式，底层检索都走 `retrieve_items` 方法。这是整个系统的检索核心，在 [src/module/retrieve.py](file:///root/projects/Chemical-Graph-RAG/src/module/retrieve.py#L2297) 中实现。

```
                        ┌──────────────┐
                        │  用户查询     │
                        └──────┬───────┘
                               │
                    ┌──────────▼──────────┐
                    │  1. 预计算 & 预加载  │
                    │  (chunks/docs/hyperedges) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  2. 查询改写 (可选)  │
                    │  LLM: 使查询更具体   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  3. 嵌入向量化       │
                    │  Embedding API       │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
  ┌───────▼───────┐   ┌───────▼───────┐   ┌───────▼───────┐
  │ 4a. Chunk 路  │   │ 4b. Node 路   │   │ 4c. Keyword 路│
  │ 向量检索      │   │ 向量检索      │   │ 精确匹配      │
  │ query↔chunk   │   │ query↔node    │   │ LLM抽词→FTS   │
  │ FAISS HNSW    │   │ →映射到chunk  │   │               │
  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  5. 三路结果合并     │
                    │  按 chunk_id 并集    │
                    │  score 取 max       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  6. 文档扩展         │
                    │  头块 + 正文组装     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  7. 终轮 Rerank     │
                    │  Reranker 重排序    │
                    │  截断到 rerank_top_k│
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  8. 切片族扩展(可选) │
                    │  长PDF同源切片全部   │
                    │  并入（按序）        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  返回检索结果列表    │
                    └─────────────────────┘
```

#### 第 1 步：预计算（`_ensure_precompute`）

在 [retrieve.py 的 `_load_precompute`](file:///root/projects/Chemical-Graph-RAG/src/module/retrieve.py#L400-L474) 中，系统将数据库中的 chunks、docs、hyperedges 一次性加载到内存字典中，构建多个索引：

- `chunk_dict`：chunk_id → chunk 的映射
- `doc_dict`：doc_id → doc 的映射
- `chunks_by_doc`：doc_id → 该文档所有 chunk 的映射（按原文序排列）
- `doc_head_chunk`：doc_id → 头块（摘要块）的映射
- `hyperedge_by_doc`：doc_id → 超边（文档摘要）的映射
- `docs_by_family`：长 PDF 切片族的索引

#### 第 2 步：查询改写（可选）

在 [retrieve.py 的 `rewrite_query`](file:///root/projects/Chemical-Graph-RAG/src/module/retrieve.py#L730-L810) 中，使用 LLM 将用户查询改写成更利于向量检索的形式。例如：

```
原查询：「外墙涂料怎么提高耐沾污」
改写后：「外墙乳胶漆提高耐沾污性常用哪些成膜物和助剂」
```

注意：在 Agentic 模式下，默认**关闭**查询改写（`enable_query_rewrite: false`），因为 LLM 已经在思考过程中自行构造了高质量的查询语句。

#### 第 3 步：嵌入向量化

使用 Embedding API 将查询文本转为向量。支持 Qwen3-Embedding 的 `Instruct:` 前缀（`_apply_query_instruct`），仅在查询侧添加，文档侧不加。

#### 第 4a 步：Chunk 向量检索

在 [retrieve.py 的 `_search_chunks_by_query`](file:///root/projects/Chemical-Graph-RAG/src/module/retrieve.py#L1546-L1561) 中：

1. 将查询向量与 chunk 向量库做 FAISS 相似度检索
2. 使用 HNSW 索引（可配置为 L2/IP/Cosine）
3. 距离转为 DPR 相似度：`similarity = 1 - (distance²) / 2`
4. 返回 top-k 个 chunk 命中

#### 第 4b 步：Node 向量检索

在 [retrieve.py 的 `_search_nodes_by_query`](file:///root/projects/Chemical-Graph-RAG/src/module/retrieve.py#L1563-L1599) 中：

1. 查询向量与 node（实体节点）向量库做 FAISS 检索
2. 每个命中的 node 映射回其所属的 chunk
3. 同一 chunk 被多个 node 命中时，取最高相似度

#### 第 4c 步：Keyword 精确匹配

这是最有特色的一路，在 [retrieve.py 的 `_keyword_path_retrieve`](file:///root/projects/Chemical-Graph-RAG/src/module/retrieve.py#L1261-L1430) 中：

1. **LLM 抽取关键词**：使用 LLM 从查询中抽取两类关键词
   - **少数词（minority）**：牌号、CAS 号、货号等低频率高区分度标识
   - **多数词（majority）**：常见字段值（如相对密度、闪点等），可配置开关
2. **FTS5 精确匹配**：在 chunk 表的 FTS5 全文索引中搜索这些关键词
   - 先查少数词，命中数够 `keyword_candidate_k` 则提前收束
   - 否则合并多数词命中
   - 支持 `keyword_max_df` 过滤过于常见的词
3. **文档级首轮 Rerank**：关键词命中先做一轮文档级 Rerank，截断到 `keyword_top_k`
4. FTS 不可用时回退为内存扫描（每块 casefold 后做 `in` 匹配）

#### 第 5 步：三路结果合并

在 [retrieve.py 的 `_union_hits_by_docs`](file:///root/projects/Chemical-Graph-RAG/src/module/retrieve.py#L1432-L1543) 中：

- Chunk 路和 Node 路先合并（`_merge_chunk_hits`）：同 chunk 取 max score，match_type 拼接
- 再与 Keyword 路做并集合并（keyword 只增不减，不删减向量命中）
- 同 chunk 的 `match_type` 合并为 `chunk+node+keyword`

#### 第 6 步：文档扩展

在 `_build_materials` 和 `_emit_doc_passages` 中，将命中结果按文档组织，包含：
- 头块（超边摘要/文档概述）
- 索引块（命中的正文块）
- 支持 `enable_full_body_context` 三种模式：
  - `off`：头块 + 命中索引块
  - `simple`：命中正文时在文首补超边头块
  - `full`：命中文档写入全部正文

#### 第 7 步：终轮 Rerank

使用 Reranker API 对文档级结果做最终重排序，截断到 `rerank_top_k`。

#### 第 8 步：切片族扩展（可选）

长 PDF 入库时被切成多段（如 `foo.pdf`、`foo.pdf_1`、`foo.pdf_2`），如果命中任一切片，自动并入同源全部切片，按 `slice_index` 原序输出。

### 4.3 并行 vs 串行执行

检索支持并行和串行两种模式（通过 `enable_parallel_paths` 配置）：

**并行模式**（默认关闭）：
```
关键词路 ────────────────┐
                         ├─→ 合并
改写+嵌入 → chunk路 ─┤
          → node路  ─┘
```
使用 `ThreadPoolExecutor` 同时启动关键词路和向量路，墙钟 ≈ max(各路耗时)。

**串行模式**（默认）：
```
改写 → 嵌入 → chunk → node → keyword → 合并
```
按顺序执行，峰值内存更低。

---

## 5. 代码逐层详解

### 5.1 入口层：API 和 DHMF

**API 入口**：[api/app.py](file:///root/projects/Chemical-Graph-RAG/api/app.py)

```python
@app.post("/agentic-query")
async def agentic_query(req: AgenticQueryRequest, request: Request):
    graph = _get_graph(request)
    respond = await _run(graph.agentic_query, req.query, pretty=False)
    return _normalize_respond(respond)
```

API 层是薄封装，将请求转发给 `DHMF.agentic_query` 方法。使用 `ThreadPoolExecutor` 在线程池中执行（避免阻塞 asyncio 事件循环）。

**DHMF 层**：[src/DHMF.py](file:///root/projects/Chemical-Graph-RAG/src/DHMF.py#L1029-L1033)

```python
def agentic_query(self, query, pretty=False):
    from .agentic.runner import run_agentic_query
    return run_agentic_query(self, query, pretty=pretty)
```

DHMF 是系统的中央协调类，它持有所有模块的引用（数据库、向量库、检索模块等）。`agentic_query` 是 `run_agentic_query` 的薄封装。

### 5.2 Runner 层：初始化与调度

**`run_agentic_query`**：[src/agentic/runner.py](file:///root/projects/Chemical-Graph-RAG/src/agentic/runner.py#L67-L139)

这个函数负责：

1. **读取配置**：从 `config.agentic` 读取 Agentic 专用配置（与 retrieve/agent 配置完全独立）
2. **构建 LLM**：`build_agentic_llm` 从 config 解析 API key 和 base URL
3. **创建 TraceSink**：如果 `log_trace=true`，创建独立日志文件记录完整交互过程
4. **创建 ToolContext**：持有 DHMF 实例、配置、检索模块引用
5. **创建 AgenticContext**：聚合 LLM、工具上下文、日志器
6. **调用 `run_agentic_loop`**：执行核心循环
7. **异常处理**：捕获异常并返回错误响应
8. **格式化输出**：`pretty=True` 时返回格式化的可读文本

### 5.3 Loop 层：核心循环

**`run_agentic_loop`**：[src/agentic/loop.py](file:///root/projects/Chemical-Graph-RAG/src/agentic/loop.py#L314-L557)

这是整个 Agentic 模式的核心。关键流程：

#### 5.3.1 初始化阶段

```python
messages = [
    {"role": "system", "content": _system_prompt(...)},
    {"role": "user", "content": q},
]
```

- 构建初始 messages，包含 system prompt（工具使用说明）和用户问题
- 根据协议选择不同的 system prompt（`SYSTEM_TOOLS` 或 `SYSTEM_JSON`）

#### 5.3.2 主循环

```python
for turn_i in range(1, max_turns + 1):
```

每一轮：

1. **Token 预算检查**：如果 prompt tokens 超过阈值（`max_prompt_tokens - prompt_token_reserve`），强制结束并输出答案
2. **调用 LLM**：`_chat(with_tools=use_openai)`
3. **错误处理**：如果 LLM 返回错误且看起来是不支持 tools，自动降级到 JSON 协议
4. **解析响应**：
   - OpenAI tools 模式：解析 `tool_calls` 字段
   - JSON 模式：解析文本中的 JSON 对象
5. **判断动作类型**：
   - 如果是 `answer`：直接返回最终答案
   - 如果是 `tool`：执行工具调用
6. **执行工具**：`ctx.tools.execute(name, arguments)`
7. **追加结果到 messages**：将工具返回结果作为新的消息追加
8. **继续下一轮**：LLM 看到新的观察结果后，决定下一步

#### 5.3.3 Token 预算与强制结束

```python
def _over_token_budget(n=None) -> bool:
    if token_threshold <= 0:
        return False
    return int(n if n is not None else last_prompt) >= token_threshold
```

当上下文接近 token 上限时，系统会注入一条强制指令：

```
检索轮次已用完。请根据到目前为止的工具结果，直接给出对用户原问题的最终答案。
```

这确保了即使 LLM 陷入循环，也能在有限轮次内给出答案。

### 5.4 Tools 层：三大工具的实现

#### 5.4.1 search 工具

在 [src/agentic/tools.py](file:///root/projects/Chemical-Graph-RAG/src/agentic/tools.py) 的 `ToolContext.search` 方法中：

1. 解析 `mode` 参数（支持多种别名，如 `mix`、`text`、`entity` 等 → 统一归一化）
2. 根据 mode 构造 `search_retrieve_kwargs`：不同 mode 开启/关闭对应的检索路径
3. 调用 `retrieve_module.retrieve_items()` 执行检索
4. 将结果压缩为 `_compact_hits`（按 doc 去重、取 top hits、截断预览文本）
5. 记录检索耗时和关键词信息
6. 返回结构化的命中列表

**`search_retrieve_kwargs` 的设计**：不同 mode 对应不同的检索参数：

| mode | chunk 路 | node 路 | keyword 路 |
|------|---------|---------|-----------|
| chunk | ✓ (k=30) | ✗ (k=0) | ✗ |
| node | ✗ (k=0) | ✓ (k=30) | ✗ |
| keyword | ✗ (k=0) | ✗ (k=0) | ✓ |
| hybrid | ✓ | ✓ | ✓ |

#### 5.4.2 read_doc 工具

1. 接收 `doc_id` 参数
2. 从预计算的 `chunks_by_doc` 中获取该文档的所有块
3. 按原文顺序排列
4. 检查是否是长 PDF 切片（`sliced=true`），如果是则标注切片信息
5. 附加超边摘要（文档概述）
6. 按 `read_doc_max_chars` 截断
7. 返回完整文档正文

#### 5.4.3 graph_neighbors 工具

1. 接收 `name` / `node_id` / `doc_id` 至少一个参数
2. 从 node 表中查找种子实体
3. 通过超边（`hyperedge_id`）和文档（`doc_id`）展开邻居
4. 去重并限制数量（`neighbors_limit`）
5. 返回种子实体和邻居实体的列表

### 5.5 多跳 Agent 模式（对比参考）

多跳 Agent 模式（`POST /multihop-query`）使用 LangGraph 编排，与 Agentic 模式的设计理念不同：

#### 5.5.1 LangGraph 工作流

在 [src/agent/graph.py](file:///root/projects/Chemical-Graph-RAG/src/agent/graph.py) 中定义：

```
START → plan → execute ⟲ → synthesize → END
```

#### 5.5.2 Plan 节点

在 [src/agent/nodes.py](file:///root/projects/Chemical-Graph-RAG/src/agent/nodes.py) 的 `plan_node` 中：

1. LLM 分析用户问题，拆解为检索步骤图
2. 输出 JSON 格式的步骤列表，包含 `id`、`question`、`depends_on`
3. 规则：**按主体链多跳，不按约束条件多跳**
   - 例如："NFPA 健康1、密度1.10、自燃>400°C 的产品的公司有什么抗氧化剂？"
   - 步骤 1：找满足所有约束的产品
   - 步骤 2：找产品的公司（依赖步骤 1）
   - 步骤 3：找公司的抗氧化剂产品（依赖步骤 2）
4. 自动注入纯 LLM 步（作为对照参考）
5. 多跳时自动注入原问题直检步（用原始问题直接检索一次）

#### 5.5.3 Execute 节点

在 `execute_node` 中：

1. 找出所有就绪步骤（依赖已满足的步骤）
2. 对有依赖的步骤，先调用 LLM 将指代消解为具体查询（如"该公司"→"巴斯夫"）
3. 调用 `QuerySkill.run` 执行检索+生成
4. 依赖图支持并行执行（多个 `depends_on=[]` 的步骤同时执行）

#### 5.5.4 Synthesize 节点

在 `synthesize_node` 中：

1. 汇总所有检索子步骤的结论
2. 对照纯 LLM 的常识草稿和原问题直检结果
3. LLM 综合生成最终答案
4. 规则：子步骤的具体型号和数值为准，常识草稿只用于补全骨架

---

## 6. 两种 Agent 模式对比

| 维度 | Agentic 模式 | 多跳 Agent 模式 |
|------|-------------|----------------|
| 编排方式 | LLM 自主决策，循环调用工具 | LangGraph 预规划，按依赖图执行 |
| 工具使用 | 3 个工具（search/read_doc/graph_neighbors） | 1 个 skill（检索+生成） |
| 检索策略 | 动态：每轮根据结果调整 | 静态：规划阶段确定 |
| 适用场景 | 复杂多变的问题，需要反复试错 | 结构化多跳问题，如实体链追踪 |
| 协议 | OpenAI function calling / JSON | LangGraph StateGraph |
| 配置段 | `config.agentic` | `config.agent` |
| 上下文管理 | 同一 messages 累积 | 每步独立检索+生成 |
| 最大轮次 | `max_turns`（默认 32） | `max_steps`（规划步骤数） |

### 6.1 何时用 Agentic？

- 问题不确定，需要探索性检索
- 需要核对文档原文中的具体数值
- 需要沿超图跳转实体关系
- 检索策略需要根据中间结果动态调整

### 6.2 何时用多跳 Agent？

- 问题结构清晰，有明显的实体链
- 需要多步推理但每步都是独立的检索+生成
- 需要并行执行多个独立子问题
- 需要纯 LLM 常识作为对照参考

---

## 附录：关键文件索引

| 文件 | 功能 |
|------|------|
| [api/app.py](file:///root/projects/Chemical-Graph-RAG/api/app.py) | FastAPI 服务入口，三个问答端点 |
| [src/DHMF.py](file:///root/projects/Chemical-Graph-RAG/src/DHMF.py) | 中央协调类，持有所有模块 |
| [src/agentic/loop.py](file:///root/projects/Chemical-Graph-RAG/src/agentic/loop.py) | Agentic 核心循环 |
| [src/agentic/tools.py](file:///root/projects/Chemical-Graph-RAG/src/agentic/tools.py) | 三大工具的实现 |
| [src/agentic/prompts.py](file:///root/projects/Chemical-Graph-RAG/src/agentic/prompts.py) | Agentic 的系统提示词 |
| [src/agentic/runner.py](file:///root/projects/Chemical-Graph-RAG/src/agentic/runner.py) | Agentic 入口，初始化与调度 |
| [src/module/retrieve.py](file:///root/projects/Chemical-Graph-RAG/src/module/retrieve.py) | 检索核心：三路召回 + 合并 + Rerank |
| [src/agent/graph.py](file:///root/projects/Chemical-Graph-RAG/src/agent/graph.py) | 多跳 Agent 的 LangGraph 工作流 |
| [src/agent/nodes.py](file:///root/projects/Chemical-Graph-RAG/src/agent/nodes.py) | 多跳 Agent 的 graph 节点实现 |
| [src/agent/skill.py](file:///root/projects/Chemical-Graph-RAG/src/agent/skill.py) | 单跳检索+生成的 QuerySkill |
| [src/agent/state.py](file:///root/projects/Chemical-Graph-RAG/src/agent/state.py) | Agent 状态类型定义与工具函数 |
| [src/agent/prompts.py](file:///root/projects/Chemical-Graph-RAG/src/agent/prompts.py) | 多跳 Agent 各阶段提示词 |
| [src/utils/config.py](file:///root/projects/Chemical-Graph-RAG/src/utils/config.py) | 配置模型：AgenticConfig、AgentConfig 等 |
| [src/utils/OpenAIAPI.py](file:///root/projects/Chemical-Graph-RAG/src/utils/OpenAIAPI.py) | LLM/Embedding/Reranker 的统一 API 封装 |