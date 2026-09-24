---
description: "工具循环问答：同一条消息上反复思考、调用 search / read_doc / read_chunk / graph_neighbors，直到作答。配置只读 config.agentic。"
kind: "package-reference"
---

# src/agentic

中文

## Summary

这个包实现 `DHMF.agentic_query`。模型决定每一轮是调用工具还是给出最终答案。工具的召回宽度、上下文上限和开关都在 `agentic` 段，不回退到 `retrieve` 或 `agent`。向量索引和嵌入、重排服务仍用已经构建好的检索模块。优先走 OpenAI tools；接口不支持或 `tool_protocol=json` 时改传 JSON 动作。

## Table of Contents

- [Use this package](#use-this-package)
- [Source map](#source-map)
- [Further Exploration](#further-exploration)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>

## Use this package

```text
respond = graph.agentic_query("……", pretty=False, history=None)
```

一轮的顺序是：组系统提示和用户问题 → 模型回复 → 若有工具调用则执行并写回观察 → 直到出现答案、达到 `max_turns`，或最近一次请求的 prompt token 接近 `max_prompt_tokens - prompt_token_reserve`。后两种情况会强制要一次最终答案（`force_answer_on_max_turns`）。`max_prompt_tokens` 应对齐当前模型的 `max_model_len`，否则强制作答来不及触发，请求会先被服务端拒绝。

工具：

| 名字 | 作用 |
| --- | --- |
| `search` | 调 `retrieve_items`。模式可以是 hybrid、chunk、node、keyword。参数来自 `agentic`，不读 `retrieve` |
| `read_doc` | 按文档 id 读 SQLite 里的正文，长度受 `read_doc_max_chars` 限制 |
| `read_chunk` | 按块阅读顺序读，不是按入库的 `chunk_id` 大小。长度受 `read_chunk_max_chars` 限制 |
| `graph_neighbors` | 同一超边或同一文档上的共现邻居，不是类型化的关系边 |

`enable_note_evidence` 打开时，核实过的短句放进证据槽。旧工具结果按轮次年龄压缩，证据槽里的内容保留。

-----

<a id="source-map"></a>

## Source map

| 文件 | 职责 |
| --- | --- |
| [`__init__.py`](__init__.py) | 导出 `run_agentic_query` |
| [`runner.py`](runner.py) | `DHMF.agentic_query` 的薄入口。建上下文、开可选轨迹日志、格式化 pretty 输出 |
| [`loop.py`](loop.py) | 循环、token 预算、强制作答、工具协议选择 |
| [`tools.py`](tools.py) | 四个工具的参数、执行和返回文本 |
| [`prompts.py`](prompts.py) | OpenAI tools 与 JSON 降级两套系统提示，以及强制作答提示 |

轨迹日志写到 `{working_path}/log/agentic_时间戳.log`，仅当 `log_trace` 为真。它不刷到控制台。

-----

<a id="further-exploration"></a>

## Further Exploration

- [`docs/agentic原理与实现详解.md`](../../docs/agentic原理与实现详解.md) — 轮次、停机条件和证据槽的说明。
- [`src/agent/README.md`](../agent/README.md) — 先规划再执行的那条路径。
- [`src/module/README.md`](../module/README.md) — `search` 底下的三路检索。

<a id="dev-note"></a>

## Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

- 给工具加参数时，同时改 `tools.py` 的 schema 和 JSON 提示。`tool_protocol=auto` 会在两种协议之间降级，只改一边会让降级路径看不懂参数。
- 压缩旧工具结果时不要动证据槽。槽满了丢最旧的一条，而不是把已核实字段截断进工具回包。

</details>
