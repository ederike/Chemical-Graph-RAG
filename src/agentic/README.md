---
description: "工具循环问答。模型在同一条消息上决定下一轮是 search、read_doc、read_chunk、graph_neighbors 还是给出答案。循环预算、工具宽度和模型都只读 config.agentic。"
kind: "package-reference"
---

# src/agentic

中文

## Summary

`DHMF.agentic_query` 把一次提问做成多轮对话，而不是事先写死检索次数。模型每轮可以调用一个或多个工具，看到工具返回的原文片段后再决定下一步。适合“先搜到产品，再核对某项限值，发现不是这份资料就换一个词”这类问题。若问题一轮就能答，它也可以不调用工具，直接给出答案。

这条路径不读取 `retrieve` 或 `agent` 的模型、轮次和 top-k。`search` 内部仍调用 `Retrieve.retrieve_items`，因此用的是同一套 FAISS 和 FTS，只是本次的候选数、重排宽度来自 `agentic` 的覆盖参数。

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Tools](#tools)
- [Source map](#source-map)
- [Known Limitations](#known-limitations)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>

## Use this package

```text
respond = graph.agentic_query("……", pretty=False, history=None)
```

成功时 `status` 为 1，`answer` 是最终正文。另外有 `turns`（每一轮是思考、工具还是答案）、`last_prompt_tokens`、`max_prompt_tokens`、`prompt_token_threshold`，以及实际使用的协议 `openai` 或 `json`。失败时 `status` 为 0，`answer` 里是原因，例如超过轮次仍未作答。

开发配置里和停机、宽度有关的字段：

| 字段 | 当前值的含义 |
| --- | --- |
| `max_turns` | 32。循环到这一轮仍没有最终答案，就进入强制作答 |
| `max_prompt_tokens` | 101072，对齐 `llm-medium` 的 `max_model_len`。0 表示不按 token 停 |
| `prompt_token_reserve` | 4096。阈值是 `max_prompt_tokens - prompt_token_reserve`。最近一次请求的 prompt token 达到阈值就强制作答，给生成留出位置 |
| `force_answer_on_max_turns` | 真。轮次用尽时再要一次不带工具的答案。为假则直接失败 |
| `tool_protocol` | `auto`。先用 OpenAI tools；接口报不支持再把系统提示换成 JSON，并在后续轮次只解析 JSON |
| `search_max_hits` | 8。一次 `search` 回给模型的命中条数 |
| `search_preview_chars` | 600。每条命中放进对话的预览长度 |
| `read_doc_max_chars` | 16000 |
| `read_chunk_max_chars` | 4000 |
| `neighbors_limit` | 20 |
| `log_trace` | 假。为真时把思考和工具原文写入 `{working_path}/log/agentic_时间戳.log`，不打印到控制台 |

四个工具可以单独关掉：`enable_search`、`enable_read_doc`、`enable_read_chunk`、`enable_graph_neighbors`。关掉的工具不会出现在 schema 里，模型调用它会被拒绝。

`enable_note_evidence` 为真时，模型可以调用证据槽工具，把已经核对过的短句记下来。槽的容量是 `evidence_slot_max`（当前 10）。旧的工具结果会被压缩，槽里的句子保留。

页面上的「Agentic 工具循环」是 `POST /api/agentic-query`。过程栏里的每一轮来自 `progress.emit`，不是再请求一次。

-----

<a id="understand-the-implementation"></a>

## Understand the implementation

`run_agentic_loop` 维护一条 `messages`。开头是系统提示和用户问题。若证据槽开着，另有一条固定的证据消息，每轮用 `_upsert_evidence_message` 原地更新，不追加重复副本。

每一轮开始前做两件事：

1. `prune_old_tool_results`。最近 `prune_keep_last_n_turns`（16）轮的工具结果保留全文。更早且长于 `prune_soft_trim_chars`（4000）的，留头部 `prune_soft_trim_head` 和尾部 `prune_soft_trim_tail`。年龄达到 `prune_hard_clear_age_turns`（28）的只留占位。证据槽不参与这次裁剪。
2. 若不是第一轮，且最近一次 prompt token 已经达到阈值，立刻 `_force_final`，不再给工具。

然后 `_chat`。协议仍是 OpenAI 且本轮允许工具时，请求带上 `tools` 和 `tool_choice=auto`。否则普通对话。若服务端拒绝 tools，`_switch_to_json` 替换系统提示，之后模型必须输出 JSON：要么 `{"kind":"answer","answer":"..."}`，要么带工具名和参数。

工具执行结果以工具消息写回 `messages`，并记进 `turns`。模型给出最终答案时结束，`progress.emit("answer", ...)`。没有答案又没到停机条件就进入下一轮。

强制作答会先再压缩一次工具结果，追加 `FORCE_ANSWER` 和原因（轮次用尽或 token 超限），然后一次不带工具的生成。若 `force_answer_on_max_turns` 为假且原因是轮次用尽，返回失败，不调用模型。

token 计数优先用接口返回的 `prompt_tokens`。压缩发生在请求之前，压缩后用 `estimate_prompt_tokens` 估一个数，避免沿用压缩前的过大值。

-----

<a id="tools"></a>

## Tools

| 工具 | 模型传入 | 代码实际做的事 |
| --- | --- | --- |
| `search` | `query`，以及 `mode`：`hybrid`、`chunk`、`node`、`keyword` | `retrieve_items`。候选数、关键词开关、重排条数用这次调用的覆盖值，来自 `agentic`，调用结束即还原 |
| `read_doc` | `doc_id` | 从 `doc` 读识别正文，截到 `read_doc_max_chars`。返回里带切片位置，方便模型知道这是长 PDF 的第几段 |
| `read_chunk` | 文档和块的定位 | 按阅读顺序（头块、`body_1`…）取块，不是按 `chunk.id` 的大小。截到 `read_chunk_max_chars` |
| `graph_neighbors` | `name`、`node_id`、`doc_id` 至少给一个 | 先找到种子节点。按名字精确查找，没有再 `instr`。邻居是同一 `hyperedge_id` 下的其他节点，不够时再补同一 `doc_id` 下的节点。最多 `neighbors_limit` 条，展开时每类 id 最多看 8 个 |
| 证据槽 | 短句字段 | `note_evidence`。满了丢最旧的一条 |

`graph_neighbors` 没有边类型，也没有“跳几跳”的参数。它只回答“和这个实体写在同一份资料总结下的还有谁”。

-----

<a id="source-map"></a>

## Source map

| 文件 | 职责 |
| --- | --- |
| [`runner.py`](runner.py) | `run_agentic_query`。建 `AgenticContext` 和 `ToolContext`，可选打开轨迹文件，格式化 pretty 输出 |
| [`loop.py`](loop.py) | `run_agentic_loop`、协议降级、压缩、强制作答、token 阈值 |
| [`tools.py`](tools.py) | 工具 schema、四个工具的执行、证据槽、检索参数的临时覆盖 |
| [`prompts.py`](prompts.py) | OpenAI 与 JSON 两套系统提示，以及 `FORCE_ANSWER` |

-----

<a id="known-limitations"></a>

## Known Limitations

- `max_prompt_tokens` 若大于模型的 `max_model_len`，请求会先被服务端以超长拒绝，强制作答不会发生。换模型后要改这个数。
- JSON 降级发生在一轮失败之后。这一轮已经失败的调用不会自动重放，下一轮才用新协议。
- `read_doc` 返回的是识别正文，不是 PDF 原页。识别错了的数字，工具循环无法靠再读一次自行改正。
- 邻居按超边和文档扩展，同名实体若分布在多份文档，精确名查找会把这些种子都带上，直到 `neighbors_limit`。

<a id="dev-note"></a>

## Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

- 给工具加参数时改三处：`tools.py` 的 schema、JSON 系统提示里的字段说明、执行函数里读取 `args` 的代码。`tool_protocol=auto` 会在两种协议之间切换，只改 schema 时 JSON 路径仍看不懂新字段。
- 压缩函数必须跳过证据槽消息。用角色或 `EVIDENCE_SLOT_MARK` 识别，不要按“最旧的一条工具消息”一刀切。
- 轨迹日志可能含资料原文。`log_trace` 默认关。打开后日志在 `working_path/log`，不要把这个目录提交进仓库。

</details>
