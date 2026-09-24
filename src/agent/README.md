---
description: "多跳规划问答：先把问题拆成检索步和可选的纯 LLM 步，再执行，最后对照汇总。配置只读 config.agent。"
kind: "package-reference"
---

# src/agent

中文

## Summary

这个包实现 `DHMF.agent_query`。它不自己打开数据库。检索通过 `QuerySkill` 回到已构建的检索模块，LLM 用 `agent` 配置段里的地址和模型。单跳是一次检索作答加一次纯 LLM 作答；多跳才按依赖执行多个检索子步。对外不要调用图里的节点函数。

## Table of Contents

- [Use this package](#use-this-package)
- [Source map](#source-map)
- [Further Exploration](#further-exploration)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>

## Use this package

```text
respond = graph.agent_query("……", pretty=False, history=None)
```

`pretty=True` 时返回可打印的字符串，否则返回带答案、来源和用量的字典。`history` 只用于把追问补全成独立问题，不改变检索索引。开关在 `config.agent`：`max_steps`、`enable_pure_llm`、`enable_direct_retrieve`、`enable_query_rewrite`。这里的 `model_args` 不读取 `retrieve.model_args`。

图的形状固定为 `START → plan → execute ⟲ → synthesize → END`。`plan` 产出步骤后，无依赖的纯 LLM 步与检索步并行。`enable_direct_retrieve` 只在多跳时额外用原问题检索一次，结果只交给汇总对照；单跳本身就是原问题，打开也不会再检一遍。

-----

<a id="source-map"></a>

## Source map

| 文件 | 职责 |
| --- | --- |
| [`__init__.py`](__init__.py) | 导出 `run_agent_query` 与提示词表 |
| [`runner.py`](runner.py) | `DHMF.agent_query` 调用的入口。组状态、跑图、整理来源和 token |
| [`graph.py`](graph.py) | LangGraph：plan、execute、synthesize 三条边 |
| [`nodes.py`](nodes.py) | 三个节点的具体动作，以及 execute 之后走循环还是汇总 |
| [`skill.py`](skill.py) | `QuerySkill`：一次检索加一次生成。不嵌套 `agent_query` |
| [`state.py`](state.py) | 计划步、步骤结果、用量合并、计划 JSON 解析 |
| [`prompts.py`](prompts.py) | 规划、改写、单跳作答、纯 LLM、汇总的提示词 |

-----

<a id="further-exploration"></a>

## Further Exploration

- [`src/agentic/README.md`](../agentic/README.md) — 另一套问答。模型自己选工具，而不是先交出完整计划。
- [`src/module/README.md`](../module/README.md) — `QuerySkill` 使用的检索阶段。
- [`src/README.md`](../README.md) — 三条入口怎么选。

<a id="dev-note"></a>

## Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

- 计划从模型文本里解析 JSON。解析失败应沿用 `state.parse_plan` 的降级，不要在节点里再写一套分词。
- `enable_thinking` 写在 `agent.model_args` 里。模型 id 含 `qwen3`，或是网关名 `llm-medium` 时，客户端才会把它放进 `chat_template_kwargs`。

</details>
