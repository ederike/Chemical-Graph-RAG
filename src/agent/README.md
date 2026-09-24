---
description: "多跳规划问答。模型先交出检索子问题，执行器按依赖完成检索或直答，最后用另一次生成把各步对照成最终答案。配置只读 config.agent。"
kind: "package-reference"
---

# src/agent

中文

## Summary

这个包实现 `DHMF.agent_query`。它解决的问题是：用户的一句话里有多个必须分开查的条件，一次 `retrieve_items` 会把条件揉在同一个向量里。规划模型把原问题拆成带依赖的子问题；没有依赖的步骤先做，依赖上游答案的步骤会先把上游结果写进子问题再检索。

它不自己持有索引。每一次检索都是 `QuerySkill`：调用已经构建好的 `Retrieve.retrieve_items`，再用 `agent` 段的模型根据命中写这一步的答案。规划和汇总也用这一段模型，不用 `retrieve.model_args`。

图是固定的，不由模型决定跳到哪个节点：

```text
START → plan → execute ⟲ → synthesize → END
```

`execute` 每跑一轮，把当前所有依赖已满足的步骤做完。还有未完成步骤就回到 `execute`，否则进入 `synthesize`。

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Source map](#source-map)
- [Known Limitations](#known-limitations)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>

## Use this package

```text
respond = graph.agent_query("……", pretty=False, history=None)
```

`pretty=False` 时字典里至少有这些字段：`status`（1 为成功）、`answer`、`retrieval_sources`、各步结果、检索耗时和 token。`history` 在进入规划之前用来补全指代，规划看到的是补全后的问句。

和这条路径有关的配置都在 `agent`：

| 字段 | 作用 |
| --- | --- |
| `base_url`、`model_args` | 规划、依赖改写、纯 LLM、汇总，以及每一步作答所用的模型 |
| `timeout` | 单次 HTTP 超时，秒 |
| `max_steps` | 规划里检索子步的上限。纯 LLM 步和原问题直检不占这个名额 |
| `enable_query_rewrite` | 为真时，`QuerySkill` 在检索前用 agent 模型再改写一次子问题。当前开发配置是关的 |
| `enable_direct_retrieve` | 仅当规划出的检索子步多于 1 时，再并行用原问题检索并作答一次，结果只交给汇总对照。单跳打开它也不会多检一遍 |
| `use_cache` | 规划、改写和作答是否走 LLM 磁盘缓存 |
| `chunk_candidate_k`、`node_candidate_k` | 非空时只覆盖这一次检索的候选数。空则用 `retrieve` 里的值 |

YAML 里的 `enable_pure_llm` 目前没有被代码读取。规划之后总会插入一步纯 LLM。

页面上的「Agent 多跳规划」对应 `POST /api/multihop-query`，内部就是 `agent_query`。

-----

<a id="understand-the-implementation"></a>

## Understand the implementation

### plan

`plan_node` 用 `PLAN_SYSTEM` / `PLAN_USER` 让模型输出 JSON 计划。`parse_plan` 去掉代码围栏后解析；解析失败时 `normalize_plan` 退回一步，问题就是原问句，保证后面仍能检索。计划被截到 `max_steps`。

然后 `inject_llm_step` 追加一步：`id` 为 `llm`，`kind` 为 `llm`，`depends_on` 为空。这一步不检索，直接用原问题让模型作答，用来和检索答案对照。如果模型自己占用了这个 id，检索步会被改名为 `r-llm`。

检索子步多于 1 且 `enable_direct_retrieve` 为真时，再追加 `kind=direct` 的一步。它调用 `QuerySkill.run(原问题)`，不递归进入 `agent_query`。

### execute

`ready_steps` 选出依赖都已出现在 `results` 里的步骤。一轮里这些步骤依次执行（图的一轮对应一次节点调用，不是每步一个图节点）。没有就绪步骤但仍有未完成步骤时，当成死锁，强制跑剩余步骤，避免图停住。

三种步骤：

- `llm`：不检索。提示词是 `LLM_DIRECT_*`，问题是原问句。
- `direct`：用原问句走一次 `QuerySkill`。
- 其余视为检索。若 `depends_on` 非空，先用 `RESOLVE_*` 把上游答案写进子问题，得到真正拿去检索的句子。改写失败时，把上游文本截到 500 字附在原计划问题后面。然后 `QuerySkill.run`。

`QuerySkill.run` 的内部顺序是：可选改写 → `retrieve_items` → 把命中格式化成上下文 → `agent` 模型作答。返回里保留来源、检索耗时和分阶段计时，供汇总和页面过程栏使用。`progress.emit` 会发出 `plan`、`step`、`synth`，HTTP 的 SSE 把它们推到过程栏。

### synthesize

全部步骤都有结果后，`route_after_execute` 走到 `synthesize_node`。提示词 `SYNTH_*` 同时看到原问题、每个检索步的答案、纯 LLM 的答案，以及可选的原问题直检答案。模型被要求对照这些材料写最终答案，而不是再发明一轮检索。来源列表由 `merge_sources` 从各步合并。

-----

<a id="source-map"></a>

## Source map

| 文件 | 职责 |
| --- | --- |
| [`runner.py`](runner.py) | `run_agent_query`。补全历史、建 `AgentContext`、跑图、组装返回字典和 pretty 文本 |
| [`graph.py`](graph.py) | `StateGraph`：三个节点和 execute 之后的条件边 |
| [`nodes.py`](nodes.py) | `plan_node`、`execute_node`、`synthesize_node`、`route_after_execute` |
| [`skill.py`](skill.py) | `QuerySkill` 与 `build_agent_llm` |
| [`state.py`](state.py) | `PlanStep`、`StepResult`、计划解析、依赖判断、用量相加 |
| [`prompts.py`](prompts.py) | `Agent_PROMPT`：规划、依赖改写、单跳作答、纯 LLM、汇总 |

-----

<a id="known-limitations"></a>

## Known Limitations

- 规划失败时退化为单次检索，不会告诉调用方“计划不可用”。日志里有 `[agent.plan]`。
- 同一轮就绪步骤是顺序执行的。依赖允许并行，实现上没有为这些步骤开线程池。
- 纯 LLM 步不看库。它的答案可能和资料矛盾，汇总提示词负责取舍，代码不自动投票。
- 子问题改写失败时，上游答案只截 500 字拼进检索句，可能丢条件。

<a id="dev-note"></a>

## Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

- 提示词里用 `_fill_prompt` 替换 `{query}` 这类占位符，不要用 `str.format`。检索正文里经常有花括号，`format` 会抛 `KeyError`。`PLAN_USER` 仍用 `.format`，是因为它的模板只有 `query` 和 `max_steps` 两个由代码控制的字段。
- 新增步骤种类时，要同时改 `is_llm_step` / `is_direct_step`、`retrieve_plan` 和 `execute_node` 的分支。漏掉的种类会被当成检索步。
- `enable_thinking` 放在 `agent.model_args`。只有模型名含 `qwen3` 或网关 id 为 `llm-medium` 时，客户端才会把它放进 `chat_template_kwargs`。

</details>
