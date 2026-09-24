---
description: "化工知识超图问答库的静态页面。一个 HTML、一份样式、一份脚本完成登录、会话、四种提问方式和过程/回答分栏，不经过打包器。"
kind: "package-reference"
---

# chemical-rag-web/web

中文

## Summary

整站在浏览器里跑，没有 React，也没有构建步骤。`index.html` 写出两块界面：未登录时的门，以及登录后的侧栏、输入框、过程栏和回答栏。`app.js` 在加载时根据 `localStorage` 决定主题和侧栏是否收起，然后把按钮绑到 `/api/*`。`styles.css` 负责暗色默认外观、分栏拖动和窄屏改为上下排列。

页面不做检索。它只把用户的问题交给 `api`，把 SSE 里的步骤画出来，并把最终答案从少量 Markdown 转成 HTML。

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Source map](#source-map)
- [Known Limitations](#known-limitations)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>

## Use this package

先按 [`api/README.md`](../../api/README.md) 启动服务，用已有账号在门页登录。登录成功后令牌放在 `localStorage` 的 `cgr_token`，之后每个请求带 `Authorization: Bearer`。401 会清掉令牌并回到门页，提示登录已失效。

输入框上方的模式决定打哪条接口。定义在 `app.js` 的 `MODES`：

| 界面文字 | `mode` 键 | 非流式路径 | 服务端行为 |
| --- | --- | --- | --- |
| Retrieve 仅检索 | `retrieve` | `POST /api/retrieve` | 只返回命中，回答栏用纯文本 |
| Dual-path 双路问答 | `dual` | `POST /api/query` | 一次检索后生成 |
| Agent 多跳规划 | `agent` | `POST /api/multihop-query` | 规划、执行、汇总 |
| Agentic 工具循环 | `agentic` | `POST /api/agentic-query` | 多轮工具 |

实际提问优先走 `POST /api/stream`，正文是 `{query, mode, history}`。`history` 来自当前会话已经显示的轮次，供服务端补全“它的闪点呢”这类追问。若服务返回 404（前面没有挂 `/api` 前缀的旧进程），脚本会退回 `MODES` 里的非流式路径再试一次。

过程栏在收到 `type=step` 时追加一张卡片，并滚到过程栏底部。`type=done` 的 `data` 交给 `fillAnswer`。`type=error` 变成异常，用 `window.alert` 显示，不写进回答栏。回答生成成功后，脚本再 `POST /api/conversations/{id}/turns` 把这一轮存进账号库。流被中途关掉时，这一轮不会出现在左侧会话里。

左侧会话列表来自 `GET /api/conversations`。点一条会 `GET /api/conversations/{id}` 并画出历史回合。点某一轮的问题文字（`.turn-q`）才把过程栏换成那一轮保存的 `stream_steps`。点回答正文不会切换过程；已经在看的那一轮再点一次问题，函数直接返回。

分栏拖动条把过程栏和回答栏的宽度比写进 `cgr_split`。双击恢复默认。视口窄于样式表里的断点时，两栏改为上下排列，比例键仍然生效。主题键是 `cgr_theme`（只认 `light`，否则暗色）。侧栏收起是 `cgr_sidebar=collapsed`。等待上限是 `cgr_timeout`。

-----

<a id="understand-the-implementation"></a>

## Understand the implementation

脚本是一个立即执行函数，没有全局导出。DOM 引用在顶部用 `getElementById` 取一次。请求集中在 `apiPost` 和流式读取函数：两者都带上令牌，超时用 `AbortController`。超时、用户停止和登出在 `catch` 里分成不同的错误文字。

回答 HTML 由 `renderMarkdown` 生成，不依赖 marked 或 DOMPurify。它先摘出代码围栏，再处理标题、列表、表格和段落，行内处理链接、粗体、删除线和行内代码。输入在替换标记之前会做 HTML 转义，所以模型输出里的 `<script>` 不会变成节点。检索模式不走这套解析，命中卡片用 `esc` 加截断，避免把半截表格渲染坏。

历史回合由 `renderThread` 按 `turn_index` 画成 `.turn-block`。当前正在看的回合带 `on`。过程栏的回放读的是该回合保存的 `stream_steps_json`，不是重新请求模型。Agent 的子步和 Agentic 的工具轮在过程栏里用各自的卡片模板，字段来自当时存下来的 JSON。

新对话会清空 `conversationId` 和已显示的回合，但不会删除服务器上的旧会话。标题默认取第一问的摘要；用户在界面上改过标题后，服务端的 `title_is_manual` 为 1，后续回合不再覆盖标题。

`index.html` 在样式表之前用一段同步脚本读 `cgr_theme` 和 `cgr_sidebar`，避免先画出暗色再跳成亮色。`robots.txt` 和页面上的 `noindex` 一起拒绝搜索引擎收录。

-----

<a id="source-map"></a>

## Source map

| 文件 | 职责 |
| --- | --- |
| [`index.html`](index.html) | 门页、侧栏、模式菜单、分栏、输入框。用 `?v=` 引用 CSS |
| [`app.js`](app.js) | 登录、四种模式、SSE、会话、Markdown、分栏和主题 |
| [`styles.css`](styles.css) | 布局、两套主题、窄屏、过程卡片和回答排版 |
| [`robots.txt`](robots.txt) | `Disallow: /` |

-----

<a id="known-limitations"></a>

## Known Limitations

- `renderMarkdown` 不是完整 CommonMark。脚注、嵌套列表和 HTML 表格不会按规范渲染。模型若输出复杂表，回答栏可能只看到转义后的原文或简化后的表。
- 过程栏只保留这次 SSE 推过来的步骤，以及后来写进 `turns.stream_steps_json` 的副本。服务端日志里更细的轨迹不会自动出现。
- 令牌在 `localStorage`。同一浏览器里换账号会覆盖 `cgr_token`。这不是多账号同时在线的设计。

<a id="dev-note"></a>

## Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

- 改 `app.js` 或 `styles.css` 后，同时改 `index.html` 里对应的 `?v=`。Docker 镜像还要重建，磁盘上的文件不会进已经烤好的镜像。
- `fillAnswer` 和 `renderThread` 都要走 `renderMarkdown`。只改其中一处，历史回合和刚生成的答案会一个渲染、一个显示星号。
- 检索命中保持 `esc`。不要为了“统一成 Markdown”把未清洗的资料片段直接送进 `innerHTML`。

</details>
