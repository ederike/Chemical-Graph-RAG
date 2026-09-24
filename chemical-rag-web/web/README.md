---
description: "化工知识超图问答库的静态前端：登录门、会话侧栏、检索/双路/多跳/工具循环四种模式，过程与回答左右分栏。"
kind: "package-reference"
---

# chemical-rag-web/web

中文

## Summary

三个源文件构成整站，没有打包器。`index.html` 提供结构，`styles.css` 提供外观，`app.js` 负责登录、会话和提问。接口都走同主机的 `/api/*`。页面标题是「化工知识超图问答库」。没有账号就不能进入，页面上没有注册入口。

## Table of Contents

- [Use this package](#use-this-package)
- [Source map](#source-map)
- [Further Exploration](#further-exploration)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>

## Use this package

本地先按 [`api/README.md`](../../api/README.md) 启动服务，浏览器打开该服务的根路径。登录后可选四种模式，对应的请求体由 `app.js` 里的 `MODES` 固定：

| 模式 | 路径 | 结果 |
| --- | --- | --- |
| Retrieve 仅检索 | `POST /api/retrieve` | 命中列表，不生成答案 |
| Dual-path 双路问答 | `POST /api/query` | 一次检索后的答案 |
| Agent 多跳规划 | `POST /api/multihop-query` | 先规划再作答 |
| Agentic 工具循环 | `POST /api/agentic-query` | 多轮工具后的答案 |

过程栏走 `POST /api/stream`，用 SSE 追加步骤。最终回答用 Markdown 渲染。静态资源的 `?v=` 用来避开旧缓存；改了 JS 或 CSS 后要换这个参数，并提示使用者强制刷新。

浏览器本地只记这些键：`cgr_token`、`cgr_user`、`cgr_theme`、`cgr_sidebar`、`cgr_split`、`cgr_timeout`。主题默认暗色。侧栏可收起。过程和回答的分栏比例记在 `cgr_split`，窄屏改为上下排列。

-----

<a id="source-map"></a>

## Source map

| 文件 | 职责 |
| --- | --- |
| [`index.html`](index.html) | 登录门、侧栏、模式菜单、过程栏、回答栏、分栏拖动条 |
| [`app.js`](app.js) | 登录与令牌、会话 CRUD、四种模式、SSE、Markdown、分栏和主题 |
| [`styles.css`](styles.css) | 布局、暗色/亮色、窄屏上下分栏 |
| [`robots.txt`](robots.txt) | 拒绝索引。页面 head 里同样写了 `noindex` |

-----

<a id="further-exploration"></a>

## Further Exploration

- [`chemical-rag-web/README.md`](../README.md) — 这个目录和 Docker、账号库的关系。
- [`api/README.md`](../../api/README.md) — `/api/*` 与无前缀路径是同一批处理函数。

<a id="dev-note"></a>

## Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

- Markdown 由 `app.js` 的 `renderMarkdown` 就地解析，没有单独的 marked 依赖。`fillAnswer` 和历史回合都要走它；检索模式的证据仍用转义文本。
- 只有点该轮问题（`.turn-q`）才切换左侧过程。点回答正文不会重绘过程栏。正在看的那一轮再次点击则直接返回。
- 不要在这个目录引入构建工具。Docker 镜像按原文件拷贝。

</details>
