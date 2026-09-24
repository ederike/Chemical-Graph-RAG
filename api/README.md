---
description: "问答网站的 HTTP 进程。负责登录、按用户隔离会话，并把检索、双路问答、多跳规划和工具循环放进同一个线程池。向量索引只在这一个进程里加载一份。"
kind: "package-reference"
---

# api

中文

## Summary

浏览器不直接调用 `DHMF`。`api` 在启动时按 `DHMF_CONFIG` 加载 YAML，构造一个进程内共享的 `DHMF`，再把页面静态文件挂到站点根路径。提问时，请求线程只负责鉴权和写 SSE；真正的检索和生成在名为 `dhmf-api` 的线程池里执行，这样事件循环不会被 FAISS 和模型请求堵住。

账号不能在页面上注册。用户、令牌和会话在 SQLite 里，默认文件是 `chemical-rag-web/data/app.db`。知识库的 `main.db` 和这份账号库不是同一个文件。

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Routes](#routes)
- [Known Limitations](#known-limitations)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>

## Use this package

在仓库根目录启动。端口、CORS 和线程池大小来自 YAML 的 `app_config`（开发配置是 `0.0.0.0:8000`、CORS `*`、线程池 4）。uvicorn 的进程数在 `run()` 里写成 1，不读取 `app_config.workers` 当作进程数。`workers` 只决定线程池能同时跑几路问答。

```text
export DHMF_CONFIG=example/a/config_open.yaml
python -m api
```

配置文件不存在时，`lifespan` 直接抛出 `FileNotFoundError`，进程起不来。文件存在时，启动末尾调用 `pin_retrieve_indexes()`，把 `search_range` 覆盖的分片读进这个进程；退出时 `unpin`。因此不要在同一个进程里边跑 `python -m api` 边跑 `vectorization()`。`GET /health` 在图已加载时返回 `ok: true` 和工作目录；`CGR_WEB_ONLY=1` 时不加载图，健康检查只表示网页试点模式。

账号命令在同一套库上操作：

```text
python -m api.accounts list
python -m api.accounts add 用户名 密码
python -m api.accounts passwd 用户名 新密码
python -m api.accounts disable 用户名
python -m api.accounts enable 用户名
```

`add --admin` 把该用户标成管理员。`passwd` 和 `disable` 会删掉该用户尚未过期的令牌，已打开的页面需要重新登录。空库第一次初始化时会种一个管理员，口令在 `store.py` 的 `_seed_admin` 里，部署后应立刻改掉。

隔离测试不碰开发库：

```text
CGR_WEB_DB=/tmp/cgr-test.db python -m api.test_store
```

调用问答需要先 `POST /auth/login`，正文是 `{"username","password"}`，响应里的 `token` 之后放在 `Authorization: Bearer`。查询正文最长 8000 字。除纯检索外，服务会用最近一轮的 prompt token 粗算历史加当前问题；达到模型上下文减预留值时返回 400，文案要求用户新开对话。

-----

<a id="understand-the-implementation"></a>

## Understand the implementation

启动顺序在 `app.py`：解析 `DHMF_CONFIG`（相对路径相对仓库根）、构造 `Config`、按 `app_config` 建立线程池、在 lifespan 里构造 `DHMF` 并探测 `/v1/models` 得到回答模型的 `max_model_len`。探测失败时上下文上限退回 `agentic.max_prompt_tokens`，再没有则用 131072。这个数只用于拒绝过长会话，不改变模型本身。

`_alias_api_prefix` 给每条已注册的业务路由复制一条 `/api` 前缀。页面写的是 `/api/query`，本机没有 nginx 时和 `/query` 执行同一个函数。复制发生在静态文件挂载之前，所以 `/api/...` 不会被 `web/` 吃掉。`CGR_WEB_ONLY=1` 时不加载 `DHMF`，只能看页面。

一次 `POST /stream` 的过程：

1. 校验令牌，非检索模式检查上下文长度。
2. 开一个 `Queue`。工作线程里 `progress.bind` 把 `emit` 放进这个队列。
3. 按 `mode` 调用 `graph.query`（`dual`）、`graph.agent_query`（`agent`）、`graph.agentic_query`（`agentic`），或其他值时只调 `retrieve_items`。
4. 结束时放入 `{"type":"done","data":...}`，异常放入 `{"type":"error"}`，最后放入 `None`。
5. 异步生成器把队列里的对象写成 SSE，直到遇到 `None`。

检索模式的问句会先经 `expand_retrieve_query`，把追问补全后再检索。生成模式把 `history` 交给 `DHMF`，由各条路径自己决定怎么用。

会话写入发生在页面拿到最终答案之后，由 `POST /conversations/{id}/turns` 完成，而不是由 `/stream` 顺带写库。因此流被浏览器中断时，这一轮不会出现在历史里。`turns` 保存问句、模式、答案、状态、耗时、来源 JSON、完整结果 JSON，以及过程栏用的 `stream_steps_json`。

令牌只存 SHA-256。校验时哈希后查找，并检查 `expires_at` 和用户是否被 `disabled`。密码同样只存摘要。

-----

<a id="routes"></a>

## Routes

| 方法与路径 | 作用 |
| --- | --- |
| `POST /auth/login` | 校验用户名和密码，签发令牌 |
| `POST /auth/logout` | 作废当前令牌 |
| `GET /auth/me` | 返回当前用户 |
| `GET /conversations?q=` | 当前用户的会话，按更新时间，最多 200 条。`q` 是标题子串 |
| `POST /conversations` | 新建会话 |
| `GET /conversations/{id}` | 会话及其回合。不是本人的 id 返回 404 |
| `PATCH /conversations/{id}` | 改标题，并标记为人工标题，之后不再被首问覆盖 |
| `DELETE /conversations/{id}` | 删除会话及其回合 |
| `POST /conversations/{id}/turns` | 追加一轮。正文含问句、模式、答案、来源和过程步骤 |
| `POST /query` | `mode` 默认 `dual_path`，调用 `DHMF.query` |
| `POST /multihop-query` | `DHMF.agent_query` |
| `POST /agentic-query` | `DHMF.agentic_query` |
| `POST /retrieve` | 只要命中列表。可带 `chunk_candidate_k`、`node_candidate_k` |
| `POST /stream` | SSE。`mode` 为 `dual`、`agent`、`agentic` 或检索 |
| `GET /health` | 进程与配置是否可用 |
| `GET /status` | 稍详细的运行信息 |
| `GET /context` | 回答模型名、上下文长度、预留 token、数字从哪来（配置或 `/v1/models`） |

除登录外，上表都要 Bearer 令牌。

-----

<a id="known-limitations"></a>

## Known Limitations

- 线程池里的任务不能被 HTTP 断连取消。浏览器停止等待后，这一轮的模型请求仍会跑完，只是结果不再写入会话。
- `/health` 返回 200 只说明进程、配置和图对象在，并且分片已经 pin。它不探测模型服务和嵌入服务是否可达，那要等一次真实问答。
- 上下文估算用的是上一轮接口报告的 prompt token，不是对历史正文重新做 tokenizer。第一轮没有这个数，不会因为估算被拒绝。
- 多进程部署会让每份进程各加载一份 FAISS。`run()` 因此固定 `workers=1`。

<a id="dev-note"></a>

## Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

- 新路由若要给页面用，直接写在 `/api` 前缀复制之前。不要只注册 `/api/...` 又假设命令行能打到不带前缀的路径，除非你接受两边不一致。
- 过程栏的新步骤在业务代码里 `progress.emit`，不要在路由函数里组装 HTML。
- 账号库和知识库的路径不要混用。测试一律设置 `CGR_WEB_DB`。

</details>
