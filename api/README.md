---
description: "FastAPI 服务：登录、按账号隔离的会话，以及检索、双路问答、多跳规划和工具循环。进程内只加载一份 FAISS。"
kind: "package-reference"
---

# api

中文

## Summary

`api` 把 `DHMF` 包成 HTTP，并托管 `chemical-rag-web/web` 的静态页。问答在线程池里跑，进度经 SSE 推给页面。账号不能在网页上注册，只能用本目录的命令在服务器上创建。uvicorn 的 worker 数固定为 1，避免一个进程里出现多份向量索引。

## Table of Contents

- [Use this package](#use-this-package)
- [Source map](#source-map)
- [Routes](#routes)
- [Further Exploration](#further-exploration)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>

## Use this package

在项目根目录启动。配置路径用环境变量 `DHMF_CONFIG`，默认 `example/a/config_open.yaml`。线程池大小、端口和 CORS 来自该文件的 `app_config`，不另设环境变量。

```text
export DHMF_CONFIG=example/a/config_open.yaml
python -m api
```

账号库默认是 `chemical-rag-web/data/app.db`，可用 `CGR_WEB_DB` 改位置。管理命令：

```text
python -m api.accounts list
python -m api.accounts add 用户名 密码
python -m api.accounts passwd 用户名 新密码
python -m api.accounts disable 用户名
python -m api.accounts enable 用户名
```

`passwd` 会使该用户已签发的登录令牌失效。隔离测试：`CGR_WEB_DB=/tmp/cgr-test.db python -m api.test_store`。

没有 nginx 时，页面请求的 `/api/*` 与去掉前缀的路径指向同一批处理函数。静态文件挂在 `/`，因此 API 路由要先注册。

-----

<a id="source-map"></a>

## Source map

| 文件 | 职责 |
| --- | --- |
| [`__main__.py`](__main__.py) | `python -m api` 入口，调用 `run()` |
| [`__init__.py`](__init__.py) | 包说明 |
| [`app.py`](app.py) | FastAPI 应用、路由、线程池、SSE、静态页挂载 |
| [`store.py`](store.py) | 账号、令牌、会话和回合的 SQLite。密码与令牌只存摘要 |
| [`accounts.py`](accounts.py) | 命令行开户、改密、停用。不提供网页注册 |
| [`test_store.py`](test_store.py) | 账号与会话按用户隔离的测试 |

-----

<a id="routes"></a>

## Routes

除登录外，下列路径都要带 `Authorization: Bearer`。`/stream` 的 `mode` 决定走哪条问答。

| 方法与路径 | 作用 |
| --- | --- |
| `POST /auth/login`、`POST /auth/logout`、`GET /auth/me` | 签发、作废、查看令牌 |
| `GET/POST /conversations`、`GET/PATCH/DELETE /conversations/{id}` | 当前用户自己的会话 |
| `POST /conversations/{id}/turns` | 追加一问一答 |
| `POST /query` | `DHMF.query`，双路检索后生成 |
| `POST /multihop-query` | `DHMF.agent_query` |
| `POST /agentic-query` | `DHMF.agentic_query` |
| `POST /retrieve` | 只返回命中，不生成答案 |
| `POST /stream` | SSE。`mode` 为 `dual`、`agent`、`agentic` 或检索 |
| `GET /health`、`GET /status`、`GET /context` | 进程是否起来、配置是否加载、回答模型的上下文长度 |

查询正文上限 8000 字。非检索模式会估算历史加当前问题的 token，接近模型上下文时拒绝并提示新开对话。

-----

<a id="further-exploration"></a>

## Further Exploration

- [`docs/API入门.md`](../docs/API入门.md) — 调用示例。
- [`chemical-rag-web/README.md`](../chemical-rag-web/README.md) — 页面和账号库文件放在哪。
- [`src/README.md`](../src/README.md) — 三条问答在库里的差别。

<a id="dev-note"></a>

## Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

- `run()` 里 `workers=1` 是有意的。多进程会各自 `faiss.read_index`，内存成倍并且互不同步。
- `CGR_WEB_ONLY=1` 时可以只看页面、不加载图。不要在这个模式下期待 `/query` 能回答。
- 进度事件通过 `progress.bind` 绑在工作线程上。新的长步骤若要出现在过程栏，就在该线程里 `emit`，不要在异步端点里直接改页面状态。

</details>
