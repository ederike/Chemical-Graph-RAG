# 从零理解这个 API 服务

你启动服务后，浏览器里看到的是：

```json
{
  "service": "Chemical Graph RAG",
  "docs": "/docs",
  "health": "/health",
  "query": "POST /query",
  "multihop-query": "POST /multihop-query",
  "retrieve": "POST /retrieve"
}
```

**这不是坏了，也不是网站首页没做完。** 你打开的是一个 **API 服务**，不是淘宝那种给人点的网页。这段 JSON 的意思是：服务活着，下面这些地址可以调用。真正给人点的操作台在 **`/docs`**。

本文用这个项目本身，把「网络服务 / API / 接口」讲清楚。不要求你先会前端或 HTTP。

---

## 1. 先记住一件事：你现在没有「网站」，只有「窗口」

以前问问题，你是在 Python 里写：

```python
graph.query('该产品的相对密度是多少？')
```

这要求：会 Python、能 import 这个仓库、电脑上装着整套库。

FastAPI 做的事只有一件：**把这句话改成「发一条网络请求」**。别的程序、别的电脑、浏览器，都可以来问同一套知识库。

可以把它想成餐厅：

| 餐厅 | 这个项目 |
|---|---|
| 后厨（炒菜） | `DHMF`：检索 + 大模型回答 |
| 传菜口 / 窗口 | FastAPI：按固定格式收问题、回话 |
| 菜单 | `http://地址:8000/docs` |
| 食客自己进厨房炒 | 以前的 `python` 脚本 |
| 装修好的大厅（点餐网页） | **目前没有**。要另做前端 |

所以你打开 `http://127.0.0.1:8000/` 时，服务给你的不是大厅装修，而是一张小纸条：「菜单在 `/docs`，点菜走 `/query`」。纸条用 JSON 写成，浏览器就原样显示出来。

**现在请打开这个地址（把端口换成你实际启动时的端口）：**

```text
http://127.0.0.1:8000/docs
```

你会看到 FastAPI 自动生成的 Swagger 页面：接口列表、参数、以及「Try it out」按钮。这才是给你用的界面。

---

## 2. 几个词，用这个服务对上号

### 服务（server）

一台一直开着的程序，等别人来问。你运行的 `uvicorn api.app:app ...` 就是服务。关掉终端，窗口就没了，别人也问不成。

### 地址（URL）

浏览器地址栏里那一串。拆开看：

```text
http://127.0.0.1:8000/docs
│     │          │    │
│     │          │    └─ 路径：这次要干什么
│     │          └─ 端口：这台机器上的几号窗口（启动时写的 8000）
│     └─ 主机：谁提供服务。127.0.0.1 = 这台电脑自己
└─ 协议：http 是明文网页/接口常用的那种
```

同一服务、不同路径，事情不一样：

| 你在浏览器打开 | 实际在干什么 |
|---|---|
| `/` | 返回那张 JSON 目录，告诉你有哪些接口 |
| `/docs` | **交互说明书**（建议从这里用） |
| `/redoc` | 另一份更像文档的说明书 |
| `/health` | 探活：库加载了没有 |
| `/query` | 单跳问答（浏览器直接打开会失败，见下一节） |
| `/multihop-query` | 多跳问答 |
| `/retrieve` | 只要检索到的证据，不要模型作答 |

`127.0.0.1` 只这台电脑能访问。若启动时 `--host 0.0.0.0`，同一局域网里别人可以用你的 IP，例如 `http://192.168.1.10:8000/docs`。

### 接口 / API（Application Programming Interface）

事先约好的「怎么问、怎么答」。本服务的约定是：

- 问：发 JSON，里面至少有 `"query": "你的问题"`
- 答：也回 JSON，里面有 `"answer"`、`"retrieval_sources"` 等

对方不必会 Python，只要会按这个约定发请求。这就是「接口」。

### 请求 和 响应

一次完整对话只有两步：

1. **请求（request）**：你 → 服务。「我要 `/query`，问题是……」
2. **响应（response）**：服务 → 你。「status=1，答案是……，来源 PDF 是……」

你在 `/docs` 里点 Execute，就是发一次请求，下面的 Response 就是响应。

---

## 3. GET 和 POST：为什么有的能直接打开，有的不能

HTTP 规定了几种「怎么问」的动词，这里只用两个。

**GET = 只看，不带大段内容。** 可以收藏、可以在浏览器地址栏直接打开。

- 打开 `http://127.0.0.1:8000/health`
- 打开 `http://127.0.0.1:8000/docs`

**POST = 提交一份数据。** 问题很长，不能全塞进地址栏，要放在请求「身体」（body）里。`/query`、`/multihop-query`、`/retrieve` 都是 POST。

所以：浏览器里直接访问 `http://127.0.0.1:8000/query` 会看到 **Method Not Allowed** 或类似错误。不是接口坏了，是你用 GET 去打了一个只收 POST 的窗口。正确用法是打开 `/docs`，找到 `POST /query`，点 Try it out。

对照：

```text
GET  /health     像「看门口有没有灯」
POST /query      像「把写好的问题纸条递进去」
```

---

## 4. JSON：两边说话用的格式

JSON 就是带引号、冒号、花括号的文本，用来表示「字段 → 值」。你根路径看到的那一坨就是。

提问时发送（请求体）：

```json
{
  "query": "该产品的相对密度是多少？"
}
```

服务回答时返回（响应体，字段会更多）：

```json
{
  "status": 1,
  "answer": "根据文档，相对密度约为 ……",
  "retrieval_sources": ["某产品TDS.pdf"],
  "retrieval_doc_ids": [123],
  "latency_s": 2.5
}
```

读的时候只看名字：

| 字段 | 含义 |
|---|---|
| `status` | `1` 成功，`0` 失败（失败时 `answer` 里常是错误原因） |
| `answer` | 模型写的回答，给人看的正文 |
| `retrieval_sources` | 用到了哪些文件名 |
| `retrieval_doc_ids` | 库里的文档编号 |
| `latency_s` | 这一问花了多少秒 |
| `plan` / `steps` | 只有 `/multihop-query` 才有，多跳每一步做了什么 |

花括号 `{ }` 是一份对象，方括号 `[ ]` 是列表。逗号分隔。这就是你需要会看的全部。

---

## 5. 动手：用 `/docs` 问第一道题

假设服务在 `8000` 端口（不是的话改数字）。

### 第 1 步：打开菜单

浏览器访问：

```text
http://127.0.0.1:8000/docs
```

页面上会列出 `GET /health`、`POST /query`、`POST /multihop-query`、`POST /retrieve`。

### 第 2 步：先探活

1. 点开 `GET /health`
2. 点 **Try it out**
3. 点 **Execute**

成功时响应类似：

```json
{
  "ok": true,
  "config_path": ".../example/a/config_open.yaml",
  "working_path": "example/a"
}
```

`ok: true` 表示知识库已经加载。若这里都失败，先别问问题，检查服务终端有没有报错。

### 第 3 步：单跳问答

1. 点开 `POST /query`
2. **Try it out**
3. Request body 改成：

```json
{
  "query": "该产品的相对密度是多少？",
  "mode": "dual_path"
}
```

`mode` 保持 `dual_path` 即可，这是当前唯一支持的单跳模式。

4. **Execute**
5. 等几秒到一两分钟（要检索 + 调大模型，不是瞬间刷新网页）
6. 往下看 **Response body** 里的 `answer`

Code `200` 表示 HTTP 层成功送到了。还要再看 JSON 里的 `status`：那才是业务成没成功。

### 第 4 步：复杂题用 multihop-query

同样步骤，用 `POST /multihop-query`，body 只需要问题：

```json
{
  "query": "满足 NFPA 评级为健康1、相对密度约1.10、自燃温度高于400°C 的产品，其生产公司还有什么抗氧化剂产品吗？"
}
```

多跳问答可能检索多轮，等的时间更长，有时几分钟。响应里会多 `plan`、`steps`。

### 第 5 步：只看检索到了什么

`POST /retrieve` 不让大模型作答，只返回命中的文档片段。适合检查「到底搜没搜到」，或怀疑答案胡编时看证据。

```json
{
  "query": "相对密度"
}
```

返回的 `items` 是一串证据：`source` 是文件名，`content` 是截断后的正文。

---

## 6. 你根路径看到的那几行，逐句翻译

```json
{
  "service": "Chemical Graph RAG",
  "docs": "/docs",
  "health": "/health",
  "query": "POST /query",
  "multihop-query": "POST /multihop-query",
  "retrieve": "POST /retrieve"
}
```

| 字段 | 人话 |
|---|---|
| `service` | 这个窗口叫什么 |
| `docs` | 说明书路径，请拼在主机后面打开 |
| `health` | 探活路径，浏览器可直接打开 |
| `query` | 单跳问答；必须 POST，不要只在地址栏敲 `/query` |
| `multihop-query` | 多跳问答；必须 POST |
| `retrieve` | 只要证据；必须 POST |

它没有「搜索框」，因为根路径只负责指路。搜索框在 `/docs` 的 Try it out 里。

---

## 7. 选哪个接口？

和以前写 Python 时一样，只是换了窗口：

| 你想做的事 | 以前 | 现在 |
|---|---|---|
| 查一个指标、一个牌号 | `graph.query(...)` | `POST /query` |
| 跨文档、多条件、问「还有什么产品」 | `graph.agent_query(...)` | `POST /multihop-query` |
| 只想看命中了哪些 PDF | `retrieve_items(...)` | `POST /retrieve` |
| 看服务还在不在 | — | `GET /health` |

简单题用 `/query`（快）；绕弯的题用 `/multihop-query`。

---

## 8. 浏览器以外怎么调用（可选）

以后若有前端或别人的程序，他们不会人工点 `/docs`，而是发 HTTP 请求。本质和你在 `/docs` 里点 Execute 相同。

在终端（服务先开着）：

```bash
# 探活（GET）
curl -s http://127.0.0.1:8000/health

# 单跳问答（POST + JSON 身体）
curl -s http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"该产品的相对密度是多少？"}'
```

`-H` 声明身体是 JSON；`-d` 后面是问题。返回的还是第 4 节那种 JSON。

Python 同样：

```python
import httpx

r = httpx.post(
    "http://127.0.0.1:8000/query",
    json={"query": "该产品的相对密度是多少？"},
    timeout=300.0,
)
print(r.json()["answer"])
```

你暂时不会这些也没关系。会用 `/docs` 就已经在用 API 了。

---

## 9. 常见现象

**页面就是那几行 JSON，没有搜索框。**  
你在 `/`。改打开 `/docs`。

**打开 `/query` 报 Method Not Allowed。**  
那是 POST 接口。用 `/docs` 里的 Try it out，不要当普通网页打开。

**Execute 之后转圈很久。**  
正常。背后要跑向量检索、Reranker、大模型。`/multihop-query` 更慢。不要当成网页卡死。

**HTTP 200 但 `status` 是 0。**  
请求送到了，业务失败。读 `answer` 里的错误（模型网关没开、超时等）。

**422 Unprocessable Entity。**  
JSON 不合格，例如没写 `query`、或 `query` 是空字符串。

**404 Not Found。**  
路径写错。没有 `/Query`、没有 `/ask`。只能是代码里登记过的那几个。

**连不上 / Connection refused。**  
服务没启动，或端口不是 8000。看你启动 uvicorn 的终端。

**换一台电脑浏览器打不开 127.0.0.1。**  
`127.0.0.1` 永远是「打开者自己这台机器」。别人要访问，用服务器的局域网 IP，且启动时要 `--host 0.0.0.0`。

---

## 10. 和「做一个网站」差在哪

| | API 服务（你现在有的） | 网站（还没有） |
|---|---|---|
| 给人看的页面 | 只有 `/docs` 这种调试页 | 输入框、按钮、排版后的答案 |
| 数据格式 | JSON | HTML |
| 谁来用 | 程序、或你会按说明书点的人 | 普通用户 |
| 本项目下一步若要做 | 另写前端，去 `POST /query` | 前端仍然调用这些接口 |

FastAPI 已经把「问答能力」暴露成窗口。要做成「打开网页、打字、显示答案」，需要再做一层页面去调用 `/query`。那是另一件事；**不会做网页，也可以一直用 `/docs` 当简易客户端。**

更偏开发的写法、启动参数、返回字段表，见同目录 [`fastapi教程.md`](fastapi教程.md)。
