# Chemical-Graph-RAG

把化工技术资料（TDS、MSDS、专利等 PDF）建成可检索的库，并在这个库上回答问题。原文、总结、文本块和实体进 SQLite，块和实体的向量进 FAISS。问答只读已经建好的库，不在一次提问里重建索引。

调用方使用 `src` 里的 `DHMF`。三条问答路径各管一种问法：

- `query`：检索一次然后生成。问句已经指向某份资料里的一段事实时用。
- `agent_query`：先把问题拆开，再逐个检索，最后汇总。一句话里有多个必须分别查证的条件时用。
- `agentic_query`：由模型决定下一轮是再搜、读原文、看同文档实体，还是作答。第一次检索不够时用。

浏览器不直接调用 `DHMF`。`api` 在一个进程里加载配置和向量索引，同时提供登录、会话和页面。

## 本地运行

配置和知识库不在 Git 里。开发机需要自己的 `example/a/config_open.yaml`，以及该配置 `working_path` 下已经建好的库。

在仓库根目录：

```bash
pip install -r requirements.txt
export DHMF_CONFIG=example/a/config_open.yaml
python -m api
```

浏览器打开 `http://127.0.0.1:8000/`。页面、登录和问答由同一个进程处理。`app_config` 里的端口以配置文件为准，开发配置是 `8000`。

只看页面、不把向量索引读进内存：

```bash
CGR_WEB_ONLY=1 python -m api
```

这种模式下可以登录和看会话，提问会提示知识库没有加载。

账号不能在网页上注册。在仓库根目录用命令行开户：

```bash
python -m api.accounts list
python -m api.accounts add 用户名 密码
python -m api.accounts passwd 用户名 新密码
python -m api.accounts disable 用户名
python -m api.accounts enable 用户名
```

空的账号库第一次启动时会写入一名管理员。部署后立刻用 `passwd` 改掉口令。账号库默认是 `chemical-rag-web/data/app.db`，和知识库的 `main.db` 不是同一个文件。

建库入口是 `main.py`（`example/a`）和 `main2.py`（`example/b`）。识别、总结、分块、抽取、建节点、向量化这几步目前是注释掉的，避免对已经建好的库重跑。要补某一步，打开对应调用；要整段重做，先调用该步的清空方法。顺序和每一步读什么表，见 `src/README.md`。

## Docker

镜像里只有代码和运行环境。`example/a`（含库）和 `cache` 留在宿主机上，由 volume 挂进去。

```bash
cp docker/env.example .env
docker compose up -d --build
```

`.env` 里填写 `AUTH_PASSWORD`。网站映射到宿主机 `8060`，API 只在 Compose 网络内的 `8000`。改页面或 Python 后需要重新构建镜像；本地 `python -m api` 则直接读磁盘上的 `chemical-rag-web/web/`。

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/` | 建库和三条问答路径 |
| `api/` | 网站所用的 HTTP 进程、登录和会话 |
| `chemical-rag-web/` | 浏览器页面和账号库目录 |
| `benchmark/` | 各套评测的运行代码 |
| `scripts/` | 延迟抽样等辅助脚本 |
| `docker/` | nginx 配置和 Compose 用的环境变量示例 |
| `main.py` / `main2.py` | 命令行建库入口 |

各目录里的 README 写该目录自己的接口和约束。

## 不在本分支里的材料

`docs/`、`results/` 和 `example/` 被 `.gitignore` 排除。部署说明、工作汇报、评测结果，以及一份去掉了密钥的示例配置，在 GitHub 的 [`extra`](https://github.com/ederike/Chemical-Graph-RAG/tree/extra) 分支，根目录有那一分支的 `README.md`。

本分支不收录知识库、原文 PDF 和带密钥的配置。
