---
description: "问答网站的静态页、账号库目录，以及一份不再被 Docker 使用的宿主机 nginx 稿。页面源码只在 web/，账号数据只在 data/。"
kind: "package-group"
---

# chemical-rag-web

中文

## Summary

这个目录给浏览器用。它不实现检索，也不保存向量。`web/` 是三个静态文件加一份 `robots.txt`，由 `api` 进程或 Docker 里的 nginx 原样送出。用户、登录令牌和会话在 `data/app.db`，由 `api/store.py` 创建。知识库仍在 `example/a/DB`（或配置里的 `working_path`）。

Docker 镜像构建时把 `web/` 拷进镜像，并使用仓库根的 `docker/nginx.conf`。本目录的 `nginx-chemical-rag.conf` 是以前放在宿主机上的稿子，根路径和反代端口都和现在的容器不一致，不要把它装进容器。

## Table of Contents

- [Packages](#packages)
- [Understand the implementation](#understand-the-implementation)
- [Related documentation](#related-documentation)
- [Dev Note](#dev-note)

-----

<a id="packages"></a>

## Packages

| 路径 | 职责 |
| --- | --- |
| [`web/`](web/README.md) | 登录门、会话侧栏、四种提问模式、过程栏和回答栏 |
| [`data/`](data/README.md) | 默认的 `app.db`。没有这个文件时，第一次 `python -m api` 或 `python -m api.accounts` 会创建 |
| [`nginx-chemical-rag.conf`](nginx-chemical-rag.conf) | 宿主机旧稿：站点根在 `/var/www/chemical-rag`，反代 `127.0.0.1:8000`。Docker 不读它 |
| [`apply-auth.sh`](apply-auth.sh) | 以前写 htpasswd 的脚本。现在的登录不走 HTTP Basic，脚本不能用来开户 |
| [`config.json`](config.json) | 早期本地口令文件。`api` 登录不读它 |

开发时不需要单独起一个前端服务器：

```text
export DHMF_CONFIG=example/a/config_open.yaml
python -m api
```

浏览器打开 `http://127.0.0.1:8000/`。页面请求 `/api/auth/login` 和 `/api/stream` 时，由同一个进程处理。

-----

<a id="understand-the-implementation"></a>

## Understand the implementation

镜像定义在仓库根的 `Dockerfile.web`。它把 `docker/nginx.conf` 放到容器内的 nginx 配置，把 `chemical-rag-web/web/` 拷到 `/usr/share/nginx/html/`。容器里没有这份目录的 volume。所以在服务器上改检出的 `web/app.js` 不会改变正在跑的容器，必须重新构建镜像再启动。

本地 `python -m api` 则是每次请求都读磁盘上的 `web/`。改完 JS 或 CSS，刷新即可，但浏览器会按 URL 里的 `?v=` 缓存。改了文件内容却没改 `index.html` 里的版本参数时，别人的浏览器可能继续用旧文件。

账号库路径由环境变量 `CGR_WEB_DB` 决定，未设置时就是 `chemical-rag-web/data/app.db`。生产环境的账号是在服务器上用 `python -m api.accounts add` 新建的，不要把开发机的 `app.db` 拷过去覆盖。

-----

<a id="related-documentation"></a>

## Related documentation

- [`api/README.md`](../api/README.md) — 页面依赖的路径、令牌和线程池。
- [`web/README.md`](web/README.md) — 四种模式分别打哪条接口，点击和分栏的行为。
- [`docs/Docker部署.md`](../docs/Docker部署.md) — 镜像和端口。

<a id="dev-note"></a>

## Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

- 不要把 `data/app.db` 当成可以提交的种子数据。空库会自己种管理员，口令在 `api/store.py`，上线后用 `python -m api.accounts passwd` 改掉。
- `config.json` 里如果还有口令，不要在文档或提交说明里复述。它不参与当前登录。
- 只改页面、不改 `api` 时，仍然要确认浏览器请求的路径在 `_alias_api_prefix` 之后能到达。新路径应先在 `api/app.py` 注册。

</details>
