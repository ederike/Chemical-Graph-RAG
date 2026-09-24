---
description: "问答网站的静态页、账号库目录，以及给宿主机 nginx 用的旧配置。Docker 镜像只烤进 web/，不使用这里的 nginx 稿。"
kind: "package-group"
---

# chemical-rag-web

中文

## Summary

这个目录是给浏览器的那一层。`web/` 是唯一的页面源码。账号和会话落在 `data/app.db`，由 `api` 创建和维护，页面不能注册。本目录的 `nginx-chemical-rag.conf` 是宿主机旧稿；Docker 里的站点使用仓库根上的 `docker/nginx.conf`，并把 `web/` 烤进镜像。只改磁盘上的 HTML 而不重建 web 镜像，线上页面不会变。

## Table of Contents

- [Packages](#packages)
- [Related documentation](#related-documentation)
- [Dev Note](#dev-note)

-----

<a id="packages"></a>

## Packages

| 路径 | 职责 |
| --- | --- |
| [`web/`](web/README.md) | 登录、会话列表、过程栏与回答栏。无构建步骤，浏览器直接加载 |
| [`data/`](data/README.md) | 默认账号库 `app.db`。运行时生成，不是页面源码 |
| [`nginx-chemical-rag.conf`](nginx-chemical-rag.conf) | 宿主机 nginx 旧稿，反代 `127.0.0.1:8000`。Docker 部署不要用它 |
| [`apply-auth.sh`](apply-auth.sh) | 旧的 htpasswd 辅助脚本。账号现已改由 `python -m api.accounts` 写入 SQLite |
| [`config.json`](config.json) | 历史本地配置。网页登录不以这份文件为准 |

开发时在仓库根目录执行 `python -m api`。`api` 发现 `web/` 存在就会把它挂到站点根路径，并把 `/api/*` 指到同一套处理函数。

-----

<a id="related-documentation"></a>

## Related documentation

- [`api/README.md`](../api/README.md) — 页面调用的路径、登录和 worker 数。
- [`docs/Docker部署.md`](../docs/Docker部署.md) — 镜像怎么烤进 `web/`。
- [`Dockerfile.web`](../Dockerfile.web) — web 镜像的构建定义。

<a id="dev-note"></a>

## Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

- 生产站点若走 Docker，改页面后需要重建 web 镜像再拉起。只覆盖服务器上的这个目录不会改变容器里的文件。
- 不要把账号库、密码或令牌写进 `web/`。页面只在登录成功后保存令牌到 `localStorage`。
- `config.json` 若含口令，不要在文档或提交说明里复述它。现行开户方式是服务器上的 `python -m api.accounts`。

</details>
