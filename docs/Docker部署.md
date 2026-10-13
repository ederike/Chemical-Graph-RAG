# 用 Docker 把项目打到云服务器

本文教你：**镜像里只装代码和 Python 环境，知识库另外拷到服务器再挂进去**。不要试图把整个 130 GB 仓库 `COPY` 进 Dockerfile，镜像会打爆，也传不动。

硬件仍按 [部署需求](部署需求.md)：内存 **≥ 32 GB**、SSD、能访问推理机。不用 Docker 的装法见 [部署文档](部署文档.md)。

---

## 1. 先搞清三块东西

| 是什么 | 体积 | 怎么走 |
|---|---|---|
| **API 镜像** `chemical-graph-rag-api` | 大约 1–2 GB | Dockerfile 构建，`docker save` 带走或在云上构建 |
| **网站镜像** `chemical-graph-rag-web` | 大约几十 MB | 登录页 + nginx |
| **数据** `example/a`（尤其 `DB/`）和可选 `cache/` | `DB` 约 13.5 GB；整项目可达 100 GB+ | **rsync / 硬盘拷**，用 volume 挂进容器 |

容器起来之后：

```text
浏览器 :80
    → web 容器 (nginx + 登录页)
        → api 容器 :8000  (uvicorn workers=1)
            → 推理机 113.108.154.212
```

宿主机 **不要再开** `autossh -R 8000`，也不要再在宿主机跑一份 uvicorn。

---

## 2. 开发机准备

需要已安装 Docker 和 Compose v2：

```bash
docker version
docker compose version
```

在**项目根目录**（有 `Dockerfile`、`docker-compose.yml` 的那一层）：

```bash
cd /root/projects/Chemical-Graph-RAG
cp docker/env.example .env
# 用编辑器把 AUTH_PASSWORD 改成网站登录密码
```

`.env` 示例：

```text
AUTH_USER=admin
AUTH_PASSWORD=你的登录密码
```

确认数据还在（问答最少要这些）：

```bash
ls example/a/config_open.yaml
ls example/a/DB/main.db
ls example/a/DB/vdb/node.shards | head
```

---

## 3. 在开发机构建镜像

```bash
cd /root/projects/Chemical-Graph-RAG
docker compose build
```

第一次会拉 `python:3.13-slim`、装 `requirements.txt`，要一段时间。成功后：

```bash
docker images | grep chemical-graph-rag
```

应看到：

- `chemical-graph-rag-api:latest`
- `chemical-graph-rag-web:latest`

本地冒烟（可选，机器内存不够就跳过，直接打包装走）：

```bash
docker compose up -d
docker compose ps
curl -sS http://127.0.0.1/api/health          # 期望 401
curl -sS -u 'admin:你的密码' http://127.0.0.1/api/health
# 期望 200 且 "ok": true
docker compose down
```

API 第一次起来要 pin 索引，大约 1 分钟内 `/health` 才会 200。`web` 会等 API 健康检查通过再起来。

---

## 4. 把镜像和数据弄到云服务器

云服务器先装 Docker（Ubuntu 示例）：

```bash
curl -fsSL https://get.docker.com | sh
systemctl enable --now docker
docker compose version || apt-get install -y docker-compose-plugin
```

内存确认：`free -h` 必须 ≥ 32 GB。

下面两条路选一条。

### 路线 A：开发机打好镜像再传（推荐，云上不必再 pip）

在开发机：

```bash
cd /root/projects/Chemical-Graph-RAG
docker save chemical-graph-rag-api:latest chemical-graph-rag-web:latest \
  | gzip > /tmp/cgr-images.tar.gz
ls -lh /tmp/cgr-images.tar.gz
```

传到云（把 `SERVER` 换成 `root@云服务器IP`）：

```bash
SERVER=root@<云服务器IP>
scp /tmp/cgr-images.tar.gz "$SERVER:/tmp/"
scp docker-compose.yml docker/env.example "$SERVER:/tmp/"   # 下面会放到项目目录
```

数据**另外传**，不要塞进镜像：

```bash
# 整项目数据（含 PDF、cache，体积很大）
rsync -aH --info=progress2 example/a/ "$SERVER:/opt/chemical-graph-rag/example/a/"
rsync -aH --info=progress2 cache/     "$SERVER:/opt/chemical-graph-rag/cache/"

# 若只要问答、不重建索引，可以只传库：
# rsync -aH --info=progress2 example/a/config_open.yaml "$SERVER:/opt/chemical-graph-rag/example/a/"
# rsync -aH --info=progress2 example/a/DB/ "$SERVER:/opt/chemical-graph-rag/example/a/DB/"
```

在云服务器：

```bash
mkdir -p /opt/chemical-graph-rag
cd /opt/chemical-graph-rag
gunzip -c /tmp/cgr-images.tar.gz | docker load
# 把 compose 放到这里
mv /tmp/docker-compose.yml ./
cp /tmp/env.example .env
# 编辑 .env 填 AUTH_PASSWORD
```

`docker-compose.yml` 里的 `./example/a`、`./cache` 是相对 compose 文件的路径，所以数据必须在 `/opt/chemical-graph-rag/example/a` 和 `cache`。

### 路线 B：代码拷到云上再构建

适合开发机 Docker 不好用、云上带宽还能拉 Python 包时。

```bash
SERVER=root@<云服务器IP>
rsync -aH --info=progress2 \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude 'cache/' \
  --exclude 'example/a/doc/' \
  --exclude 'example/a/doc_*/' \
  --exclude 'example/a/log/' \
  ./ "$SERVER:/opt/chemical-graph-rag/"

# 数据照样要单独保证 example/a/DB 在服务器上
rsync -aH --info=progress2 example/a/DB/ "$SERVER:/opt/chemical-graph-rag/example/a/DB/"
```

云上：

```bash
cd /opt/chemical-graph-rag
cp docker/env.example .env
# 编辑 .env
docker compose build
```

---

## 5. 在云上启动

```bash
cd /opt/chemical-graph-rag
# 确认
test -f example/a/config_open.yaml && test -f example/a/DB/main.db && echo data_ok
grep AUTH_PASSWORD .env

docker compose up -d
docker compose ps
docker compose logs -f api
```

日志里要看到 `Application startup complete` 和 `Uvicorn running`。然后：

```bash
curl -sS -o /dev/null -w "%{http_code}\n" http://127.0.0.1/api/health
# 401 = nginx 鉴权已开

curl -sS -u 'admin:你的密码' http://127.0.0.1/api/health
# 200 + ok:true

curl -sS http://127.0.0.1/ | grep '化工知识超图问答库'
```

浏览器打开 `http://云服务器IP/`，用 `.env` 里的用户名密码登录提问。

推理机必须从**容器里**能访问。若健康检查通、提问失败：

```bash
docker compose exec api curl -sS -m 8 http://113.108.154.212:8030/v1/models | head
docker compose exec api curl -sS -m 8 http://113.108.154.212:9998/v1/models | head
```

---

## 6. 日常命令

```bash
cd /opt/chemical-graph-rag

docker compose logs -f api          # API 日志
docker compose logs -f web          # nginx
docker compose restart api          # 改了 example/a/config_open.yaml 后
docker compose down                 # 停
docker compose up -d                # 再起

# 改登录密码：改 .env 的 AUTH_PASSWORD 后
docker compose up -d web --force-recreate
```

更新代码：重新 `docker compose build api`（或 `docker load` 新镜像）再 `docker compose up -d api`。

更新知识库：先 `docker compose stop api`，rsync 新的 `example/a/DB`，再 `docker compose start api`。不要在 API 跑着的时候覆盖 `main.db`。

---

## 7. 常见错误

| 你看到的 | 原因 | 怎么办 |
|---|---|---|
| `AUTH_PASSWORD` 相关 compose 报错 | 没 `.env` 或没填密码 | 根目录放 `.env` |
| build 时 COPY 巨慢 / 磁盘暴涨 | 没走 `.dockerignore`，把 DB/PDF 打进镜像了 | 确认 `.dockerignore` 在项目根，且包含 `example/a/DB/`、`cache/` |
| `config_path` / `FileNotFoundError` | 没挂 `example/a` 或路径不对 | compose 必须在数据目录那一层启动 |
| web 一直 starting、api unhealthy | pin 索引还没完，或 DB 没挂上 | `docker compose logs api`，等 3 分钟 |
| 登录 502 | API 挂了，或旧隧道占着宿主机 8000（本方案 web 走容器网，一般不是这个） | `docker compose ps`；宿主机不要再跑 uvicorn |
| 登录成功但没答案 | 容器出网到不了推理机 | 在 api 容器里 curl 8030/9998 |
| OOMKilled | 机器不足 32 GB，或又开了一份 API | `docker inspect` 看 OOM；workers 必须为 1 |

---

## 8. 不要做的事

1. 不要把 `example/a/DB`、`cache/`、PDF 写进 Dockerfile 的 `COPY`。
2. 不要把 `--workers` 改成大于 1。
3. 不要把 `AUTH_PASSWORD` 写进 Dockerfile 或打进镜像层。
4. 不要在云上同时跑「Docker 里的 API」和「宿主机 uvicorn / autossh」。
5. 不要用内存小于 32 GB 的云主机硬上。
