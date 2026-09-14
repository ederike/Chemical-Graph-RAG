# 问答 API。知识库 example/a/DB 用 volume 挂进来，不要 COPY。
FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DHMF_CONFIG=example/a/config_open.yaml \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --timeout 120 -r requirements.txt

COPY api/ ./api/
COPY src/ ./src/
COPY main.py ./

EXPOSE 8000

# pin 索引约 40–60s，start-period 给足时间
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
