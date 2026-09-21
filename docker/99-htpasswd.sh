#!/bin/sh
# nginx 官方镜像会在启动时执行 /docker-entrypoint.d/*.sh
set -eu
if [ -z "${AUTH_PASSWORD:-}" ]; then
    echo "AUTH_PASSWORD empty; skip htpasswd (API 自己做登录)" >&2
    exit 0
fi
DIGEST=$(printf '%s' "$AUTH_PASSWORD" | openssl dgst -binary -sha1 | openssl base64 -A)
printf '%s:{SHA}%s\n' "${AUTH_USER:-admin}" "$DIGEST" > /etc/nginx/chemical-rag.htpasswd
chmod 644 /etc/nginx/chemical-rag.htpasswd
echo "htpasswd ready user=${AUTH_USER:-admin}"
