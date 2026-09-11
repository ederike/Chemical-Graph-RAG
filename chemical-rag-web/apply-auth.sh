#!/bin/bash
# 修改 config.json 里的 username / password 后执行本脚本，凭据立即生效。
#   bash /root/chemical-rag-web/apply-auth.sh
set -euo pipefail
ROOT=/root/chemical-rag-web
WEB_DST=/var/www/chemical-rag
HTPASSWD=/etc/nginx/chemical-rag.htpasswd
CFG="$ROOT/config.json"

if [[ ! -f "$CFG" ]]; then
  echo "missing $CFG" >&2
  exit 1
fi

python3 - "$CFG" "$HTPASSWD" <<'PY'
import json, hashlib, base64, sys, pathlib
cfg_path, out_path = sys.argv[1], sys.argv[2]
cfg = json.loads(pathlib.Path(cfg_path).read_text(encoding="utf-8"))
user = str(cfg.get("username") or "").strip()
pw = str(cfg.get("password") or "")
if not user or not pw:
    raise SystemExit("config.json 里 username / password 不能为空")
if ":" in user:
    raise SystemExit("username 不能包含冒号")
digest = base64.b64encode(hashlib.sha1(pw.encode("utf-8")).digest()).decode("ascii")
pathlib.Path(out_path).write_text(f"{user}:{{SHA}}{digest}\n", encoding="utf-8")
print(f"updated {out_path} (user={user})")
PY

chown root:www-data "$HTPASSWD"
chmod 640 "$HTPASSWD"
chmod 600 "$CFG"

mkdir -p "$WEB_DST"
rm -rf "$WEB_DST"/*
cp -a "$ROOT/web/." "$WEB_DST/"
chown -R root:www-data "$WEB_DST"
find "$WEB_DST" -type d -exec chmod 755 {} \;
find "$WEB_DST" -type f -exec chmod 644 {} \;

nginx -t
systemctl reload nginx
echo "nginx reloaded. 静态页与登录凭据已生效。"
