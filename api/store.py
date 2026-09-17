"""SQLite store for web accounts, login tokens, and per-user conversations.

Accounts cannot see each other's conversations. The knowledge-graph DB is
untouched; this file is only the website's user/session data.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "chemical-rag-web" / "data" / "app.db"
CONFIG_JSON = ROOT / "chemical-rag-web" / "config.json"

PBKDF2_ITERS = 210_000
TOKEN_DAYS = 30

_lock = threading.Lock()
_db_path: Optional[Path] = None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s: str) -> datetime:
    s = (s or "").replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERS
    )
    return "pbkdf2$sha256$%d$%s$%s" % (
        PBKDF2_ITERS,
        salt.hex(),
        digest.hex(),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        kind, algo, iters_s, salt_hex, hash_hex = stored.split("$")
        if kind != "pbkdf2" or algo != "sha256":
            return False
        iters = int(iters_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        got = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
        return hmac.compare_digest(got, expected)
    except Exception:
        return False


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def db_path() -> Path:
    raw = os.environ.get("CGR_WEB_DB", "").strip()
    path = Path(raw) if raw else DEFAULT_DB
    if not path.is_absolute():
        path = ROOT / path
    return path


def connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=8, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db() -> Path:
    global _db_path
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        conn = connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                  password_hash TEXT NOT NULL,
                  is_admin INTEGER NOT NULL DEFAULT 0,
                  disabled INTEGER NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS auth_tokens (
                  token_hash TEXT PRIMARY KEY,
                  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                  created_at TEXT NOT NULL,
                  expires_at TEXT NOT NULL,
                  last_used_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                  id TEXT PRIMARY KEY,
                  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                  title TEXT NOT NULL,
                  title_is_manual INTEGER NOT NULL DEFAULT 0,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                  turn_index INTEGER NOT NULL,
                  query TEXT NOT NULL,
                  mode TEXT NOT NULL,
                  answer TEXT NOT NULL DEFAULT '',
                  status INTEGER NOT NULL DEFAULT 0,
                  latency_s REAL,
                  sources_json TEXT NOT NULL DEFAULT '[]',
                  result_json TEXT NOT NULL DEFAULT '{}',
                  stream_steps_json TEXT NOT NULL DEFAULT '[]',
                  created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conv_user_updated
                  ON conversations(user_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_turns_conv
                  ON turns(conversation_id, turn_index);
                CREATE INDEX IF NOT EXISTS idx_tokens_user
                  ON auth_tokens(user_id);
                """
            )
            conn.commit()
        finally:
            conn.close()
        _seed_admin()
    _db_path = path
    return path


def _seed_admin() -> None:
    conn = connect()
    try:
        n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if n:
            return
        username, password = "admin", "kaiyin"
        if CONFIG_JSON.is_file():
            try:
                cfg = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
                username = str(cfg.get("username") or username).strip() or username
                password = str(cfg.get("password") or password)
            except Exception:
                pass
        conn.execute(
            "INSERT INTO users(username, password_hash, is_admin, disabled, created_at) "
            "VALUES (?, ?, 1, 0, ?)",
            (username, hash_password(password), _now()),
        )
        conn.commit()
    finally:
        conn.close()


def _user_row(row: sqlite3.Row | None) -> Optional[dict]:
    if row is None:
        return None
    return {
        "id": int(row["id"]),
        "username": row["username"],
        "is_admin": bool(row["is_admin"]),
        "disabled": bool(row["disabled"]),
        "created_at": row["created_at"],
    }


def create_user(username: str, password: str, *, is_admin: bool = False) -> dict:
    username = (username or "").strip()
    if not username:
        raise ValueError("用户名不能为空")
    if ":" in username or "/" in username or " " in username:
        raise ValueError("用户名不能包含空格、冒号或斜杠")
    if len(username) > 64:
        raise ValueError("用户名过长")
    if not password:
        raise ValueError("密码不能为空")
    init_db()
    with _lock:
        conn = connect()
        try:
            conn.execute(
                "INSERT INTO users(username, password_hash, is_admin, disabled, created_at) "
                "VALUES (?, ?, ?, 0, ?)",
                (username, hash_password(password), 1 if is_admin else 0, _now()),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
            return _user_row(row)
        except sqlite3.IntegrityError as e:
            raise ValueError(f"用户已存在：{username}") from e
        finally:
            conn.close()


def list_users() -> list[dict]:
    init_db()
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT id, username, is_admin, disabled, created_at FROM users ORDER BY id"
        ).fetchall()
        return [_user_row(r) for r in rows]
    finally:
        conn.close()


def get_user_by_name(username: str) -> Optional[dict]:
    init_db()
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
        return _user_row(row)
    finally:
        conn.close()


def set_password(username: str, password: str) -> None:
    if not password:
        raise ValueError("密码不能为空")
    init_db()
    with _lock:
        conn = connect()
        try:
            cur = conn.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (hash_password(password), username.strip()),
            )
            if cur.rowcount == 0:
                raise ValueError(f"没有这个用户：{username}")
            conn.commit()
        finally:
            conn.close()
    revoke_user_tokens(username)


def set_disabled(username: str, disabled: bool) -> None:
    init_db()
    with _lock:
        conn = connect()
        try:
            cur = conn.execute(
                "UPDATE users SET disabled = ? WHERE username = ?",
                (1 if disabled else 0, username.strip()),
            )
            if cur.rowcount == 0:
                raise ValueError(f"没有这个用户：{username}")
            conn.commit()
        finally:
            conn.close()
    if disabled:
        revoke_user_tokens(username)


def authenticate(username: str, password: str) -> Optional[dict]:
    init_db()
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
        if row is None or row["disabled"]:
            return None
        if not verify_password(password, row["password_hash"]):
            return None
        return _user_row(row)
    finally:
        conn.close()


def issue_token(user_id: int) -> str:
    init_db()
    token = uuid.uuid4().hex + uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    exp = now + timedelta(days=TOKEN_DAYS)
    with _lock:
        conn = connect()
        try:
            conn.execute(
                "INSERT INTO auth_tokens(token_hash, user_id, created_at, expires_at, last_used_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    hash_token(token),
                    int(user_id),
                    now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    exp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    return token


def user_from_token(token: str) -> Optional[dict]:
    if not token:
        return None
    init_db()
    conn = connect()
    try:
        row = conn.execute(
            "SELECT u.* FROM auth_tokens t JOIN users u ON u.id = t.user_id "
            "WHERE t.token_hash = ?",
            (hash_token(token),),
        ).fetchone()
        if row is None or row["disabled"]:
            return None
        meta = conn.execute(
            "SELECT expires_at FROM auth_tokens WHERE token_hash = ?",
            (hash_token(token),),
        ).fetchone()
        if meta is None:
            return None
        if _parse_iso(meta["expires_at"]) < datetime.now(timezone.utc):
            conn.execute(
                "DELETE FROM auth_tokens WHERE token_hash = ?", (hash_token(token),)
            )
            conn.commit()
            return None
        conn.execute(
            "UPDATE auth_tokens SET last_used_at = ? WHERE token_hash = ?",
            (_now(), hash_token(token)),
        )
        conn.commit()
        return _user_row(row)
    finally:
        conn.close()


def revoke_token(token: str) -> None:
    init_db()
    with _lock:
        conn = connect()
        try:
            conn.execute(
                "DELETE FROM auth_tokens WHERE token_hash = ?", (hash_token(token),)
            )
            conn.commit()
        finally:
            conn.close()


def revoke_user_tokens(username: str) -> None:
    init_db()
    with _lock:
        conn = connect()
        try:
            conn.execute(
                "DELETE FROM auth_tokens WHERE user_id IN "
                "(SELECT id FROM users WHERE username = ?)",
                (username.strip(),),
            )
            conn.commit()
        finally:
            conn.close()


def _owned_conversation(conn: sqlite3.Connection, conv_id: str, user_id: int):
    return conn.execute(
        "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
        (conv_id, user_id),
    ).fetchone()


def create_conversation(user_id: int, title: str) -> dict:
    init_db()
    title = (title or "").strip() or "新对话"
    cid = str(uuid.uuid4())
    now = _now()
    with _lock:
        conn = connect()
        try:
            conn.execute(
                "INSERT INTO conversations(id, user_id, title, title_is_manual, created_at, updated_at) "
                "VALUES (?, ?, ?, 0, ?, ?)",
                (cid, int(user_id), title[:80], now, now),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
            return _conv_row(conn, row)
        finally:
            conn.close()


def list_conversations(user_id: int, q: str = "", limit: int = 200) -> list[dict]:
    init_db()
    conn = connect()
    try:
        sql = (
            "SELECT c.*, "
            "(SELECT COUNT(*) FROM turns t WHERE t.conversation_id = c.id) AS turn_count, "
            "(SELECT query FROM turns t WHERE t.conversation_id = c.id "
            " ORDER BY t.turn_index DESC LIMIT 1) AS last_query "
            "FROM conversations c WHERE c.user_id = ?"
        )
        args: list[Any] = [int(user_id)]
        q = (q or "").strip()
        if q:
            sql += " AND (c.title LIKE ? OR EXISTS (SELECT 1 FROM turns t WHERE t.conversation_id = c.id AND t.query LIKE ?))"
            like = f"%{q}%"
            args.extend([like, like])
        sql += " ORDER BY c.updated_at DESC LIMIT ?"
        args.append(int(limit))
        rows = conn.execute(sql, args).fetchall()
        out = []
        for r in rows:
            item = _conv_row(conn, r, extra=True)
            out.append(item)
        return out
    finally:
        conn.close()


def get_conversation(user_id: int, conv_id: str) -> Optional[dict]:
    init_db()
    conn = connect()
    try:
        row = _owned_conversation(conn, conv_id, int(user_id))
        if row is None:
            return None
        data = _conv_row(conn, row, extra=True)
        data["turns"] = [
            _turn_row(t)
            for t in conn.execute(
                "SELECT * FROM turns WHERE conversation_id = ? AND user_id = ? "
                "ORDER BY turn_index ASC",
                (conv_id, int(user_id)),
            ).fetchall()
        ]
        return data
    finally:
        conn.close()


def rename_conversation(user_id: int, conv_id: str, title: str) -> Optional[dict]:
    title = (title or "").strip()
    if not title:
        raise ValueError("标题不能为空")
    init_db()
    with _lock:
        conn = connect()
        try:
            row = _owned_conversation(conn, conv_id, int(user_id))
            if row is None:
                return None
            conn.execute(
                "UPDATE conversations SET title = ?, title_is_manual = 1, updated_at = ? "
                "WHERE id = ? AND user_id = ?",
                (title[:80], _now(), conv_id, int(user_id)),
            )
            conn.commit()
            row = _owned_conversation(conn, conv_id, int(user_id))
            return _conv_row(conn, row, extra=True)
        finally:
            conn.close()


def delete_conversation(user_id: int, conv_id: str) -> bool:
    init_db()
    with _lock:
        conn = connect()
        try:
            row = _owned_conversation(conn, conv_id, int(user_id))
            if row is None:
                return False
            conn.execute(
                "DELETE FROM turns WHERE conversation_id = ? AND user_id = ?",
                (conv_id, int(user_id)),
            )
            conn.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conv_id, int(user_id)),
            )
            conn.commit()
            return True
        finally:
            conn.close()


def add_turn(
    user_id: int,
    conv_id: str,
    *,
    query: str,
    mode: str,
    answer: str = "",
    status: int = 0,
    latency_s: Optional[float] = None,
    sources: Optional[list] = None,
    result: Optional[dict] = None,
    stream_steps: Optional[list] = None,
) -> Optional[dict]:
    init_db()
    with _lock:
        conn = connect()
        try:
            row = _owned_conversation(conn, conv_id, int(user_id))
            if row is None:
                return None
            n = conn.execute(
                "SELECT COALESCE(MAX(turn_index), 0) FROM turns WHERE conversation_id = ?",
                (conv_id,),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO turns(conversation_id, user_id, turn_index, query, mode, answer, "
                "status, latency_s, sources_json, result_json, stream_steps_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    conv_id,
                    int(user_id),
                    int(n) + 1,
                    query,
                    mode,
                    answer or "",
                    int(status),
                    latency_s,
                    json.dumps(sources or [], ensure_ascii=False),
                    json.dumps(result or {}, ensure_ascii=False),
                    json.dumps(stream_steps or [], ensure_ascii=False),
                    _now(),
                ),
            )
            title = row["title"]
            if not row["title_is_manual"] and (not title or title == "新对话"):
                title = (query or "").strip().replace("\n", " ")[:40] or "新对话"
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (title, _now(), conv_id, int(user_id)),
            )
            conn.commit()
            turn = conn.execute(
                "SELECT * FROM turns WHERE conversation_id = ? ORDER BY id DESC LIMIT 1",
                (conv_id,),
            ).fetchone()
            return _turn_row(turn)
        finally:
            conn.close()


def _conv_row(conn: sqlite3.Connection, row: sqlite3.Row, extra: bool = False) -> dict:
    data = {
        "id": row["id"],
        "title": row["title"],
        "title_is_manual": bool(row["title_is_manual"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    if extra:
        keys = row.keys()
        data["turn_count"] = int(row["turn_count"]) if "turn_count" in keys else 0
        last = row["last_query"] if "last_query" in keys else None
        if last is None:
            last_row = conn.execute(
                "SELECT query FROM turns WHERE conversation_id = ? ORDER BY turn_index DESC LIMIT 1",
                (row["id"],),
            ).fetchone()
            last = last_row["query"] if last_row else ""
            cnt = conn.execute(
                "SELECT COUNT(*) FROM turns WHERE conversation_id = ?", (row["id"],)
            ).fetchone()[0]
            data["turn_count"] = int(cnt)
        data["last_query"] = last or ""
    return data


def _turn_row(row: sqlite3.Row) -> dict:
    def _load(raw, default):
        try:
            return json.loads(raw) if raw else default
        except Exception:
            return default

    return {
        "id": int(row["id"]),
        "turn_index": int(row["turn_index"]),
        "query": row["query"],
        "mode": row["mode"],
        "answer": row["answer"],
        "status": int(row["status"]),
        "latency_s": row["latency_s"],
        "sources": _load(row["sources_json"], []),
        "result": _load(row["result_json"], {}),
        "stream_steps": _load(row["stream_steps_json"], []),
        "created_at": row["created_at"],
    }
