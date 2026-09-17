"""后台账号管理。不能在网页上注册，只在服务器上加用户。

    python -m api.accounts list
    python -m api.accounts add 用户名 密码
    python -m api.accounts passwd 用户名 新密码
    python -m api.accounts disable 用户名
    python -m api.accounts enable 用户名
"""
from __future__ import annotations

import argparse
import sys

from api.store import (
    create_user,
    db_path,
    init_db,
    list_users,
    set_disabled,
    set_password,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m api.accounts",
        description="化工问答网站账号（SQLite）。网页不能注册。",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出全部账号")

    p_add = sub.add_parser("add", help="新增账号")
    p_add.add_argument("username")
    p_add.add_argument("password")
    p_add.add_argument("--admin", action="store_true")

    p_pw = sub.add_parser("passwd", help="改密码（会踢掉该用户已登录会话）")
    p_pw.add_argument("username")
    p_pw.add_argument("password")

    p_off = sub.add_parser("disable", help="停用账号")
    p_off.add_argument("username")
    p_on = sub.add_parser("enable", help="启用账号")
    p_on.add_argument("username")

    args = parser.parse_args(argv)
    init_db()
    print(f"db: {db_path()}", file=sys.stderr)

    try:
        if args.cmd == "list":
            users = list_users()
            if not users:
                print("(empty)")
                return 0
            for u in users:
                flag = []
                if u["is_admin"]:
                    flag.append("admin")
                if u["disabled"]:
                    flag.append("disabled")
                extra = f" ({', '.join(flag)})" if flag else ""
                print(f"{u['id']}\t{u['username']}{extra}\t{u['created_at']}")
            return 0
        if args.cmd == "add":
            u = create_user(args.username, args.password, is_admin=args.admin)
            print(f"created {u['username']} id={u['id']}")
            return 0
        if args.cmd == "passwd":
            set_password(args.username, args.password)
            print(f"password updated for {args.username}")
            return 0
        if args.cmd == "disable":
            set_disabled(args.username, True)
            print(f"disabled {args.username}")
            return 0
        if args.cmd == "enable":
            set_disabled(args.username, False)
            print(f"enabled {args.username}")
            return 0
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
